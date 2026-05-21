# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
"""GraphAtlasLatentForecaster — probabilistic predictive training in latent space.

Trains :class:`anemoi.models.models.latent_dit_wrapper.AnemoiLatentDiTModel`
with FGN-style noise injection. The model produces a latent residual
``r_lat`` from ``(z_t, z_{t-1}, ξ)``; the loss is fair-CRPS in LATENT
space (NOT full-res) with optional spectral-CRPS reg.

Variant B normalization (locked 2026-05-21):

  * Model emits LATENT residuals in MEAN-STD space (NOT divided by
    σ_tend).
  * AR feedback runs in mean-std space: ``z_{t+1} = z_t + r_lat``. No
    denormalize-then-renormalize round-trip per step.
  * Tendency normalization lives in the LOSS via
    :class:`anemoi.training.losses.scalers.LatentVarTendencyScaler` — per
    channel the L1/CRPS is divided by σ_lat_tend (latent-space
    tendency stdev pre-computed by
    ``grafai/datasets/compute_latent_tendency_stats.py`` and appended to
    the training zarr as
    ``statistics_tendencies_<freqstr>_latent_stdev``).

Forcings are bilinear-encoded to the latent grid and fed to the
predictive model as conditioning (HGT, land/sea, lat/lon, time-of-day) —
locked decision 2026-05-21.

Decoder is NOT involved at training time: the latent task trains the
predictive model alone, with truth latent residuals as the CRPS target.
At inference the trained predictive feeds its samples to the trained
decoder via :class:`AnemoiAtlasModel`.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import einops
import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.layers.bilinear_encoder import bilinear_downsample
from anemoi.models.models.latent_dit_wrapper import AnemoiLatentDiTModel
from anemoi.training.train.tasks.base import BaseGraphModule

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

    from anemoi.training.schemas.base_schema import BaseSchema

LOGGER = logging.getLogger(__name__)


class GraphAtlasLatentForecaster(BaseGraphModule):
    """Probabilistic latent-space rollout trainer for the v30 architecture.

    Inherits scaler / optimizer / loss / metric infrastructure from
    :class:`BaseGraphModule`. Override ``_step`` to run the Atlas latent
    pipeline (bilinear-encode → predictive forward with noise → CRPS in
    latent → mean-std AR feedback).

    Parameters
    ----------
    nens_per_device : int
        Number of ensemble members sampled per (batch, device). Atlas/FGN
        defaults: 2 (minimum for fair-CRPS bias correction).
    rollout : int
        AR rollout length (1 = single step; >1 = multi-step rollout with
        mean-std-space feedback).
    spectral_crps_weight : float
        Weight on the spectral-CRPS regularization term computed at the
        latent grid shape. 0.0 disables.
    crps_alpha : float
        Fair-CRPS blend parameter passed to the configured CRPS loss.
        1.0 = fully fair (Atlas); 0.95 = almost-fair (FGN); 0.0 = MAE.

    Notes
    -----
    The model resolution is the LATENT grid (e.g. 63×63), not full-res.
    The dataloader yields full-res samples; this task bilinear-encodes
    each frame on the fly so no pre-processing step is needed.
    """

    def __init__(
        self,
        *,
        config: "BaseSchema",
        graph_data: "HeteroData",
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: IndexCollection,
        metadata: dict,
        supporting_arrays: dict,
        nens_per_device: int = 2,
        rollout: int = 1,
        spectral_crps_weight: float = 0.0,
        crps_alpha: float = 1.0,
    ) -> None:
        super().__init__(
            config=config,
            graph_data=graph_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )
        inner = self.model.model if hasattr(self.model, "model") else self.model
        if not isinstance(inner, AnemoiLatentDiTModel):
            error = (
                f"GraphAtlasLatentForecaster requires AnemoiLatentDiTModel; "
                f"got {type(inner).__name__}. Check config.model._target_."
            )
            raise TypeError(error)
        self.predictive: AnemoiLatentDiTModel = inner

        self.multi_step = self.predictive.history_len
        self.rollout = rollout
        self.nens_per_device = nens_per_device
        self.spectral_crps_weight = spectral_crps_weight
        self.crps_alpha = crps_alpha
        self.latent_shape = tuple(self.predictive.latent_shape)
        self.noise_vector_dim = self.predictive.noise_vector_dim
        self.forcings_channels = self.predictive.forcings_channels
        LOGGER.info(
            "GraphAtlasLatentForecaster: latent=%s history=%d rollout=%d "
            "nens=%d noise_dim=%d forcings=%d crps_alpha=%.2f spec_w=%.3f",
            self.latent_shape, self.multi_step, self.rollout,
            self.nens_per_device, self.noise_vector_dim, self.forcings_channels,
            self.crps_alpha, self.spectral_crps_weight,
        )

    # ------------------------------------------------------------------
    # Encoding helpers (no learned params; bilinear on whatever device).
    # ------------------------------------------------------------------
    def _encode_prog_and_forcings(
        self,
        x_full: torch.Tensor,
        n_prog: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Split x into prognostic + forcings and bilinear-encode each.

        Returns ``(z, forcings_latent)`` where ``forcings_latent`` is
        None when the model is configured forcings-less.
        """
        prog = x_full[:, :n_prog]
        z = bilinear_downsample(prog, target_shape=self.latent_shape)
        f_lat = None
        if self.forcings_channels > 0:
            if x_full.shape[1] < n_prog + self.forcings_channels:
                error = (
                    f"x has {x_full.shape[1]} channels; expected at least "
                    f"{n_prog + self.forcings_channels}."
                )
                raise ValueError(error)
            f_phys = x_full[:, n_prog : n_prog + self.forcings_channels]
            f_lat = bilinear_downsample(f_phys, target_shape=self.latent_shape)
        return z, f_lat

    # ------------------------------------------------------------------
    # Core: latent CRPS rollout step.
    # ------------------------------------------------------------------
    def _atlas_latent_step(
        self,
        x_hist: torch.Tensor,
        x_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the latent CRPS rollout.

        Parameters
        ----------
        x_hist : torch.Tensor
            ``(B, history_len, C_in, H, W)`` full-res history (prog + forcings).
            history_len == self.multi_step (Atlas: 2).
        x_targets : torch.Tensor
            ``(B, rollout, C_prog, H, W)`` ground-truth prognostic states at
            rollout steps t+1, t+2, ..., t+rollout.

        Returns
        -------
        loss : torch.Tensor
            Scalar training loss = CRPS_latent + spectral_crps_weight * CRPS_spec_latent.
        r_pred_list : list[torch.Tensor]
            Per-rollout-step predicted latent residuals (ensemble),
            shape ``(B, E, C_prog, h_lat, w_lat)`` each. Useful for metrics.
        """
        B = x_hist.shape[0]
        E = self.nens_per_device
        n_prog = x_targets.shape[2]

        # Encode the history: z_hist[i] = B(x_hist[:, i, :prog])
        z_hist = []
        f_hist = []
        for i in range(self.multi_step):
            z_i, f_i = self._encode_prog_and_forcings(x_hist[:, i], n_prog)
            z_hist.append(z_i)
            f_hist.append(f_i)
        z_curr_init = z_hist[-1]                     # z_t  (B, C_prog, h, w)
        z_prev_init = z_hist[-2] if self.multi_step >= 2 else z_hist[-1]
        f_curr_init = f_hist[-1]
        f_prev_init = f_hist[-2] if self.multi_step >= 2 else f_hist[-1]

        # Replicate the history across ensemble members and fold E into batch
        # so the predictive model sees (B*E, ...) for a single forward pass.
        def _expand(t):
            if t is None:
                return None
            return t.unsqueeze(1).expand(B, E, *t.shape[1:]).reshape(B * E, *t.shape[1:])

        z_curr = _expand(z_curr_init)
        z_prev = _expand(z_prev_init)
        f_curr = _expand(f_curr_init)
        f_prev = _expand(f_prev_init)

        # Latent ensemble state for AR feedback (Variant B: stays in mean-std).
        r_pred_list: list[torch.Tensor] = []
        loss = z_curr.new_zeros(())
        for step in range(self.rollout):
            # Sample a fresh noise vector per (batch, member).
            if self.noise_vector_dim > 0:
                noise_flat = torch.randn(
                    B * E, self.noise_vector_dim,
                    device=z_curr.device, dtype=z_curr.dtype,
                )
            else:
                noise_flat = None

            # Predictive forward (already in folded (B*E, ...) form).
            r_lat_flat = self.predictive(
                z_curr, z_prev, noise_flat,
                forcings_curr=f_curr, forcings_prev=f_prev,
            )                                        # (B*E, C_prog, h, w)

            # Truth latent residual for this step.
            z_target = bilinear_downsample(
                x_targets[:, step], target_shape=self.latent_shape,
            )                                        # (B, C_prog, h, w)

            # CRPS in latent space: re-fold predicted to (B, E, C, h, w),
            # truth has a singleton ensemble dim. The configured loss
            # (GraphCastCRPSLoss with LatentVarTendencyScaler + optional
            # spectral CRPS) does the per-channel scaling and GraphCast-style
            # reduction internally.
            r_pred = einops.rearrange(r_lat_flat, "(b e) c h w -> b e c h w", b=B, e=E)
            r_pred_list.append(r_pred)

            # Flatten spatial into "cell" for the standard scaler/loss path,
            # mirroring how other tasks present their (B, E, cell, V) tensor.
            r_pred_flat_cell = einops.rearrange(r_pred, "b e c h w -> b e (h w) c")
            r_true_unnorm = z_target - z_curr_init   # (B, C, h, w) — truth diff
            r_true_flat_cell = einops.rearrange(r_true_unnorm, "b c h w -> b 1 (h w) c")

            step_loss = self.loss(
                r_pred_flat_cell, r_true_flat_cell, squash=True,
            )
            loss = loss + step_loss

            # AR feedback: roll history and replace last-step latent.
            # Variant B: z_next = z_t + r_lat in mean-std space.
            z_prev = z_curr
            z_curr = z_curr + r_lat_flat            # in-place residual update

            # Forcings advance: bilinear-encode forcings at the new
            # target time so the next predictive forward sees the right
            # time-of-day / solar conditioning. This requires the
            # dataloader to provide forcings at t+step+1, which is
            # standard in the multistep batch.
            # TODO(wire-from-dataloader): pull forcings at t+step+1 from
            # the batch tensor rather than re-using the current ones.
            f_prev = f_curr

        # Mean over rollout steps so the loss magnitude is comparable
        # across different rollout lengths.
        loss = loss / max(self.rollout, 1)
        return loss, r_pred_list

    # ------------------------------------------------------------------
    # Anemoi task hook.
    # ------------------------------------------------------------------
    def _step(
        self,
        batch: torch.Tensor,
        batch_idx: int,                              # noqa: ARG002
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, dict, list]:
        """Compute one batch's CRPS-latent loss.

        Batch layout: ``(B, T, ensemble=1, cell, V)`` with
        ``T == multi_step + rollout``. We reshape ``cell -> (H, W)``
        from ``data_indices.data.input.field_shape``.

        TODO(verify): field_shape attribute name. Same caveat as
        GraphAtlasDecoderForecaster.
        """
        # TODO: replace with the correct field_shape access path once we
        # smoke against the real IndexCollection.
        field_shape = self.data_indices.data.input.field_shape
        H, W = field_shape

        # Slice history + targets.
        x_hist = batch[:, :self.multi_step, 0, :, self.data_indices.data.input.full]
        x_targets = batch[
            :, self.multi_step : self.multi_step + self.rollout, 0,
            :, self.data_indices.data.input.prognostic,
        ]

        B = x_hist.shape[0]
        # Reshape (B, T, cell, V) -> (B, T, V, H, W) for spatial ops.
        x_hist = x_hist.permute(0, 1, 3, 2).reshape(B, self.multi_step, -1, H, W).contiguous()
        x_targets = x_targets.permute(0, 1, 3, 2).reshape(B, self.rollout, -1, H, W).contiguous()

        loss, r_pred_list = self._atlas_latent_step(x_hist, x_targets)

        metrics: dict = {}
        return loss, metrics, r_pred_list

    # ------------------------------------------------------------------
    # Lightning hooks.
    # ------------------------------------------------------------------
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss, _metrics, _pred = self._step(batch, batch_idx, validation_mode=False)
        self.log("train_atlas_latent_crps", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss, _metrics, _pred = self._step(batch, batch_idx, validation_mode=True)
        self.log("val_atlas_latent_crps", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss
