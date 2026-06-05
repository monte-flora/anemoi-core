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
from anemoi.models.preprocessing.latent_residual_normalizer import LatentResidualNormalizer
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
        full_res_shape: tuple[int, int] = (250, 250),
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

        # Honor config-level overrides for these knobs. The training framework
        # only passes 7 kwargs to model_task(**kwargs) (train.py:202), so
        # nens/rollout/crps_alpha/spec_w default unless we pull them from
        # config.training here. Discovered 2026-05-24 — v30b silently ran with
        # function defaults instead of YAML values.
        train_cfg = self.config.training
        if hasattr(train_cfg, "ensemble_size_per_device") and train_cfg.ensemble_size_per_device is not None:
            nens_per_device = int(train_cfg.ensemble_size_per_device)
        rollout_cfg = getattr(train_cfg, "rollout", None)
        if rollout_cfg is not None and hasattr(rollout_cfg, "start") and rollout_cfg.start is not None:
            rollout = int(rollout_cfg.start)
        loss_cfg = getattr(train_cfg, "training_loss", None)
        if loss_cfg is not None:
            if hasattr(loss_cfg, "alpha") and loss_cfg.alpha is not None:
                crps_alpha = float(loss_cfg.alpha)
            if hasattr(loss_cfg, "spectral_crps_weight") and loss_cfg.spectral_crps_weight is not None:
                spectral_crps_weight = float(loss_cfg.spectral_crps_weight)

        self.multi_step = self.predictive.history_len
        self.rollout = rollout
        self.nens_per_device = nens_per_device
        self.spectral_crps_weight = spectral_crps_weight
        self.crps_alpha = crps_alpha
        self.latent_shape = tuple(self.predictive.latent_shape)
        self.noise_vector_dim = self.predictive.noise_vector_dim
        self.forcings_channels = self.predictive.forcings_channels
        # Full-res spatial shape of the dataloader's batch (cell -> H × W reshape).
        # TODO: derive this from metadata['field_shape'] once we confirm the path.
        self.full_res_shape = tuple(full_res_shape)

        # Optional tendency-normalized residual training (v17 ResidualNormalizer
        # ported to latent grid). v30b/v30b-det without this collapsed to per-
        # variable shortcuts in BOTH MSE and CRPS recipes; v17's tendency-norm
        # trick prevents that. Discovered 2026-05-24.
        #
        # Config field: training.latent_residual_normalizer.latent_stats_path
        # When set: the predictive model is TRAINED to output tendency-normalized
        # latent residuals (every channel target ~O(1)). At inference time the
        # AnemoiAtlasModel composer denormalizes before passing to the decoder
        # (which was trained on mean-std residuals).
        self.latent_residual_normalizer: Optional[LatentResidualNormalizer] = None
        norm_cfg = getattr(train_cfg, "latent_residual_normalizer", None)
        if norm_cfg is not None and getattr(norm_cfg, "latent_stats_path", None):
            prog_idx_list = (
                data_indices.model.input.prognostic.tolist()
                if hasattr(data_indices.model.input.prognostic, "tolist")
                else list(data_indices.model.input.prognostic)
            )
            input_idx_to_name = {idx: name for name, idx in data_indices.data.input.name_to_index.items()}
            prog_channel_names = [input_idx_to_name[int(i)] for i in prog_idx_list]
            self.latent_residual_normalizer = LatentResidualNormalizer.from_zarr(
                latent_stats_path=str(norm_cfg.latent_stats_path),
                prog_channel_names=prog_channel_names,
                latent_stats_key=getattr(norm_cfg, "latent_stats_key", None),
                min_stdev=float(getattr(norm_cfg, "min_stdev", 1e-7)),
            )
            # Attach to the inner predictive model so the buffer travels with
            # the inference-only checkpoint and AnemoiAtlasModel can find it.
            self.predictive.latent_residual_normalizer = self.latent_residual_normalizer

        LOGGER.info(
            "GraphAtlasLatentForecaster: latent=%s history=%d rollout=%d "
            "nens=%d noise_dim=%d forcings=%d crps_alpha=%.2f spec_w=%.3f "
            "tendency_norm=%s",
            self.latent_shape, self.multi_step, self.rollout,
            self.nens_per_device, self.noise_vector_dim, self.forcings_channels,
            self.crps_alpha, self.spectral_crps_weight,
            "ON" if self.latent_residual_normalizer is not None else "OFF",
        )

    # ------------------------------------------------------------------
    # Encoding helpers (no learned params; bilinear on whatever device).
    # ------------------------------------------------------------------
    def _encode_prog_and_forcings(
        self,
        x_prog: torch.Tensor,
        x_forcings: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Bilinear-encode prog + forcings tensors (already pre-sliced).

        Earlier this method positional-sliced ``x_full[:, :n_prog]`` and
        ``x_full[:, n_prog:n_prog+forcings_channels]`` — silently picking
        the wrong channels for the ufs2arco-built dataset where forcings
        are interleaved (e.g. cos_latitude at index 2, land_sea_mask at
        index 7). Callers now do the slicing with ``data_indices`` lists
        and pass pre-separated tensors. See
        ``[[feedback-smoke-correctness-gap]]``.

        Returns ``(z, forcings_latent)`` where ``forcings_latent`` is
        None when the model is configured forcings-less or x_forcings is None.
        """
        z = bilinear_downsample(x_prog, target_shape=self.latent_shape)
        f_lat = None
        if self.forcings_channels > 0 and x_forcings is not None:
            f_lat = bilinear_downsample(x_forcings, target_shape=self.latent_shape)
        return z, f_lat

    # ------------------------------------------------------------------
    # Core: latent CRPS rollout step.
    # ------------------------------------------------------------------
    def _atlas_latent_step(
        self,
        x_hist_prog: torch.Tensor,
        x_hist_forcings: Optional[torch.Tensor],
        x_targets_prog: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the latent CRPS rollout.

        Parameters
        ----------
        x_hist_prog : torch.Tensor
            ``(B, history_len, C_prog, H, W)`` PROGNOSTIC-only history
            (pre-sliced by ``data_indices.data.input.prognostic``).
        x_hist_forcings : torch.Tensor or None
            ``(B, history_len, C_forcings, H, W)`` FORCINGS-only history
            (pre-sliced by ``data_indices.data.input.forcing``). None if
            the predictive is configured forcings-less.
        x_targets_prog : torch.Tensor
            ``(B, rollout, C_prog, H, W)`` ground-truth prognostic states
            at rollout steps t+1, t+2, ..., t+rollout.

        Returns
        -------
        loss : torch.Tensor
            Scalar training loss = CRPS_latent + spectral_crps_weight * CRPS_spec_latent.
        r_pred_list : list[torch.Tensor]
            Per-rollout-step predicted latent residuals (ensemble),
            shape ``(B, E, C_prog, h_lat, w_lat)`` each. Useful for metrics.
        """
        B = x_hist_prog.shape[0]
        E = self.nens_per_device

        # Encode the history (pre-sliced prog + optional forcings).
        z_hist = []
        f_hist = []
        for i in range(self.multi_step):
            x_p = x_hist_prog[:, i]
            x_f = x_hist_forcings[:, i] if x_hist_forcings is not None else None
            z_i, f_i = self._encode_prog_and_forcings(x_p, x_f)
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
                x_targets_prog[:, step], target_shape=self.latent_shape,
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
            r_true_meanstd = z_target - z_curr_init   # (B, C, h, w) — truth diff in mean-std space
            # If tendency-norm training is active, transform the TARGET so the
            # model is trained to predict tendency-normalized residuals (v17
            # recipe). Model output stays as-is — its natural output is the
            # tendency-normalized prediction.
            if self.latent_residual_normalizer is not None:
                r_true_for_loss = self.latent_residual_normalizer.transform(r_true_meanstd)
            else:
                r_true_for_loss = r_true_meanstd
            r_true_flat_cell = einops.rearrange(r_true_for_loss, "b c h w -> b 1 (h w) c")

            # Diagnostic: dump per-channel min/max of the tensors going into
            # the loss when ANY of them has non-finite values. Identifies
            # which channel(s) are overflowing so we can localize the cause.
            if not (torch.isfinite(r_pred_flat_cell).all() and torch.isfinite(r_true_flat_cell).all()):
                LOGGER.error("=== NON-FINITE INPUT TO LOSS — per-channel diagnostic ===")
                with torch.no_grad():
                    pmin = r_pred_flat_cell.amin(dim=(0, 1, 2)).float()
                    pmax = r_pred_flat_cell.amax(dim=(0, 1, 2)).float()
                    tmin = r_true_flat_cell.amin(dim=(0, 1, 2)).float()
                    tmax = r_true_flat_cell.amax(dim=(0, 1, 2)).float()
                    pfin = torch.isfinite(r_pred_flat_cell).all(dim=(0, 1, 2))
                    tfin = torch.isfinite(r_true_flat_cell).all(dim=(0, 1, 2))
                names = list(self.data_indices.model.output.name_to_index.keys())
                for i in range(min(len(names), r_pred_flat_cell.shape[-1])):
                    nm = names[i] if i < len(names) else f"ch{i}"
                    LOGGER.error(
                        "  ch%3d %-25s  PRED[fin=%s] min=%+.3e max=%+.3e   TRUE[fin=%s] min=%+.3e max=%+.3e",
                        i, nm, pfin[i].item(), pmin[i].item(), pmax[i].item(),
                        tfin[i].item(), tmin[i].item(), tmax[i].item(),
                    )
                LOGGER.error("=== end diagnostic ===")

            step_loss = self.loss(
                r_pred_flat_cell, r_true_flat_cell, squash=True,
            )
            loss = loss + step_loss

            # AR feedback: roll history and replace last-step latent.
            # Variant B: z_next = z_t + r_lat in mean-std space.
            # If tendency-norm training is active, the model's output is
            # tendency-normalized; denormalize back to mean-std space before
            # adding to z_curr (which lives in mean-std space).
            z_prev = z_curr
            if self.latent_residual_normalizer is not None:
                r_lat_meanstd = self.latent_residual_normalizer.inverse_transform(r_lat_flat)
                z_curr = z_curr + r_lat_meanstd
            else:
                z_curr = z_curr + r_lat_flat

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
        # Full-res shape of the dataloader's batch (set at task init).
        H, W = self.full_res_shape

        # NOTE: `batch` is ALREADY normalized here — BaseGraphModule's
        # on_after_batch_transfer → _normalize_batch calls
        # self.model.pre_processors(batch) before _step runs.
        # See anemoi-core/training/src/anemoi/training/train/tasks/base.py:515.

        # Slice history + targets using data_indices lists (NOT positional)
        # — forcings and prognostics are interleaved in the input channel
        # order (ufs2arco sort_channels_by_levels=True). Mirrors the
        # pattern in GraphResidualForecaster:residualforecaster.py:75.
        prog_idx = self.data_indices.data.input.prognostic
        forcing_idx = self.data_indices.data.input.forcing

        x_hist_prog = batch[:, :self.multi_step, 0, :, prog_idx]              # (B, T, cell, C_prog)
        x_targets_prog = batch[
            :, self.multi_step : self.multi_step + self.rollout, 0, :, prog_idx,
        ]                                                                       # (B, R, cell, C_prog)

        if self.forcings_channels > 0:
            x_hist_forcings = batch[:, :self.multi_step, 0, :, forcing_idx]   # (B, T, cell, C_forcings)
        else:
            x_hist_forcings = None

        B = x_hist_prog.shape[0]
        # Reshape (B, T, cell, V) -> (B, T, V, H, W) for spatial ops.
        x_hist_prog = x_hist_prog.permute(0, 1, 3, 2).reshape(
            B, self.multi_step, -1, H, W,
        ).contiguous()
        x_targets_prog = x_targets_prog.permute(0, 1, 3, 2).reshape(
            B, self.rollout, -1, H, W,
        ).contiguous()
        if x_hist_forcings is not None:
            x_hist_forcings = x_hist_forcings.permute(0, 1, 3, 2).reshape(
                B, self.multi_step, -1, H, W,
            ).contiguous()

        # Smoke-time invariant check (cheap, runs every step).
        if x_hist_prog.shape[2] != self.predictive.in_channels:
            error = (
                f"x_hist_prog has {x_hist_prog.shape[2]} channels but "
                f"predictive.in_channels={self.predictive.in_channels}. "
                f"data_indices/config mismatch."
            )
            raise RuntimeError(error)
        if x_hist_forcings is not None and x_hist_forcings.shape[2] != self.forcings_channels:
            error = (
                f"x_hist_forcings has {x_hist_forcings.shape[2]} channels but "
                f"self.forcings_channels={self.forcings_channels}. "
                f"data_indices/config mismatch."
            )
            raise RuntimeError(error)

        loss, r_pred_list = self._atlas_latent_step(x_hist_prog, x_hist_forcings, x_targets_prog)

        # NaN guard — fail fast rather than silently train through NaN.
        if not torch.isfinite(loss):
            error = (
                f"Non-finite loss at step (validation_mode={validation_mode}): "
                f"{loss.item() if loss.numel()==1 else loss}. "
                "Check tendency-scaler division, forcings slicing, or input NaNs."
            )
            raise RuntimeError(error)

        metrics: dict = {}
        return loss, metrics, r_pred_list

    # ------------------------------------------------------------------
    # DDPEnsGroupStrategy hooks (called from the trainer setup path).
    # These just stash the comm-group info so downstream code can use it;
    # we don't currently dispatch ensemble-wise reductions internally, so
    # the storage is sufficient for the trainer to proceed.
    # ------------------------------------------------------------------
    def set_ens_comm_group(
        self,
        ens_comm_group,
        ens_comm_group_id: int,
        ens_comm_group_rank: int,
        ens_comm_num_groups: int,
        ens_comm_group_size: int,
    ) -> None:
        self.ens_comm_group = ens_comm_group
        self.ens_comm_group_id = ens_comm_group_id
        self.ens_comm_group_rank = ens_comm_group_rank
        self.ens_comm_num_groups = ens_comm_num_groups
        self.ens_comm_group_size = ens_comm_group_size

    def set_ens_comm_subgroup(
        self,
        ens_comm_subgroup,
        ens_comm_subgroup_id: int,
        ens_comm_subgroup_rank: int,
        ens_comm_subgroup_num_groups: int,
        ens_comm_subgroup_size: int,
    ) -> None:
        self.ens_comm_subgroup = ens_comm_subgroup
        self.ens_comm_subgroup_id = ens_comm_subgroup_id
        self.ens_comm_subgroup_rank = ens_comm_subgroup_rank
        self.ens_comm_subgroup_num_groups = ens_comm_subgroup_num_groups
        self.ens_comm_subgroup_size = ens_comm_subgroup_size

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
