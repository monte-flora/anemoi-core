# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
"""GraphAtlasDecoderForecaster — single-step decoder pre-training for v30 Atlas.

Trains :class:`anemoi.models.models.decoder_dit_wrapper.AnemoiDecoderDiTModel`
deterministically on (x_t, x_{t+1}) pairs, with no autoregressive rollout.
The decoder learns to recover a full-resolution residual conditioned on:

  * ``r_lat = B(x_{t+1, prog}) - B(x_{t, prog})``  — the TRUTH latent residual,
    produced by bilinear downsample of the prognostic slice. **No learned
    encoder.** No noise.
  * ``x_t`` — the full-resolution current state including forcings.

Loss: L1 (MAE) on the predicted full-resolution residual, scaled with the
existing GraphCast level-grouped reduction (``GraphCastBaseLoss``) so the
loss values are directly comparable to v17-era runs.

Variant B normalization (locked 2026-05-21):

  * Decoder emits residuals in MEAN-STD-NORMALIZED space (the Anemoi data-
    loader's natural representation). NO tendency normalization at the model
    boundary.
  * Tendency normalization lives in the LOSS as a scaler
    (``VarTendencyScaler`` divides each channel's L1 by σ_tend²).
  * No ResidualNormalizer pre-processor for this task.

Single-step (no rollout): the decoder is a deterministic function of (r_lat,
x_t). We never feed its output back as input. Decoder is frozen at inference
time and re-used by every latent predictive model that follows.

Output mode of the underlying decoder: REPLACES the standard EPD path.
``self.model`` is an :class:`AnemoiDecoderDiTModel` (NOT
``AnemoiModelInterface``), so this task overrides the typical model setup
inherited from ``BaseGraphModule`` accordingly.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.layers.bilinear_encoder import bilinear_downsample
from anemoi.models.models.decoder_dit_wrapper import AnemoiDecoderDiTModel
from anemoi.training.train.tasks.base import BaseGraphModule

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

    from anemoi.training.schemas.base_schema import BaseSchema

LOGGER = logging.getLogger(__name__)


class GraphAtlasDecoderForecaster(BaseGraphModule):
    """Single-step decoder pre-training for the Atlas-style architecture.

    Inherits checkpointing, optimizer, scaler, and metric infrastructure
    from :class:`BaseGraphModule`. Overrides ``_step`` to bypass the
    standard EPD forward path and instead run the Atlas decoder pipeline.

    Parameters
    ----------
    config, graph_data, statistics, statistics_tendencies, data_indices,
    metadata, supporting_arrays : see :class:`BaseGraphModule`.
        Standard Anemoi task kwargs.
    latent_shape : tuple[int, int]
        Spatial shape of the latent grid the decoder is conditioned on.
        Must match the predictive model's latent_shape at composed-
        inference time.
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

        # The Atlas decoder is constructed by AnemoiModelInterface via the
        # standard Hydra ``model._target_`` field. We expect that target to
        # resolve to AnemoiDecoderDiTModel.
        inner = self.model.model if hasattr(self.model, "model") else self.model
        if not isinstance(inner, AnemoiDecoderDiTModel):
            error = (
                f"GraphAtlasDecoderForecaster requires the underlying model to be "
                f"AnemoiDecoderDiTModel; got {type(inner).__name__}. Check "
                f"config.model._target_."
            )
            raise TypeError(error)
        self.decoder: AnemoiDecoderDiTModel = inner

        # Locked: single AR step.
        self.multi_step = 1
        self.rollout = 1
        self.latent_shape = tuple(self.decoder.latent_shape)
        LOGGER.info(
            "GraphAtlasDecoderForecaster: decoder %s; full_res=%s, latent=%s",
            type(self.decoder).__name__, self.decoder.full_res_shape, self.latent_shape,
        )

    # ------------------------------------------------------------------
    # Core single-step forward.
    # ------------------------------------------------------------------
    def _atlas_decoder_step(
        self,
        x_t_full: torch.Tensor,
        x_t_prog: torch.Tensor,
        x_tp1_prog: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One decoder forward pass.

        Parameters
        ----------
        x_t_full : torch.Tensor
            ``(B, C_in, H, W)`` full-res state with **prognostic AND forcings
            in their native interleaved order** (sliced by
            ``data_indices.data.input.full``). Passed to the decoder as
            full-res conditioning context.
        x_t_prog : torch.Tensor
            ``(B, C_prog, H, W)`` PROGNOSTIC-ONLY current state (sliced by
            ``data_indices.data.input.prognostic``). Used for bilinear
            encoding into latent space — forcings are excluded because the
            latent residual is computed in prognostic space only.
        x_tp1_prog : torch.Tensor
            ``(B, C_prog, H, W)`` PROGNOSTIC-ONLY next-step state.

        Returns
        -------
        delta_pred, delta_truth : (B, C_prog, H, W) each.

        Notes
        -----
        Earlier versions of this method positional-sliced ``x_t[:,:n_prog]``
        which silently picked the WRONG channels (the ufs2arco recipe
        interleaves forcings within the prognostic level ordering). The
        caller now pre-slices using ``data_indices`` lists. See
        ``[[feedback-smoke-correctness-gap]]`` for the diagnostic story.
        """
        # Truth latent residual: bilinear-downsample the prognostic-only
        # difference. (Forcings excluded — see docstring.)
        with torch.no_grad():
            z_t = bilinear_downsample(x_t_prog, target_shape=self.latent_shape)
            z_tp1 = bilinear_downsample(x_tp1_prog, target_shape=self.latent_shape)
            r_lat_truth = z_tp1 - z_t

        # Decoder forward: gets the FULL state (prog + forcings) as
        # conditioning context, plus the latent residual.
        delta_pred = self.decoder(r_lat_truth, x_t_full)

        # Truth full-res residual: prog-only diff (matches delta_pred).
        delta_truth = x_tp1_prog - x_t_prog
        return delta_pred, delta_truth

    # ------------------------------------------------------------------
    # Anemoi task hook (training + validation step).
    # ------------------------------------------------------------------
    def _step(
        self,
        batch: torch.Tensor,
        batch_idx: int,                          # noqa: ARG002
        validation_mode: bool = False,           # noqa: ARG002
    ) -> tuple[torch.Tensor, dict, list]:
        """Compute loss for a single (x_t, x_{t+1}) pair.

        Returns
        -------
        loss : torch.Tensor
            Scalar training loss (mean across the batch + reduction by
            ``self.loss``).
        metrics : dict
            Validation metrics dict (empty for now; populated by parent
            class hooks during validation).
        y_pred_list : list
            Per-rollout-step prediction list — single-entry here.

        Notes
        -----
        Batch layout follows the standard Anemoi dataloader:
        ``(B, T=multi_step+rollout=2, ensemble=1, cell, V)``.

        We slice ``t=0`` as the current state (with forcings) and
        ``t=1`` for the prognostic next-step truth.
        """
        # Decoder knows its own spatial shape; use it (full_res_shape ==
        # the dataloader's flat cell -> 2-D mapping).
        H, W = self.decoder.full_res_shape

        # NOTE: `batch` is ALREADY normalized here — BaseGraphModule's
        # on_after_batch_transfer → _normalize_batch calls
        # self.model.pre_processors(batch) before _step runs.
        # See anemoi-core/training/src/anemoi/training/train/tasks/base.py:515.

        # Slice using data_indices lists (NOT positional) — forcings and
        # prognostics are interleaved in the input channel order. Mirrors
        # the pattern in GraphResidualForecaster:residualforecaster.py:75.
        x_full = batch[:, 0, 0, :, self.data_indices.data.input.full]          # (B, cell, V_in=116)
        x_t_prog = batch[:, 0, 0, :, self.data_indices.data.input.prognostic]  # (B, cell, V_prog=105)
        x_tp1_prog = batch[:, 1, 0, :, self.data_indices.data.input.prognostic]

        B = x_full.shape[0]
        x_full = x_full.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
        x_t_prog = x_t_prog.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
        x_tp1_prog = x_tp1_prog.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()

        # Smoke-time invariant check (cheap, runs every step but trivially fast).
        # Mismatched counts → the data_indices and config disagree.
        if x_full.shape[1] != self.decoder.xt_tokenizer.in_channels:
            error = (
                f"x_full has {x_full.shape[1]} channels, but decoder.xt_tokenizer "
                f"expects {self.decoder.xt_tokenizer.in_channels}. data_indices/config mismatch."
            )
            raise RuntimeError(error)

        # Run decoder + truth diff.
        delta_pred, delta_truth = self._atlas_decoder_step(x_full, x_t_prog, x_tp1_prog)

        # ---- Loss path -------------------------------------------------
        # Flatten back to anemoi's (B, ensemble=1, cell, V) layout so the
        # standard GraphCast loss (with general_variable / level_average /
        # limited_area_mask / VarTendencyScaler) applies unchanged.
        delta_pred_flat = delta_pred.reshape(B, delta_pred.shape[1], -1).permute(0, 2, 1).unsqueeze(1)
        delta_truth_flat = delta_truth.reshape(B, delta_truth.shape[1], -1).permute(0, 2, 1).unsqueeze(1)

        # GraphCast-style reduction is provided by self.loss (which is
        # GraphCastMAELoss in the v30 config); per-channel σ_tend scaling
        # comes from the configured scalers list (StdevTendencyScaler).
        loss = self.loss(
            delta_pred_flat, delta_truth_flat,
            squash=True,
        )

        # NaN guard — fail fast rather than silently train through NaN
        # (the v30b smoke produced nan.0 losses for 5 steps + checkpoint
        # save before being noticed; we don't want that to happen again).
        if not torch.isfinite(loss):
            error = (
                f"Non-finite loss (validation_mode={validation_mode}): "
                f"{loss.item() if loss.numel()==1 else loss}. "
                "Check tendency-scaler division, channel slicing, or input NaNs."
            )
            raise RuntimeError(error)

        # Per the parent class contract, also surface y_pred for metrics
        # and any downstream hooks.
        y_pred_list = [delta_pred_flat]
        metrics: dict = {}
        return loss, metrics, y_pred_list

    # ------------------------------------------------------------------
    # Lightning hooks — minimal pass-throughs.
    # ------------------------------------------------------------------
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss, _metrics, _ypred = self._step(batch, batch_idx, validation_mode=False)
        self.log("train_atlas_decoder_l1_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss, _metrics, _ypred = self._step(batch, batch_idx, validation_mode=True)
        self.log("val_atlas_decoder_l1_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss
