# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.utils.checkpoint import checkpoint

from anemoi.training.train.tasks.rollout import BaseRolloutGraphModule

if TYPE_CHECKING:
    from collections.abc import Generator

LOGGER = logging.getLogger(__name__)


class GraphResidualForecaster(BaseRolloutGraphModule):
    """Graph neural-network forecaster that predicts *normalized residuals*.

    Δx_norm = (x_{t+1} - x_t) / std_Δx  (mean difference assumed zero).
    Matches the GraphCast framework.
    """

    # Diagnostic logging control
    _diag_log_interval = 100  # Log every N batches
    _diag_batch_counter = 0

    @staticmethod
    def _log_tensor_stats(name: str, tensor: torch.Tensor, log_fn=LOGGER.info) -> dict:
        """Log statistics for a tensor and return them as a dict."""
        with torch.no_grad():
            t = tensor.float()
            has_nan = torch.isnan(t).any().item()
            has_inf = torch.isinf(t).any().item()
            if has_nan or has_inf:
                nan_count = torch.isnan(t).sum().item()
                inf_count = torch.isinf(t).sum().item()
                log_fn(
                    "  %s: NaN=%d, Inf=%d, shape=%s",
                    name, nan_count, inf_count, list(tensor.shape)
                )
                return {"nan": nan_count, "inf": inf_count}

            stats = {
                "mean": t.mean().item(),
                "std": t.std().item(),
                "min": t.min().item(),
                "max": t.max().item(),
                "abs_max": t.abs().max().item(),
            }
            log_fn(
                "  %s: mean=%.4f, std=%.4f, min=%.4f, max=%.4f, |max|=%.4f",
                name, stats["mean"], stats["std"], stats["min"], stats["max"], stats["abs_max"]
            )
            return stats

    @staticmethod
    def _get_normalizer_buffers(processors) -> tuple[torch.Tensor, torch.Tensor]:
        for processor in processors.processors.values():
            if hasattr(processor, "_norm_mul") and hasattr(processor, "_norm_add"):
                return processor._norm_mul, processor._norm_add
        raise RuntimeError("InputNormalizer buffers not found in pre_processors.")

    def _rollout_step(
        self,
        batch: torch.Tensor,
        rollout: int | None = None,
        validation_mode: bool = False,
    ) -> Generator[tuple[torch.Tensor | None, dict, list]]:
        """Rollout step for the residual forecaster (GraphCast-style).

        The model predicts *normalized residuals*:
            Δx_norm = (y_true - x_last) / σ_Δx
        where y_true and x_last are in physical space.

        The loss is computed in this normalized residual space.

        At each step, the next-state prediction is reconstructed as:
            x̂_{t+1} = x_last + Δx̂_norm * σ_Δx
        """
        # Start rollout of preprocessed batch
        x = batch[
            :,
            0 : self.multi_step,
            ...,
            self.data_indices.data.input.full,
        ]  # (bs, multi_step, latlon, nvar)

        msg = (
            "Batch length not sufficient for requested multi_step length!"
            f", {batch.shape[1]} !>= {rollout + self.multi_step}"
        )
        assert batch.shape[1] >= rollout + self.multi_step, msg

        # Get indices for prognostic and diagnostic variables
        # model.output indices are for the model output tensor
        model_prog_idx = self.data_indices.model.output.prognostic
        model_diag_idx = self.data_indices.model.output.diagnostic
        # data.input indices are for the input/batch tensor (used to index normalizer buffers)
        input_prog_idx = self.data_indices.data.input.prognostic

        # Get normalizer buffers for unnormalization
        # IMPORTANT: norm_mul and norm_add are indexed by INPUT variable positions,
        # so we must use data.input.prognostic (not data.output.prognostic) to access them.
        norm_mul, norm_add = self._get_normalizer_buffers(self.model.pre_processors)

        # Increment batch counter for diagnostic logging
        GraphResidualForecaster._diag_batch_counter += 1
        should_log_diag = (GraphResidualForecaster._diag_batch_counter % GraphResidualForecaster._diag_log_interval == 1)

        for rollout_step in range(rollout or self.rollout):
            # Forward prediction (normalized residual)
            # model_output shape: (batch, ensemble, grid, n_output)
            model_output = self(x)

            # ============================================================
            # Extract PROGNOSTIC variables for residual prediction
            # ============================================================
            # Slice model output to only prognostic variables
            Δx̂_norm_prog = model_output[..., model_prog_idx]  # (batch, ensemble, grid, n_prog)

            # Get normalized values from batch (only prognostic)
            # Both x and batch use input variable ordering, so use input_prog_idx
            x_last_norm = x[:, -1, ..., input_prog_idx]  # (batch, ensemble, grid, n_prog)
            y_true_norm = batch[
                :,
                self.multi_step + rollout_step,
                ...,
                input_prog_idx,
            ]  # (batch, ensemble, grid, n_prog)

            # ============================================================
            # GraphCast-style: Compute residuals in PHYSICAL space
            # ============================================================

            # Unnormalize to physical space (in float32 for precision)
            # x_norm = x_phys * norm_mul + norm_add  =>  x_phys = (x_norm - norm_add) / norm_mul
            x_last_phys = (x_last_norm.float() - norm_add[input_prog_idx].float()) / norm_mul[input_prog_idx].float()
            y_true_phys = (y_true_norm.float() - norm_add[input_prog_idx].float()) / norm_mul[input_prog_idx].float()

            # Compute target residual in physical space, then normalize by diff_std only
            Δx_true_norm = self.model.residual_normalizer.transform(
                x_last_phys,
                y_true_phys,
                in_place=False,
            )

            # ============================================================
            # DIAGNOSTIC LOGGING (every N batches, rollout_step 0 only)
            # ============================================================
            if should_log_diag and rollout_step == 0:
                LOGGER.info("=" * 60)
                LOGGER.info("RESIDUAL FORECASTER DIAGNOSTICS (batch %d, step %d)",
                           GraphResidualForecaster._diag_batch_counter, rollout_step)
                LOGGER.info("=" * 60)
                LOGGER.info("Input tensors (normalized state space):")
                self._log_tensor_stats("x_last_norm", x_last_norm)
                self._log_tensor_stats("y_true_norm", y_true_norm)
                LOGGER.info("Physical space tensors:")
                self._log_tensor_stats("x_last_phys", x_last_phys)
                self._log_tensor_stats("y_true_phys", y_true_phys)
                LOGGER.info("Residuals (normalized):")
                self._log_tensor_stats("Δx_true_norm (target)", Δx_true_norm)
                self._log_tensor_stats("Δx̂_norm_prog (model)", Δx̂_norm_prog)
                LOGGER.info("Residual difference (pred - target):")
                residual_diff = Δx̂_norm_prog.float() - Δx_true_norm.float()
                self._log_tensor_stats("Δx̂ - Δx_true", residual_diff)
                LOGGER.info("-" * 60)

            # ============================================================
            # GraphCast-style: Reconstruct in PHYSICAL space, then renormalize
            # ============================================================

            # Reconstruct next state in physical space (only prognostic)
            y_pred_phys_prog = self.model.residual_normalizer.inverse_transform(
                x_last_phys,
                Δx̂_norm_prog,
                in_place=False,
            )

            # Renormalize prognostic predictions for next rollout step (normalized state space)
            y_pred_prog = (y_pred_phys_prog * norm_mul[input_prog_idx].float() + norm_add[input_prog_idx].float()).to(model_output.dtype)

            # Diagnostic logging for reconstruction
            if should_log_diag and rollout_step == 0:
                LOGGER.info("Reconstruction:")
                self._log_tensor_stats("y_pred_phys_prog", y_pred_phys_prog)
                self._log_tensor_stats("y_pred_prog (renormalized)", y_pred_prog)
                # Check if prediction drifted significantly from target
                pred_vs_true = y_pred_prog.float() - y_true_norm.float()
                self._log_tensor_stats("y_pred - y_true (norm space)", pred_vs_true)
                LOGGER.info("=" * 60)

            # ============================================================
            # Build full prediction tensor with prognostic + diagnostic
            # ============================================================
            # y_pred needs to have shape compatible with model output for _advance_input
            # which expects y_pred[..., model.output.prognostic]
            n_output = len(self.data_indices.model.output.full)
            y_pred = torch.zeros(
                *model_output.shape[:-1], n_output,
                dtype=model_output.dtype, device=model_output.device
            )
            y_pred[..., model_prog_idx] = y_pred_prog

            # Handle diagnostic variables if present (direct prediction, no residual)
            if len(model_diag_idx) > 0:
                # Diagnostics are predicted directly - keep as normalized for metrics
                y_pred[..., model_diag_idx] = model_output[..., model_diag_idx]

            # ============================================================
            # Loss in normalized RESIDUAL space (for backprop)
            # Only compute loss on prognostic variables (residual prediction)
            # ============================================================
            Δx̂_norm_full, Δx_true_norm_full, grid_shard_slice = self._prepare_tensors_for_loss(
                Δx̂_norm_prog,
                Δx_true_norm,
                validation_mode,
            )

            loss = checkpoint(
                self._compute_loss,
                Δx̂_norm_full,
                Δx_true_norm_full,
                grid_shard_slice,
                use_reentrant=False,
            )

            # Log loss and check for issues at every step (not just step 0)
            if should_log_diag:
                with torch.no_grad():
                    loss_val = loss.item() if loss.numel() == 1 else loss.sum().item()
                    has_nan = torch.isnan(loss).any().item()
                    has_inf = torch.isinf(loss).any().item()
                    if has_nan or has_inf or loss_val > 100:
                        LOGGER.warning(
                            "LOSS ALERT (batch %d, step %d): loss=%.4f, has_nan=%s, has_inf=%s",
                            GraphResidualForecaster._diag_batch_counter, rollout_step,
                            loss_val, has_nan, has_inf
                        )
                        # Log state at problematic step
                        LOGGER.warning("State at problematic step:")
                        self._log_tensor_stats("x (input)", x, LOGGER.warning)
                        self._log_tensor_stats("Δx̂_norm_prog", Δx̂_norm_prog, LOGGER.warning)
                        self._log_tensor_stats("Δx_true_norm", Δx_true_norm, LOGGER.warning)
                    elif rollout_step == 0:
                        LOGGER.info("Loss at step %d: %.4f", rollout_step, loss_val)

            # Log multi-step rollout degradation
            if should_log_diag and rollout_step > 0 and rollout_step % 2 == 0:
                LOGGER.info("Rollout step %d - checking for drift:", rollout_step)
                self._log_tensor_stats("  x (model input)", x)
                self._log_tensor_stats("  Δx̂_norm_prog (model output)", Δx̂_norm_prog)

            # ============================================================
            # Validation metrics in STATE space (not residual space!)
            # The post_processors expect tensors with shape matching data.output.full
            # so they can correctly index the normalization buffers.
            # ============================================================
            metrics_next = {}
            if validation_mode:
                # Get ground truth with full output shape (to match y_pred)
                y_true_for_metrics = batch[
                    :,
                    self.multi_step + rollout_step,
                    ...,
                    self.data_indices.data.output.full,
                ]

                # Pass full tensors (n_output shape) so post_processors can
                # correctly apply denormalization using _output_idx
                metrics_next = self.calculate_val_metrics(
                    y_pred,
                    y_true_for_metrics,
                    step=rollout_step,
                    grid_shard_slice=self.grid_shard_slice,
                )

            # Feed next-state prediction back into input window
            x = self._advance_input(x, y_pred, batch, rollout_step)

            yield loss, metrics_next, y_pred
