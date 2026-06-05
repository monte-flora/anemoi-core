# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Deterministic residual forecaster with random conditioning injection.

Used as a warmup stage before CRPS training (v24_b stage 1): the model
sees random per-batch noise on every forward, so the adaLN modulation
weights (W in physicsnemo's ``adaptive_modulation = Linear(D, 6D)``,
zero-initialized) start receiving non-zero gradient signal even though
the loss is the standard deterministic GraphCast MSE.

Rationale. During v17 training the conditioning vector was identically
zero. With c=0, ``adaptive_modulation(c) = b`` (bias only) and the
gradient w.r.t. W is ``∇W = ∇out @ c.T = 0``. So W stayed at zero
through 175k training steps. When v24 then introduces a real noise
vector via the matmul encoder, W has to learn from zero in only ~25K
FT steps — too few to converge, which we hypothesise as the dominant
cause of v24's rollout failure.

This task lets W see non-zero ``c`` (via ``forward_with_noise``) under
the v17 deterministic loss, giving the adaLN heads ~5K steps of
"structured but loss-aligned" gradient signal before CRPS pivots them
to actually USE c for spread.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from anemoi.training.train.tasks.residualforecaster import GraphResidualForecaster

if TYPE_CHECKING:
    from collections.abc import Generator

LOGGER = logging.getLogger(__name__)


class GraphResidualForecasterRandomCondition(GraphResidualForecaster):
    """Deterministic residual forecaster + random conditioning vector."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cfg = kwargs.get("config") or args[0]
        self.noise_vector_dim = int(getattr(cfg.training, "noise_vector_dim", 32))
        LOGGER.info(
            "GraphResidualForecasterRandomCondition: noise_vector_dim=%d (deterministic warmup; MSE-style loss)",
            self.noise_vector_dim,
        )

    def _rollout_step(
        self,
        batch: torch.Tensor,
        rollout: int | None = None,
        validation_mode: bool = False,
    ) -> Generator[tuple[torch.Tensor | None, dict, list]]:
        """Rollout step identical to GraphResidualForecaster._rollout_step
        except the model's forward is routed through ``forward_with_noise``
        with a freshly sampled noise vector per call. The ensemble dim
        is the trivial singleton (B, 1, ...) for compatibility with the
        upstream pre-processor + residual_normalizer pipeline.
        """
        x = batch[
            :,
            0 : self.multi_step,
            ...,
            self.data_indices.data.input.full,
        ]

        msg = (
            "Batch length not sufficient for requested multi_step length!"
            f", {batch.shape[1]} !>= {rollout + self.multi_step}"
        )
        assert batch.shape[1] >= rollout + self.multi_step, msg

        model_prog_idx = self.data_indices.model.output.prognostic
        model_diag_idx = self.data_indices.model.output.diagnostic
        input_prog_idx = self.data_indices.data.input.prognostic
        norm_mul, norm_add = self._get_normalizer_buffers(self.model.pre_processors)

        if not hasattr(self.model, "forward_with_noise"):
            msg = (
                "GraphResidualForecasterRandomCondition requires the model to implement "
                "forward_with_noise(x, noise_vec). Use AnemoiDiTModel with "
                "noise_vector_dim configured."
            )
            raise AttributeError(msg)

        for rollout_step in range(rollout or self.rollout):
            # Fresh noise per training step. The ensemble dim is trivially 1
            # so forward_with_noise's (b e) fold is a no-op — c reaches the
            # adaLN heads with shape (B, hidden_size).
            B = x.shape[0]
            E = x.shape[2] if x.ndim == 5 else 1
            noise_vec = torch.randn(
                B, E, self.noise_vector_dim, device=x.device, dtype=x.dtype,
            )
            model_output = self.model.forward_with_noise(x, noise_vec)

            Δx̂_norm_prog = model_output[..., model_prog_idx]

            x_last_norm = x[:, -1, ..., input_prog_idx]
            y_true_norm = batch[
                :,
                self.multi_step + rollout_step,
                ...,
                input_prog_idx,
            ]

            x_last_phys = (x_last_norm.float() - norm_add[input_prog_idx].float()) / norm_mul[input_prog_idx].float()
            y_true_phys = (y_true_norm.float() - norm_add[input_prog_idx].float()) / norm_mul[input_prog_idx].float()

            Δx_true_norm = self.model.residual_normalizer.transform(
                x_last_phys,
                y_true_phys,
                in_place=False,
            )

            y_pred_phys_prog = self.model.residual_normalizer.inverse_transform(
                x_last_phys,
                Δx̂_norm_prog,
                in_place=False,
            )

            y_pred_prog = (
                y_pred_phys_prog * norm_mul[input_prog_idx].float() + norm_add[input_prog_idx].float()
            ).to(model_output.dtype)

            n_output = len(self.data_indices.model.output.full)
            y_pred = torch.zeros(
                *model_output.shape[:-1], n_output,
                dtype=model_output.dtype, device=model_output.device,
            )
            y_pred[..., model_prog_idx] = y_pred_prog
            if len(model_diag_idx) > 0:
                y_pred[..., model_diag_idx] = model_output[..., model_diag_idx]

            # Reuse the parent class's loss pipeline (MSE in normalized
            # residual space). Mirror GraphResidualForecaster._rollout_step
            # tail: _prepare_tensors_for_loss → _compute_loss → metrics.
            from torch.utils.checkpoint import checkpoint

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

            if hasattr(self.loss, "_last_per_group_losses") and self.loss._last_per_group_losses is not None:
                for group_name, group_loss in self.loss._last_per_group_losses.items():
                    self.log(
                        f"train_loss/{group_name}",
                        group_loss,
                        on_step=True,
                        on_epoch=False,
                        logger=True,
                        rank_zero_only=True,
                    )

            metrics_next = {}
            if validation_mode:
                y_true_for_metrics = batch[
                    :,
                    self.multi_step + rollout_step,
                    ...,
                    self.data_indices.data.output.full,
                ]
                metrics_next = self.calculate_val_metrics(
                    y_pred,
                    y_true_for_metrics,
                    step=rollout_step,
                    grid_shard_slice=self.grid_shard_slice,
                )

            x = self._advance_input(x, y_pred, batch, rollout_step)

            yield loss, metrics_next, y_pred
