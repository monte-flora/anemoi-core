# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""MC-dropout residual ensemble forecaster.

Implements the U-Cast (Cachay, Watson-Parris, Yu, arXiv:2604.09041)
recipe on top of our DiT-NATTEN backbone: the model is run multiple
times with independent dropout realisations as the source of ensemble
diversity, and trained via fair-CRPS. No noise encoder, no adaLN
weight retraining, no architectural surgery — the dropout layers
already present in the DiT (attn_drop_rate, proj_drop_rate,
drop_path_rate) carry the stochasticity.

Each ensemble member is one row of the ``(b e)`` flat-batch fold;
PyTorch's ``nn.Dropout`` samples per-element so each row gets an
independent mask in a single forward pass — same wall-clock cost
per microbatch as deterministic forward (the model is just doing
twice the work because flat-batch dim is 2× larger).

Companion to GraphEnsResidualForecaster (FGN-style noise vector) —
use whichever the loaded checkpoint supports.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.utils.checkpoint import checkpoint

from anemoi.training.train.tasks.ensresidualforecaster import GraphEnsResidualForecaster
from anemoi.training.train.tasks.residualforecaster import GraphResidualForecaster

if TYPE_CHECKING:
    from collections.abc import Generator

LOGGER = logging.getLogger(__name__)


class GraphEnsResidualForecasterDropout(GraphEnsResidualForecaster):
    """Ensemble residual forecaster with MC-dropout stochasticity.

    Differs from the parent class in exactly one place: the model
    forward is the standard deterministic path ``self.model(x)`` instead
    of ``self.model.forward_with_noise(x, noise_vec)``. The flat-batch
    fold of ``(B, T, E, G, V)`` to ``(B*E, ...)`` causes each row to
    receive an independent dropout realisation.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        LOGGER.info(
            "GraphEnsResidualForecasterDropout: nens_per_device=%d, no noise_encoder "
            "(stochasticity from dropout/DropPath layers in the DiT)",
            self.nens_per_device,
        )

    def _rollout_step(
        self,
        batch: torch.Tensor,
        rollout: int | None = None,
        validation_mode: bool = False,
    ) -> Generator[tuple[torch.Tensor | None, dict, list]]:
        """Ensemble residual rollout where dropout provides member diversity."""
        x = batch[
            :,
            0 : self.multi_step,
            ...,
            self.data_indices.data.input.full,
        ]
        x = torch.cat([x] * self.nens_per_device, dim=2)
        assert x.shape[2] == self.nens_per_device

        msg = (
            "Batch length not sufficient for requested multi_step length!"
            f", {batch.shape[1]} !>= {rollout + self.multi_step}"
        )
        assert batch.shape[1] >= rollout + self.multi_step, msg

        model_prog_idx = self.data_indices.model.output.prognostic
        model_diag_idx = self.data_indices.model.output.diagnostic
        input_prog_idx = self.data_indices.data.input.prognostic

        norm_mul, norm_add = self._get_normalizer_buffers(self.model.pre_processors)
        nm_mul_prog = norm_mul[input_prog_idx].float()
        nm_add_prog = norm_add[input_prog_idx].float()

        for rollout_step in range(rollout or self.rollout):
            # Standard deterministic forward — ensemble diversity comes from
            # the dropout layers in the DiT, which sample independently per
            # flat-batch row (=per ensemble member).
            model_output = self.model(x)  # (B, nens_per_device, G, n_output)

            Δx̂_norm_prog = model_output[..., model_prog_idx]
            x_last_norm = x[:, -1, ..., input_prog_idx]
            y_true_norm = batch[
                :,
                self.multi_step + rollout_step,
                ...,
                input_prog_idx,
            ]

            x_last_phys = (x_last_norm.float() - nm_add_prog) / nm_mul_prog
            y_true_phys = (y_true_norm.float() - nm_add_prog) / nm_mul_prog

            Δx_true_norm = self.model.residual_normalizer.transform(
                x_last_phys, y_true_phys, in_place=False,
            )

            y_pred_phys_prog = self.model.residual_normalizer.inverse_transform(
                x_last_phys, Δx̂_norm_prog, in_place=False,
            )

            y_pred_prog = (
                y_pred_phys_prog * nm_mul_prog + nm_add_prog
            ).to(model_output.dtype)

            n_output = len(self.data_indices.model.output.full)
            y_pred = torch.zeros(
                *model_output.shape[:-1], n_output,
                dtype=model_output.dtype, device=model_output.device,
            )
            y_pred[..., model_prog_idx] = y_pred_prog
            if len(model_diag_idx) > 0:
                y_pred[..., model_diag_idx] = model_output[..., model_diag_idx]

            Δx_true_for_loss = Δx_true_norm[:, 0:1]

            loss = checkpoint(
                self.compute_loss_metrics_residual,
                Δx̂_norm_prog,
                Δx_true_for_loss.squeeze(1),
                rollout_step,
                validation_mode,
                use_reentrant=False,
            )[0]

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
                    y_pred.mean(dim=1, keepdim=True),
                    y_true_for_metrics,
                    step=rollout_step,
                    grid_shard_slice=self.grid_shard_slice,
                )

            x = self._advance_input(x, y_pred, batch, rollout_step)

            yield loss, metrics_next, y_pred
