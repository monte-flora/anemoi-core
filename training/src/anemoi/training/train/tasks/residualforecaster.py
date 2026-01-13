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
    """Graph neural-network forecaster that predicts *normalized residuals*
    Δx_norm = (x_{t+1} - x_t) / std_Δx  (mean difference assumed zero).
    
    Matches the GraphCast framework.
    """
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

        Parameters
        ----------
        batch : torch.Tensor
            Batch to use for rollout
        rollout : Optional[int], optional
            Number of times to rollout for, by default None
            If None, will use self.rollout
        training_mode : bool, optional
            Whether in training mode and to calculate the loss, by default True
            If False, loss will be None
        validation_mode : bool, optional
            Whether in validation mode, and to calculate validation metrics, by default False
            If False, metrics will be empty

        Yields
        ------
        Generator[tuple[Union[torch.Tensor, None], dict, list], None, None]
            Loss value, metrics, and predictions (per step)

        """        
        # start rollout of preprocessed batch
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

        for rollout_step in range(rollout or self.rollout):

            # forward prediction (normalized residual), shape = (bs, latlon, nvar)
            # Note: For input to self, x needs to be normalized.
            # At this point, we want to run the
            # self.model.pre_processors as otherwise, we
            # want an "x" in physical space for the remaining computations.
            Δx̂_norm = self(x)

            norm_mul, _norm_add = self._get_normalizer_buffers(self.model.pre_processors)
            x_last_norm = x[:, -1, ..., self.data_indices.data.input.prognostic]
            y_true_norm = batch[
                :,
                self.multi_step + rollout_step,
                ...,
                self.data_indices.data.output.prognostic,
            ]

            # Compute normalized true residual directly in normalized space
            Δx_true_norm = self.model.residual_normalizer.transform_from_normalized(
                x_last_norm,
                y_true_norm,
                norm_mul,
                in_place=False,
            )
            
            # loss in normalized residual space
            loss, metrics_next, y_pred = checkpoint(
                self.compute_loss_metrics,
                Δx̂_norm,
                Δx_true_norm,
                step=rollout_step,
                validation_mode=validation_mode,
                use_reentrant=False,
            )

            # Reconstruct next-state prediction in normalized space for rollout
            y_pred = self.model.residual_normalizer.inverse_transform_to_normalized(
                x_last_norm,
                Δx̂_norm,
                norm_mul,
                in_place=False,
            )
                        
            # feed next-state prediction back into input window
            x = self._advance_input(x, y_pred, batch, rollout_step)
            
            yield loss, metrics_next, y_pred
