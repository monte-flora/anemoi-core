# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""GraphCast-style MSE loss with level-grouped variable reduction.

This module provides an MSE loss that follows the reduction semantics from
the GraphCast paper (Lam et al., 2023), where:
- Variables are grouped by their base name (e.g., all temperature levels together)
- Loss is averaged over levels within each group before summing across groups
- This ensures equal weighting per physical quantity, not per output channel

Example YAML configuration:
    training:
      training_loss:
        _target_: anemoi.training.losses.GraphCastMSELoss
        scalers: ['node_weights']
        ignore_nans: False
"""

from typing import TYPE_CHECKING

import torch

from anemoi.training.losses.base import GraphCastBaseLoss

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup


class GraphCastMSELoss(GraphCastBaseLoss):
    """Mean Squared Error loss with GraphCast-style reduction.

    Computes MSE between predictions and targets, then reduces using
    the GraphCast reduction order:
    1. Mean over vertical levels within each variable group
    2. Mean over spatial (grid) and ensemble dimensions
    3. Sum over variable groups
    4. Weighted mean over batch (optionally downweighting extreme samples)

    This differs from standard Anemoi MSELoss which treats each level
    as an independent variable, effectively weighting 3D variables
    by their number of levels.

    Parameters
    ----------
    ignore_nans : bool, optional
        If True, use nanmean/nansum to ignore NaN values. Default False.
    sample_weighting : bool, optional
        If True, downweight samples with extreme target values. Default False.
    sample_weight_threshold : float, optional
        Target magnitude threshold for weighting. Default 10.0.
    sample_weight_min : float, optional
        Minimum weight for extreme samples. Default 0.01.

    Example
    -------
    >>> loss_fn = GraphCastMSELoss()
    >>> loss_fn.set_data_indices(data_indices)  # Required for variable grouping
    >>> loss = loss_fn(pred, target)

    >>> # With sample weighting to handle extreme convective events
    >>> loss_fn = GraphCastMSELoss(sample_weighting=True, sample_weight_threshold=10.0)
    """

    name: str = "graphcast_mse"

    def __init__(
        self,
        ignore_nans: bool = False,
        sample_weighting: bool = False,
        sample_weight_threshold: float = 10.0,
        sample_weight_min: float = 0.01,
    ) -> None:
        super().__init__(
            ignore_nans=ignore_nans,
            sample_weighting=sample_weighting,
            sample_weight_threshold=sample_weight_threshold,
            sample_weight_min=sample_weight_min,
        )

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the squared difference between prediction and target.

        Computation is performed in float32 for numerical stability with bfloat16
        inputs. The loss remains in float32 for proper gradient computation.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)

        Returns
        -------
        torch.Tensor
            Squared difference tensor in float32.
        """
        # Compute loss in float32 for numerical stability
        # This is critical for bfloat16 training where squaring small differences
        # can lose precision. Loss stays in float32 for proper backward pass.
        diff = pred.float() - target.float()
        squared = torch.square(diff)
        return squared  # Keep in float32

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: "ProcessGroup | None" = None,
        **kwargs,
    ) -> torch.Tensor:
        """Calculates the area-weighted scaled loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)
        squash : bool, optional
            Average last dimension, by default True
        scaler_indices: tuple[int,...], optional
            Indices to subset the calculated scaler with, by default None
        without_scalers: list[str] | list[int] | None, optional
            list of scalers to exclude from scaling
        grid_shard_slice : slice, optional
            Slice of the grid if x comes sharded
        group: ProcessGroup, optional
            Distributed group, by default None

        Returns
        -------
        torch.Tensor
            Weighted loss
        """
        is_sharded = grid_shard_slice is not None

        # Compute sample weights before calculating difference (need original target scale)
        sample_weights = None
        #if self.sample_weighting:
        #    sample_weights = self._compute_sample_weights(target)

        out = self.calculate_difference(pred, target)

        # Scale the output (general variable, by vertical level)
        out = self.scale(out, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)

        return self.reduce(out, squash, group=group if is_sharded else None, sample_weights=sample_weights)


class WeightedGraphCastMSELoss(GraphCastMSELoss):
    """GraphCast per-physical-quantity MSE reduction + per-sample EDM noise weight.

    Identical reduction to GraphCastMSELoss (mean-over-levels within each variable
    group, so 3D fields are NOT overweighted by level count and a 2D field like
    comp_refl counts as one physical quantity), but additionally applies the
    diffusion task's per-sample EDM weight lambda(sigma) as ``sample_weights`` in
    the batch reduction. This is the loss for GraphDiffusionDenoiser /
    GraphDiffusionForecaster with a DiT: ``GraphCastMSELoss`` would silently
    swallow the ``weights`` kwarg (via ``**kwargs``) and ignore the noise
    weighting; ``anemoi.training.losses.WeightedMSELoss`` applies the noise
    weight but reduces per-CHANNEL (overweighting 3D vars by level count).

    The diffusion task calls ``self.loss(pred, target, weights=lambda_sigma, ...)``
    with ``weights`` of shape (B, 1, 1, 1); we route it to ``sample_weights`` (B,),
    giving a self-normalized weighted batch mean ``sum(lambda*L)/sum(lambda)``.
    """

    name: str = "weighted_graphcast_mse"

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: "ProcessGroup | None" = None,
        weights: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        is_sharded = grid_shard_slice is not None
        out = self.calculate_difference(pred, target)
        out = self.scale(out, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)
        # diffusion EDM noise weight lambda(sigma): (B,1,1,1) -> per-sample (B,)
        sample_weights = None if weights is None else weights.reshape(weights.shape[0]).to(out.dtype)
        return self.reduce(out, squash, group=group if is_sharded else None, sample_weights=sample_weights)
