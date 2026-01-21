# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""GraphCast-style Huber loss with level-grouped variable reduction.

This module provides a Huber loss that follows the reduction semantics from
the GraphCast paper (Lam et al., 2023), where:
- Variables are grouped by their base name (e.g., all temperature levels together)
- Loss is averaged over levels within each group before summing across groups
- This ensures equal weighting per physical quantity, not per output channel

Huber loss is less sensitive to outliers than MSE, transitioning from quadratic
to linear behavior for errors larger than delta. This makes it suitable for
weather data with extreme events (e.g., deep convection) that can cause
100+ sigma outliers in normalized tendency space.

For |error| < delta: loss = 0.5 * error^2  (MSE behavior)
For |error| >= delta: loss = delta * |error| - 0.5 * delta^2  (MAE behavior)

Example YAML configuration:
    training:
      training_loss:
        _target_: anemoi.training.losses.GraphCastHuberLoss
        delta: 3.0
        scalers: ['node_weights']
        ignore_nans: False
"""

from typing import TYPE_CHECKING

import torch

from anemoi.training.losses.base import GraphCastBaseLoss

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup


class GraphCastHuberLoss(GraphCastBaseLoss):
    """Huber loss with GraphCast-style reduction.

    Computes Huber loss between predictions and targets, then reduces using
    the GraphCast reduction order:
    1. Mean over vertical levels within each variable group
    2. Mean over spatial (grid) and ensemble dimensions
    3. Sum over variable groups
    4. Mean over batch

    Huber loss provides robustness to outliers by using:
    - Quadratic loss (MSE) for small errors: |error| < delta
    - Linear loss (MAE) for large errors: |error| >= delta

    This is particularly useful for storm-scale weather prediction where
    deep convection can produce extreme tendencies (100+ sigma) that would
    otherwise dominate MSE loss.

    Parameters
    ----------
    delta : float, optional
        Threshold for switching from quadratic to linear loss. Default 1.0.
        For normalized tendency space, delta=3.0 means errors < 3 sigma use
        MSE, while larger errors use MAE-like behavior.
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
    >>> loss_fn = GraphCastHuberLoss(delta=3.0)
    >>> loss_fn.set_data_indices(data_indices)  # Required for variable grouping
    >>> loss = loss_fn(pred, target)

    >>> # With sample weighting for extra outlier robustness
    >>> loss_fn = GraphCastHuberLoss(delta=3.0, sample_weighting=True)

    Notes
    -----
    Choosing delta:
    - delta=1.0: Tight threshold, switches to linear quickly
    - delta=3.0: Reasonable for normalized data, allows 3-sigma errors to use MSE
    - delta=5.0: More permissive, only extreme outliers get linear treatment

    For a 100-sigma error:
    - MSE contribution: 10,000
    - Huber(delta=3) contribution: 3 * 100 - 0.5 * 9 = 295.5 (34x reduction)
    """

    name: str = "graphcast_huber"

    def __init__(
        self,
        delta: float = 1.0,
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
        self.delta = delta

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the Huber loss between prediction and target.

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
            Huber loss tensor in float32.
        """
        diff = torch.abs(pred - target)
        return torch.where(diff < self.delta, 0.5 * torch.square(diff), self.delta * (diff - 0.5 * self.delta))
        
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
        """Calculates the area-weighted scaled Huber loss.

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
        if self.sample_weighting:
            sample_weights = self._compute_sample_weights(target)

        out = self.calculate_difference(pred, target)

        # Scale the output (general variable, by vertical level)
        out = self.scale(out, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)

        return self.reduce(out, squash, group=group if is_sharded else None, sample_weights=sample_weights)
