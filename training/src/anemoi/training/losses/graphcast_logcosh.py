# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""GraphCast-style LogCosh loss with level-grouped variable reduction.

This module provides a LogCosh loss that follows the reduction semantics from
the GraphCast paper (Lam et al., 2023), where:
- Variables are grouped by their base name (e.g., all temperature levels together)
- Loss is averaged over levels within each group before summing across groups
- This ensures equal weighting per physical quantity, not per output channel

LogCosh loss is log(cosh(error)), which behaves like:
- MSE for small errors: ~0.5 * error^2 when |error| << 1
- MAE for large errors: ~|error| - log(2) when |error| >> 1

This provides smooth transition between MSE and MAE behavior, making it
robust to outliers while maintaining differentiability everywhere.

Example YAML configuration:
    training:
      training_loss:
        _target_: anemoi.training.losses.GraphCastLogCoshLoss
        scalers: ['node_weights']
        ignore_nans: False
"""

from typing import TYPE_CHECKING

import numpy as np
import torch

from anemoi.training.losses.base import GraphCastBaseLoss

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup


class LogCosh(torch.autograd.Function):
    """LogCosh custom autograd function for numerical stability.

    Uses the identity: log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)
    This avoids numerical overflow for large |x| values.
    """

    @staticmethod
    def forward(ctx, inp: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(inp)
        abs_input = torch.abs(inp)
        return abs_input + torch.nn.functional.softplus(-2 * abs_input) - np.log(2)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (inp,) = ctx.saved_tensors
        return grad_output * torch.tanh(inp)


class GraphCastLogCoshLoss(GraphCastBaseLoss):
    """LogCosh loss with GraphCast-style reduction.

    Computes LogCosh loss between predictions and targets, then reduces using
    the GraphCast reduction order:
    1. Mean over vertical levels within each variable group
    2. Mean over spatial (grid) and ensemble dimensions
    3. Sum over variable groups
    4. Mean over batch

    LogCosh provides a smooth interpolation between MSE and MAE:
    - For small errors (|e| << 1): loss ≈ 0.5 * e^2 (MSE-like)
    - For large errors (|e| >> 1): loss ≈ |e| - log(2) (MAE-like)

    This makes it naturally robust to outliers without requiring a
    threshold parameter like Huber loss.

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
    >>> loss_fn = GraphCastLogCoshLoss()
    >>> loss_fn.set_data_indices(data_indices)  # Required for variable grouping
    >>> loss = loss_fn(pred, target)

    >>> # With sample weighting for extra outlier robustness
    >>> loss_fn = GraphCastLogCoshLoss(sample_weighting=True)

    Notes
    -----
    Comparison with other losses for various error magnitudes:

    | Error | MSE    | MAE | Huber(δ=1) | LogCosh |
    |-------|--------|-----|------------|---------|
    | 0.1   | 0.01   | 0.1 | 0.005      | 0.005   |
    | 1.0   | 1.0    | 1.0 | 0.5        | 0.43    |
    | 10.0  | 100    | 10  | 9.5        | 9.31    |
    | 100.0 | 10000  | 100 | 99.5       | 99.31   |

    LogCosh is smoother than Huber at the transition point and doesn't
    require tuning a delta parameter.
    """

    name: str = "graphcast_logcosh"

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
        """Calculate the LogCosh loss between prediction and target.

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
            LogCosh loss tensor in float32.
        """
        # Compute loss in float32 for numerical stability
        diff = pred.float() - target.float()
        return LogCosh.apply(diff)

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
        """Calculates the area-weighted scaled LogCosh loss.

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
