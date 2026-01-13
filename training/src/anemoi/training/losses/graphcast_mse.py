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

import logging
from typing import TYPE_CHECKING

import torch

from anemoi.training.losses.base import GraphCastBaseLoss

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup

LOGGER = logging.getLogger(__name__)


# Debug flag - set to True to enable detailed NaN diagnostics
_DEBUG_NAN = False
_EXTREME_VALUE_THRESHOLD = 1e6


def _check_tensor_for_nan(
    tensor: torch.Tensor,
    step_name: str,
    variable_groups: dict | None = None,
    group_slices: list | None = None,
) -> bool:
    """Check tensor for NaN/Inf and extreme values, log details if found.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to check
    step_name : str
        Name of the computation step for logging
    variable_groups : dict | None
        Variable group mapping for detailed per-variable reporting
    group_slices : list | None
        Group slices for detailed per-group reporting

    Returns
    -------
    bool
        True if tensor is clean, False if NaN/Inf/extreme values found
    """
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    has_extreme = (torch.abs(tensor) > _EXTREME_VALUE_THRESHOLD).any().item()

    if has_nan or has_inf:
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        LOGGER.error(
            "NaN/Inf DETECTED at step '%s': %d NaN, %d Inf (shape=%s)",
            step_name,
            nan_count,
            inf_count,
            list(tensor.shape),
        )

        # Report per-variable-group details if available
        if variable_groups is not None and group_slices is not None and len(tensor.shape) >= 1:
            group_names = list(variable_groups.keys())
            for i, (start_idx, end_idx) in enumerate(group_slices):
                if tensor.shape[-1] > start_idx:
                    end_idx = min(end_idx, tensor.shape[-1])
                    group_slice = tensor[..., start_idx:end_idx]
                    g_nan = torch.isnan(group_slice).sum().item()
                    g_inf = torch.isinf(group_slice).sum().item()
                    if g_nan > 0 or g_inf > 0:
                        LOGGER.error(
                            "  Variable group '%s' (idx %d-%d): %d NaN, %d Inf",
                            group_names[i] if i < len(group_names) else f"group_{i}",
                            start_idx,
                            end_idx,
                            g_nan,
                            g_inf,
                        )
        return False

    if has_extreme:
        extreme_count = (torch.abs(tensor) > _EXTREME_VALUE_THRESHOLD).sum().item()
        max_val = torch.abs(tensor).max().item()
        LOGGER.warning(
            "EXTREME VALUES at step '%s': %d values > %g (max=%.6e, shape=%s)",
            step_name,
            extreme_count,
            _EXTREME_VALUE_THRESHOLD,
            max_val,
            list(tensor.shape),
        )

        # Report per-variable-group details if available
        if variable_groups is not None and group_slices is not None and len(tensor.shape) >= 1:
            group_names = list(variable_groups.keys())
            for i, (start_idx, end_idx) in enumerate(group_slices):
                if tensor.shape[-1] > start_idx:
                    end_idx = min(end_idx, tensor.shape[-1])
                    group_slice = tensor[..., start_idx:end_idx]
                    g_extreme = (torch.abs(group_slice) > _EXTREME_VALUE_THRESHOLD).sum().item()
                    if g_extreme > 0:
                        g_max = torch.abs(group_slice).max().item()
                        LOGGER.warning(
                            "  Variable group '%s' (idx %d-%d): %d extreme values (max=%.6e)",
                            group_names[i] if i < len(group_names) else f"group_{i}",
                            start_idx,
                            end_idx,
                            g_extreme,
                            g_max,
                        )

    return True


class GraphCastMSELoss(GraphCastBaseLoss):
    """Mean Squared Error loss with GraphCast-style reduction.

    Computes MSE between predictions and targets, then reduces using
    the GraphCast reduction order:
    1. Mean over vertical levels within each variable group
    2. Mean over spatial (grid) and ensemble dimensions
    3. Sum over variable groups
    4. Mean over batch

    This differs from standard Anemoi MSELoss which treats each level
    as an independent variable, effectively weighting 3D variables
    by their number of levels.

    Parameters
    ----------
    ignore_nans : bool, optional
        If True, use nanmean/nansum to ignore NaN values. Default False.

    Example
    -------
    >>> loss_fn = GraphCastMSELoss()
    >>> loss_fn.set_data_indices(data_indices)  # Required for variable grouping
    >>> loss = loss_fn(pred, target)
    """

    name: str = "graphcast_mse"

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the squared difference between prediction and target.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)

        Returns
        -------
        torch.Tensor
            Squared difference tensor, same shape as inputs.
        """
        if _DEBUG_NAN:
            # Check inputs first
            _check_tensor_for_nan(
                pred,
                "calculate_difference:pred_input",
                self.variable_groups,
                self.group_slices,
            )
            _check_tensor_for_nan(
                target,
                "calculate_difference:target_input",
                self.variable_groups,
                self.group_slices,
            )

        diff = pred - target

        if _DEBUG_NAN:
            _check_tensor_for_nan(
                diff,
                "calculate_difference:diff",
                self.variable_groups,
                self.group_slices,
            )

        squared = torch.square(diff)

        if _DEBUG_NAN:
            _check_tensor_for_nan(
                squared,
                "calculate_difference:squared",
                self.variable_groups,
                self.group_slices,
            )

        return squared

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
        """Calculates the area-weighted scaled loss with NaN debugging.

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

        # Step 1: Calculate difference (squared error)
        out = self.calculate_difference(pred, target)

        if _DEBUG_NAN:
            _check_tensor_for_nan(
                out,
                "forward:after_calculate_difference",
                self.variable_groups,
                self.group_slices,
            )

        # Step 2: Scale the output
        out = self.scale(out, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)

        if _DEBUG_NAN:
            _check_tensor_for_nan(
                out,
                "forward:after_scale",
                self.variable_groups,
                self.group_slices,
            )

        # Step 3: Reduce
        result = self.reduce(out, squash, group=group if is_sharded else None)

        if _DEBUG_NAN:
            _check_tensor_for_nan(result, "forward:after_reduce")

        return result
