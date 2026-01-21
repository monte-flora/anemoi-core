# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""GraphCast-style Gaussian NLL loss with learned diagonal covariance.

This module provides a Gaussian negative log-likelihood loss that learns
a diagonal covariance matrix in normalized residual space. This addresses
the assumption that Cov(r_norm) = I, which may not hold even after normalization.

Loss formula (up to constants):
    L = mean(r² / σ² + log(σ²))

where r = Δx_norm - Δx̂_norm is the residual in normalized space.

Example YAML configuration:
    training:
      training_loss:
        _target_: anemoi.training.losses.GraphCastGaussianNLLLoss
        variance_mode: per_variable_per_level
        variance_init: 0.0
        scalers: ['node_weights']
        ignore_nans: False
"""

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn

from anemoi.training.losses.base import GraphCastBaseLoss
from anemoi.models.data_indices.collection import IndexCollection

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup

LOGGER = logging.getLogger(__name__)


class GraphCastGaussianNLLLoss(GraphCastBaseLoss):
    """Gaussian negative log-likelihood loss with learned diagonal covariance.

    Learns a diagonal covariance matrix B = diag(σ²) in normalized residual space,
    allowing the model to adapt to heteroscedastic errors across variables/levels.

    The loss computes:
        L = (1/2) * mean(r²/σ² + log(σ²))

    where r = pred - target is the residual in normalized tendency space.

    Supports two parameterization modes:
    - 'per_variable': One variance per variable group (e.g., one for all qv levels)
    - 'per_variable_per_level': One variance per variable-level (e.g., separate for qv_0, qv_1, ...)

    Parameters
    ----------
    variance_mode : str, optional
        How to parameterize variances: 'per_variable' or 'per_variable_per_level'.
        Default 'per_variable'.
    variance_init : float, optional
        Initial value for log_var (var = exp(log_var)). Default 0.0 (var=1, equivalent to MSE).
    variance_eps : float, optional
        Small constant added to variance for numerical stability. Default 1e-6.
    variance_trainable : bool, optional
        Whether to learn the variances. If False, keeps them fixed at init. Default True.
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
    >>> # YAML configuration
    >>> training:
    >>>   training_loss:
    >>>     _target_: anemoi.training.losses.GraphCastGaussianNLLLoss
    >>>     variance_mode: per_variable_per_level
    >>>     variance_init: 0.0
    >>>     scalers: ['node_weights']

    >>> # Python usage
    >>> loss_fn = GraphCastGaussianNLLLoss(variance_mode='per_variable_per_level')
    >>> loss_fn.set_data_indices(data_indices)  # Required for variable grouping
    >>> loss = loss_fn(pred, target)

    Notes
    -----
    - Initializing with variance_init=0.0 makes this identical to MSE at start of training
    - The log_var parameterization ensures σ² > 0 via var = exp(log_var)
    - Uses the same GraphCast reduction semantics as GraphCastMSELoss
    - The 1/2 prefactor is omitted as it's a constant that doesn't affect optimization
    """

    name: str = "graphcast_gaussian_nll"

    def __init__(
        self,
        variance_mode: str = "per_variable",
        variance_init: float = 0.0,
        variance_eps: float = 1e-6,
        variance_trainable: bool = True,
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

        if variance_mode not in ["per_variable", "per_variable_per_level"]:
            raise ValueError(
                f"variance_mode must be 'per_variable' or 'per_variable_per_level', "
                f"got '{variance_mode}'"
            )

        self.variance_mode = variance_mode
        self.variance_init = variance_init
        self.variance_eps = variance_eps
        self.variance_trainable = variance_trainable

        # Will be initialized in set_data_indices
        self.log_var: nn.Parameter | None = None

    def set_data_indices(self, data_indices: IndexCollection) -> None:
        """Initialize variable groups and variance parameters.

        Parameters
        ----------
        data_indices : IndexCollection
            Collection of data indices from the model.
        """
        # Initialize variable groups via parent
        super().set_data_indices(data_indices)

        # Build log_var parameter based on mode
        if self.variance_mode == "per_variable":
            # One variance per variable group
            n_groups = len(self.variable_groups)
            shape = (n_groups,)
            LOGGER.info(
                "Initializing per-variable variances with shape %s for %d groups: %s",
                shape,
                n_groups,
                list(self.variable_groups.keys()),
            )
        else:  # per_variable_per_level
            # One variance per (variable, level) pair
            n_vars = sum(len(indices) for indices in self.variable_groups.values())
            shape = (n_vars,)
            LOGGER.info(
                "Initializing per-variable-per-level variances with shape %s for %d variables",
                shape,
                n_vars,
            )

        # Initialize log_var parameter
        log_var_init = torch.full(shape, self.variance_init, dtype=torch.float32)
        self.log_var = nn.Parameter(log_var_init, requires_grad=self.variance_trainable)

        LOGGER.info(
            "GraphCastGaussianNLLLoss initialized: mode=%s, init_var=%.4f, trainable=%s",
            self.variance_mode,
            torch.exp(torch.tensor(self.variance_init)).item(),
            self.variance_trainable,
        )

    def _get_variance_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Build variance tensor with correct shape for broadcasting.

        Returns variance tensor that can broadcast with (B, E, G, V) tensors.

        Parameters
        ----------
        device : torch.device
            Target device for the variance tensor
        dtype : torch.dtype
            Target dtype for computation

        Returns
        -------
        torch.Tensor
            Variance tensor with shape appropriate for broadcasting:
            - per_variable mode: (1, 1, 1, V) where entries are repeated per group
            - per_variable_per_level mode: (1, 1, 1, V) with per-variable-level values
        """
        if self.log_var is None:
            raise RuntimeError(
                "log_var not initialized. Call set_data_indices() before forward pass."
            )

        # Compute var = exp(log_var) + eps
        # Cast to target dtype for computation
        var = torch.exp(self.log_var.to(dtype=dtype, device=device)) + self.variance_eps

        if self.variance_mode == "per_variable":
            # var has shape (n_groups,)
            # Need to expand to (1, 1, 1, n_vars_total) by repeating each group's var
            # across its levels

            var_expanded = []
            for i, (start_idx, end_idx) in enumerate(self.group_slices):
                n_levels = end_idx - start_idx
                # Repeat this group's variance for all its levels
                var_expanded.append(var[i].expand(n_levels))

            # Concatenate: (n_vars_total,)
            var_full = torch.cat(var_expanded, dim=0)

            # Reshape for broadcasting: (1, 1, 1, n_vars_total)
            return var_full.view(1, 1, 1, -1)

        else:  # per_variable_per_level
            # var already has shape (n_vars_total,)
            # Just reshape for broadcasting: (1, 1, 1, n_vars_total)
            return var.view(1, 1, 1, -1)

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate Gaussian NLL between prediction and target.

        Computes: r²/σ² + log(σ²)
        where r = pred - target is the residual in normalized tendency space.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (B, E, G, V)
        target : torch.Tensor
            Target tensor, shape (B, E, G, V)

        Returns
        -------
        torch.Tensor
            NLL tensor in float32, shape (B, E, G, V)
        """
        # Compute residual in float32 for numerical stability
        r = pred.float() - target.float()
        r_squared = torch.square(r)

        # Get variance tensor: (1, 1, 1, V)
        var = self._get_variance_tensor(device=r.device, dtype=r.dtype)

        # Compute NLL: r²/σ² + log(σ²)
        # Broadcasting: (B,E,G,V) / (1,1,1,V) + (1,1,1,V) -> (B,E,G,V)
        nll = r_squared / var + torch.log(var)

        # Scale by 0.5 (optional - can be omitted as it's a constant)
        # nll = 0.5 * nll

        return nll  # Keep in float32

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
        """Calculates the Gaussian NLL loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (B, E, G, V)
        target : torch.Tensor
            Target tensor, shape (B, E, G, V)
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
            Scalar loss value
        """
        is_sharded = grid_shard_slice is not None

        # Compute sample weights before calculating difference (need original target scale)
        sample_weights = None
        if self.sample_weighting:
            sample_weights = self._compute_sample_weights(target)

        # Compute NLL: (B, E, G, V)
        out = self.calculate_difference(pred, target)

        # Scale by node weights (spatial weighting)
        out = self.scale(out, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)

        # Reduce using GraphCast semantics
        return self.reduce(out, squash, group=group if is_sharded else None, sample_weights=sample_weights)

    def get_learned_variances(self) -> dict[str, torch.Tensor]:
        """Get the learned variances for inspection.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary mapping variable names to their learned variances.
            For per_variable mode, each group gets one variance.
            For per_variable_per_level mode, each variable-level gets one variance.
        """
        if self.log_var is None:
            raise RuntimeError("log_var not initialized. Call set_data_indices() first.")

        var = torch.exp(self.log_var).detach().cpu()

        result = {}
        if self.variance_mode == "per_variable":
            # One variance per group
            for i, (basename, _) in enumerate(self.variable_groups.items()):
                result[basename] = var[i].item()
        else:  # per_variable_per_level
            # One variance per variable-level
            idx = 0
            for basename, indices in self.variable_groups.items():
                for level_idx in range(len(indices)):
                    var_name = f"{basename}_{level_idx}" if len(indices) > 1 else basename
                    result[var_name] = var[idx].item()
                    idx += 1

        return result
