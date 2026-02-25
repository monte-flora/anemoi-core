# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

"""Robust GraphCast-style losses for training with extreme values.

This module provides robust loss functions that maintain GraphCast reduction
semantics while preventing gradient explosion from extreme values. These are
designed for modeling convective weather phenomena (thunderstorms) where:
- The model should make sharp, confident predictions
- Extreme values (50-100σ) occur naturally in the data
- Standard MSE can cause gradient explosion when predictions are confidently wrong

The losses here allow the model to confidently predict extreme events while
bounding the gradient magnitude when those predictions are incorrect.

NOTE: For LogCosh loss with GraphCast reduction, use GraphCastLogCoshLoss
from graphcast_logcosh.py. This module provides ClippedMSE and PseudoHuber.

Example YAML configuration:
    training:
      training_loss:
        _target_: anemoi.training.losses.GraphCastClippedMSELoss
        clip_value: 100.0  # Clip squared error to prevent explosion
        scalers: ['general_variable', 'limited_area_mask']
        ignore_nans: False
"""

from typing import TYPE_CHECKING

import torch

from anemoi.training.losses.base import GraphCastBaseLoss

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup

LOGGER = logging.getLogger(__name__)


class GraphCastClippedMSELoss(GraphCastBaseLoss):
    """MSE loss with per-element clipping to prevent gradient explosion.

    Computes MSE between predictions and targets, but clips the squared error
    contribution of each element to a maximum value. This prevents extreme
    mispredictions from causing gradient explosion while still allowing the
    model to make confident predictions.

    For example, with clip_value=100:
    - Prediction: +50σ, Target: 0σ -> Squared error: 2500 -> Clipped to: 100
    - Prediction: +5σ, Target: 0σ -> Squared error: 25 -> Not clipped

    This preserves the quadratic loss for normal predictions while bounding
    the gradient contribution of extreme errors.

    Parameters
    ----------
    clip_value : float, optional
        Maximum squared error per element. For normalized residuals with
        variance-based normalization, recommended values:
        - 100: clips errors >10σ (conservative)
        - 400: clips errors >20σ (moderate)
        - 900: clips errors >30σ (aggressive)
        Default 100.0.
    ignore_nans : bool, optional
        If True, use nanmean/nansum to ignore NaN values. Default False.
    log_clipping_stats : bool, optional
        If True, log statistics about clipping behavior. Default True.
    log_per_variable_loss : bool, optional
        If True, log loss per variable group. Default True.
    log_interval : int, optional
        Log statistics every N forward passes. Default 100.

    Example
    -------
    >>> loss_fn = GraphCastClippedMSELoss(clip_value=100.0)
    >>> loss_fn.set_data_indices(data_indices)
    >>> loss = loss_fn(pred, target)

    Notes
    -----
    This loss is more aggressive than Huber loss in encouraging sharp predictions.
    It remains quadratic (MSE-like) below the clip threshold, only saturating
    at extreme errors. This makes it suitable for applications where sharpness
    is critical (e.g., thunderstorm forecasting).
    """

    name: str = "graphcast_clipped_mse"

    def __init__(
        self,
        clip_value: float = 100.0,
        ignore_nans: bool = False,
        log_clipping_stats: bool = True,
        log_per_variable_loss: bool = True,
        log_interval: int = 100,
    ) -> None:
        super().__init__(ignore_nans=ignore_nans)
        self.clip_value = clip_value
        self.log_clipping_stats = log_clipping_stats
        self.log_per_variable_loss = log_per_variable_loss
        self.log_interval = log_interval
        self._step_counter = 0

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the clipped squared difference.

        NOTE: This method is not used when logging is enabled. See forward() for
        the implementation with logging support.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)

        Returns
        -------
        torch.Tensor
            Clipped squared difference tensor in float32.
        """
        # Compute in float32 for numerical stability with bfloat16
        diff = pred.float() - target.float()
        squared = torch.square(diff)

        # Clip squared error to prevent gradient explosion
        clipped = torch.clamp(squared, max=self.clip_value)

        return clipped  # Keep in float32

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
        """Calculates the area-weighted scaled loss with optional logging.

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
        self._step_counter += 1
        should_log = (self._step_counter % self.log_interval) == 0

        # Compute squared error (before clipping)
        # Compute in float32 for numerical stability with bfloat16
        diff = pred.float() - target.float()
        squared_unclipped = torch.square(diff)

        # Log clipping statistics (must detach to avoid gradient checkpointing errors)
        # PyTorch's gradient checkpointing saves intermediate activations during forward pass
        # and recomputes them during backward. If logging code creates branches in the
        # computational graph, recomputation can fail with shape mismatches. Detaching
        # the tensors before logging ensures logging code doesn't affect the graph.
        if should_log and self.log_clipping_stats:
            self._log_clipping_statistics(squared_unclipped.detach())

        # Apply clipping - this is the actual loss computation
        out = torch.clamp(squared_unclipped, max=self.clip_value)

        # Scale the output (general variable, by vertical level)
        out_scaled = self.scale(out, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)

        # Log per-variable losses (detached to avoid graph issues)
        if should_log and self.log_per_variable_loss and self.variable_groups is not None:
            self._log_per_variable_losses(out_scaled.detach())

        return self.reduce(out_scaled, squash, group=group if is_sharded else None)

    def _log_clipping_statistics(self, squared_error: torch.Tensor) -> None:
        """Log statistics about clipping behavior.

        Parameters
        ----------
        squared_error : torch.Tensor
            Unclipped squared error tensor, shape (B, E, G, V)
        """
        # Only log from rank 0 in distributed training
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() != 0:
                return

        # Overall clipping statistics
        total_elements = squared_error.numel()
        clipped_mask = squared_error > self.clip_value
        num_clipped = clipped_mask.sum().item()
        pct_clipped = 100.0 * num_clipped / total_elements

        max_value = squared_error.max().item()
        max_sigma = torch.sqrt(torch.tensor(max_value)).item()
        clip_sigma = torch.sqrt(torch.tensor(self.clip_value)).item()

        if num_clipped > 0:
            LOGGER.info(
                "ClipMSE | Clipped: %.3f%% (%d/%d) | Max: %.1f (%.1fσ) | Threshold: %.1f (%.1fσ)",
                pct_clipped,
                num_clipped,
                total_elements,
                max_value,
                max_sigma,
                self.clip_value,
                clip_sigma,
            )

            # Per-variable clipping statistics
            if self.variable_groups is not None:
                self._log_per_variable_clipping(squared_error, clipped_mask)
        else:
            LOGGER.info(
                "ClipMSE | No clipping | Max: %.1f (%.1fσ)",
                max_value,
                max_sigma,
            )

    def _log_per_variable_clipping(self, squared_error: torch.Tensor, clipped_mask: torch.Tensor) -> None:
        """Log clipping statistics per variable group.

        Parameters
        ----------
        squared_error : torch.Tensor
            Unclipped squared error tensor, shape (B, E, G, V)
        clipped_mask : torch.Tensor
            Boolean mask of clipped elements, shape (B, E, G, V)
        """
        # Compute per-variable clipping percentages
        var_clip_stats = []
        for var_name, (start_idx, end_idx) in zip(self.variable_groups.keys(), self.group_slices):
            var_squared = squared_error[..., start_idx:end_idx]
            var_clipped = clipped_mask[..., start_idx:end_idx]

            var_total = var_squared.numel()
            var_num_clipped = var_clipped.sum().item()

            if var_num_clipped > 0:
                var_pct_clipped = 100.0 * var_num_clipped / var_total
                var_max = var_squared.max().item()
                var_max_sigma = torch.sqrt(torch.tensor(var_max)).item()
                var_clip_stats.append((var_name, var_pct_clipped, var_max_sigma, var_num_clipped))

        # Log top variables with most clipping (compact format)
        if var_clip_stats:
            var_clip_stats.sort(key=lambda x: x[2], reverse=True)  # Sort by max_sigma
            top_vars = [f"{name}:{max_sig:.1f}σ({pct:.2f}%)"
                       for name, pct, max_sig, _ in var_clip_stats[:5]]
            LOGGER.info("  Top clipped vars: %s", " | ".join(top_vars))

    def _log_per_variable_losses(self, scaled_error: torch.Tensor) -> None:
        """Log loss contribution per variable group.

        Parameters
        ----------
        scaled_error : torch.Tensor
            Scaled squared error tensor, shape (B, E, G, V)
        """
        # Only log from rank 0 in distributed training
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() != 0:
                return

        # Use the GraphCast reduction to get per-variable losses
        # This gives us (B, n_groups)
        per_var_losses = self._reduce_per_variable(scaled_error)

        # Average over batch: (n_groups,)
        per_var_losses_mean = per_var_losses.mean(dim=0)

        # Format as compact single line
        loss_strs = [f"{var}: {loss_val.item():.4f}" for var, loss_val in
                     zip(self.variable_groups.keys(), per_var_losses_mean)]
        LOGGER.info("Per-variable losses: %s", " | ".join(loss_strs))


class GraphCastPseudoHuberLoss(GraphCastBaseLoss):
    """Pseudo-Huber loss (smooth L1) with GraphCast reduction semantics.

    The pseudo-Huber loss is a smooth approximation to the Huber loss:
    L(x) = δ² * (sqrt(1 + (x/δ)²) - 1)

    Properties:
    - Smooth everywhere (unlike Huber which has a discontinuous derivative)
    - Approximately quadratic for |x| < δ
    - Approximately linear for |x| > δ
    - More aggressive transition than log-cosh

    This loss is similar to Huber but with a smooth transition, making it
    suitable for gradient-based optimization. The parameter δ controls the
    transition point from quadratic to linear behavior.

    Parameters
    ----------
    delta : float, optional
        Transition point from quadratic to linear. For normalized residuals:
        - delta=5.0: transition at 5σ error (conservative)
        - delta=10.0: transition at 10σ error (moderate)
        - delta=20.0: transition at 20σ error (aggressive)
        Default 10.0.
    ignore_nans : bool, optional
        If True, use nanmean/nansum to ignore NaN values. Default False.

    Example
    -------
    >>> loss_fn = GraphCastPseudoHuberLoss(delta=10.0)
    >>> loss_fn.set_data_indices(data_indices)
    >>> loss = loss_fn(pred, target)

    Notes
    -----
    The pseudo-Huber loss is more robust than MSE but less so than log-cosh.
    It's a good middle ground if you want to preserve sharp predictions while
    having moderate protection against gradient explosion.

    Reference: https://en.wikipedia.org/wiki/Huber_loss#Pseudo-Huber_loss_function
    """

    name: str = "graphcast_pseudo_huber"

    def __init__(self, delta: float = 10.0, ignore_nans: bool = False) -> None:
        super().__init__(ignore_nans=ignore_nans)
        self.delta = delta

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the pseudo-Huber loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)

        Returns
        -------
        torch.Tensor
            Pseudo-Huber loss tensor in float32.
        """
        # Compute in float32 for numerical stability
        diff = pred.float() - target.float()

        # Pseudo-Huber: δ² * (sqrt(1 + (x/δ)²) - 1)
        term = 1.0 + (diff / self.delta) ** 2
        loss = self.delta**2 * (torch.sqrt(term) - 1.0)

        return loss  # Keep in float32

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

        out = self.calculate_difference(pred, target)

        # Scale the output (general variable, by vertical level)
        out = self.scale(out, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)

        return self.reduce(out, squash, group=group if is_sharded else None)
