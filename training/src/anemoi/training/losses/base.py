# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import functools
import logging
from abc import ABC
from abc import abstractmethod

import torch
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.distributed.graph import reduce_tensor
from anemoi.training.losses.scaler_tensor import ScaleTensor
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


class BaseLoss(nn.Module, ABC):
    """Base loss."""

    scaler: ScaleTensor

    def __init__(self, ignore_nans: bool = False) -> None:
        """Node- and feature_weighted Loss.

        Exposes:
        - self.avg_function: torch.nanmean or torch.mean
        - self.sum_function: torch.nansum or torch.sum
        depending on the value of `ignore_nans`

        Registers:
        - self.scaler: ScaleTensor modified with `add_scaler` and `update_scaler`

        These losses are designed for use within the context of
        the anemoi-training configuration, where scalars are added
        after initialisation. If being used outside of this
        context, call `add_scalar` and `update_scalar` to add or
        update the scale tensors.

        Parameters
        ----------
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False

        """
        super().__init__()

        self.add_module("scaler", ScaleTensor())

        self.avg_function = torch.nanmean if ignore_nans else torch.mean
        self.sum_function = torch.nansum if ignore_nans else torch.sum

        self.supports_sharding = True
        self.num_scales = 1

    @functools.wraps(ScaleTensor.add_scaler)
    def add_scaler(self, dimension: int | tuple[int], scaler: torch.Tensor, *, name: str | None = None) -> None:
        self.scaler.add_scaler(dimension=dimension, scaler=scaler, name=name)

    @functools.wraps(ScaleTensor.update_scaler)
    def update_scaler(self, name: str, scaler: torch.Tensor, *, override: bool = False) -> None:
        self.scaler.update_scaler(name=name, scaler=scaler, override=override)

    def set_data_indices(self, data_indices: IndexCollection) -> None:
        """Hook to set the data indices for the loss."""

    def scale(
        self,
        x: torch.Tensor,
        subset_indices: tuple[int, ...] | None = None,
        *,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
    ) -> torch.Tensor:
        """Scale a tensor by the variable_scaling.

        Parameters
        ----------
        x : torch.Tensor
            Tensor to be scaled, shape (bs, ensemble, lat*lon, n_outputs)
        subset_indices: tuple[int,...], optional
            Indices to subset the calculated scaler and `x` tensor with, by default None.
        without_scalers: list[str] | list[int] | None, optional
            list of scalers to exclude from scaling. Can be list of names or dimensions to exclude.
            By default None
        grid_shard_slice : slice, optional
            Slice of the grid if x comes sharded, by default None

        Returns
        -------
        torch.Tensor
            Scaled error tensor
        """
        if subset_indices is None:
            subset_indices = [Ellipsis]

        if len(self.scaler) == 0:
            return x[subset_indices]

        if TensorDim.GRID not in self.scaler:
            error_msg = (
                "Scaler tensor must be at least applied to the GRID dimension. "
                "Please add a scaler here, use `UniformWeights` for simple uniform scaling.",
            )
            raise RuntimeError(error_msg)

        scale_tensor = self.scaler
        if without_scalers is not None and len(without_scalers) > 0:
            if isinstance(without_scalers[0], str):
                scale_tensor = self.scaler.without(without_scalers)
            else:
                scale_tensor = self.scaler.without_by_dim(without_scalers)

        return scale_tensor.scale_iteratively(
            x,
            subset_indices=subset_indices,
            grid_shard_slice=grid_shard_slice,
        )

    def reduce(
        self,
        out: torch.Tensor,
        squash: bool = True,
        squash_mode: str = "avg", # Originally avg
        group: ProcessGroup | None = None,
    ) -> torch.Tensor:
        """Reduce the out of the loss.

        If `squash` is True, the last dimension is averaged.

        Irrespective of `squash`, the output is reduced over the
        batch, ensemble and grid dimensions.

        Parameters
        ----------
        out : torch.Tensor
            Difference tensor, of shape TensorDim
        squash : bool, optional
            Whether to squash the variable dimension, by default True
        squash_mode : str, optional
            Mode to use for squashing the variable dimension, by default "avg"
            If "avg", the last dimension is averaged.
            If "sum", the last dimension is summed.

        Returns
        -------
        torch.Tensor
            Reduced output tensor

        Raises
        ------
        ValueError
            If squash_mode is not one of ['avg', 'sum']
        """
        #[B, Ens, Grid, Variable] 
        
        if squash:
            if squash_mode == "avg":
                out = self.avg_function(out, dim=TensorDim.VARIABLE)
            elif squash_mode == "sum":
                out = self.sum_function(out, dim=TensorDim.VARIABLE)
            else:
                msg = f"Invalid squash_mode '{squash_mode}'. Supported modes are: 'avg', 'sum'"
                raise ValueError(msg)

        # Monte: commented out to see what happens with avg all dimensions. 
        # here the grid dimension is summed because the normalisation is handled in the node weighting
        grid_summed = self.sum_function(out, dim=(TensorDim.GRID))

        out = self.avg_function(
            grid_summed,
            #out,
            dim=(
                TensorDim.BATCH_SIZE,
                TensorDim.ENSEMBLE_DIM,
                #TensorDim.GRID Add this in in not grid summing
            ),
        )

        return out if group is None else reduce_tensor(out, group)

    @property
    def name(self) -> str:
        """Used for logging identification purposes."""
        return self.__class__.__name__.lower()

    @abstractmethod
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
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
            list of scalers to exclude from scaling. Can be list of names or dimensions to exclude.
            By default None
        grid_shard_slice : slice, optional
            Slice of the grid if x comes sharded, by default None
        group: ProcessGroup, optional
            Distributed group to reduce over, by default None

        Returns
        -------
        torch.Tensor
            Weighted loss
        """


class FunctionalLoss(BaseLoss):
    """Loss which a user can subclass and provide `calculate_difference`.

    `calculate_difference` should calculate the difference between the prediction and target.
    All scaling and weighting is handled by the parent class.

    Example:
    --------
    ```python
    class MyLoss(FunctionalLoss):
        def calculate_difference(self, pred, target):
            return pred - target
    ```
    """

    @abstractmethod
    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate difference between prediction and target."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        **kwargs,  # noqa: ARG002
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
            list of scalers to exclude from scaling. Can be list of names or dimensions to exclude.
            By default None
        grid_shard_slice : slice, optional
            Slice of the grid if x comes sharded, by default None
        group: ProcessGroup, optional
            Distributed group, by default None

        Returns
        -------
        torch.Tensor
            Weighted loss
        """
        is_sharded = grid_shard_slice is not None
        out = self.calculate_difference(pred, target)
        out = self.scale(out, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)

        return self.reduce(out, squash, group=group if is_sharded else None)
    
    
class GraphCastBaseLoss(FunctionalLoss):
    """Loss with GraphCast-style reduction semantics.

    Implements the reduction order from Lam et al. (2023) GraphCast paper:
    1. Mean over vertical levels within each variable group
    2. Mean over spatial (grid) and ensemble dimensions
    3. Sum over variable groups
    4. Mean over batch (optionally weighted by sample extremity)

    This ensures each physical quantity (e.g., temperature, humidity) contributes
    equally to the loss regardless of the number of vertical levels, unlike
    the default Anemoi reduction which weights by level count.

    Optionally supports sample weighting to downweight samples with extreme
    target values, which helps with heavy-tailed distributions common in
    convective weather prediction.

    Reference:
        Lam, R., et al. (2023). "Learning skillful medium-range global weather
        forecasting." Science, 382(6677), 1416-1421.
    """

    def __init__(
        self,
        *args,
        sample_weighting: bool = False,
        sample_weight_threshold: float = 10.0,
        sample_weight_min: float = 0.01,
        **kwargs,
    ) -> None:
        """Initialize GraphCastBaseLoss.

        Parameters
        ----------
        sample_weighting : bool, optional
            If True, downweight samples with extreme target values. Default False.
        sample_weight_threshold : float, optional
            Target magnitude threshold for weighting. Samples with max |target| above
            this value get progressively downweighted. Default 10.0 (10 sigma for
            normalized tendencies).
        sample_weight_min : float, optional
            Minimum weight for extreme samples. Default 0.01 (1% of normal weight).
        """
        super().__init__(*args, **kwargs)
        self.variable_groups: dict[str, list[int]] | None = None
        self.group_slices: list[tuple[int, int]] | None = None
        self.group_sizes: torch.Tensor | None = None

        # Sample weighting parameters
        self.sample_weighting = sample_weighting
        self.sample_weight_threshold = sample_weight_threshold
        self.sample_weight_min = sample_weight_min

        # Per-variable group loss logging
        self._group_loss_log_interval = 250  # Log every N calls (matches total loss MLflow interval)
        self._group_loss_call_count = 0

        # Store last per-group losses for MLflow logging by the training task
        self._last_per_group_losses: dict[str, float] | None = None

    def _build_variable_groups(self) -> dict[str, list[int]]:
        """Build variable groups from data indices.

        Groups variables by their base name (e.g., "temperature_500" -> "temperature").
        Only includes prognostic variables in the grouping.

        IMPORTANT: The indices returned are in the "prognostic-only" space (0 to n_prognostic-1),
        not the full output space. This is because the loss function receives tensors that have
        already been indexed to only include prognostic variables.

        Returns
        -------
        dict[str, list[int]]
            Mapping from base variable name to list of prognostic-space indices.
        """
        prognostic_indices = self.data_indices.data.output.prognostic
        name_to_idx = self.data_indices.data.output.name_to_index

        # Create mapping from full output index to prognostic-only index
        # e.g., if prognostic_indices = [2, 5, 6, 7, 84, 85], then:
        #   full_to_prognostic = {2: 0, 5: 1, 6: 2, 7: 3, 84: 4, 85: 5}
        sorted_prognostic = sorted(prognostic_indices.tolist() if hasattr(prognostic_indices, 'tolist') else list(prognostic_indices))
        full_to_prognostic = {full_idx: prog_idx for prog_idx, full_idx in enumerate(sorted_prognostic)}

        LOGGER.debug(
            "Building variable groups: %d prognostic variables, full indices range [%d, %d]",
            len(sorted_prognostic),
            min(sorted_prognostic) if sorted_prognostic else -1,
            max(sorted_prognostic) if sorted_prognostic else -1,
        )

        groups: dict[str, list[int]] = {}
        for name, idx in name_to_idx.items():
            if idx not in prognostic_indices:
                continue

            # Parse variable name: "qv_7" -> base="qv", or "t2m" -> base="t2m"
            if "_" in name and name.split("_")[-1].isdigit():
                base = "_".join(name.split("_")[:-1])
            else:
                base = name

            # Convert from full output index to prognostic-only index
            prog_idx = full_to_prognostic[idx]
            groups.setdefault(base, []).append(prog_idx)

        # Sort indices within each group for consistent ordering
        for base in groups:
            groups[base].sort()

        return groups

    def set_data_indices(self, data_indices: IndexCollection) -> None:
        """Initialize variable groups from data indices.

        Parameters
        ----------
        data_indices : IndexCollection
            Collection of data indices from the model.

        Raises
        ------
        ValueError
            If any variable group has non-contiguous indices.
        """
        self.data_indices = data_indices
        self.variable_groups = self._build_variable_groups()

        # Build contiguous slices for efficient tensor indexing
        self.group_slices = []
        for basename, idxs in self.variable_groups.items():
            start_idx = min(idxs)
            end_idx = max(idxs) + 1

            # Verify indices are contiguous for efficient slicing
            expected = list(range(start_idx, end_idx))
            if idxs != expected:
                error_msg = (
                    f"Variable group '{basename}' has non-contiguous indices: {idxs}. "
                    f"Expected contiguous range: {expected}. "
                    "GraphCast loss requires variables to be stored contiguously by level."
                )
                raise ValueError(error_msg)

            self.group_slices.append((start_idx, end_idx))

        self.group_sizes = torch.tensor(
            [len(idxs) for idxs in self.variable_groups.values()],
            dtype=torch.float,
        )

        LOGGER.info(
            "GraphCastBaseLoss initialized with %d variable groups: %s",
            len(self.variable_groups),
            list(self.variable_groups.keys()),
        )

        if self.sample_weighting:
            LOGGER.info(
                "Sample weighting enabled: threshold=%.1f, min_weight=%.3f",
                self.sample_weight_threshold,
                self.sample_weight_min,
            )

    def _compute_sample_weights(self, target: torch.Tensor) -> torch.Tensor:
        """Compute per-sample weights based on target extremity.

        Samples with extreme target values (large |target|) are downweighted
        to reduce the influence of outliers during training.

        The weighting scheme:
        - weight = 1.0 for samples with max|target| <= threshold
        - weight decreases for samples with max|target| > threshold
        - weight is clamped to sample_weight_min to avoid zero weights

        Parameters
        ----------
        target : torch.Tensor
            Target tensor of shape (B, E, G, V)

        Returns
        -------
        torch.Tensor
            Per-sample weights of shape (B,), normalized to sum to B
        """
        # Compute max absolute target value per sample: (B,)
        # Max over ensemble, grid, and variable dimensions
        max_abs_target = torch.abs(target).amax(dim=(1, 2, 3))  # (B,)

        # Compute weights: 1.0 for normal samples, decreasing for extreme samples
        # weight = threshold / max(threshold, max_abs_target)
        weights = self.sample_weight_threshold / torch.clamp(
            max_abs_target, min=self.sample_weight_threshold
        )

        # Clamp to minimum weight
        weights = torch.clamp(weights, min=self.sample_weight_min)

        # Normalize weights so they sum to batch_size (preserves loss scale)
        weights = weights * (weights.numel() / weights.sum())

        return weights

    def _reduce_per_variable(self, out: torch.Tensor) -> torch.Tensor:
        """Reduce loss tensor to per-variable-group values.

        Implements GraphCast reduction:
        1. Mean over levels within each variable group
        2. Sum over grid dimension (requires unit-sum normalized weights)
        3. Mean over ensemble dimension

        IMPORTANT: This method assumes loss weights (e.g., limited_area_mask, node_weights)
        are normalized to sum to 1 (norm: "unit-sum"). When weights sum to 1, summing
        the weighted losses gives the correct weighted mean. If weights are not normalized,
        the loss will be incorrectly scaled.

        Parameters
        ----------
        out : torch.Tensor
            Loss tensor of shape (B, E, G, V_flat), already scaled by weights

        Returns
        -------
        torch.Tensor
            Reduced tensor of shape (B, n_groups)
        """
        per_group_means = []
        for i, (start_idx, end_idx) in enumerate(self.group_slices):
            # Extract contiguous slice: (B, E, G, n_levels)
            group_data = out[..., start_idx:end_idx]

            # Mean over levels: (B, E, G)
            group_mean = group_data.mean(dim=-1)

            per_group_means.append(group_mean)

        # Stack: (B, E, G, n_groups)
        per_group = torch.stack(per_group_means, dim=-1)

        # Sum over grid (weights are unit-sum normalized), then mean over ensemble: (B, n_groups)
        # NOTE: Using sum over grid because weights should sum to 1 globally.
        # This matches FunctionalLoss.reduce() behavior and correctly handles
        # masked losses (e.g., limited_area_mask) when norm="unit-sum".
        result = per_group.sum(dim=TensorDim.GRID).mean(dim=TensorDim.ENSEMBLE_DIM)

        return result

    def _log_per_group_losses(self, per_group_loss: torch.Tensor, group: "ProcessGroup | None" = None) -> None:
        """Log per-variable-group loss statistics and store for MLflow logging.

        Only logs from rank 0 to avoid interleaved output from multiple GPUs.
        Always stores per-group losses in self._last_per_group_losses for
        retrieval by the training task (MLflow logging).

        Parameters
        ----------
        per_group_loss : torch.Tensor
            Loss tensor of shape (B, n_groups) after reduction over levels, ensemble, and grid.
            When grid is sharded, this contains LOCAL sums.
        group : ProcessGroup | None
            Model parallel group for distributed reduction. Used to scale local values
            to global estimates.
        """
        if self.variable_groups is None:
            return

        # Only log from rank 0 to avoid interleaved output
        is_rank_zero = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if not is_rank_zero:
            return

        with torch.no_grad():
            # Mean over batch: (n_groups,)
            group_means = per_group_loss.mean(dim=0)

            # Scale by model parallel group size to show global values
            if group is not None:
                model_parallel_size = torch.distributed.get_world_size(group)
                group_means = group_means * model_parallel_size

            # Get group names and sort by loss (descending)
            group_names = list(self.variable_groups.keys())
            losses_with_names = [(name, group_means[i].item()) for i, name in enumerate(group_names)]

            # Store for MLflow logging by the training task
            self._last_per_group_losses = {name: val for name, val in losses_with_names}

            losses_with_names.sort(key=lambda x: x[1], reverse=True)

            LOGGER.info("=" * 70)
            LOGGER.info("PER-VARIABLE GROUP LOSS (sorted by loss, descending)")
            LOGGER.info("=" * 70)
            LOGGER.info(f"  {'Group':<25} {'Loss':>12} {'Levels':>8}")
            LOGGER.info(f"  {'-'*25} {'-'*12} {'-'*8}")

            total_loss = 0.0
            for name, loss_val in losses_with_names:
                n_levels = len(self.variable_groups[name])
                total_loss += loss_val
                LOGGER.info(f"  {name:<25} {loss_val:>12.4f} {n_levels:>8}")

            LOGGER.info(f"  {'-'*25} {'-'*12} {'-'*8}")
            LOGGER.info(f"  {'TOTAL':<25} {total_loss:>12.4f}")
            LOGGER.info("=" * 70)

    def reduce(
        self,
        out: torch.Tensor,
        squash: bool = True,
        squash_mode: str = "sum",  # Ignored, always sums for GraphCast semantics
        group: ProcessGroup | None = None,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reduce loss tensor using GraphCast semantics.

        Reduction order:
        1. Mean over levels within each variable group
        2. Mean over ensemble and grid dimensions
        3. Sum over variable groups (if squash=True)
        4. Weighted mean over batch (if sample_weights provided)

        Parameters
        ----------
        out : torch.Tensor
            Loss tensor of shape (B, E, G, V)
        squash : bool, optional
            If True, sum over variables and return scalar.
            If False, return per-variable-group losses. Default True.
        squash_mode : str, optional
            Ignored. GraphCast always uses sum over variables.
        group : ProcessGroup | None, optional
            Distributed process group for reduction. Default None.
        sample_weights : torch.Tensor | None, optional
            Per-sample weights of shape (B,). If provided, used for weighted
            mean over batch dimension. Default None.

        Returns
        -------
        torch.Tensor
            Scalar loss if squash=True, else tensor of shape (n_groups,)
        """
        # (B, E, G, V) -> (B, n_groups)
        per_group_loss = self._reduce_per_variable(out)

        # Log per-variable group losses periodically and store for MLflow
        self._group_loss_call_count += 1
        if self._group_loss_call_count % self._group_loss_log_interval == 1:
            self._log_per_group_losses(per_group_loss, group=group)
        else:
            self._last_per_group_losses = None

        out = per_group_loss

        if squash:
            # Sum over variable groups: (B,)
            out = self.sum_function(out, dim=-1)

            # For distributed training with sharded grid:
            # 1. First reduce across GPUs to get global grid sum (per batch sample)
            # 2. Then average over batch
            # NOTE: No divide by world_size because weights are unit-sum normalized.
            # When weights sum to 1 globally, all_reduce SUM of partial sums = global weighted mean.
            if group is not None:
                out = reduce_tensor(out, group)

            # Weighted mean over batch: scalar
            if sample_weights is not None:
                # Weighted average: sum(out * weights) / sum(weights)
                out = (out * sample_weights).sum() / sample_weights.sum()
            else:
                out = self.avg_function(out)

            return out

        # For per-variable diagnostics: reduce across GPUs first, then batch
        if group is not None:
            out = reduce_tensor(out, group)

        if sample_weights is not None:
            # Weighted average per group: (n_groups,)
            out = (out * sample_weights.unsqueeze(-1)).sum(dim=0) / sample_weights.sum()
        else:
            out = self.avg_function(out, dim=TensorDim.BATCH_SIZE)

        return out
          