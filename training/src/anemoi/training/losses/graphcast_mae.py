# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""GraphCast-style MAE loss with level-grouped variable reduction.

Same per-channel reduction semantics as :class:`GraphCastMSELoss` but
with the per-cell error term replaced by |y - x| (absolute) instead of
(y - x)² (squared). Use this for v30 Variant-B decoder training (and
other absolute-error pipelines) where the σ-scaler is the linear
:class:`StdevTendencyScaler` rather than the squared
:class:`VarTendencyScaler`. The pairing rule (match powers between the
error term and the σ scaler) is documented in
``feedback-loss-tendency-pairing``.

Example YAML configuration:
    training:
      training_loss:
        _target_: anemoi.training.losses.GraphCastMAELoss
        scalers: ['general_variable', 'stdev_tendency', 'limited_area_mask']
        ignore_nans: False
"""

from typing import TYPE_CHECKING

import torch

from anemoi.training.losses.base import GraphCastBaseLoss

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup


class GraphCastMAELoss(GraphCastBaseLoss):
    """Mean Absolute Error loss with GraphCast-style reduction.

    Per-cell metric is ``|pred - target|``; everything downstream of
    ``calculate_difference`` (scale + reduce + level grouping + variable-
    group sum + batch weighting) mirrors :class:`GraphCastMSELoss`. The
    only difference is the error term's power (1 vs 2).

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

    Notes
    -----
    Pair with :class:`StdevTendencyScaler` (NOT
    :class:`VarTendencyScaler`) when the model emits mean-std residuals.
    The MAE term has magnitude ``~σ_tend/σ_var`` per channel, so the
    correct compensating weight is ``(σ_var/σ_tend)¹``. ``v17``-style
    MSE used the squared pairing; v30 decoder uses this linear one.
    """

    name: str = "graphcast_mae"

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
        """Per-cell absolute error in float32.

        Compute in float32 for numerical stability with bfloat16 inputs
        (matches GraphCastMSELoss policy). The MAE has no overflow risk
        the way MSE does, but consistency with the MSE pipeline keeps
        the rest of the reduction code identical.
        """
        diff = pred.float() - target.float()
        return torch.abs(diff)

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
        """Identical to GraphCastMSELoss.forward — only the per-cell metric differs."""
        is_sharded = grid_shard_slice is not None
        sample_weights = None
        out = self.calculate_difference(pred, target)
        out = self.scale(
            out, scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )
        return self.reduce(
            out, squash,
            group=group if is_sharded else None,
            sample_weights=sample_weights,
        )
