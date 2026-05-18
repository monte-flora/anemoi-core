# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""GraphCast-style CRPS loss with level-grouped variable reduction.

Provides a fair / almost-fair kernel CRPS loss that follows the same
reduction semantics as ``GraphCastMSELoss`` — variables grouped by base
name, mean over levels within a group, sum over groups, weighted batch
mean. Lets a CRPS training run be a single-knob swap (MSE → CRPS) from
a ``GraphCastMSELoss`` deterministic recipe.

Math is ported from ``kcrps.AlmostFairKernelCRPS`` (the bare
``BaseLoss``-derived variant). The only structural change is that the
CRPS value (which collapses the ensemble dimension) is reshaped back to
``(B, 1, G, V)`` so the inherited ``GraphCastBaseLoss.reduce()`` works
unchanged.

Example YAML:

    training:
      training_loss:
        _target_: anemoi.training.losses.GraphCastCRPSLoss
        alpha: 1.0                     # 1.0 = fair, 0.0 = unfair, blend in between
        scalers: ['general_variable', 'limited_area_mask']
        ignore_nans: False
"""
from typing import TYPE_CHECKING

import einops
import torch

from anemoi.training.losses.base import GraphCastBaseLoss

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup


class GraphCastCRPSLoss(GraphCastBaseLoss):
    """Almost-fair kernel CRPS loss with GraphCast-style reduction.

    Parameters
    ----------
    alpha : float, default 1.0
        Linear blend between fair (1.0) and unfair (0.0) CRPS.
        Fair CRPS uses ``coef = 1 / (2 * N * (N - 1))`` to correct the
        finite-ensemble bias; this matches FGN's training objective.
    no_autocast : bool, default True
        Disable autocast for the CRPS calculation. Recommended when
        training in bf16 — the pairwise differences are small and
        precision matters.
    ignore_nans : bool, default False
        Use nanmean / nansum-style reductions to skip NaN cells.
    sample_weighting : bool, default False
        Downweight samples with extreme target values. Same machinery as
        ``GraphCastMSELoss``.
    sample_weight_threshold : float, default 10.0
        Max-abs-target threshold above which samples get progressively
        downweighted.
    sample_weight_min : float, default 0.01
        Lower bound on per-sample weight.

    Notes
    -----
    The CRPS estimator (Hersbach 2000, fair-CRPS variant from Ferro
    et al. 2008):

        fCRPS(x^{1:N}, y) = (1/N) Σ_n |x^n - y|
                         - (1/(2 N (N-1))) Σ_{n,n'} |x^n - x^{n'}|

    ``alpha`` blends this with the naive (unfair) form
    ``(1/(2N²)) Σ |x^n - x^{n'}|`` via the ``epsilon = (1-alpha)/N``
    parameter, exactly as in ``AlmostFairKernelCRPS``.
    """

    name: str = "graphcast_crps"

    def __init__(
        self,
        alpha: float = 1.0,
        no_autocast: bool = True,
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
        self.alpha = float(alpha)
        self.no_autocast = bool(no_autocast)

    def _kernel_crps(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """Per-cell-per-variable almost-fair kernel CRPS.

        Parameters
        ----------
        preds : torch.Tensor
            Ensemble predictions, shape ``(B, V, G, E)``.
        targets : torch.Tensor
            Single-realisation target, shape ``(B, V, G)``.
        alpha : float
            Fair/unfair blend coefficient (1.0 = fair).

        Returns
        -------
        torch.Tensor
            Pointwise CRPS values, shape ``(B, V, G)``.
        """
        ens_size = preds.shape[-1]
        if ens_size < 2:
            msg = (
                f"GraphCastCRPSLoss requires ensemble size >= 2 to evaluate "
                f"fair-CRPS, got E={ens_size}."
            )
            raise ValueError(msg)

        epsilon = (1.0 - alpha) / ens_size

        # |x^n - x^{n'}| for all (n, n') including the diagonal — broadcast.
        # Shape: (B, V, G, E, E)
        var = torch.abs(preds.unsqueeze(dim=-1) - preds.unsqueeze(dim=-2))
        diag = torch.eye(ens_size, dtype=torch.bool, device=preds.device)

        # |x^n - y| broadcast to (B, V, G, E, E), masked to off-diagonal.
        err_r = einops.repeat(
            torch.abs(preds - targets.unsqueeze(dim=-1)),
            "b v g e -> b v g n e",
            n=ens_size,
        )
        mem_err = err_r * (~diag)
        mem_err_transpose = mem_err.transpose(-1, -2)

        # Coefficient for the fair-CRPS skill term; the (1 - epsilon) factor
        # in front of `var` linearly interpolates between fair and unfair.
        coef = 1.0 / (2.0 * ens_size * (ens_size - 1))
        return coef * torch.sum(
            mem_err + mem_err_transpose - (1.0 - epsilon) * var,
            dim=(-1, -2),
        )

    def calculate_difference(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the per-cell-per-variable CRPS contribution.

        ``GraphCastBaseLoss.forward`` calls this, then applies ``scale``
        and the GraphCast ``reduce`` (mean-over-levels-per-group →
        sum-over-grid → mean-over-ensemble → sum-over-groups → batch
        mean). Because the CRPS estimator inherently collapses the
        ensemble dimension, we return a singleton ``E=1`` so the
        downstream reduction is unchanged.

        Parameters
        ----------
        pred : torch.Tensor
            Ensemble predictions, shape ``(B, E, G, V)`` with ``E >= 2``.
        target : torch.Tensor
            Truth, shape ``(B, 1, G, V)`` (singleton ensemble dim) or
            ``(B, G, V)``.

        Returns
        -------
        torch.Tensor
            Pointwise CRPS values, shape ``(B, 1, G, V)``. The value at
            ``[b, 0, g, v]`` is the fair-CRPS of the ``E`` predictions
            at cell ``g``, variable ``v``, sample ``b`` against the truth.
        """
        # Collapse any singleton ensemble dimension on the target.
        if target.ndim == 4:
            if target.shape[1] != 1:
                # Multi-realisation target shouldn't happen at training
                # time, but average defensively rather than erroring.
                target = target.mean(dim=1)
            else:
                target = target.squeeze(1)

        # Cast to float32 for the CRPS math — pairwise abs-diffs are
        # precision-sensitive under bf16.
        preds_f = pred.float()
        target_f = target.float()

        # Permute to the (B, V, G, E) / (B, V, G) convention used by
        # AlmostFairKernelCRPS._kernel_crps so the math is byte-identical.
        preds_p = einops.rearrange(preds_f, "b e g v -> b v g e")
        target_p = einops.rearrange(target_f, "b g v -> b v g")

        if self.no_autocast:
            # The pairwise-abs-diff computation accumulates many small
            # terms; disable autocast so we stay in fp32.
            with torch.amp.autocast(device_type="cuda", enabled=False):
                crps_per_cell = self._kernel_crps(preds_p, target_p, self.alpha)
        else:
            crps_per_cell = self._kernel_crps(preds_p, target_p, self.alpha)

        # Reshape back to (B, E=1, G, V) for the inherited scale + reduce.
        return einops.rearrange(crps_per_cell, "b v g -> b 1 g v")

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
        **kwargs,  # noqa: ARG002
    ) -> torch.Tensor:
        """Run CRPS → scale → GraphCast reduce.

        Mirrors ``GraphCastMSELoss.forward``. The only override of the
        ``FunctionalLoss`` base path is that we pre-compute
        ``sample_weights`` from the target BEFORE ``calculate_difference``
        collapses the ensemble dim, matching the MSE loss behaviour.
        """
        is_sharded = grid_shard_slice is not None

        sample_weights = None
        if self.sample_weighting:
            # Use the target shape pre-squeeze so the reduction matches
            # `_compute_sample_weights`'s expected (B, E, G, V) signature.
            if target.ndim == 3:
                sw_target = target.unsqueeze(1)
            else:
                sw_target = target
            sample_weights = self._compute_sample_weights(sw_target)

        out = self.calculate_difference(pred, target)
        out = self.scale(
            out,
            scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )

        return self.reduce(
            out,
            squash,
            group=group if is_sharded else None,
            sample_weights=sample_weights,
        )
