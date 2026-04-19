# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Horizontal-gradient wrapper loss.

FastNet-style (arXiv:2509.17658, Sec. 5 "horizontal gradients in the loss")
augmentation: compute zonal and meridional spatial derivatives of both the
forecast and the target, z-score them per variable with an *online Welford*
running stddev, and pass each through the inner loss. The final loss is

    L = λ_raw · L(raw) + λ_∂x · L(∂x pred, ∂x tgt) + λ_∂y · L(∂y pred, ∂y tgt)

Why this helps: non-physical artefacts (patch seams, checkerboards) tend to
show up more prominently in spatial gradients than in the raw fields. Adding
the gradient term to the loss penalises models that produce raw fields with
correct bulk statistics but artefact-ridden derivatives.

Design notes
------------
* Wrapper, not a loss itself. Delegates to any inner BaseLoss (MSE, MSH,
  CombinedLoss). Inherits from BaseLoss directly (same pattern as
  CombinedLoss).
* 4th-order central-difference operator applied as a depth-wise Conv2d on
  the (H, W) reshape of the input. Reflect-padding at the 2-pixel boundary.
  Stencil is a fixed buffer; gradients through it are trivial.
* Per-variable σ_∂x, σ_∂y are accumulated online with Chan's parallel
  Welford algorithm in fp64 buffers. After the first batch (B·G ≈ 242k
  samples/var on graf-oklahoma) σ is already accurate to ~0.14%; further
  batches tighten it. Resumes across checkpoints via registered buffers.
* Mean subtraction is skipped: for MSE it cancels in the pred−tgt
  difference; for MSH a constant shift only affects the k=0 FFT bin
  (negligible in coherence computation). Only σ matters.
* DDP-safe: when ``distributed_stats=True`` the per-batch partial stats
  are all-reduced across ranks via Chan's pooled update so every rank
  uses the same σ.
* Operates in residual-normalized space (what the loss is handed by
  GraphResidualForecaster). By linearity of the finite-difference
  operator, MSE-inner residual-space and state-space gradient loss are
  mathematically identical up to the implicit 1/σ²_Δx factor; for MSH
  they differ slightly but residual-space is strictly more targeted at
  artefacts introduced by the model itself (x_last contribution cancels).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.combined import CombinedLoss

LOGGER = logging.getLogger(__name__)

# 4th-order central-difference weights: f'(i) ≈ (f(i-2) - 8 f(i-1) + 8 f(i+1) - f(i+2)) / 12
_CENTRAL_DIFF_4 = torch.tensor([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0
_STENCIL_PAD = 2  # half-width of the 5-point stencil


# ============================================================================
# Online Welford statistics
# ============================================================================
class _OnlineGradStats(nn.Module):
    """Parallel-Welford aggregator for per-variable mean/variance.

    Buffers are fp64 for long-term numerical stability. Returns fp32 σ on
    demand. Persistent across checkpoints.
    """

    def __init__(self, n_vars: int) -> None:
        super().__init__()
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("mean",  torch.zeros(n_vars, dtype=torch.float64))
        self.register_buffer("m2",    torch.zeros(n_vars, dtype=torch.float64))

    @torch.no_grad()
    def update(self, x: torch.Tensor, distributed: bool = False) -> None:
        """Accept a new tensor of shape (..., V) and fold it into the running stats."""
        x = x.reshape(-1, x.shape[-1]).to(torch.float64)
        n_new = torch.tensor(float(x.shape[0]), dtype=torch.float64, device=x.device)
        if n_new == 0:
            return
        mean_new = x.mean(dim=0)
        m2_new = ((x - mean_new) ** 2).sum(dim=0)

        if distributed and dist.is_available() and dist.is_initialized():
            # Pooled Welford across ranks: reduce (count, count*mean, M2+count*mean²) then reassemble.
            # This is the Chan/Welford parallel combine with world_size as the number of partitions.
            count_mean = n_new * mean_new
            # all-reduce sums
            dist.all_reduce(n_new, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_mean, op=dist.ReduceOp.SUM)
            global_mean = count_mean / n_new.clamp_min(1.0)
            # For M2, pooled formula: ΣM2 + Σ n_i (μ_i - μ)²
            # We only have local m2_new / mean_new here; approximate by sum-of-per-rank-m2
            # which slightly under-estimates. The error shrinks as ranks' stats converge, and
            # on the first few batches is bounded by (world_size - 1)/world_size × var. For
            # practical purposes at 8 GPUs this is a few percent on the first batch, vanishing
            # thereafter. A rigorous exchange would need two allreduces; this one is close
            # enough for our stddev use-case.
            dist.all_reduce(m2_new, op=dist.ReduceOp.SUM)
            # correction term for between-rank variance
            local_count = x.shape[0]
            local_dev = (mean_new - global_mean) ** 2 * local_count
            dist.all_reduce(local_dev, op=dist.ReduceOp.SUM)
            m2_new = m2_new + local_dev
            mean_new = global_mean

        if float(self.count) == 0.0:
            self.count.copy_(n_new)
            self.mean.copy_(mean_new)
            self.m2.copy_(m2_new)
            return

        delta = mean_new - self.mean
        total = self.count + n_new
        self.mean.add_(delta * n_new / total.clamp_min(1.0))
        self.m2.add_(m2_new + delta.pow(2) * self.count * n_new / total.clamp_min(1.0))
        self.count.copy_(total)

    @property
    def std(self) -> torch.Tensor:
        """Unbiased stddev per variable (fp32, safe for downstream math)."""
        denom = (self.count - 1.0).clamp_min(1.0)
        return (self.m2 / denom).sqrt().clamp_min(1e-6).float()


# ============================================================================
# Finite-difference operators
# ============================================================================
def _apply_stencil_x(x_2d: torch.Tensor, stencil: torch.Tensor) -> torch.Tensor:
    """Depth-wise 1D convolution along the W (x) axis with reflect padding.

    x_2d shape: (N, V, H, W). Returns same shape.
    """
    _N, V, _H, _W = x_2d.shape
    kx = stencil.view(1, 1, 1, -1).expand(V, 1, 1, -1).to(x_2d.dtype)
    xp = F.pad(x_2d, (_STENCIL_PAD, _STENCIL_PAD, 0, 0), mode="reflect")
    return F.conv2d(xp, kx, groups=V)


def _apply_stencil_y(x_2d: torch.Tensor, stencil: torch.Tensor) -> torch.Tensor:
    """Depth-wise 1D convolution along the H (y) axis with reflect padding."""
    _N, V, _H, _W = x_2d.shape
    ky = stencil.view(1, 1, -1, 1).expand(V, 1, -1, 1).to(x_2d.dtype)
    xp = F.pad(x_2d, (0, 0, _STENCIL_PAD, _STENCIL_PAD), mode="reflect")
    return F.conv2d(xp, ky, groups=V)


# ============================================================================
# Wrapper loss
# ============================================================================
class HorizontalGradientLoss(CombinedLoss):
    r"""Gradient-augmented wrapper around an inner loss.

    L_total = λ_raw · L(pred, tgt) + λ_∂x · L(∂x pred, ∂x tgt)
                                   + λ_∂y · L(∂y pred, ∂y tgt)

    Parameters
    ----------
    inner_loss : BaseLoss
        Any BaseLoss subclass (MSE, MSH, CombinedLoss, …). Applied
        identically to raw, ∂x, ∂y inputs; its scalers / reductions apply
        unchanged.
    x_dim, y_dim : int
        Spatial dimensions of the 2D grid; x_dim * y_dim must equal the
        flattened grid size of the input tensors.
    n_vars : int
        Number of prognostic+diagnostic output variables. Used to size the
        online-stats buffers.
    raw_weight, dx_weight, dy_weight : float
        Scalar mixing weights λ_raw, λ_∂x, λ_∂y.
    normalize_gradients : bool
        If True, normalize ∂pred_x / ∂tgt_x by an online-running per-variable
        σ (same for ∂y). Default True.
    distributed_stats : bool
        If True, pool Welford partial stats across DDP ranks each step.
        Default True.
    """

    def __init__(
        self,
        inner_loss: BaseLoss | dict | DictConfig,
        x_dim: int,
        y_dim: int,
        n_vars: int,
        *,
        raw_weight: float = 1.0,
        dx_weight: float = 1.0,
        dy_weight: float = 1.0,
        normalize_gradients: bool = True,
        distributed_stats: bool = True,
        **inner_kwargs: Any,
    ) -> None:
        # CombinedLoss.__init__ instantiates child loss configs (dict/DictConfig
        # or callable/BaseLoss) and populates self.losses / self._loss_scaler_specification.
        # We inherit its scaler routing (add_scaler / update_scaler) and its
        # recursive handling in losses/utils.py:print_variable_scaling.
        # Pass the inner as a single child with weight 1.0; our forward()
        # override ignores the weighted-sum logic and applies gradient
        # augmentation instead.
        CombinedLoss.__init__(
            self,
            losses=[inner_loss],
            loss_weights=[1.0],
            **inner_kwargs,
        )
        self.inner = self.losses[0]
        self.x_dim = int(x_dim)
        self.y_dim = int(y_dim)
        self.n_vars = int(n_vars)
        self.raw_weight = float(raw_weight)
        self.dx_weight = float(dx_weight)
        self.dy_weight = float(dy_weight)
        self.normalize_gradients = bool(normalize_gradients)
        self.distributed_stats = bool(distributed_stats)

        self.register_buffer("_diff_kernel", _CENTRAL_DIFF_4.clone(), persistent=False)

        if self.normalize_gradients:
            self.stats_x = _OnlineGradStats(n_vars)
            self.stats_y = _OnlineGradStats(n_vars)
        else:
            self.stats_x = None
            self.stats_y = None

        LOGGER.info(
            "HorizontalGradientLoss initialised: field=%dx%d, n_vars=%d, "
            "raw=%.2f, dx=%.2f, dy=%.2f, normalize=%s, distributed_stats=%s",
            self.x_dim, self.y_dim, self.n_vars,
            self.raw_weight, self.dx_weight, self.dy_weight,
            self.normalize_gradients, self.distributed_stats,
        )

    # ------------------------------------------------------------------
    # Reshape helpers: (B, E, G, V)  <->  (B*E, V, H, W)
    # ------------------------------------------------------------------
    def _to_2d(self, x: torch.Tensor) -> torch.Tensor:
        B, E, G, V = x.shape
        assert G == self.x_dim * self.y_dim, (
            f"HorizontalGradientLoss: flat grid={G} but x_dim*y_dim={self.x_dim * self.y_dim}"
        )
        return x.reshape(B * E, self.x_dim, self.y_dim, V).permute(0, 3, 1, 2).contiguous()

    def _from_2d(self, x_2d: torch.Tensor, B: int, E: int) -> torch.Tensor:
        # (B*E, V, H, W) -> (B, E, H*W, V)
        return x_2d.permute(0, 2, 3, 1).reshape(B, E, self.x_dim * self.y_dim, -1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        B, E, _G, _V = pred.shape
        kernel = self._diff_kernel

        # Raw-field contribution (inner's scalers etc. apply as usual)
        L = self.raw_weight * self.inner(pred, target, **kwargs)

        pred_2d = self._to_2d(pred)
        tgt_2d = self._to_2d(target)

        # ∂x
        if self.dx_weight != 0.0:
            dpred_x = _apply_stencil_x(pred_2d, kernel)
            dtgt_x = _apply_stencil_x(tgt_2d, kernel)
            if self.normalize_gradients:
                if self.training:
                    # Welford update on target (not pred) so model output distribution
                    # can't leak into normalization.
                    # Shape for update: (N_flat, V)
                    self.stats_x.update(
                        dtgt_x.permute(0, 2, 3, 1).reshape(-1, self.n_vars),
                        distributed=self.distributed_stats,
                    )
                sigma_x = self.stats_x.std.to(dtype=dpred_x.dtype, device=dpred_x.device)
                sigma_x = sigma_x.view(1, -1, 1, 1)
                dpred_x = dpred_x / sigma_x
                dtgt_x = dtgt_x / sigma_x
            L = L + self.dx_weight * self.inner(
                self._from_2d(dpred_x, B, E),
                self._from_2d(dtgt_x, B, E),
                **kwargs,
            )

        # ∂y
        if self.dy_weight != 0.0:
            dpred_y = _apply_stencil_y(pred_2d, kernel)
            dtgt_y = _apply_stencil_y(tgt_2d, kernel)
            if self.normalize_gradients:
                if self.training:
                    self.stats_y.update(
                        dtgt_y.permute(0, 2, 3, 1).reshape(-1, self.n_vars),
                        distributed=self.distributed_stats,
                    )
                sigma_y = self.stats_y.std.to(dtype=dpred_y.dtype, device=dpred_y.device)
                sigma_y = sigma_y.view(1, -1, 1, 1)
                dpred_y = dpred_y / sigma_y
                dtgt_y = dtgt_y / sigma_y
            L = L + self.dy_weight * self.inner(
                self._from_2d(dpred_y, B, E),
                self._from_2d(dtgt_y, B, E),
                **kwargs,
            )

        return L

    # ------------------------------------------------------------------
    # Delegation to inner. We override CombinedLoss.add_scaler / update_scaler
    # because the parent implementation accesses `self.losses[i].scaler`
    # directly — fine when children are leaf losses, but broken when the
    # child is itself a CombinedLoss (GraphCastCombinedLoss), which has
    # `self.scaler` deleted and routes via its own add_scaler instead. We
    # call inner.add_scaler(...) and let it handle the fan-out.
    # ------------------------------------------------------------------
    def set_data_indices(self, data_indices) -> None:
        """Forward data_indices down to inner (+ grandchildren via inner's own routing)."""
        if data_indices is None:
            return
        if hasattr(self.inner, "set_data_indices"):
            self.inner.set_data_indices(data_indices)

    def add_scaler(self, dimension, scaler, *, name: str | None = None) -> None:
        self.inner.add_scaler(dimension, scaler, name=name)

    def update_scaler(self, name: str, scaler, *, override: bool = False) -> None:
        self.inner.update_scaler(name, scaler=scaler, override=override)
