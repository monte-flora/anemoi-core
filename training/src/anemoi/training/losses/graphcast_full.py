# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Flat FastNet-style loss for storm-scale LAM training.

Replaces the nested ``HorizontalGradientLoss → GraphCastWindAwareLoss →
GraphCastCombinedLoss[MSE + MSH]`` stack with a single class that computes
every term in one pass:

    L =  w_raw_mse   · MSE(pred, target)
       + w_raw_msh   · MSH(pred, target)
       + w_dx        · MSE(∂x pred, ∂x target)
       + w_dy        · MSE(∂y pred, ∂y target)
       + w_wind_spd  · MSE(zero except s at u_idx)
       + w_wind_dir  · MSE(u/v slots replaced with d_i, d_j)

Leaf-forward count per step is ~6 (1 MSH + 5 MSE) vs 12 for the nested
stack (6 MSH + 6 MSE). MSH is the expensive term; evaluating it only on
the raw field — which is where the double-penalty-suppression
justification actually lives — gets us back ~80% of the throughput lost
to the wrapper composition.

Ablate any term by setting its weight to 0. Weights for every term are
independent; term-level gates short-circuit the transform so ablating
gradient or wind saves the associated compute.
"""

from __future__ import annotations

import logging
from typing import Any

import einops
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.combined import CombinedLoss
from anemoi.training.losses.horizontal_gradient import _CENTRAL_DIFF_4
from anemoi.training.losses.horizontal_gradient import _OnlineGradStats
from anemoi.training.losses.horizontal_gradient import _apply_stencil_x
from anemoi.training.losses.horizontal_gradient import _apply_stencil_y

LOGGER = logging.getLogger(__name__)


class GraphCastFullLoss(CombinedLoss):
    r"""Single-class flat FastNet loss.

    Inherits CombinedLoss so ``isinstance(loss, CombinedLoss)`` is True
    (utils.py:print_variable_scaling recurses correctly). Internally
    holds two child losses — MSE and MSH — and calls each of them at
    most once per step on transformed inputs.

    Parameters
    ----------
    x_dim, y_dim : int
        Spatial dimensions of the 2D grid. x_dim * y_dim must equal the
        flattened grid size.
    n_vars : int
        Number of output variables. Used to size the online gradient
        statistics buffers.
    raw_mse_weight : float, default 1.0
        λ for MSE on the raw field. 0 ablates.
    raw_msh_weight : float, default 1.0
        λ for MSH on the raw field. 0 ablates MSH entirely.
    grad_x_weight, grad_y_weight : float, default 1.0
        λ for MSE on the zonal / meridional gradient fields.
    wind_speed_weight : float, default 5.0
        λ for MSE on wind-speed contributions (FastNet Eq. 10).
    wind_dir_weight : float, default 1.0
        λ for MSE on direction-decomposed wind u/v.
    coherence_weight : float, default 1.0
        φ inside MSH's AMSE_k. 0 reduces MSH to amplitude-only.
    use_gamma_k : bool, default True
        MSH γ_k = max(N_k · k^√3, 1.0) wavenumber weighting.
    gamma_k_min : float, default 1.0
        Floor for γ_k.
    use_variable_normalization : bool, default True
        MSH β_j = 1/⟨AMSE_j⟩ online Welford normalisation.
    normalize_gradients : bool, default True
        Divide ∂x, ∂y by online Welford σ per variable.
    epsilon : float, default 1e-6
        Guard inside sqrt(u² + v² + ε²) for wind speed.
    u_v_pairs : list[tuple[str, str]] | None, default None
        Explicit (u_name, v_name) pairs; auto-detect if None.
    distributed_stats : bool, default True
        All-reduce Welford partial stats across DDP ranks.
    ignore_nans : bool, default False
    mse_scalers : list[str] | None
        Scaler names routed to the internal MSE leaf. Defaults to
        ``['general_variable', 'limited_area_mask']``.
    msh_scalers : list[str] | None
        Scaler names routed to the internal MSH leaf. Defaults to
        ``['general_variable']``.
    """

    def __init__(
        self,
        *,
        x_dim: int,
        y_dim: int,
        n_vars: int,
        raw_mse_weight: float = 1.0,
        raw_msh_weight: float = 1.0,
        grad_x_weight: float = 1.0,
        grad_y_weight: float = 1.0,
        wind_speed_weight: float = 5.0,
        wind_dir_weight: float = 1.0,
        coherence_weight: float = 1.0,
        use_gamma_k: bool = True,
        gamma_k_min: float = 1.0,
        use_variable_normalization: bool = True,
        normalize_gradients: bool = True,
        epsilon: float = 1e-6,
        u_v_pairs: list | None = None,
        distributed_stats: bool = True,
        ignore_nans: bool = False,
        mse_scalers: list[str] | None = None,
        msh_scalers: list[str] | None = None,
        precomputed_stats_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.precomputed_stats_path = str(precomputed_stats_path) if precomputed_stats_path else None

        mse_cfg = OmegaConf.create(
            {
                "_target_": "anemoi.training.losses.GraphCastMSELoss",
                "scalers": list(mse_scalers)
                if mse_scalers is not None
                else ["general_variable", "limited_area_mask"],
                "ignore_nans": bool(ignore_nans),
            }
        )
        msh_child_cfg = {
            "_target_": "anemoi.training.losses.graphcast_msh.GraphCastMSHLoss",
            "scalers": list(msh_scalers)
            if msh_scalers is not None
            else ["general_variable"],
            "x_dim": int(x_dim),
            "y_dim": int(y_dim),
            "n_vars": int(n_vars),
            "coherence_weight": float(coherence_weight),
            "use_gamma_k": bool(use_gamma_k),
            "gamma_k_min": float(gamma_k_min),
            "use_variable_normalization": bool(use_variable_normalization),
            "distributed_stats": bool(distributed_stats),
            "ignore_nans": bool(ignore_nans),
        }
        if self.precomputed_stats_path is not None:
            msh_child_cfg["precomputed_stats_path"] = self.precomputed_stats_path
        msh_cfg = OmegaConf.create(msh_child_cfg)
        # MSH is the expensive leaf; skip instantiating it when its weight is zero.
        children: list = [mse_cfg]
        if raw_msh_weight != 0.0:
            children.append(msh_cfg)
        CombinedLoss.__init__(
            self,
            losses=children,
            loss_weights=[1.0] * len(children),
            **kwargs,
        )
        self.mse = self.losses[0]
        self.msh = self.losses[1] if len(self.losses) > 1 else None

        self.x_dim = int(x_dim)
        self.y_dim = int(y_dim)
        self.n_vars = int(n_vars)
        self.raw_mse_weight = float(raw_mse_weight)
        self.raw_msh_weight = float(raw_msh_weight)
        self.grad_x_weight = float(grad_x_weight)
        self.grad_y_weight = float(grad_y_weight)
        self.wind_speed_weight = float(wind_speed_weight)
        self.wind_dir_weight = float(wind_dir_weight)
        self.normalize_gradients = bool(normalize_gradients)
        self.distributed_stats = bool(distributed_stats)
        self.epsilon = float(epsilon)
        self.u_v_pairs_override = (
            [tuple(p) for p in u_v_pairs] if u_v_pairs is not None else None
        )

        # Finite-difference stencil buffer
        self.register_buffer("_diff_kernel", _CENTRAL_DIFF_4.clone(), persistent=False)

        # Gradient σ per variable. Two sources:
        #   (1) precomputed_stats_path — reads statistics_gradient_{x,y}_stdev
        #       from the zarr via raw zarr access (same pattern as
        #       Store.statistics_tendencies). Data→model subsetting is
        #       resolved in set_data_indices().
        #   (2) Online Welford on ∂target during training (fallback).
        self._data_idx_for_model_output: torch.Tensor | None = None
        self._precomputed_sigma_x: torch.Tensor | None = None
        self._precomputed_sigma_y: torch.Tensor | None = None

        need_sigma_x = self.normalize_gradients and (self.grad_x_weight != 0.0)
        need_sigma_y = self.normalize_gradients and (self.grad_y_weight != 0.0)

        if self.precomputed_stats_path is not None and (need_sigma_x or need_sigma_y):
            # Lazy import: zarr only needed when precomputed path is set
            import zarr as _zarr

            _z = _zarr.open(self.precomputed_stats_path, mode="r")
            zarr_variables = list(_z.attrs.get("variables", []))
            # RAW-zarr index lookup by name — bypasses anemoi-datasets' `drop`
            # filter, which maps names into a compressed (dropped) index space
            # that doesn't match our raw (V_data=147) stats arrays.
            self._zarr_name_to_index: dict[str, int] = {
                name: i for i, name in enumerate(zarr_variables)
            }
            # Residual-space σ — paired with GraphResidualForecaster's loss which
            # already has the tendency divided by statistics_tendencies_<freq>_stdev.
            _freq = _z.attrs.get("frequency", "15m")
            _sx_key = f"statistics_tendencies_{_freq}_gradient_x_stdev"
            _sy_key = f"statistics_tendencies_{_freq}_gradient_y_stdev"
            if need_sigma_x:
                sx = torch.as_tensor(_z[_sx_key][:], dtype=torch.float32)
                if sx.shape[0] != len(zarr_variables):
                    raise ValueError(
                        f"zarr 'variables' attr ({len(zarr_variables)}) != "
                        f"{_sx_key!r} length ({sx.shape[0]})"
                    )
                self.register_buffer("_precomputed_sigma_x_data", sx, persistent=False)
            if need_sigma_y:
                sy = torch.as_tensor(_z[_sy_key][:], dtype=torch.float32)
                self.register_buffer("_precomputed_sigma_y_data", sy, persistent=False)
            self.grad_stats_x = None
            self.grad_stats_y = None
            LOGGER.info(
                "GraphCastFullLoss: using precomputed gradient σ from %s (freq=%s)",
                self.precomputed_stats_path, _freq,
            )
        else:
            self.grad_stats_x = _OnlineGradStats(self.n_vars) if need_sigma_x else None
            self.grad_stats_y = _OnlineGradStats(self.n_vars) if need_sigma_y else None

        # Wind indices — resolved at set_data_indices
        self._u_idx: torch.Tensor | None = None
        self._v_idx: torch.Tensor | None = None
        self._pair_names: list[tuple[str, str]] = []

        LOGGER.info(
            "GraphCastFullLoss initialised: field=%dx%d n_vars=%d  weights=(raw_mse=%.2f, raw_msh=%.2f, "
            "dx=%.2f, dy=%.2f, wind_speed=%.2f, wind_dir=%.2f)  MSH=%s",
            self.x_dim, self.y_dim, self.n_vars,
            self.raw_mse_weight, self.raw_msh_weight,
            self.grad_x_weight, self.grad_y_weight,
            self.wind_speed_weight, self.wind_dir_weight,
            "enabled" if self.msh is not None else "disabled",
        )

    # ------------------------------------------------------------------
    # Scaler / data-indices routing (CombinedLoss children need it)
    # ------------------------------------------------------------------
    def set_data_indices(self, data_indices) -> None:
        if data_indices is None:
            return
        if hasattr(self.mse, "set_data_indices"):
            self.mse.set_data_indices(data_indices)
        if self.msh is not None and hasattr(self.msh, "set_data_indices"):
            self.msh.set_data_indices(data_indices)

        # Resolve (V_model,) → (V_raw_zarr,) mapping for precomputed-σ lookup
        # by name — bypasses the `drop`-filtered data_indices.data mapping
        # so our raw 147-entry arrays are indexed correctly.
        if hasattr(self, "_precomputed_sigma_x_data") or hasattr(self, "_precomputed_sigma_y_data"):
            output_n2i = data_indices.model.output.name_to_index
            n_model_out = len(output_n2i)
            idx_map = torch.zeros(n_model_out, dtype=torch.long)
            missing: list[str] = []
            for name, model_idx in output_n2i.items():
                if name in self._zarr_name_to_index:
                    idx_map[model_idx] = int(self._zarr_name_to_index[name])
                else:
                    missing.append(name)
            if missing:
                raise KeyError(
                    f"GraphCastFullLoss: {len(missing)} output variable(s) not in "
                    f"zarr 'variables' attr: {missing[:10]}"
                )
            self._data_idx_for_model_output = idx_map

            # Sanity: log stats at the selected raw-zarr indices
            if hasattr(self, "_precomputed_sigma_x_data"):
                sx = self._precomputed_sigma_x_data[idx_map]
                n_zero_x = int((sx == 0).sum())
                LOGGER.info(
                    "GraphCastFullLoss: σ_∂x selected for %d outputs; zeros=%d; min_nonzero=%.3e max=%.3e",
                    n_model_out, n_zero_x,
                    float(sx[sx > 0].min()) if (sx > 0).any() else float("nan"),
                    float(sx.max()),
                )
                if n_zero_x:
                    zero_names = [name for name in output_n2i
                                  if self._precomputed_sigma_x_data[self._zarr_name_to_index[name]] == 0]
                    LOGGER.warning("  zero σ_∂x outputs (will clamp to 1e-12): %s", zero_names[:10])

        # Resolve wind indices
        if self.wind_speed_weight == 0.0 and self.wind_dir_weight == 0.0:
            return
        name_to_idx = dict(data_indices.model.output.name_to_index)
        if self.u_v_pairs_override is not None:
            pairs = self.u_v_pairs_override
        else:
            pairs = _autodetect_wind_pairs(list(name_to_idx.keys()))
        pairs = [p for p in pairs if p[0] in name_to_idx and p[1] in name_to_idx]
        if not pairs:
            LOGGER.warning("GraphCastFullLoss: no (u, v) pairs resolved; wind terms will be skipped.")
            return
        self._u_idx = torch.tensor([name_to_idx[u] for u, _ in pairs], dtype=torch.long)
        self._v_idx = torch.tensor([name_to_idx[v] for _, v in pairs], dtype=torch.long)
        self._pair_names = pairs
        LOGGER.info(
            "GraphCastFullLoss: resolved %d (u, v) pairs: %s",
            len(pairs),
            ", ".join(f"{u}↔{v}" for u, v in pairs),
        )

    def add_scaler(self, dimension, scaler, *, name: str | None = None) -> None:
        self.mse.add_scaler(dimension, scaler, name=name)
        if self.msh is not None:
            self.msh.add_scaler(dimension, scaler, name=name)

    def update_scaler(self, name: str, scaler, *, override: bool = False) -> None:
        self.mse.update_scaler(name, scaler=scaler, override=override)
        if self.msh is not None:
            self.msh.update_scaler(name, scaler=scaler, override=override)

    def _sigma_for_axis(
        self,
        dpred: torch.Tensor,
        dtgt: torch.Tensor,
        *,
        axis: str,
    ) -> torch.Tensor:
        """Return the per-variable gradient σ broadcast-ready (1, V, 1, 1).

        Precomputed path (reads from zarr) is preferred; otherwise updates and
        reads the online Welford estimator.
        """
        assert axis in ("x", "y")
        precomp_buf = f"_precomputed_sigma_{axis}_data"
        if hasattr(self, precomp_buf):
            if self._data_idx_for_model_output is None:
                raise RuntimeError(
                    "GraphCastFullLoss: precomputed σ enabled but set_data_indices() "
                    "has not been called yet — no model-output→data-index mapping.",
                )
            idx = self._data_idx_for_model_output.to(dpred.device)
            sigma_data = getattr(self, precomp_buf).to(device=dpred.device, dtype=dpred.dtype)
            sigma_var = sigma_data[idx]  # (V_model,)
            # Guard against zeros on static/constant variables: they contribute 0/0 otherwise
            sigma_var = torch.clamp(sigma_var, min=1.0e-12)
            return sigma_var.view(1, -1, 1, 1)

        welford = getattr(self, f"grad_stats_{axis}")
        if welford is None:
            raise RuntimeError(f"GraphCastFullLoss: no σ source for axis {axis!r}")
        if self.training:
            welford.update(
                dtgt.permute(0, 2, 3, 1).reshape(-1, self.n_vars),
                distributed=self.distributed_stats,
            )
        return welford.std.to(dtype=dpred.dtype, device=dpred.device).view(1, -1, 1, 1)

    # ------------------------------------------------------------------
    # Forward: compute all transforms once, sum weighted leaf losses
    # ------------------------------------------------------------------
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        B, E, G, V = pred.shape
        L: torch.Tensor | float = 0.0

        # --- 1. Raw-field terms ---------------------------------------
        if self.raw_mse_weight != 0.0:
            L = L + self.raw_mse_weight * self.mse(pred, target, **kwargs)
        if self.raw_msh_weight != 0.0 and self.msh is not None:
            L = L + self.raw_msh_weight * self.msh(pred, target, **kwargs)

        # --- 2. Gradient terms (compute transforms once, MSE each) ----
        need_grad = self.grad_x_weight != 0.0 or self.grad_y_weight != 0.0
        if need_grad:
            # (B, E, G, V) → (B*E, V, H, W)
            pred_2d = einops.rearrange(
                pred, "b e (h w) v -> (b e) v h w", h=self.x_dim, w=self.y_dim
            )
            tgt_2d = einops.rearrange(
                target, "b e (h w) v -> (b e) v h w", h=self.x_dim, w=self.y_dim
            )
            kernel = self._diff_kernel

            if self.grad_x_weight != 0.0:
                dpred = _apply_stencil_x(pred_2d, kernel)
                dtgt = _apply_stencil_x(tgt_2d, kernel)
                if self.normalize_gradients:
                    sigma = self._sigma_for_axis(
                        dpred, dtgt, axis="x",
                    )
                    dpred = dpred / sigma
                    dtgt = dtgt / sigma
                dpred_flat = einops.rearrange(
                    dpred, "(b e) v h w -> b e (h w) v", b=B
                )
                dtgt_flat = einops.rearrange(
                    dtgt, "(b e) v h w -> b e (h w) v", b=B
                )
                L = L + self.grad_x_weight * self.mse(dpred_flat, dtgt_flat, **kwargs)

            if self.grad_y_weight != 0.0:
                dpred = _apply_stencil_y(pred_2d, kernel)
                dtgt = _apply_stencil_y(tgt_2d, kernel)
                if self.normalize_gradients:
                    sigma = self._sigma_for_axis(
                        dpred, dtgt, axis="y",
                    )
                    dpred = dpred / sigma
                    dtgt = dtgt / sigma
                dpred_flat = einops.rearrange(
                    dpred, "(b e) v h w -> b e (h w) v", b=B
                )
                dtgt_flat = einops.rearrange(
                    dtgt, "(b e) v h w -> b e (h w) v", b=B
                )
                L = L + self.grad_y_weight * self.mse(dpred_flat, dtgt_flat, **kwargs)

        # --- 3. Wind terms (decomposition + speed) --------------------
        need_wind = (
            (self.wind_speed_weight != 0.0 or self.wind_dir_weight != 0.0)
            and self._u_idx is not None
            and self._v_idx is not None
            and self._u_idx.numel() > 0
        )
        if need_wind:
            u_idx = self._u_idx.to(pred.device)
            v_idx = self._v_idx.to(pred.device)
            u_pred = pred.index_select(-1, u_idx)
            v_pred = pred.index_select(-1, v_idx)
            u_tgt = target.index_select(-1, u_idx)
            v_tgt = target.index_select(-1, v_idx)

            eps_sq = self.epsilon * self.epsilon
            s_pred = torch.sqrt(u_pred * u_pred + v_pred * v_pred + eps_sq)
            s_tgt = torch.sqrt(u_tgt * u_tgt + v_tgt * v_tgt + eps_sq)

            if self.wind_dir_weight != 0.0:
                d_i_pred = u_pred / s_pred
                d_j_pred = v_pred / s_pred
                d_i_tgt = u_tgt / s_tgt
                d_j_tgt = v_tgt / s_tgt
                pred_dir = pred.clone()
                tgt_dir = target.clone()
                pred_dir.index_copy_(-1, u_idx, d_i_pred)
                pred_dir.index_copy_(-1, v_idx, d_j_pred)
                tgt_dir.index_copy_(-1, u_idx, d_i_tgt)
                tgt_dir.index_copy_(-1, v_idx, d_j_tgt)
                L = L + self.wind_dir_weight * self.mse(pred_dir, tgt_dir, **kwargs)

            if self.wind_speed_weight != 0.0:
                pred_spd = torch.zeros_like(pred)
                tgt_spd = torch.zeros_like(target)
                pred_spd.index_copy_(-1, u_idx, s_pred)
                tgt_spd.index_copy_(-1, u_idx, s_tgt)
                L = L + self.wind_speed_weight * self.mse(pred_spd, tgt_spd, **kwargs)

        return L


def _autodetect_wind_pairs(names: list[str]) -> list[tuple[str, str]]:
    """Match `u_<suffix>` ↔ `v_<suffix>`. Shared with GraphCastWindAwareLoss."""
    name_set = set(names)
    pairs: list[tuple[str, str]] = []
    for n in names:
        if not n.startswith("u_"):
            continue
        suffix = n[len("u_"):]
        v_name = "v_" + suffix
        if v_name in name_set:
            pairs.append((n, v_name))
    pairs.sort()
    return pairs
