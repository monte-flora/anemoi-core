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
    grad_var_weights : dict[str, float] | None, default None
        Per-variable weights applied **only to the gradient terms**
        (∂x and ∂y). The raw MSE/MSH and wind paths are unaffected.
        Keys may be either an exact output-variable name (e.g.
        ``qv_33``) or a level-stripped group stem (e.g. ``pressure``,
        which matches every ``pressure_<N>``). Exact-name matches take
        precedence over group matches; unmatched variables fall back
        to the value of ``default`` (1.0 if absent). All values must
        be ≥ 0; setting a weight to 0 fully ablates that variable's
        gradient contribution. Use this to neutralise variables whose
        residual-space σ_∂ is anomalously small (e.g. ``pressure``
        with σ_∂x ≈ 0.19 vs the median 0.36, or ``qv_33`` with
        σ_∂x ≈ 0.002 — both over-amplify their gradient term under
        z-score normalisation).
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
        grad_var_weights: dict | None = None,
        column_mass_flux_weight: float = 0.0,
        w_var_names: list | None = None,
        w_level_pressure_weights: list | None = None,
        hydrostatic_weight: float = 0.0,
        hydrostatic_alphas: list | None = None,
        hydrostatic_z_levels: list | None = None,
        hydrostatic_p_var_names: list | None = None,
        hydrostatic_theta_var_names: list | None = None,
        hydrostatic_qv_var_names: list | None = None,
        **kwargs: Any,
    ) -> None:
        self.precomputed_stats_path = str(precomputed_stats_path) if precomputed_stats_path else None

        # Per-variable gradient weights (validated; resolved against
        # data_indices in set_data_indices). Stored as a plain dict so
        # OmegaConf DictConfig and dict both work.
        if grad_var_weights is None:
            self._grad_var_weights_cfg: dict[str, float] = {}
        else:
            self._grad_var_weights_cfg = {
                str(k): float(v) for k, v in dict(grad_var_weights).items()
            }
            for k, v in self._grad_var_weights_cfg.items():
                if v < 0.0 or v != v:  # NaN check via self-comparison
                    msg = (
                        f"grad_var_weights[{k!r}] = {v}; must be ≥ 0 and finite."
                    )
                    raise ValueError(msg)

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

        # --- Column-mass-flux conservation term ----------------------------
        # Penalises domain-mean(Σ w_lev · Δp_lev) mismatch between pred & target.
        # See diag_mass_flux.py: v17 develops |domain mean ∫ρw·dz| swings of
        # ~150 kg/(m²·s) at 18 h (vs <8 in truth) — a known
        # ML-emulator failure mode (compensating-subsidence absence).
        self.column_mass_flux_weight = float(column_mass_flux_weight)
        self.w_var_names_override = (
            [str(v) for v in w_var_names] if w_var_names is not None else None
        )
        self.w_level_pressure_weights = (
            [float(p) for p in w_level_pressure_weights]
            if w_level_pressure_weights is not None
            else None
        )
        # Resolved in set_data_indices()
        self._w_idx: torch.Tensor | None = None
        self._w_level_dp: torch.Tensor | None = None

        # --- Hydrostatic-balance soft constraint (DLESyM-style) -------------
        # Soft constraint that penalises deviations from hydrostatic balance
        # via an error-tolerant loss
        #     f(r_k) = (r/α)² / (1 + exp(1 - (r/α)²))
        # where the residual at adjacent level pair (k-1, k) is
        #     r_k = T_v_mean - (g/R) · (z_k - z_{k-1}) / ln(p_{k-1}/p_k)
        # Below α_k the loss is ~0 (tolerates GRAF's natural non-hydrostatic
        # imbalance from convection); above α_k it approaches MSE. α_k is
        # precomputed from the natural percentile distribution of r_k in
        # the training zarr via the Lambert-W scaling
        #     α_k = Q_k(p) · sqrt(W₀(1) + 1).
        self.hydrostatic_weight = float(hydrostatic_weight)
        self.hydrostatic_alphas_cfg = (
            [float(a) for a in hydrostatic_alphas] if hydrostatic_alphas is not None else None
        )
        self.hydrostatic_z_levels_cfg = (
            [float(z) for z in hydrostatic_z_levels] if hydrostatic_z_levels is not None else None
        )
        self.hyd_p_names_override     = list(hydrostatic_p_var_names)     if hydrostatic_p_var_names     is not None else None
        self.hyd_theta_names_override = list(hydrostatic_theta_var_names) if hydrostatic_theta_var_names is not None else None
        self.hyd_qv_names_override    = list(hydrostatic_qv_var_names)    if hydrostatic_qv_var_names    is not None else None
        # Resolved in set_data_indices():
        self._hyd_p_idx: torch.Tensor | None = None
        self._hyd_theta_idx: torch.Tensor | None = None
        self._hyd_qv_idx: torch.Tensor | None = None
        self._hyd_dz: torch.Tensor | None = None       # (n_pairs,) — z_{k} − z_{k-1}
        self._hyd_alpha: torch.Tensor | None = None    # (n_pairs,)

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
    @staticmethod
    def _level_stripped_stem(name: str) -> str:
        """``pressure_14`` → ``pressure``; ``t2m`` → ``t2m``.

        Strips a single trailing ``_<digits>`` suffix. Used so a single
        ``grad_var_weights`` key like ``pressure`` covers every
        ``pressure_<level>``.
        """
        import re
        m = re.match(r"^(.+?)(?:_\d+)?$", name)
        return m.group(1) if m else name

    def set_data_indices(self, data_indices) -> None:
        if data_indices is None:
            return
        if hasattr(self.mse, "set_data_indices"):
            self.mse.set_data_indices(data_indices)
        if self.msh is not None and hasattr(self.msh, "set_data_indices"):
            self.msh.set_data_indices(data_indices)

        # Per-variable gradient weights (V_model,) — exact-name match
        # takes precedence over level-stripped group stem; default 1.0.
        if self._grad_var_weights_cfg:
            output_n2i = data_indices.model.output.name_to_index
            n_out = len(output_n2i)
            default_w = float(self._grad_var_weights_cfg.get("default", 1.0))
            w = torch.full((n_out,), default_w, dtype=torch.float32)
            unmatched = set(self._grad_var_weights_cfg.keys()) - {"default"}
            for name, model_idx in output_n2i.items():
                if name in self._grad_var_weights_cfg:
                    w[model_idx] = float(self._grad_var_weights_cfg[name])
                    unmatched.discard(name)
                else:
                    stem = self._level_stripped_stem(name)
                    if stem in self._grad_var_weights_cfg:
                        w[model_idx] = float(self._grad_var_weights_cfg[stem])
                        unmatched.discard(stem)
            if unmatched:
                LOGGER.warning(
                    "GraphCastFullLoss.grad_var_weights: %d key(s) matched no "
                    "model-output variable: %s",
                    len(unmatched), sorted(unmatched)[:10],
                )
            n_zero = int((w == 0.0).sum())
            n_nondefault = int((w != default_w).sum())
            LOGGER.info(
                "GraphCastFullLoss.grad_var_weights resolved: default=%.3f, "
                "n_zero=%d, n_nondefault=%d (e.g. %s)",
                default_w, n_zero, n_nondefault,
                {n: float(w[i]) for n, i in output_n2i.items() if w[i] != default_w}
            )
            self.register_buffer("_grad_var_weights", w, persistent=False)
        else:
            self._grad_var_weights = None

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

        name_to_idx = dict(data_indices.model.output.name_to_index)

        # Resolve wind indices (only when wind terms are active)
        if self.wind_speed_weight != 0.0 or self.wind_dir_weight != 0.0:
            if self.u_v_pairs_override is not None:
                pairs = self.u_v_pairs_override
            else:
                pairs = _autodetect_wind_pairs(list(name_to_idx.keys()))
            pairs = [p for p in pairs if p[0] in name_to_idx and p[1] in name_to_idx]
            if not pairs:
                LOGGER.warning("GraphCastFullLoss: no (u, v) pairs resolved; wind terms will be skipped.")
            else:
                self._u_idx = torch.tensor([name_to_idx[u] for u, _ in pairs], dtype=torch.long)
                self._v_idx = torch.tensor([name_to_idx[v] for _, v in pairs], dtype=torch.long)
                self._pair_names = pairs
                LOGGER.info(
                    "GraphCastFullLoss: resolved %d (u, v) pairs: %s",
                    len(pairs),
                    ", ".join(f"{u}↔{v}" for u, v in pairs),
                )

        # --- Resolve W indices for column-mass-flux loss --------------------
        if self.column_mass_flux_weight != 0.0:
            if self.w_var_names_override is not None:
                w_names = list(self.w_var_names_override)
            else:
                # Autodetect: model outputs named "w_NN" sorted by level int
                w_pat = []
                for name in name_to_idx:
                    if name.startswith("w_"):
                        suffix = name[2:]
                        if suffix.isdigit():
                            w_pat.append((int(suffix), name))
                w_pat.sort()
                w_names = [n for _, n in w_pat]
            w_names = [n for n in w_names if n in name_to_idx]
            if not w_names:
                LOGGER.warning(
                    "GraphCastFullLoss.column_mass_flux_weight != 0 but no W "
                    "outputs found; mass-flux term disabled."
                )
                self.column_mass_flux_weight = 0.0
            else:
                self._w_idx = torch.tensor(
                    [name_to_idx[n] for n in w_names], dtype=torch.long
                )
                if self.w_level_pressure_weights is not None:
                    if len(self.w_level_pressure_weights) != len(w_names):
                        msg = (
                            f"w_level_pressure_weights length "
                            f"({len(self.w_level_pressure_weights)}) does not match "
                            f"number of resolved W outputs ({len(w_names)})."
                        )
                        raise ValueError(msg)
                    dp = torch.tensor(self.w_level_pressure_weights, dtype=torch.float32)
                else:
                    dp = torch.ones(len(w_names), dtype=torch.float32)
                # Regular attribute (not register_buffer) — mirrors how _u_idx
                # is stored, and avoids a "buffer already exists" conflict with
                # the pre-init `self._w_level_dp = None` in __init__.
                self._w_level_dp = dp
                LOGGER.info(
                    "GraphCastFullLoss.column_mass_flux_weight=%.3e enabled on "
                    "%d W outputs: %s (Δp weights sum=%.1f)",
                    self.column_mass_flux_weight, len(w_names), w_names,
                    float(dp.sum()),
                )

        # --- Resolve hydrostatic-balance constraint -------------------------
        if self.hydrostatic_weight != 0.0:
            # Helper: autodetect "<prefix>_<level_int>" sorted by level
            def _autodetect_3d(prefix: str) -> list[str]:
                out = []
                for name in name_to_idx:
                    if name.startswith(prefix + "_"):
                        suffix = name[len(prefix) + 1:]
                        if suffix.isdigit():
                            out.append((int(suffix), name))
                out.sort()
                return [n for _, n in out]
            p_names     = self.hyd_p_names_override     or _autodetect_3d("pressure")
            theta_names = self.hyd_theta_names_override or _autodetect_3d("theta")
            qv_names    = self.hyd_qv_names_override    or _autodetect_3d("qv")
            p_names     = [n for n in p_names     if n in name_to_idx]
            theta_names = [n for n in theta_names if n in name_to_idx]
            qv_names    = [n for n in qv_names    if n in name_to_idx]

            n_lev = len(p_names)
            consistent = (
                n_lev > 1
                and len(theta_names) == n_lev
                and len(qv_names) == n_lev
            )
            if not consistent:
                LOGGER.warning(
                    "GraphCastFullLoss.hydrostatic_weight != 0 but could not "
                    "find consistent pressure/theta/qv 3-D outputs "
                    "(p=%d, theta=%d, qv=%d). Hydrostatic term disabled.",
                    len(p_names), len(theta_names), len(qv_names),
                )
                self.hydrostatic_weight = 0.0
            elif self.hydrostatic_z_levels_cfg is None or self.hydrostatic_alphas_cfg is None:
                LOGGER.warning(
                    "GraphCastFullLoss.hydrostatic_weight != 0 but "
                    "hydrostatic_z_levels (%s) or hydrostatic_alphas (%s) "
                    "missing. Hydrostatic term disabled.",
                    self.hydrostatic_z_levels_cfg, self.hydrostatic_alphas_cfg,
                )
                self.hydrostatic_weight = 0.0
            else:
                if len(self.hydrostatic_z_levels_cfg) != n_lev:
                    msg = (
                        f"hydrostatic_z_levels length ({len(self.hydrostatic_z_levels_cfg)}) "
                        f"must match number of resolved p/theta/qv levels ({n_lev})."
                    )
                    raise ValueError(msg)
                if len(self.hydrostatic_alphas_cfg) != n_lev - 1:
                    msg = (
                        f"hydrostatic_alphas length ({len(self.hydrostatic_alphas_cfg)}) "
                        f"must match number of adjacent level pairs ({n_lev - 1})."
                    )
                    raise ValueError(msg)
                self._hyd_p_idx     = torch.tensor([name_to_idx[n] for n in p_names],     dtype=torch.long)
                self._hyd_theta_idx = torch.tensor([name_to_idx[n] for n in theta_names], dtype=torch.long)
                self._hyd_qv_idx    = torch.tensor([name_to_idx[n] for n in qv_names],    dtype=torch.long)
                z = torch.tensor(self.hydrostatic_z_levels_cfg, dtype=torch.float32)
                self._hyd_dz    = (z[1:] - z[:-1]).clone()                                   # (n_lev-1,)
                self._hyd_alpha = torch.tensor(self.hydrostatic_alphas_cfg, dtype=torch.float32)
                LOGGER.info(
                    "GraphCastFullLoss.hydrostatic_weight=%.3e enabled on %d "
                    "level pairs.  p_names=%s  theta_names=%s  qv_names=%s",
                    self.hydrostatic_weight, n_lev - 1,
                    p_names, theta_names, qv_names,
                )
                LOGGER.info(
                    "  z_levels (m): %s",
                    [f"{z[i].item():.1f}" for i in range(n_lev)],
                )
                LOGGER.info(
                    "  α per pair (K): %s",
                    [f"{a:.3f}" for a in self.hydrostatic_alphas_cfg],
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

            # Per-variable gradient weights (√w applied to both pred and
            # tgt so the inner MSE picks up exactly w per variable).
            sqrt_w = None
            if self._grad_var_weights is not None:
                sqrt_w = self._grad_var_weights.to(pred_2d.device).sqrt().view(1, -1, 1, 1)

            if self.grad_x_weight != 0.0:
                dpred = _apply_stencil_x(pred_2d, kernel)
                dtgt = _apply_stencil_x(tgt_2d, kernel)
                if self.normalize_gradients:
                    sigma = self._sigma_for_axis(
                        dpred, dtgt, axis="x",
                    )
                    dpred = dpred / sigma
                    dtgt = dtgt / sigma
                if sqrt_w is not None:
                    dpred = dpred * sqrt_w
                    dtgt = dtgt * sqrt_w
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
                if sqrt_w is not None:
                    dpred = dpred * sqrt_w
                    dtgt = dtgt * sqrt_w
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

        # --- 4. Column mass-flux conservation -------------------------------
        # Domain-coherent column-integrated W (pressure-weighted) — the loss
        # is the squared difference of the *domain mean* of (Σ w_lev · Δp_lev)
        # between pred and target. By averaging over spatial cells first, the
        # term is blind to local storm-scale W variance (real convection) and
        # only catches column-coherent bias — i.e. compensating-subsidence
        # failure that produces a net upward/downward mass flux across the
        # whole domain.
        if (
            self.column_mass_flux_weight != 0.0
            and self._w_idx is not None
            and self._w_idx.numel() > 0
        ):
            w_idx = self._w_idx.to(pred.device)
            dp = self._w_level_dp.to(pred.device)            # (n_lev,)
            w_pred = pred.index_select(-1, w_idx)            # (B, E, G, n_lev)
            w_tgt  = target.index_select(-1, w_idx)
            cf_pred = (w_pred * dp).sum(dim=-1)              # (B, E, G)
            cf_tgt  = (w_tgt  * dp).sum(dim=-1)
            mu_pred = cf_pred.mean(dim=-1)                   # (B, E) domain mean
            mu_tgt  = cf_tgt.mean(dim=-1)
            mass_loss = ((mu_pred - mu_tgt) ** 2).mean()
            L = L + self.column_mass_flux_weight * mass_loss

        # --- 5. Hydrostatic-balance soft constraint -------------------------
        # Penalises deviations from hydrostatic balance via an error-tolerant
        # loss (DLESyM-style). Computed per-pixel per adjacent-level-pair:
        #
        #   T  = θ · (p/p0)^(R_d/c_p)
        #   T_v = T · (1 + 0.61·qv)
        #   r_k = mean(T_v_{k-1}, T_v_k) - (g/R_d) · (z_k - z_{k-1}) / ln(p_{k-1}/p_k)
        #   x   = r / α_k
        #   f   = x² / (1 + exp(1 - x²))                          # ≈0 for |x|<1
        #
        # α_k is the K-tolerance below which the loss is essentially flat;
        # supplied as a precomputed per-pair scalar from the training-zarr
        # natural distribution.
        if (
            self.hydrostatic_weight != 0.0
            and self._hyd_p_idx is not None
            and self._hyd_p_idx.numel() > 1
        ):
            R_d = 287.05
            c_p = 1004.0
            p0 = 1.0e5
            g = 9.80665
            p_idx  = self._hyd_p_idx.to(pred.device)
            th_idx = self._hyd_theta_idx.to(pred.device)
            qv_idx = self._hyd_qv_idx.to(pred.device)
            dz     = self._hyd_dz.to(device=pred.device, dtype=pred.dtype)
            alpha  = self._hyd_alpha.to(device=pred.device, dtype=pred.dtype)

            p_lvl  = pred.index_select(-1, p_idx)            # (B, E, G, n_lev)
            th_lvl = pred.index_select(-1, th_idx)
            qv_lvl = pred.index_select(-1, qv_idx)
            # Guard against pathological zero-/negative-pressure or negative-qv
            # noise the model may produce during early steps.
            p_lvl  = p_lvl.clamp(min=1.0)
            qv_lvl = qv_lvl.clamp(min=0.0)

            T   = th_lvl * (p_lvl / p0).pow(R_d / c_p)
            T_v = T * (1.0 + 0.61 * qv_lvl)
            T_v_mean = 0.5 * (T_v[..., 1:] + T_v[..., :-1])       # (B, E, G, n_pairs)
            p_lo = p_lvl[..., :-1]                                # lower-altitude (higher p)
            p_hi = p_lvl[..., 1:]                                 # upper-altitude (lower p)
            log_ratio = torch.log(p_lo / p_hi).clamp(min=1.0e-6)
            rhs = (g / R_d) * dz / log_ratio                       # broadcasts over (B,E,G)
            r = T_v_mean - rhs                                     # (B, E, G, n_pairs)  [K]
            x = r / alpha
            x2 = x * x
            f = x2 / (1.0 + torch.exp(1.0 - x2))
            L = L + self.hydrostatic_weight * f.mean()

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
