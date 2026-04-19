# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Wind-aware loss wrapper: decompose (u, v) into (direction, speed).

Problem
-------
MSE training on (u, v) wind components rewards magnitude dampening: shrinking
predicted wind vectors reduces directional-error penalties at the cost of
under-forecasting wind speed — particularly harmful for extreme-wind events.

Solution (after FastNet Sec. "Wind speed + direction decomposition")
--------------------------------------------------------------------
Leave the model's inputs / outputs as (u, v). *Only inside the loss*:

    for each pair (u_k, v_k):
        s_k   = sqrt(u_k² + v_k² + ε²)
        d_i_k = u_k / s_k       # unit direction (zonal)
        d_j_k = v_k / s_k       # unit direction (meridional)

and compute:

    L_total =    inner( pred with u→d_i, v→d_j, ... )      ← direction (+ non-wind)
              +  λ_speed · inner( speed-only mask,  ... )   ← magnitude
             with λ_speed = 5.0 (per FastNet).

Unit-vector form of direction avoids atan2 in the loss (no gradient
discontinuities at quadrant boundaries).

Residual-normalized space note
------------------------------
The loss receives Δu_norm, Δv_norm (via GraphResidualForecaster). So s is
the magnitude of the *wind change* over one step, not the state wind speed.
Physical interpretation shifts, but the anti-dampening mechanism still
applies: a model that shrinks Δu to reduce directional penalty also shrinks
s_delta below truth, which the speed term detects directly.

Auto-detection
--------------
At ``set_data_indices`` the wrapper scans model-output variable names for
pairs with a shared suffix after ``u_`` / ``v_`` (e.g. ``u_0``↔``v_0``,
``u_10m``↔``v_10m``). Override via the ``u_v_pairs`` kwarg for non-standard
naming.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.combined import CombinedLoss

LOGGER = logging.getLogger(__name__)

# Default pair detection: `u_<suffix>` pairs with `v_<suffix>`. Examples that
# match: u_0/v_0, u_33/v_33, u_10m/v_10m. Fine-grained overrides can be
# provided via the ``u_v_pairs`` constructor kwarg.
_DEFAULT_U_PREFIX = "u_"
_DEFAULT_V_PREFIX = "v_"


class GraphCastWindAwareLoss(CombinedLoss):
    r"""Wind-aware loss wrapper.

    Wraps an inner loss. For each detected (u, v) pair in the model output
    variables, replaces ``u`` with ``u/s`` and ``v`` with ``v/s`` at the
    loss input, and adds a separately-weighted ``s = sqrt(u² + v² + ε²)``
    term routed through the inner loss at the ``u``-variable slots.

    Parameters
    ----------
    inner_loss : BaseLoss | dict | DictConfig
        Inner loss (leaf or CombinedLoss). Passed through CombinedLoss's
        init, which handles Hydra instantiation from dict/DictConfig.
    u_v_pairs : list[tuple[str, str]] | None, optional
        Explicit pairs of (u_name, v_name). When None (default), auto-
        detect from model-output variable names using prefix matching.
    speed_weight : float, optional
        Multiplier applied to the speed contribution. Default 5.0 per
        FastNet.
    epsilon : float, optional
        Small constant inside the sqrt for numerical stability
        (s = sqrt(u² + v² + ε²)). Default 1e-6.
    """

    def __init__(
        self,
        inner_loss: BaseLoss | dict | DictConfig,
        *,
        u_v_pairs: list[tuple[str, str]] | None = None,
        speed_weight: float = 5.0,
        epsilon: float = 1e-6,
        **inner_kwargs: Any,
    ) -> None:
        # Inherit CombinedLoss init: handles Hydra DictConfig instantiation,
        # scaler routing, and isinstance(CombinedLoss) → utils.py recursion.
        CombinedLoss.__init__(
            self,
            losses=[inner_loss],
            loss_weights=[1.0],
            **inner_kwargs,
        )
        self.inner = self.losses[0]
        self.u_v_pairs_override = (
            [tuple(p) for p in u_v_pairs] if u_v_pairs is not None else None
        )
        self.speed_weight = float(speed_weight)
        self.epsilon = float(epsilon)

        # Resolved at set_data_indices time.
        self._u_idx: torch.Tensor | None = None
        self._v_idx: torch.Tensor | None = None
        self._pair_names: list[tuple[str, str]] = []

        LOGGER.info(
            "GraphCastWindAwareLoss initialised: speed_weight=%.2f, epsilon=%.0e, "
            "u_v_pairs_override=%s",
            self.speed_weight, self.epsilon,
            "yes" if u_v_pairs is not None else "no (auto-detect)",
        )

    # ------------------------------------------------------------------
    # Pair resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _autodetect_pairs(names: list[str]) -> list[tuple[str, str]]:
        """Find (u_X, v_X) pairs by shared suffix after ``u_`` / ``v_``."""
        name_set = set(names)
        pairs: list[tuple[str, str]] = []
        for n in names:
            if not n.startswith(_DEFAULT_U_PREFIX):
                continue
            suffix = n[len(_DEFAULT_U_PREFIX):]
            v_name = _DEFAULT_V_PREFIX + suffix
            if v_name in name_set:
                pairs.append((n, v_name))
        # Keep a stable, name-sorted order so the behaviour is deterministic.
        pairs.sort()
        return pairs

    def set_data_indices(self, data_indices) -> None:
        """Forward to inner + resolve u/v index tensors."""
        if data_indices is None:
            return
        if hasattr(self.inner, "set_data_indices"):
            self.inner.set_data_indices(data_indices)

        name_to_idx = dict(data_indices.model.output.name_to_index)
        if self.u_v_pairs_override is not None:
            pairs = self.u_v_pairs_override
        else:
            pairs = self._autodetect_pairs(list(name_to_idx.keys()))

        missing = [p for p in pairs if p[0] not in name_to_idx or p[1] not in name_to_idx]
        if missing:
            LOGGER.warning(
                "GraphCastWindAwareLoss: skipping pairs missing from model output: %s",
                missing,
            )
            pairs = [p for p in pairs if p not in missing]

        if not pairs:
            LOGGER.warning(
                "GraphCastWindAwareLoss: no (u, v) pairs resolved; wrapper will "
                "pass through untouched."
            )
            self._pair_names = []
            self._u_idx = None
            self._v_idx = None
            return

        u_idx = torch.tensor([name_to_idx[u] for u, _ in pairs], dtype=torch.long)
        v_idx = torch.tensor([name_to_idx[v] for _, v in pairs], dtype=torch.long)
        # Use plain attributes (index tensors don't need to be buffers — they're
        # tiny and constant for the life of the run).
        self._u_idx = u_idx
        self._v_idx = v_idx
        self._pair_names = pairs
        LOGGER.info(
            "GraphCastWindAwareLoss: resolved %d (u, v) pairs: %s",
            len(pairs),
            ", ".join(f"{u}↔{v}" for u, v in pairs),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self._u_idx is None or self._v_idx is None or self._u_idx.numel() == 0:
            # No pairs resolved (e.g. set_data_indices was never called, or
            # the output has no winds). Transparent passthrough.
            return self.inner(pred, target, **kwargs)

        u_idx = self._u_idx.to(device=pred.device)
        v_idx = self._v_idx.to(device=pred.device)

        # Extract the u / v slices
        u_pred = pred.index_select(-1, u_idx)
        v_pred = pred.index_select(-1, v_idx)
        u_tgt = target.index_select(-1, u_idx)
        v_tgt = target.index_select(-1, v_idx)

        # s = sqrt(u² + v² + ε²); safe against s → 0.
        eps_sq = self.epsilon * self.epsilon
        s_pred = torch.sqrt(u_pred * u_pred + v_pred * v_pred + eps_sq)
        s_tgt = torch.sqrt(u_tgt * u_tgt + v_tgt * v_tgt + eps_sq)
        d_i_pred = u_pred / s_pred
        d_j_pred = v_pred / s_pred
        d_i_tgt = u_tgt / s_tgt
        d_j_tgt = v_tgt / s_tgt

        # Build decomposed tensors: replace u with d_i, v with d_j at wind slots.
        pred_dir = pred.clone()
        target_dir = target.clone()
        pred_dir.index_copy_(-1, u_idx, d_i_pred)
        pred_dir.index_copy_(-1, v_idx, d_j_pred)
        target_dir.index_copy_(-1, u_idx, d_i_tgt)
        target_dir.index_copy_(-1, v_idx, d_j_tgt)

        # Speed-only tensors: zeros everywhere, s at u_idx slots. Inner's
        # per-variable weights on the u-group apply; zero entries elsewhere
        # contribute nothing.
        pred_spd = torch.zeros_like(pred)
        target_spd = torch.zeros_like(target)
        pred_spd.index_copy_(-1, u_idx, s_pred)
        target_spd.index_copy_(-1, u_idx, s_tgt)

        L_dir = self.inner(pred_dir, target_dir, **kwargs)
        L_spd = self.inner(pred_spd, target_spd, **kwargs)
        return L_dir + self.speed_weight * L_spd

    # ------------------------------------------------------------------
    # Scaler routing: same override as HorizontalGradientLoss — CombinedLoss's
    # default add_scaler reaches into self.losses[i].scaler which is deleted
    # when the child is itself a CombinedLoss. Delegating to inner.add_scaler
    # lets the inner handle further routing.
    # ------------------------------------------------------------------
    def add_scaler(self, dimension, scaler, *, name: str | None = None) -> None:
        self.inner.add_scaler(dimension, scaler, name=name)

    def update_scaler(self, name: str, scaler, *, override: bool = False) -> None:
        self.inner.update_scaler(name, scaler=scaler, override=override)
