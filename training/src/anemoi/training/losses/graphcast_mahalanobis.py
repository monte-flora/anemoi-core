# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Mahalanobis-distance loss with precomputed tendency correlation.

Implements the loss from Agarwal et al., "Skillful Global Ocean Emulation
and the Role of Correlation-Aware Loss" (arXiv:2604.18727). Penalizes
residuals via a static, precomputed full correlation matrix of normalized
tendencies — penalizing errors that violate the climatological
cross-variable correlation structure.

Per-cell loss:

    L = r^T Σ^{-1} r            # use_sqrt=False (squared, default)
    L = sqrt(r^T Σ^{-1} r)      # use_sqrt=True (paper's exact form)

where r = pred - target in normalized tendency space (the residual the
training task already produces) and Σ is read from a zarr root array
``statistics_tendencies_<freq>_correlation`` shape (V_raw, V_raw).

Σ is loaded once at ``set_data_indices`` time, subsetted to the V_model
prognostic variables in model-output order, regularized with a small
diagonal jitter, and Cholesky-factorized to obtain ``L_inv`` such that

    r^T Σ^{-1} r  =  || L_inv @ r ||^2

Comparison with ``GraphCastGaussianNLLLoss``: both are members of the
Mahalanobis family. The NLL loss uses a *diagonal* Σ that is *learned
online* (`+ log(σ²)` regularizer required); this loss uses the *full*
Σ that is *static and precomputed* (no log-det term needed). They are
not interchangeable — the off-diagonal entries here capture
cross-variable correlations the diagonal cannot represent.

Example YAML configuration:

    training:
      training_loss:
        _target_: anemoi.training.losses.GraphCastMahalanobisLoss
        precomputed_stats_path: ${system.input.dataset}
        use_sqrt: False
        jitter: 1.0e-6
        scalers: ['limited_area_mask']
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses.base import BaseLoss
from anemoi.training.utils.enums import TensorDim

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup

LOGGER = logging.getLogger(__name__)


class GraphCastMahalanobisLoss(BaseLoss):
    """Mahalanobis-distance loss over a precomputed tendency correlation matrix.

    Parameters
    ----------
    precomputed_stats_path : str
        Path to the zarr root containing the correlation matrix. The loss
        reads ``statistics_tendencies_<freq>_correlation`` and uses the
        zarr's ``frequency`` attribute to resolve the key.
    correlation_key : str | None, optional
        Override the default key. If None (default), uses
        ``statistics_tendencies_<freq>_correlation``.
    use_sqrt : bool, optional
        If True, returns ``sqrt(r^T Σ^{-1} r)`` per cell (paper's exact form).
        If False (default), returns the squared form — cleaner gradient,
        equivalent ranking.
    jitter : float, optional
        Diagonal load applied to Σ before factorization, expressed as a
        fraction of the trace (i.e. ``Σ + jitter * (trace(Σ)/V) * I``).
        Default 1e-6.
    sqrt_eps : float, optional
        Numerical guard inside ``sqrt(... + eps)`` when ``use_sqrt=True``.
        Default 1e-8.
    ignore_nans : bool, optional
        If True, use ``nansum``/``nanmean`` for spatial reduction. Default False.

    Notes
    -----
    - This loss does NOT apply ``general_variable`` style per-variable
      scalers, because variable weighting is already encoded in
      ``Σ^{-1}``. Pass only spatial scalers (e.g. ``limited_area_mask``,
      ``node_weights``) via the ``scalers`` config.
    - The loss receives tensors of shape ``(B, E, G, V_model)`` where
      ``V_model`` is the count of prognostic outputs. The precomputed Σ
      is on the raw zarr index space; ``set_data_indices`` builds the
      mapping by variable name.
    """

    name: str = "graphcast_mahalanobis"

    def __init__(
        self,
        precomputed_stats_path: str,
        correlation_key: str | None = None,
        use_sqrt: bool = False,
        jitter: float = 1.0e-6,
        sqrt_eps: float = 1.0e-8,
        ignore_nans: bool = False,
    ) -> None:
        super().__init__(ignore_nans=ignore_nans)

        self.precomputed_stats_path = str(precomputed_stats_path)
        self.correlation_key_override = correlation_key
        self.use_sqrt = bool(use_sqrt)
        self.jitter = float(jitter)
        self.sqrt_eps = float(sqrt_eps)

        # Lazy import: zarr is only needed when this loss is constructed
        import zarr as _zarr

        z = _zarr.open(self.precomputed_stats_path, mode="r")
        zarr_variables = list(z.attrs.get("variables", []))
        if not zarr_variables:
            raise ValueError(
                f"GraphCastMahalanobisLoss: zarr {self.precomputed_stats_path!r} "
                "has no 'variables' attr."
            )
        self._zarr_name_to_index: dict[str, int] = {
            name: i for i, name in enumerate(zarr_variables)
        }

        freq = z.attrs.get("frequency", "15m")
        key = self.correlation_key_override or f"statistics_tendencies_{freq}_correlation"
        if key not in z:
            raise KeyError(
                f"GraphCastMahalanobisLoss: zarr {self.precomputed_stats_path!r} "
                f"is missing array {key!r}. Run "
                "grafai/datasets/compute_tendency_correlation.py to populate it."
            )
        sigma_full = torch.as_tensor(z[key][:], dtype=torch.float64)
        if sigma_full.shape != (len(zarr_variables), len(zarr_variables)):
            raise ValueError(
                f"GraphCastMahalanobisLoss: {key!r} shape {tuple(sigma_full.shape)} "
                f"!= (V_raw, V_raw)=({len(zarr_variables)}, {len(zarr_variables)})"
            )

        # Stash on the module; subsetting + factorization happens in
        # set_data_indices once we know the V_model -> raw mapping.
        self.register_buffer("_sigma_full", sigma_full, persistent=False)

        # Filled in by set_data_indices()
        self._L_inv: torch.Tensor | None = None
        self._n_vars: int | None = None
        self._var_names_in_order: list[str] | None = None
        LOGGER.info(
            "GraphCastMahalanobisLoss: loaded Σ from %s (key=%r, shape=%s, freq=%s)",
            self.precomputed_stats_path, key, tuple(sigma_full.shape), freq,
        )
        LOGGER.info(
            "GraphCastMahalanobisLoss: use_sqrt=%s jitter=%.2e",
            self.use_sqrt, self.jitter,
        )

    def set_data_indices(self, data_indices: IndexCollection) -> None:
        """Subset Σ to the prognostic outputs in model-output order, factorize."""
        if data_indices is None:
            return

        # The loss receives V_model = count of prognostic outputs. Build the
        # ordered list of names in model-output order, restricted to prognostics.
        prognostic_idxs = data_indices.data.output.prognostic
        prognostic_set = set(
            prognostic_idxs.tolist()
            if hasattr(prognostic_idxs, "tolist")
            else list(prognostic_idxs)
        )
        name_to_idx = data_indices.model.output.name_to_index
        # Sort prognostic names by their model-output index — that is the
        # ordering of V in the loss tensor.
        ordered = sorted(
            ((idx, name) for name, idx in name_to_idx.items() if idx in prognostic_set),
            key=lambda p: p[0],
        )
        ordered_names = [name for _, name in ordered]

        missing = [n for n in ordered_names if n not in self._zarr_name_to_index]
        if missing:
            raise KeyError(
                f"GraphCastMahalanobisLoss: {len(missing)} prognostic name(s) not in "
                f"zarr 'variables' attr: {missing[:10]}"
            )
        zarr_idx = torch.tensor(
            [self._zarr_name_to_index[n] for n in ordered_names], dtype=torch.long
        )

        # Subset Σ to (V_model, V_model), in model-output order
        sigma = self._sigma_full[zarr_idx][:, zarr_idx].to(torch.float64)
        V = sigma.shape[0]

        # Symmetrize numerically (correlation matrices off-disk can drift slightly)
        sigma = 0.5 * (sigma + sigma.T)

        # Diagonal jitter scaled to the trace, then factorize Σ = L L^T
        diag_load = self.jitter * (sigma.diagonal().abs().mean() + 1.0e-12)
        sigma = sigma + diag_load * torch.eye(V, dtype=sigma.dtype)

        try:
            L = torch.linalg.cholesky(sigma)
        except RuntimeError as exc:
            raise RuntimeError(
                f"GraphCastMahalanobisLoss: Cholesky failed even with jitter={self.jitter}. "
                "Increase the jitter or inspect the precomputed correlation matrix."
            ) from exc

        # L_inv solves L L^T x = r in two triangular sweeps; precomputing L_inv
        # lets the forward pass be a single dense matmul.  Σ^{-1} = L_inv^T L_inv,
        # so r^T Σ^{-1} r = || L_inv @ r ||^2.
        L_inv = torch.linalg.solve_triangular(
            L, torch.eye(V, dtype=L.dtype), upper=False
        )

        # Cast back to float32 — bf16 forward will upcast as needed.
        self.register_buffer("_L_inv", L_inv.to(torch.float32), persistent=False)
        self._n_vars = V
        self._var_names_in_order = ordered_names

        cond = float(torch.linalg.cond(sigma).item())
        LOGGER.info(
            "GraphCastMahalanobisLoss: Σ subsetted to V=%d, cond(Σ+jitter)=%.3e, "
            "diag_load=%.3e",
            V, cond, float(diag_load),
        )

    def _whitened_squared_residual(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Return ``r^T Σ^{-1} r`` per cell, shape ``(B, E, G, 1)``."""
        if self._L_inv is None:
            raise RuntimeError(
                "GraphCastMahalanobisLoss: set_data_indices() must run before forward()."
            )
        # Compute residual in float32 for numerical stability with bf16 inputs.
        r = pred.float() - target.float()  # (B, E, G, V)
        L_inv = self._L_inv.to(device=r.device, dtype=r.dtype)
        # r_w[..., i] = sum_j L_inv[i, j] r[..., j]
        r_w = torch.einsum("begv,iv->begi", r, L_inv)
        per_cell = r_w.pow(2).sum(dim=-1, keepdim=True)  # (B, E, G, 1)
        return per_cell

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,  # noqa: ARG002 — single output dim; squash is a no-op
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: "ProcessGroup | None" = None,
        **kwargs,  # noqa: ARG002
    ) -> torch.Tensor:
        is_sharded = grid_shard_slice is not None

        per_cell = self._whitened_squared_residual(pred, target)  # (B, E, G, 1)

        if self.use_sqrt:
            per_cell = torch.sqrt(per_cell + self.sqrt_eps)

        # Spatial scalers (e.g. limited_area_mask, node_weights) act on the
        # GRID dim and broadcast against the singleton V dim. Per-variable
        # scalers (general_variable, variable_level_scaler) MUST NOT be in the
        # config — variable weighting is encoded in Σ^{-1} already.
        per_cell = self.scale(
            per_cell,
            scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )

        # Reduce: sum over GRID (unit-sum weights), mean over BATCH/ENSEMBLE,
        # squeeze trailing V=1.  Mirrors BaseLoss.reduce but skips the
        # squash-mode branch since V is already collapsed.
        per_cell = per_cell.squeeze(-1)  # (B, E, G)
        grid_summed = self.sum_function(per_cell, dim=TensorDim.GRID)  # (B, E)
        out = self.avg_function(
            grid_summed, dim=(TensorDim.BATCH_SIZE, TensorDim.ENSEMBLE_DIM)
        )

        if is_sharded and group is not None:
            from anemoi.models.distributed.graph import reduce_tensor

            out = reduce_tensor(out, group)
        return out
