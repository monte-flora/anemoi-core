# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""GraphCast-style Modified Spherical Harmonic (MSH) loss for 2D Cartesian LAM grids.

On a regular 2D grid the MSH loss reduces to a radially-binned 2D-FFT loss
following the formulation of Subich et al. 2025 (arXiv:2502.14506) and
FastNet (arXiv:2509.17658, Eq. 3):

    AMSE_k(x̂, x) = ( √PSD_k(x̂) − √PSD_k(x) )²                        [amplitude term]
                 + 2 · max(PSD_k(x̂), PSD_k(x)) · ( 1 − Coh_k(x̂, x) ) [coherence term]

with PSD_k(x) = Σ_{l at radius k} |α_x(k,l)|² and
Coh_k(x̂,x) = Σ_l Re[α_x̂(k,l) · α_x*(k,l)] / √(PSD_k(x̂) · PSD_k(x)).

Optional γ_k weighting (FastNet Eq. 9) emphasises small-scale errors that
the k⁻³ / k⁻⁵⁄³ atmospheric PSD decay would otherwise under-weight:

    γ_k = max( N_k · k / √3 , 1.0 )   with N_k = 1 / mean_k(k/√3)

The loss inherits GraphCastBaseLoss reduction semantics (mean-within-group,
sum-across-groups) so multi-level variables like u_0..u_33 contribute
equally with 2D variables like t2m.

Residual-normalised-space note: when used inside GraphResidualForecaster the
inputs `Δx̂_norm = (y_pred − x_last) / σ_Δx` already have per-variable
inverse-variance-of-diffs (FastNet s_j) applied implicitly. No extra scaler
needed for s_j — γ_k and the variable/LAM scalers layer on top.
"""

from __future__ import annotations

import logging

import einops
import numpy as np
import torch
import torch.distributed as dist
import torch.fft
import torch.nn as nn

from anemoi.training.losses.base import GraphCastBaseLoss
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


def _load_precomputed_array_from_zarr(zarr_path: str, name: str) -> np.ndarray:
    """Load a 1-D statistics array from the zarr root.

    Matches the raw-zarr access pattern used by
    ``anemoi.datasets.data.stores.Store.statistics_tendencies``: opens the
    store read-only and slices the named dataset into a numpy array.
    """
    import zarr  # lazy import — only needed when precomputed path is set

    z = zarr.open(zarr_path, mode="r")
    return np.asarray(z[name][:])


# ============================================================================
# Helpers
# ============================================================================
def _build_radial_bins(
    x_dim: int,
    y_dim: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Precompute index + conjugate-pair weight + k-centres for an rfft2 layout.

    The 2D real-to-complex FFT (``torch.fft.rfft2``) returns only the
    ``(H, W//2+1)`` half-plane of the full ``(H, W)`` complex spectrum — the
    missing half follows by conjugate symmetry ``F(-kx, -ky) = F*(kx, ky)``.
    To make the radial-bin PSD match ``|fft2|²`` exactly, we weight the
    interior columns (``0 < ky < W/2``) by 2× before scatter-add; the DC
    (``ky=0``) and Nyquist (``ky=W/2``, even ``W``) columns are already
    self-conjugate along ``kx`` and keep weight 1.

    Returns
    -------
    flat_bin_idx : LongTensor, shape (x_dim * (y_dim//2 + 1),)
        For each rfft bin, the integer radial wavenumber it belongs to
        (k = round(sqrt(kx² + ky²))).
    flat_weight : FloatTensor, same shape as flat_bin_idx
        Per-bin conjugate-pair multiplier (1.0 for DC/Nyquist columns, 2.0 for
        interior columns).
    k_centres : FloatTensor, shape (n_bins,)
        The integer k value at each bin centre.
    n_bins : int
        Total number of radial bins (= 1 + floor(sqrt((x_dim/2)² + (y_dim/2)²))).
    """
    kx = torch.fft.fftfreq(x_dim, device=device, dtype=dtype) * x_dim
    ky = torch.fft.rfftfreq(y_dim, device=device, dtype=dtype) * y_dim
    kxv, kyv = torch.meshgrid(kx, ky, indexing="ij")
    k_mag = torch.sqrt(kxv**2 + kyv**2)  # (x_dim, y_dim//2 + 1)
    bin_idx = torch.round(k_mag).to(torch.long)

    # Conjugate-pair weight: non-DC-non-Nyquist rfft columns represent two
    # modes in the full fft2 spectrum (F and F*), so count them twice.
    weight = torch.full_like(k_mag, 2.0)
    weight[:, 0] = 1.0  # DC column (ky = 0)
    if y_dim % 2 == 0:
        weight[:, -1] = 1.0  # Nyquist column (ky = y_dim/2)

    n_bins = int(bin_idx.max().item()) + 1
    k_centres = torch.arange(n_bins, device=device, dtype=dtype)
    return bin_idx.flatten(), weight.flatten(), k_centres, n_bins


def _gamma_k_weights(
    k_centres: torch.Tensor,
    *,
    normalise: bool = True,
    min_weight: float = 1.0,
    exponent: float = 3**0.5,  # FastNet Eq. 9: k^√3
) -> torch.Tensor:
    """FastNet Eq. 9: γ_k = max(N_k · k^√3, 1.0).

    Aggressive high-k emphasis that partially compensates the atmospheric
    k^-3 / k^-5/3 PSD decay. N_k is a per-spectrum normaliser so that the
    mean γ_k across bins stays ≈ 1 before the floor is applied; this
    preserves the overall AMSE magnitude for a given variable while
    redistributing weight toward small scales.
    """
    # k^exponent; guard k=0 (0^anything = 0 → clamp_min later).
    raw = k_centres.clamp_min(0.0) ** exponent
    if normalise:
        nonzero = raw[raw > 0]
        if nonzero.numel():
            raw = raw / max(float(nonzero.mean().item()), 1e-12)
    return torch.clamp(raw, min=min_weight)


class _OnlineAMSEStats(nn.Module):
    """Per-variable AMSE running mean via Chan's parallel Welford (fp64).

    Used for FastNet's β_j = 1 / <AMSE_j>_year correction. We compute the
    running mean of the per-variable AMSE observed during training (pred
    vs target) and use β_j = 1/mean as a per-variable multiplier so each
    variable contributes ~O(1) to the loss regardless of its native AMSE
    scale. Update is no-grad; β_j value changes only between optimizer
    steps, so this can't destabilise training.
    """

    def __init__(self, n_vars: int) -> None:
        super().__init__()
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("mean",  torch.zeros(n_vars, dtype=torch.float64))

    @torch.no_grad()
    def update(self, amse_per_var: torch.Tensor, distributed: bool = False) -> None:
        x = amse_per_var.reshape(-1, amse_per_var.shape[-1]).to(torch.float64)
        n_new = torch.tensor(float(x.shape[0]), dtype=torch.float64, device=x.device)
        if float(n_new) == 0.0:
            return
        mean_new = x.mean(dim=0)

        if distributed and dist.is_available() and dist.is_initialized():
            # Pool rank-local (n_i, μ_i) via count-weighted sum → global μ.
            count_mean = n_new * mean_new
            dist.all_reduce(n_new, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_mean, op=dist.ReduceOp.SUM)
            mean_new = count_mean / n_new.clamp_min(1.0)

        delta = mean_new - self.mean
        total = self.count + n_new
        self.mean.add_(delta * n_new / total.clamp_min(1.0))
        self.count.copy_(total)

    @property
    def beta(self) -> torch.Tensor:
        """β_j = 1 / <AMSE_j>, fp32 with a small floor to avoid blowup pre-warmup."""
        return (1.0 / self.mean.clamp_min(1e-10)).float()


def _radially_bin_sum(
    field: torch.Tensor,
    flat_bin_idx: torch.Tensor,
    n_bins: int,
) -> torch.Tensor:
    """Scatter-sum `field` over its last-but-one dim into radial k bins.

    Parameters
    ----------
    field : Tensor of shape (..., n_pixels, V)
        Values at each FFT bin. Typically real-valued (e.g. |α|²).
    flat_bin_idx : LongTensor of shape (n_pixels,)
        Bin assignment for each pixel.
    n_bins : int
        Number of output radial bins.

    Returns
    -------
    Tensor of shape (..., n_bins, V).
    """
    # scatter-add along the pixel axis
    *lead, n_pix, V = field.shape
    out = field.new_zeros((*lead, n_bins, V))
    # expand bin index to (1..., n_pix, 1) for scatter
    idx = flat_bin_idx.view(*([1] * len(lead)), n_pix, 1).expand_as(field)
    out.scatter_add_(-2, idx, field)
    return out


# ============================================================================
# The MSH loss
# ============================================================================
class GraphCastMSHLoss(GraphCastBaseLoss):
    r"""GraphCast-style MSH loss on a 2D Cartesian LAM grid.

    FastNet-faithful implementation with radial k-binning, coherence term,
    and optional γ_k weighting. Per-variable-group reduction is inherited
    from GraphCastBaseLoss.

    Parameters
    ----------
    x_dim, y_dim : int
        Spatial dims of the 2D grid. Must satisfy x_dim * y_dim == grid_size.
    ignore_nans : bool, optional
        Use nan-safe reductions, by default False.
    coherence_weight : float, optional
        Multiplier on the coherence (phase) term. 1.0 matches FastNet Eq. 3.
        Set to 0 to reduce to pure amplitude-spectrum loss. Default 1.0.
    use_gamma_k : bool, optional
        Apply γ_k wavenumber weighting per FastNet Eq. 9. Default True.
    gamma_k_min : float, optional
        Floor for γ_k (FastNet's `max(·, 1.0)`). Default 1.0.
    n_vars : int, optional
        Number of output variables. Required if `use_variable_normalization=True`
        — sizes the online-AMSE Welford buffers. Default None (disables β_j).
    use_variable_normalization : bool, optional
        Apply FastNet's β_j = 1/<AMSE_j> per-variable correction (Eq. 8).
        Uses an online Welford estimator updated from training batches; no
        offline pass required. Default False (opt-in).
    distributed_stats : bool, optional
        If True, reduce the Welford partial stats across DDP ranks each
        update. Default True.
    """

    # Public alias used in class docs / error messages.
    _loss_name = "GraphCastMSHLoss"

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        ignore_nans: bool = False,
        coherence_weight: float = 1.0,
        use_gamma_k: bool = True,
        gamma_k_min: float = 1.0,
        n_vars: int | None = None,
        use_variable_normalization: bool = False,
        distributed_stats: bool = True,
        precomputed_stats_path: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(ignore_nans=ignore_nans, **kwargs)
        self.x_dim = int(x_dim)
        self.y_dim = int(y_dim)
        self.coherence_weight = float(coherence_weight)
        self.use_gamma_k = bool(use_gamma_k)
        self.gamma_k_min = float(gamma_k_min)
        self.use_variable_normalization = bool(use_variable_normalization)
        self.distributed_stats = bool(distributed_stats)
        self.precomputed_stats_path = str(precomputed_stats_path) if precomputed_stats_path else None

        # Precompute bin lookup + conjugate-pair weight + γ_k as buffers so
        # they move with .to(device).
        flat_bin_idx, flat_weight, k_centres, n_bins = _build_radial_bins(self.x_dim, self.y_dim)
        gamma_k = _gamma_k_weights(k_centres, min_weight=self.gamma_k_min) if self.use_gamma_k else torch.ones_like(k_centres)
        self.register_buffer("_flat_bin_idx", flat_bin_idx, persistent=False)
        self.register_buffer("_flat_weight", flat_weight, persistent=False)
        self.register_buffer("_k_centres", k_centres, persistent=False)
        self.register_buffer("_gamma_k", gamma_k, persistent=False)
        self._n_bins = int(n_bins)
        self._y_rfft = int(self.y_dim // 2 + 1)

        # β_j setup. Two sources:
        #   (1) precomputed_stats_path — reads `statistics_msh_beta` from the
        #       zarr via raw zarr access (same pattern as
        #       anemoi.datasets.data.stores.Store.statistics_tendencies).
        #       Data-dim (V_data) → model-output-dim (V_model) subsetting is
        #       resolved in set_data_indices().
        #   (2) Online Welford on AMSE(pred, target) during training.
        # Path (1) skips the Welford allocation and update entirely.
        self.n_vars = int(n_vars) if n_vars is not None else None

        if self.use_variable_normalization and self.precomputed_stats_path is not None:
            import zarr as _zarr
            _z = _zarr.open(self.precomputed_stats_path, mode="r")
            # Residual-space β matches what GraphResidualForecaster's loss sees
            # at training time — the tendency has already been divided by
            # statistics_tendencies_<freq>_stdev by the model's ResidualNormalizer.
            _freq = _z.attrs.get("frequency", "15m")
            _beta_key = f"statistics_tendencies_{_freq}_msh_beta"
            beta_data = np.asarray(_z[_beta_key][:])
            zarr_variables = list(_z.attrs.get("variables", []))
            if len(zarr_variables) != beta_data.shape[0]:
                raise ValueError(
                    f"zarr 'variables' attr ({len(zarr_variables)}) != "
                    f"statistics_msh_beta length ({beta_data.shape[0]})"
                )
            # RAW-zarr index lookup by name — bypasses anemoi-datasets' `drop`
            # filter, which would otherwise map names into a compressed
            # (dropped) index space that doesn't match our raw stats array.
            self._zarr_name_to_index: dict[str, int] = {
                name: i for i, name in enumerate(zarr_variables)
            }
            self.register_buffer(
                "_precomputed_beta_data",
                torch.as_tensor(beta_data, dtype=torch.float32),
                persistent=False,
            )
            self.amse_stats = None
            self._data_idx_for_model_output: torch.Tensor | None = None
            LOGGER.info(
                "%s: using precomputed β from %s (zarr dim V_data=%d)",
                self._loss_name, self.precomputed_stats_path, beta_data.shape[0],
            )
        elif self.use_variable_normalization:
            if n_vars is None:
                raise ValueError(
                    "GraphCastMSHLoss: n_vars is required when "
                    "use_variable_normalization=True and no precomputed_stats_path is given",
                )
            self.amse_stats = _OnlineAMSEStats(int(n_vars))
            self._data_idx_for_model_output = None
        else:
            self.amse_stats = None
            self._data_idx_for_model_output = None

        # Register a trivial GRID-dim scaler of size 1 so BaseLoss.scale()'s
        # "scaler tensor must be at least applied to the GRID dimension" check
        # passes. Our calculate_difference reduces the grid dimension to 1
        # (radial-binned & summed), so a unit weight there is the identity.
        self.add_scaler(
            TensorDim.GRID.value,
            torch.ones(1, dtype=torch.float32),
            name="msh_grid_placeholder",
        )

        LOGGER.info(
            "%s initialised: field=%dx%d, n_bins=%d, coherence_weight=%.2f, use_gamma_k=%s",
            self._loss_name, self.x_dim, self.y_dim, self._n_bins,
            self.coherence_weight, self.use_gamma_k,
        )

    # ------------------------------------------------------------------
    # Core math
    # ------------------------------------------------------------------
    def _amse_per_variable(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the FastNet AMSE summed over radial bins, per variable.

        Returns a tensor of shape (B, E, 1, V) with a sentinel grid dim so
        downstream scalers / reductions (which expect (B, E, G, V)) still work.
        """
        grid_size = pred.shape[TensorDim.GRID]
        assert self.x_dim * self.y_dim == grid_size, (
            f"{self._loss_name}: x_dim*y_dim={self.x_dim * self.y_dim} "
            f"!= grid_size={grid_size}"
        )

        # Reshape (B, E, H*W, V) -> (B, E, H, W, V) -> FFT over (H, W)
        # then flatten pixels back for scatter-add: -> (B, E, H*W, V)
        pred_2d = einops.rearrange(
            pred, "b e (h w) v -> b e h w v", h=self.x_dim, w=self.y_dim
        )
        target_2d = einops.rearrange(
            target, "b e (h w) v -> b e h w v", h=self.x_dim, w=self.y_dim
        )

        # Real-to-complex FFT (rfft2): returns (H, W//2+1) half-plane; the
        # missing half is recovered by the conjugate-pair weight baked into
        # self._flat_weight. ~50% memory and ~1.5-2× faster than full fft2.
        # FP32 for numerical stability under bf16-mixed precision.
        orig_dtype = pred.dtype
        alpha_pred = torch.fft.rfft2(pred_2d.float(), dim=(-3, -2))
        alpha_true = torch.fft.rfft2(target_2d.float(), dim=(-3, -2))

        # |α|² per (H, W//2+1) bin and Re[α_pred · α_true*]
        power_pred = alpha_pred.real**2 + alpha_pred.imag**2
        power_true = alpha_true.real**2 + alpha_true.imag**2
        cross = alpha_pred.real * alpha_true.real + alpha_pred.imag * alpha_true.imag

        # Flatten pixels, apply conjugate-pair weight, scatter-add into radial bins.
        # power/cross shapes: (..., H, W//2+1, V) -> (..., H*(W//2+1), V)
        n_pix_rfft = self.x_dim * self._y_rfft
        shape_flat = power_pred.shape[:-3] + (n_pix_rfft, power_pred.shape[-1])
        power_pred = power_pred.reshape(shape_flat)
        power_true = power_true.reshape(shape_flat)
        cross = cross.reshape(shape_flat)

        # Weight shape (n_pix_rfft,) -> broadcast over leading dims + last V
        w = self._flat_weight.view(*([1] * (power_pred.dim() - 2)), n_pix_rfft, 1)
        power_pred = power_pred * w
        power_true = power_true * w
        cross = cross * w

        psd_pred = _radially_bin_sum(power_pred, self._flat_bin_idx, self._n_bins)
        psd_true = _radially_bin_sum(power_true, self._flat_bin_idx, self._n_bins)
        cross_k = _radially_bin_sum(cross, self._flat_bin_idx, self._n_bins)

        # Amplitude term: (√PSD_pred − √PSD_true)²
        eps = torch.finfo(psd_pred.dtype).eps
        amp_err = (torch.sqrt(psd_pred + eps) - torch.sqrt(psd_true + eps)) ** 2

        if self.coherence_weight > 0.0:
            # Coh_k = cross_k / √(PSD_pred · PSD_true), clamped to [-1, 1]
            denom = torch.sqrt(psd_pred * psd_true + eps)
            coh_k = torch.clamp(cross_k / denom, min=-1.0, max=1.0)
            coh_err = 2.0 * torch.maximum(psd_pred, psd_true) * (1.0 - coh_k)
        else:
            coh_err = torch.zeros_like(amp_err)

        # (B, E, n_bins, V)
        amse_k = amp_err + self.coherence_weight * coh_err

        # γ_k along the bin axis: broadcast (n_bins,) to (1, 1, n_bins, 1)
        gamma_shape = (1,) * (amse_k.dim() - 2) + (self._n_bins, 1)
        amse_k = amse_k * self._gamma_k.view(*gamma_shape)

        # Sum over k bins -> (B, E, V)
        per_var = amse_k.sum(dim=-2)

        # FastNet β_j (Eq. 8): per-variable normalization so every variable
        # contributes ~O(1) regardless of its native AMSE scale.
        # Source order: (1) pre-computed stats from zarr → (2) online Welford.
        if self.use_variable_normalization and getattr(self, "_precomputed_beta_data", None) is not None:
            if self._data_idx_for_model_output is None:
                raise RuntimeError(
                    "GraphCastMSHLoss: precomputed β enabled but set_data_indices() "
                    "has not been called yet — no model-output→data-index mapping.",
                )
            idx = self._data_idx_for_model_output.to(per_var.device)
            beta = self._precomputed_beta_data.to(device=per_var.device, dtype=per_var.dtype)[idx]
            per_var = per_var * beta.view(1, 1, -1)
        elif self.use_variable_normalization and self.amse_stats is not None:
            if self.training:
                self.amse_stats.update(per_var.detach(), distributed=self.distributed_stats)
            beta = self.amse_stats.beta.to(device=per_var.device, dtype=per_var.dtype)
            per_var = per_var * beta.view(1, 1, -1)

        # Add sentinel grid dim -> (B, E, 1, V)
        per_var = per_var.unsqueeze(TensorDim.GRID)
        return per_var.to(dtype=orig_dtype)

    # ------------------------------------------------------------------
    # FunctionalLoss contract: calculate_difference returns the per-element
    # error that the scaler framework multiplies against.
    # ------------------------------------------------------------------
    def calculate_difference(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Return (B, E, 1, V) AMSE summed over radial k bins."""
        return self._amse_per_variable(pred, target)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group=None,
        **kwargs,  # noqa: ARG002
    ) -> torch.Tensor:
        """Compute scaled AMSE, reduce per variable group and batch.

        Excludes grid-dim scalers (limited_area_mask, node_weights) by default
        because the loss output has a sentinel G=1 dim; those scalers would
        mis-broadcast. They're moot anyway when the training config supplies
        the LAM-masked region as the input grid.
        """
        # Exclude grid-space scalers — they expect (B, E, G, V) with full G
        grid_scalers = ["limited_area_mask", "node_weights", "nan_mask_weights"]
        user_exclude = list(without_scalers or [])
        merged_exclude = list({*user_exclude, *grid_scalers})

        out = self.calculate_difference(pred, target)
        out = self.scale(
            out,
            scaler_indices,
            without_scalers=merged_exclude,
            grid_shard_slice=None,  # no grid dim to shard
        )
        # GraphCastBaseLoss.reduce does mean-within-variable-group, sum-across
        return self.reduce(out, squash, group=None)

    # ------------------------------------------------------------------
    # Data-indices resolution — needed when precomputed_stats_path is set
    # so we can map (V_model,) → (V_data,) for the β lookup. Same lookup
    # pattern as BaseTendencyScaler.get_scaling_values.
    # ------------------------------------------------------------------
    def set_data_indices(self, data_indices) -> None:
        super().set_data_indices(data_indices)
        if data_indices is None:
            return
        if getattr(self, "_precomputed_beta_data", None) is None:
            return
        output_n2i = data_indices.model.output.name_to_index
        n_model_out = len(output_n2i)
        idx_map = torch.zeros(n_model_out, dtype=torch.long)
        resolved = 0
        missing: list[str] = []
        for name, model_idx in output_n2i.items():
            if name in self._zarr_name_to_index:
                idx_map[model_idx] = int(self._zarr_name_to_index[name])
                resolved += 1
            else:
                missing.append(name)
        if missing:
            raise KeyError(
                f"{self._loss_name}: {len(missing)} output variable(s) not found in "
                f"zarr 'variables' attr: {missing[:10]}"
            )
        self._data_idx_for_model_output = idx_map
        LOGGER.info(
            "%s: resolved %d/%d model→raw-zarr index entries for β lookup",
            self._loss_name, resolved, n_model_out,
        )


# Public aliases
SpectralAmplitudeLoss = GraphCastMSHLoss  # legacy name kept for back-compat
MSHLoss = GraphCastMSHLoss
