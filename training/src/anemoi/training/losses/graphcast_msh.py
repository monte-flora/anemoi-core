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
import torch
import torch.fft

from anemoi.training.losses.base import GraphCastBaseLoss
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================
def _build_radial_bins(
    x_dim: int,
    y_dim: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Precompute the (flat-pixel → k-bin) index and k-centre tensors.

    Returns
    -------
    flat_bin_idx : LongTensor, shape (x_dim * y_dim,)
        For each FFT bin, the integer radial wavenumber it belongs to
        (k = round(sqrt(kx² + ky²))).
    k_centres : FloatTensor, shape (n_bins,)
        The integer k value at each bin centre, in same dtype as dtype.
    n_bins : int
        Total number of radial bins (= 1 + floor(sqrt((x_dim/2)² + (y_dim/2)²))).
    """
    kx = torch.fft.fftfreq(x_dim, device=device, dtype=dtype) * x_dim
    ky = torch.fft.fftfreq(y_dim, device=device, dtype=dtype) * y_dim
    kxv, kyv = torch.meshgrid(kx, ky, indexing="ij")
    k_mag = torch.sqrt(kxv**2 + kyv**2)  # (x_dim, y_dim)
    bin_idx = torch.round(k_mag).to(torch.long)  # integer radial wavenumber per bin
    n_bins = int(bin_idx.max().item()) + 1
    k_centres = torch.arange(n_bins, device=device, dtype=dtype)
    return bin_idx.flatten(), k_centres, n_bins


def _gamma_k_weights(
    k_centres: torch.Tensor,
    *,
    normalise: bool = True,
    min_weight: float = 1.0,
) -> torch.Tensor:
    """FastNet Eq. 9: γ_k = max(N_k · k/√3, 1.0)."""
    raw = k_centres / (3**0.5)
    if normalise:
        # N_k preserves the magnitude of the AMSE calculation — pick a scale
        # that keeps mean(raw) == mean(γ) when γ doesn't hit the floor.
        nonzero = raw[raw > 0]
        n_k = 1.0 if nonzero.numel() == 0 else float(nonzero.mean().item()) ** 0
        # use multiplicative N_k so that γ_k averaged over the spectrum is ~1.
        # N_k = 1 / mean(raw) ensures mean(N_k · raw) = 1, then the floor at
        # min_weight prevents the low-k bins from collapsing the signal.
        raw = raw / max(float(nonzero.mean().item()), 1e-12) if nonzero.numel() else raw
    return torch.clamp(raw, min=min_weight)


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
        **kwargs,
    ) -> None:
        super().__init__(ignore_nans=ignore_nans, **kwargs)
        self.x_dim = int(x_dim)
        self.y_dim = int(y_dim)
        self.coherence_weight = float(coherence_weight)
        self.use_gamma_k = bool(use_gamma_k)
        self.gamma_k_min = float(gamma_k_min)

        # Precompute bin lookup + γ_k as buffers so they move with .to(device).
        flat_bin_idx, k_centres, n_bins = _build_radial_bins(self.x_dim, self.y_dim)
        gamma_k = _gamma_k_weights(k_centres, min_weight=self.gamma_k_min) if self.use_gamma_k else torch.ones_like(k_centres)
        self.register_buffer("_flat_bin_idx", flat_bin_idx, persistent=False)
        self.register_buffer("_k_centres", k_centres, persistent=False)
        self.register_buffer("_gamma_k", gamma_k, persistent=False)
        self._n_bins = int(n_bins)

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

        # FFT in float32 for numerical stability under bf16-mixed precision
        orig_dtype = pred.dtype
        alpha_pred = torch.fft.fft2(pred_2d.float(), dim=(-3, -2))
        alpha_true = torch.fft.fft2(target_2d.float(), dim=(-3, -2))

        # |α|² per (H, W) bin and Re[α_pred · α_true*]
        power_pred = alpha_pred.real**2 + alpha_pred.imag**2
        power_true = alpha_true.real**2 + alpha_true.imag**2
        cross = alpha_pred.real * alpha_true.real + alpha_pred.imag * alpha_true.imag

        # Flatten pixels, scatter-add into radial bins
        shape_flat = power_pred.shape[:-3] + (self.x_dim * self.y_dim, power_pred.shape[-1])
        power_pred = power_pred.reshape(shape_flat)
        power_true = power_true.reshape(shape_flat)
        cross = cross.reshape(shape_flat)

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

        # Sum over k bins -> (B, E, V); add sentinel grid dim -> (B, E, 1, V)
        per_var = amse_k.sum(dim=-2)
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


# Public aliases
SpectralAmplitudeLoss = GraphCastMSHLoss  # legacy name kept for back-compat
MSHLoss = GraphCastMSHLoss
