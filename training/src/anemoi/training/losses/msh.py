# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Modified Spherical Harmonic (MSH) spectral amplitude loss.

On 2D Cartesian LAM grids this reduces to an amplitude-only 2D FFT loss:

    MSH = mean_k [ (|S_pred(k)| - |S_truth(k)|)^2 * w(k) ]

Reference: Subich et al. 2025 (arxiv.org/abs/2502.14506). Separating the
amplitude penalty from the phase penalty eliminates the scale-dependent
double-penalty suppression that afflicts pure MSE at high wavenumbers.
"""

import logging

import torch
import torch.fft

from anemoi.training.losses.base import FunctionalLoss
from anemoi.training.losses.spatial import amplitude
from anemoi.training.losses.spatial import get_spectra

LOGGER = logging.getLogger(__name__)


def spectral_amplitude_squared_error(
    predicted_output: torch.Tensor,
    real_output: torch.Tensor,
    dims: tuple[int, int],
    high_k_weight_exponent: float = 0.0,
) -> torch.Tensor:
    r"""Per-wavenumber squared error between FFT amplitude spectra.

    Returns a tensor of shape (..., x_dim, y_dim, variable) of
    ``(|S_pred| - |S_truth|)^2 * w(k)`` where w(k) = (k / k_max) ** exponent.
    """
    power_spectra_real, power_spectra_pred = get_spectra(predicted_output, real_output, dims)
    amp_real = amplitude(power_spectra_real)
    amp_pred = amplitude(power_spectra_pred)
    sq_err = (amp_pred - amp_real) ** 2

    if high_k_weight_exponent != 0.0:
        x_dim, y_dim = dims
        # Build a 2D wavenumber-magnitude grid matching torch.fft.fft2 layout.
        kx = torch.fft.fftfreq(x_dim, device=sq_err.device, dtype=sq_err.dtype) * x_dim
        ky = torch.fft.fftfreq(y_dim, device=sq_err.device, dtype=sq_err.dtype) * y_dim
        kxv, kyv = torch.meshgrid(kx, ky, indexing="ij")
        k_mag = torch.sqrt(kxv**2 + kyv**2)
        k_max = k_mag.max().clamp(min=1.0)
        weight = (k_mag / k_max) ** high_k_weight_exponent
        # Broadcast across leading dims and the variable dim.
        while weight.dim() < sq_err.dim():
            weight = weight.unsqueeze(0)
        weight = weight.unsqueeze(-1)
        sq_err = sq_err * weight

    return sq_err


class SpectralAmplitudeLoss(FunctionalLoss):
    r"""Spectral amplitude loss — the 2D-Cartesian reduction of MSH.

    Computes the mean squared error between the FFT amplitude spectra of the
    prediction and the target. Pairs well with MSE inside a CombinedLoss:
    MSE supplies the phase/pointwise signal, MSH supplies the amplitude
    signal that MSE's double-penalty tends to suppress.

    Parameters
    ----------
    x_dim : int
        X dimension of the 2D grid (must satisfy x_dim * y_dim == grid size).
    y_dim : int
        Y dimension of the 2D grid.
    ignore_nans : bool, optional
        Use nan-safe reductions in the parent class, by default False.
    high_k_weight_exponent : float, optional
        If non-zero, multiply per-bin squared errors by ``(k / k_max) ** exponent``
        to emphasize (exponent > 0) or de-emphasize (< 0) small-scale errors.
        Defaults to 0.0 (uniform weighting), matching Subich et al.'s first-pass
        formulation; all variable/channel weighting is delegated to the scaler
        infrastructure.
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        ignore_nans: bool = False,
        high_k_weight_exponent: float = 0.0,
    ) -> None:
        super().__init__(ignore_nans)
        LOGGER.warning(
            "SpectralAmplitudeLoss (MSH) can only be used with data on 2D grids.",
        )
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.high_k_weight_exponent = high_k_weight_exponent

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sq_err = spectral_amplitude_squared_error(
            pred,
            target,
            dims=(self.x_dim, self.y_dim),
            high_k_weight_exponent=self.high_k_weight_exponent,
        )
        return sq_err.reshape(pred.shape)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        **kwargs,  # noqa: ARG002
    ) -> torch.Tensor:
        result = super().forward(pred, target, squash, scaler_indices=scaler_indices, without_scalers=without_scalers)
        # Plain mean so the loss scales linearly with amplitude error^2,
        # matching Subich et al.'s formulation (no outer sqrt).
        return torch.mean(result)


# Public alias — MSH is the name used in the literature; SpectralAmplitudeLoss
# describes what the computation actually is on a Cartesian grid.
MSHLoss = SpectralAmplitudeLoss
