# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Low-pass-filtered MSE for regular-grid LAM training (multi-scale loss term).

Motivation (GRAF AI v40b): plain grid-point MSE on 15-minute tendencies is
dominated by convective-scale variance, so the meso-alpha/synoptic component
of the tendency — small in amplitude but decisive for long-rollout stability —
contributes little gradient. ``LowPassMSELoss`` low-passes BOTH prediction and
target with a fixed separable Gaussian on the (H, W) grid and takes the MSE of
the filtered pair, making the large-scale dynamics an explicit training
signal. Combine with the grid-scale loss via ``CombinedLoss``::

    training_loss:
      _target_: anemoi.training.losses.combined.CombinedLoss
      losses:
        - _target_: anemoi.training.losses.graphcast_full.GraphCastFullLoss
          ...
        - _target_: anemoi.training.losses.lowpass.LowPassMSELoss
          x_dim: 375
          y_dim: 375
          cutoff_km: 200.0
          cell_km: 4.29
      loss_weights: [1.0, 1.0]

The filter's half-power wavelength is ``cutoff_km``: a Gaussian with spatial
std sigma attenuates wavelength lambda by exp(-2 pi^2 sigma^2 / lambda^2), so
sigma = cutoff_km * sqrt(ln 2 / 2) / pi. The filter is linear and applied
identically to pred and target, so all BaseLoss scaler machinery (variable
weights, limited-area mask, level scalers) applies unchanged to the filtered
difference.
"""

import logging
import math

import torch
import torch.nn.functional as F

from anemoi.training.losses.base import FunctionalLoss

LOGGER = logging.getLogger(__name__)


class LowPassMSELoss(FunctionalLoss):
    """MSE of Gaussian-low-passed prediction vs low-passed target."""

    name: str = "lowpass_mse"

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        cutoff_km: float = 200.0,
        cell_km: float = 4.29,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(ignore_nans=ignore_nans, **kwargs)
        self.x_dim = int(x_dim)
        self.y_dim = int(y_dim)
        self.cutoff_km = float(cutoff_km)
        # Half-power wavelength -> Gaussian std (in cells).
        sigma_km = self.cutoff_km * math.sqrt(math.log(2.0) / 2.0) / math.pi
        sigma = sigma_km / float(cell_km)
        ksize = 2 * int(math.ceil(3.0 * sigma)) + 1
        ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2.0
        k1d = torch.exp(-(ax**2) / (2.0 * sigma**2))
        k1d = (k1d / k1d.sum()).view(1, 1, 1, ksize)
        self.register_buffer("_k1d", k1d, persistent=False)
        self._pad = ksize // 2
        LOGGER.info(
            "LowPassMSELoss: cutoff=%.0f km (sigma=%.2f cells, kernel=%d), grid=(%d, %d)",
            self.cutoff_km, sigma, ksize, self.y_dim, self.x_dim,
        )

    def _lowpass(self, x: torch.Tensor) -> torch.Tensor:
        """Separable Gaussian low-pass over the grid dim of (bs, ens, grid, vars)."""
        bs, ens, grid, nv = x.shape
        if grid != self.y_dim * self.x_dim:
            msg = f"LowPassMSELoss: grid={grid} != y_dim*x_dim={self.y_dim * self.x_dim}"
            raise ValueError(msg)
        k = self._k1d.to(dtype=x.dtype)
        g = x.permute(0, 1, 3, 2).reshape(bs * ens * nv, 1, self.y_dim, self.x_dim)
        g = F.pad(g, (self._pad, self._pad, 0, 0), mode="reflect")
        g = F.conv2d(g, k)
        g = F.pad(g, (0, 0, self._pad, self._pad), mode="reflect")
        g = F.conv2d(g, k.transpose(-1, -2))
        return g.reshape(bs, ens, nv, grid).permute(0, 1, 3, 2)

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Squared difference of the low-passed fields (bs, ensemble, lat*lon, n_outputs)."""
        return torch.square(self._lowpass(pred) - self._lowpass(target))
