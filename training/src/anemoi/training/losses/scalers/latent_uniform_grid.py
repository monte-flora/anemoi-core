# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
"""Uniform grid-dim scaler at a configurable latent grid size.

Most loss functions require AT LEAST one scaler covering the GRID dim
(see ``losses/base.py:scale``). For v30 latent-space CRPS training, the
loss operates at the latent grid (e.g., 63×63 = 3969 cells), NOT the
full-res 250×250 = 62500. The standard ``GraphNodeAttributeScaler`` /
``Boolean1DMask``-based scalers are sized for the full-res cell grid
(from the dataloader's graph), so they can't be used directly.

This scaler emits a uniform tensor of shape ``(h_lat * w_lat,)``. Default
``norm="unit-sum"`` — the BaseScaler ``normalise`` path then rescales
the ones array so they sum to 1, which is what
``GraphCastBaseLoss._reduce_per_variable`` needs (it does SUM over the
grid dim with the implicit contract that grid weights sum to 1).
Without unit-sum normalization, the per-cell loss is multiplied by
``n_cells`` (e.g. ~3969× for a 63×63 latent grid), which corrupts loss
magnitudes — see ``[[feedback-loss-tendency-pairing]]`` and the
2026-05-22 v30b smoke for the failure pattern.
"""
from __future__ import annotations

import logging

import torch

from anemoi.training.losses.scalers.base_scaler import BaseScaler
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


class LatentUniformGridScaler(BaseScaler):
    """Uniform ones at the latent grid size; satisfies the loss's grid-dim contract.

    Parameters
    ----------
    latent_shape : list[int] | tuple[int, int]
        ``(h_lat, w_lat)`` — the latent grid the loss operates on. The
        emitted scaler has shape ``(h_lat * w_lat,)`` of ones.
    """

    scale_dims: TensorDim = TensorDim.GRID

    def __init__(
        self,
        latent_shape: list[int] | tuple[int, int],
        norm: str | None = "unit-sum",
        **kwargs,
    ) -> None:
        super().__init__(norm=norm)
        del kwargs  # accept and ignore auto-injected create_scalers kwargs
        self.latent_shape = tuple(int(v) for v in latent_shape)
        if len(self.latent_shape) != 2:
            error_msg = f"latent_shape must be a 2-tuple, got {self.latent_shape}"
            raise ValueError(error_msg)
        self._n_cells = self.latent_shape[0] * self.latent_shape[1]
        LOGGER.info(
            "LatentUniformGridScaler: emitting ones at (%d,) for latent grid %s "
            "with norm=%r (≈1/n_cells per cell after unit-sum normalization).",
            self._n_cells, self.latent_shape, norm,
        )

    def get_scaling_values(self, **_kwargs) -> torch.Tensor:
        return torch.ones((self._n_cells,), dtype=torch.float32)
