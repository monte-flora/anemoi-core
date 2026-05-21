# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import warnings
from abc import abstractmethod

import numpy as np
import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses.scalers.base_scaler import BaseScaler
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


class BaseTendencyScaler(BaseScaler):
    """Configurable method to scale prognostic variables based on data statistics and statistics_tendencies."""

    scale_dims: TensorDim = TensorDim.VARIABLE

    def __init__(
        self,
        data_indices: IndexCollection,
        statistics: dict,
        statistics_tendencies: dict,
        norm: str | None = None,
        **kwargs,
    ) -> None:
        """Initialise variable level scaler.

        Parameters
        ----------
        data_indices : IndexCollection
            Collection of data indices.
        statistics : dict
            Data statistics dictionary
        statistics_tendencies : dict
            Data statistics dictionary for tendencies
        norm : str, optional
            Type of normalization to apply. Options are None, unit-sum, unit-mean and l1.
        """
        super().__init__(norm=norm)
        del kwargs
        self.data_indices = data_indices
        self.statistics = statistics
        self.statistics_tendencies = statistics_tendencies

        if not self.statistics_tendencies:
            warnings.warn("Dataset has no tendency statistics! Are you sure you want to use a tendency scaler?")

    @abstractmethod
    def get_level_scaling(self, variable_level: int) -> float: ...

    def get_scaling_values(self, **_kwargs) -> torch.Tensor:
        variable_level_scaling = torch.ones((len(self.data_indices.data.output.full),), dtype=torch.float32)

        for key, idx in self.data_indices.model.output.name_to_index.items():
            if idx in self.data_indices.model.output.prognostic and self.data_indices.data.output.name_to_index.get(
                key,
            ):
                prog_idx = self.data_indices.data.output.name_to_index[key]
                variable_stdev = self.statistics["stdev"][prog_idx] if self.statistics_tendencies else 1
                variable_tendency_stdev = (
                    self.statistics_tendencies["stdev"][prog_idx] if self.statistics_tendencies else 1
                )
                scaling = self.get_level_scaling(variable_stdev, variable_tendency_stdev)
                variable_level_scaling[idx] *= scaling

        return variable_level_scaling


class NoTendencyScaler(BaseTendencyScaler):
    """No scaling by tendency statistics."""

    def get_level_scaling(self, variable_stdev: float, variable_tendency_stdev: float) -> float:
        del variable_stdev, variable_tendency_stdev
        return 1.0


class StdevTendencyScaler(BaseTendencyScaler):
    """Scale loses by standard deviation of tendency statistics."""

    def get_level_scaling(self, variable_stdev: float, variable_tendency_stdev: float) -> float:
        return variable_stdev / variable_tendency_stdev


class VarTendencyScaler(BaseTendencyScaler):
    """Scale loses by variance of tendency statistics."""

    def get_level_scaling(self, variable_stdev: float, variable_tendency_stdev: float) -> float:
        return variable_stdev**2 / variable_tendency_stdev**2


class _LatentTendencyMixin:
    """Shared init logic for latent-space tendency scalers.

    Loads ``statistics_tendencies_<freqstr>_latent_stdev`` directly from a
    zarr and substitutes it for the physical ``statistics_tendencies['stdev']``
    that ``create_scalers`` auto-injects. Subclasses pair this with a
    ``get_level_scaling`` (squared for MSE-family losses, linear for
    MAE/CRPS-family losses).
    """

    def __init__(
        self,
        data_indices: IndexCollection,
        statistics: dict,
        statistics_tendencies: dict,
        norm: str | None = None,
        *,
        latent_stats_path: str,
        latent_stats_key: str | None = None,
        **kwargs,
    ) -> None:
        import zarr

        z = zarr.open(latent_stats_path, mode="r")
        freqstr = z.attrs.get("frequency")
        if not freqstr:
            error = (
                f"{self.__class__.__name__}: zarr at {latent_stats_path!r} has no "
                "'frequency' attribute; cannot resolve latent stats key."
            )
            raise RuntimeError(error)
        key = latent_stats_key or f"statistics_tendencies_{freqstr}_latent_stdev"
        if key not in z:
            error = (
                f"{self.__class__.__name__}: {key!r} missing from "
                f"{latent_stats_path!r}. Run grafai/datasets/"
                "compute_latent_tendency_stats.py first."
            )
            raise RuntimeError(error)
        latent_stdev = np.asarray(z[key][:])
        LOGGER.info(
            "%s: loaded latent tendency stdev %s from %s (range [%.3e, %.3e])",
            self.__class__.__name__, key, latent_stats_path,
            float(latent_stdev.min()), float(latent_stdev.max()),
        )

        tendencies_latent = dict(statistics_tendencies or {})
        tendencies_latent["stdev"] = latent_stdev

        super().__init__(
            data_indices=data_indices,
            statistics=statistics,
            statistics_tendencies=tendencies_latent,
            norm=norm,
            **kwargs,
        )


class LatentVarTendencyScaler(_LatentTendencyMixin, BaseTendencyScaler):
    """Variance-of-tendency scaler in LATENT space.

    Pairs with **MSE-style** losses (squared error). Multiplies each
    per-variable loss contribution by ``(σ_var / σ_lat_tend)²`` so a
    mean-std-space residual lands on O(1) magnitude per channel.

    Use this when the underlying loss is squared (MSE, GaussianNLL).

    For MAE/CRPS-style absolute-error losses, use
    :class:`LatentStdevTendencyScaler` instead — the linear σ pairing.

    Parameters
    ----------
    latent_stats_path : str
        Path to the training zarr containing
        ``statistics_tendencies_<freqstr>_latent_stdev`` (computed by
        ``grafai/datasets/compute_latent_tendency_stats.py``).
    latent_stats_key : str, optional
        Override for the array key.
    """

    def get_level_scaling(self, variable_stdev: float, variable_tendency_stdev: float) -> float:
        return variable_stdev**2 / variable_tendency_stdev**2


class LatentStdevTendencyScaler(_LatentTendencyMixin, BaseTendencyScaler):
    """Stdev-of-tendency scaler in LATENT space.

    Pairs with **MAE / CRPS / absolute-error** losses. Multiplies each
    per-variable loss contribution by ``(σ_var / σ_lat_tend)¹`` so a
    mean-std-space residual lands on O(1) magnitude per channel.

    Use this for the v30 latent predictive task (CRPS in latent) and any
    other L1-family losses computed against mean-std residuals.

    For MSE-style squared-error losses, use
    :class:`LatentVarTendencyScaler` instead.

    Parameters as for :class:`LatentVarTendencyScaler`.
    """

    def get_level_scaling(self, variable_stdev: float, variable_tendency_stdev: float) -> float:
        return variable_stdev / variable_tendency_stdev
