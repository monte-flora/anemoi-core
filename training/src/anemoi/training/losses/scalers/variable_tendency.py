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

        # Defensive instrumentation: forcing-like variables (cos_julian_day,
        # land_sea_mask, ...) have σ_tend = 0 because they're time-invariant
        # or rigorously periodic. If one slips into the prognostic index set
        # via a misconfigured `forcing:` list, the σ_phys/σ_tend or σ_phys²/σ_tend²
        # scaler factor becomes inf and silently corrupts training. Collect
        # offenders and emit one summary line at the end.
        zero_tend_offenders: list[str] = []
        extreme_offenders: list[tuple[str, float]] = []

        for key, idx in self.data_indices.model.output.name_to_index.items():
            if idx in self.data_indices.model.output.prognostic and self.data_indices.data.output.name_to_index.get(
                key,
            ):
                prog_idx = self.data_indices.data.output.name_to_index[key]
                variable_stdev = self.statistics["stdev"][prog_idx] if self.statistics_tendencies else 1
                variable_tendency_stdev = (
                    self.statistics_tendencies["stdev"][prog_idx] if self.statistics_tendencies else 1
                )

                # Zero σ_tend → forcing slipped into prognostic.
                if float(variable_tendency_stdev) <= 0.0:
                    zero_tend_offenders.append(key)
                    # Skip the divide-by-zero; leave scaler = 1 for this channel.
                    continue

                scaling = self.get_level_scaling(variable_stdev, variable_tendency_stdev)

                # Extreme scaler value — usually means σ_tend is much smaller than
                # σ_phys for reasons other than ordinary climatology (e.g. clipped
                # stats, near-constant variable). Flag for the user to verify.
                if not np.isfinite(scaling) or float(scaling) > 1e4 or float(scaling) < 1e-4:
                    extreme_offenders.append((key, float(scaling)))

                variable_level_scaling[idx] *= scaling

        if zero_tend_offenders:
            warnings.warn(
                f"{self.__class__.__name__}: {len(zero_tend_offenders)} prognostic "
                f"variable(s) have σ_tend = 0 — strongly suggests a forcing-like "
                f"variable is misclassified as prognostic. Setting scaler = 1 for "
                f"these channels (otherwise the scaler would be inf). Offenders: "
                f"{zero_tend_offenders}. Add them to your `forcing:` list in "
                f"data/vars.",
                stacklevel=2,
            )
        if extreme_offenders:
            offenders_str = ", ".join(f"{n}={v:.3e}" for n, v in extreme_offenders[:10])
            LOGGER.warning(
                "%s: %d prognostic variable(s) have extreme scaler values (|v|>1e4 or <1e-4). "
                "Verify their σ_tend / σ_phys ratios are not artifacts. First %d: %s",
                self.__class__.__name__, len(extreme_offenders),
                min(10, len(extreme_offenders)), offenders_str,
            )

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
        latent_stdev_raw = np.asarray(z[key][:])

        # IMPORTANT: latent_stdev_raw is indexed by the RAW zarr's variable
        # ordering (138 channels). But BaseTendencyScaler indexes the
        # tendency-stdev array with `prog_idx = data.output.name_to_index[name]`
        # — i.e. the POST-DROP output space (e.g. 99 channels). If we just
        # pass latent_stdev_raw, channel index misalignment makes some
        # prognostic vars pick up a FORCING's σ_lat_tend (which is 0 for
        # time-invariant forcings) → division by zero → inf scaler → inf
        # loss. (Bug found 2026-05-22 from v30b smoke; see
        # [[feedback-smoke-correctness-gap]].)
        #
        # Re-index by NAME. `data.output.name_to_index[name]` returns
        # positions in the POST-DROP INPUT space (0..N_input_after_drop-1),
        # NOT positions within output.full (which has fewer entries). So
        # we size the re-indexed array by data.input.full (or equivalent
        # input-space size). Indexing matches what statistics["stdev"] uses.
        zarr_names = list(z.attrs.get("variables", []))
        if not zarr_names or len(zarr_names) != len(latent_stdev_raw):
            error = (
                f"{self.__class__.__name__}: zarr's `variables` attr "
                f"({len(zarr_names)}) does not match the latent-stdev "
                f"array length ({len(latent_stdev_raw)}). Cannot re-index."
            )
            raise RuntimeError(error)
        zarr_name_to_idx = {n: i for i, n in enumerate(zarr_names)}
        out_n2i = data_indices.data.output.name_to_index
        # Size by the max index value (+1) so writes never go out-of-bounds,
        # matching whatever index space `out_n2i` lives in.
        n_array = max(out_n2i.values()) + 1
        latent_stdev = np.ones((n_array,), dtype=np.float32)
        n_filled = 0
        # Defensive instrumentation — track which prognostic names came back
        # with σ_lat_tend = 0 (indicates a forcing-like variable slipped in,
        # which would otherwise produce inf scaler at the next stage).
        zero_tend_prognostic: list[str] = []
        # The set of prognostic output indices, for the σ=0 check below.
        prog_idx_set = set(int(i) for i in data_indices.model.output.prognostic.tolist())
        # Reverse map from output index → model output name (for the warning).
        model_idx_to_name = {idx: name for name, idx in data_indices.model.output.name_to_index.items()}
        for name, out_idx in out_n2i.items():
            if name not in zarr_name_to_idx:
                LOGGER.warning(
                    "%s: output variable %r not in latent stats array; "
                    "leaving scaler = 1 for this channel.",
                    self.__class__.__name__, name,
                )
                continue
            val = float(latent_stdev_raw[zarr_name_to_idx[name]])
            latent_stdev[out_idx] = val
            n_filled += 1
            # If this output index is prognostic AND σ_lat_tend = 0, flag it.
            model_idx = data_indices.model.output.name_to_index.get(name)
            if model_idx is not None and int(model_idx) in prog_idx_set and val <= 0.0:
                zero_tend_prognostic.append(name)

        LOGGER.info(
            "%s: re-indexed latent tendency stdev for %d/%d output names "
            "into a %d-d array (raw zarr had %d). Filled range [%.3e, %.3e].",
            self.__class__.__name__, n_filled, len(out_n2i),
            n_array, len(latent_stdev_raw),
            float(latent_stdev.min()), float(latent_stdev.max()),
        )
        if zero_tend_prognostic:
            warnings.warn(
                f"{self.__class__.__name__}: {len(zero_tend_prognostic)} prognostic "
                f"variable(s) have σ_lat_tend = 0 in the latent stats array — "
                f"this WILL produce inf scaler values and inf training loss. "
                f"Likely cause: a forcing-like variable (cos_julian_day, "
                f"land_sea_mask, ...) is misclassified as prognostic. Add it to "
                f"your `forcing:` list in data/vars. Offenders: "
                f"{zero_tend_prognostic}.",
                stacklevel=2,
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
