# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""LevelAverageScaler — GraphCast-style per-variable level averaging.

Default Anemoi ``WeightedMSELoss`` treats every ``(variable, level)`` slot
as an independent field in the loss. For storm-scale configs with ~17
pressure levels each for theta/qv/u/v/w/pressure (102 fields) and a few
surface variables (e.g. t2m, comp_refl, apcp), the 3-D variables would
dominate the loss by ~34× without compensation.

``LevelAverageScaler`` rescales each ``(variable, level)`` weight to
``1.0 / N_levels(variable)``, so every 3-D variable contributes **one
aggregate unit** of loss across all its levels — matching what
``GraphCastFullLoss`` did implicitly via its reduction pipeline. Combine
with ``GeneralVariableLossScaler`` for explicit surface-variable
re-weighting (e.g. ``t2m: 0.1``).

Variables outside the configured scaling group, or any variable for
which the metadata extractor reports ``level is None`` (e.g. surface
variables that nonetheless got matched), keep weight 1.0.
"""

import logging
from collections import defaultdict

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses.scalers.variable_level import BaseVariableLevelScaler
from anemoi.training.utils.variables_metadata import ExtractVariableGroupAndLevel

LOGGER = logging.getLogger(__name__)


class LevelAverageScaler(BaseVariableLevelScaler):
    """Weight every level of a multi-level variable by ``1 / N_levels``.

    Parameters
    ----------
    data_indices : IndexCollection
        Standard anemoi data indices.
    group : str
        Variable group to apply averaging to (e.g. ``"pl"``).
    metadata_extractor : ExtractVariableGroupAndLevel
        Variable-metadata helper.
    norm : str | None
        Optional normalisation applied after weighting. Inherited from base.
    """

    def __init__(
        self,
        data_indices: IndexCollection,
        group: str,
        metadata_extractor: ExtractVariableGroupAndLevel,
        norm: str | None = None,
        **kwargs,
    ) -> None:
        del kwargs
        # slope / y_intercept are required by the base class but unused here;
        # the override below replaces the standard get_scaling_values path.
        super().__init__(
            data_indices,
            group,
            y_intercept=1.0,
            slope=0.0,
            metadata_extractor=metadata_extractor,
            norm=norm,
        )

    @staticmethod
    def get_level_scaling(variable_level: float) -> torch.Tensor:
        """Unused — the standard per-level dispatch can't know the parent
        variable's level count. Kept to satisfy the abstract base class.
        """
        del variable_level
        return 1.0

    def get_scaling_values(self, **_kwargs) -> torch.Tensor:
        n_out = len(self.data_indices.data.output.full)
        scaling = torch.ones((n_out,), dtype=torch.float32)

        # First pass: count levels per variable stem within the group.
        levels_per_param: dict[str, int] = defaultdict(int)
        for variable_name in self.data_indices.model.output.name_to_index:
            grp, param, lvl = self.variable_metadata_extractor.get_group_and_level(variable_name)
            if grp == self.scaling_group and lvl is not None:
                levels_per_param[param] += 1

        if not levels_per_param:
            LOGGER.warning(
                "LevelAverageScaler: no multi-level variables found in group %r; "
                "scaling is a no-op.",
                self.scaling_group,
            )
            return scaling

        LOGGER.info(
            "LevelAverageScaler: group=%s  per-param level counts: %s",
            self.scaling_group, dict(levels_per_param),
        )

        # Second pass: assign weight = 1 / N_levels(param).
        for variable_name, idx in self.data_indices.model.output.name_to_index.items():
            grp, param, lvl = self.variable_metadata_extractor.get_group_and_level(variable_name)
            if grp != self.scaling_group or lvl is None:
                continue
            n = levels_per_param.get(param, 1)
            scaling[idx] = 1.0 / float(n)

        return scaling
