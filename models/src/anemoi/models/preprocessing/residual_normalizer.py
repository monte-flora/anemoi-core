# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Optional

import numpy as np
import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import BasePreprocessor

LOGGER = logging.getLogger(__name__)

class ResidualNormalizer(BasePreprocessor):
    """Normalizes residuals (tendencies) using provided statistics.

    Δx_norm = (y_true - x_last) / std_tendency
    """

    def __init__(self, data_indices: IndexCollection, statistics_tendencies: dict, min_stdev: float = 1e-8):
        # Passing an empty dict for the config
        super().__init__({}, data_indices, statistics_tendencies)

        self.min_stdev = min_stdev
        name_to_index_training_input = self.data_indices.data.input.name_to_index

        stddev_tendency = statistics_tendencies["stdev"]

        # For the default, we want =1 for the division.
        # Otherwise, use the stored stdevs for the prognostic variables.
        _stdev = np.ones((stddev_tendency.size,), dtype=np.float32)
        for name, i in name_to_index_training_input.items():
            if i in self.data_indices.data.output.full:
                stdev_val = stddev_tendency[i]
                # Apply minimum threshold to prevent division by very small numbers
                if stdev_val < min_stdev:
                    LOGGER.warning(f"ResidualNormalizer: {name} (idx={i}) has very small stdev={stdev_val:.10f}, "
                                   f"clipping to {min_stdev}")
                    stdev_val = min_stdev
                _stdev[i] = stdev_val

        # register as buffers so they move automatically with the model
        self.register_buffer("_std_tendency", torch.from_numpy(_stdev), persistent=True)
        # Swapped out the ..output.full for output.prognostic as we do want 
        # to include eventual diagnostic variables in the residual calculations. 
        self.register_buffer("_prog_idx", self.data_indices.data.output.prognostic, persistent=True)

    # --------------------------------------------------------------
    # Forward / inverse normalization for residuals
    # --------------------------------------------------------------
    def transform(self, x_last: torch.Tensor, y_true: torch.Tensor, in_place: bool = True) -> torch.Tensor:
        """Compute normalized residual Δx_norm = (y_true - x_last) / std_tendency."""
        if not in_place:
            y_true = y_true.clone()

        #x_last, y_true shape = (B, T, cell, n_prognostic)        
        # Compute the residual in physical space and then 
        # divide by the 1-step difference stdev to normalize. 
        Δx = y_true - x_last
        # Since y_true and x_last are only prognostic variables, 
        # we only want to grab the tendencies for the prognostic variables.. 
        Δx.div_(self._std_tendency[self._prog_idx])
        return Δx

    def inverse_transform(self, x_last: torch.Tensor, Δx_norm: torch.Tensor, in_place: bool = True) -> torch.Tensor:
        """Reconstruct next state from normalized residuals."""
        if not in_place:
            Δx_norm = Δx_norm.clone()
        
        # The AI model predicts the normalized residual (Δx_norm)
        # To get back the physical space, multiple by the time-diff stdev
        # the residual in physical space is then added onto the 
        # last timestep. 
        Δx_phys = Δx_norm * self._std_tendency[self._prog_idx]
        return x_last + Δx_phys

    def transform_from_normalized(
        self,
        x_last_norm: torch.Tensor,
        y_true_norm: torch.Tensor,
        norm_mul: torch.Tensor,
        in_place: bool = True,
    ) -> torch.Tensor:
        """Compute normalized residual directly from normalized inputs."""
        if not in_place:
            y_true_norm = y_true_norm.clone()

        Δx_norm = y_true_norm.sub_(x_last_norm)
        Δx_norm.div_(norm_mul[self._prog_idx] * self._std_tendency[self._prog_idx])
        return Δx_norm

    def inverse_transform_to_normalized(
        self,
        x_last_norm: torch.Tensor,
        Δx_norm: torch.Tensor,
        norm_mul: torch.Tensor,
        in_place: bool = True,
    ) -> torch.Tensor:
        """Reconstruct next state in normalized space."""
        if not in_place:
            Δx_norm = Δx_norm.clone()

        Δx_norm.mul_(self._std_tendency[self._prog_idx] * norm_mul[self._prog_idx])
        return x_last_norm + Δx_norm

    def inverse_transform_physical_from_normalized(
        self,
        x_last_norm: torch.Tensor,
        Δx_norm: torch.Tensor,
        norm_mul: torch.Tensor,
        norm_add: torch.Tensor,
        in_place: bool = True,
    ) -> torch.Tensor:
        """Reconstruct next state in physical space from normalized inputs."""
        if not in_place:
            Δx_norm = Δx_norm.clone()

        Δx_phys = Δx_norm.mul(self._std_tendency[self._prog_idx])
        x_last_phys = (x_last_norm - norm_add[self._prog_idx]) / norm_mul[self._prog_idx]
        return x_last_phys + Δx_phys
