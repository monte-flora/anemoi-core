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

# Debug flag for NaN detection in residual normalization
_DEBUG_RESIDUAL_NAN = False  # Set to True to enable verbose NaN checking
_debug_residual_call_count = 0


def _check_tensor_nan_residual(tensor: torch.Tensor, name: str, call_count: int) -> bool:
    """Check tensor for NaN/Inf and log details. Returns True if clean."""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    if has_nan or has_inf:
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        total = tensor.numel()
        # Per-variable stats (last dim is variables)
        if tensor.dim() >= 1:
            nan_per_var = torch.isnan(tensor).sum(dim=tuple(range(tensor.dim() - 1)))
            nan_vars = torch.where(nan_per_var > 0)[0].tolist()[:10]
        else:
            nan_vars = []
        LOGGER.error(
            "RESIDUAL_NORM NaN: call=%d, tensor=%s, NaN=%d/%d (%.1f%%), Inf=%d, shape=%s, vars=%s",
            call_count, name, nan_count, total, 100*nan_count/total, inf_count, list(tensor.shape), nan_vars
        )
        return False

    # Also check for extreme values
    valid = tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)]
    if valid.numel() > 0:
        max_val = torch.abs(valid).max().item()
        if max_val > 1e4:
            extreme_per_var = (torch.abs(tensor) > 1e4).sum(dim=tuple(range(tensor.dim() - 1)))
            extreme_vars = torch.where(extreme_per_var > 0)[0].tolist()[:10]
            LOGGER.warning(
                "RESIDUAL_NORM EXTREME: call=%d, tensor=%s, max=%.2e, shape=%s, vars=%s",
                call_count, name, max_val, list(tensor.shape), extreme_vars
            )
    return True

class ResidualNormalizer(BasePreprocessor):
    """Normalizes residuals (tendencies) using provided statistics.

    Δx_norm = (y_true - x_last) / std_tendency

    Note: All normalization computations are performed in float32 to avoid
    precision issues with bfloat16, then cast back to the original dtype.
    """

    def __init__(self, data_indices: IndexCollection, statistics_tendencies: dict, min_stdev: float = 1e-6):
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
        # Keep statistics in float32 for numerical stability
        self.register_buffer("_std_tendency", torch.from_numpy(_stdev), persistent=True)
        # IMPORTANT: _std_tendency is indexed by INPUT variable positions (built from name_to_index_training_input),
        # so we must use data.input.prognostic (not data.output.prognostic) to access it correctly.
        self.register_buffer("_prog_idx", self.data_indices.data.input.prognostic, persistent=True)

        LOGGER.info("ResidualNormalizer: Statistics buffers stored in float32 for numerical stability")

    # --------------------------------------------------------------
    # Forward / inverse normalization for residuals
    # All computations done in float32 for numerical stability
    # --------------------------------------------------------------
    def transform(self, x_last: torch.Tensor, y_true: torch.Tensor, in_place: bool = True) -> torch.Tensor:
        """Compute normalized residual Δx_norm = (y_true - x_last) / std_tendency.

        Performs computation in float32 for numerical stability.
        """
        original_dtype = y_true.dtype

        # Cast to float32 for numerical stability
        y_true_f32 = y_true.float()
        x_last_f32 = x_last.float()

        # x_last, y_true shape = (B, T, cell, n_prognostic)
        # Compute the residual in physical space and then
        # divide by the 1-step difference stdev to normalize.
        Δx = y_true_f32 - x_last_f32
        # Since y_true and x_last are only prognostic variables,
        # we only want to grab the tendencies for the prognostic variables..
        Δx = Δx / self._std_tendency[self._prog_idx].float()
        return Δx.to(original_dtype)

    def inverse_transform(self, x_last: torch.Tensor, Δx_norm: torch.Tensor, in_place: bool = True) -> torch.Tensor:
        """Reconstruct next state from normalized residuals.

        Performs computation in float32 for numerical stability.
        """
        original_dtype = Δx_norm.dtype

        # Cast to float32 for numerical stability
        Δx_f32 = Δx_norm.float()
        x_last_f32 = x_last.float()

        # The AI model predicts the normalized residual (Δx_norm)
        # To get back the physical space, multiply by the time-diff stdev
        # the residual in physical space is then added onto the
        # last timestep.
        Δx_phys = Δx_f32 * self._std_tendency[self._prog_idx].float()
        result = x_last_f32 + Δx_phys
        return result.to(original_dtype)

    def transform_from_normalized(
        self,
        x_last_norm: torch.Tensor,
        y_true_norm: torch.Tensor,
        norm_mul: torch.Tensor,
        in_place: bool = True,
    ) -> torch.Tensor:
        """Compute normalized residual directly from normalized inputs.

        Performs computation in float32 for numerical stability, then casts
        back to the original dtype.
        """
        # Save original dtype for casting back
        original_dtype = y_true_norm.dtype

        # Always work on a copy when doing dtype conversion
        # Cast to float32 for numerical stability
        y_true_f32 = y_true_norm.float()
        x_last_f32 = x_last_norm.float()

        if _DEBUG_RESIDUAL_NAN:
            global _debug_residual_call_count
            _debug_residual_call_count += 1
            call_count = _debug_residual_call_count

            # Check inputs
            _check_tensor_nan_residual(x_last_f32, "x_last_norm_input", call_count)
            _check_tensor_nan_residual(y_true_f32, "y_true_norm_input", call_count)

            # Check divisor components (compute in float32)
            divisor = norm_mul[self._prog_idx].float() * self._std_tendency[self._prog_idx].float()
            min_div = divisor.min().item()
            max_div = divisor.max().item()
            if min_div < 1e-6:
                small_idx = torch.where(divisor < 1e-6)[0].tolist()[:10]
                LOGGER.warning(
                    "RESIDUAL_NORM SMALL_DIVISOR: call=%d, min=%.2e, max=%.2e, small_vars=%s",
                    call_count, min_div, max_div, small_idx
                )

        # Compute residual in float32
        Δx_norm = y_true_f32 - x_last_f32

        if _DEBUG_RESIDUAL_NAN:
            _check_tensor_nan_residual(Δx_norm, "Δx_after_sub", call_count)

        # Division in float32 (using float32 buffers)
        divisor = norm_mul[self._prog_idx].float() * self._std_tendency[self._prog_idx].float()
        Δx_norm = Δx_norm / divisor

        if _DEBUG_RESIDUAL_NAN:
            _check_tensor_nan_residual(Δx_norm, "Δx_after_div", call_count)

        # Cast back to original dtype
        return Δx_norm.to(original_dtype)

    def inverse_transform_to_normalized(
        self,
        x_last_norm: torch.Tensor,
        Δx_norm: torch.Tensor,
        norm_mul: torch.Tensor,
        in_place: bool = True,
    ) -> torch.Tensor:
        """Reconstruct next state in normalized space.

        Performs computation in float32 for numerical stability, then casts
        back to the original dtype.
        """
        # Save original dtype for casting back
        original_dtype = Δx_norm.dtype

        # Cast to float32 for numerical stability
        Δx_f32 = Δx_norm.float()
        x_last_f32 = x_last_norm.float()

        if _DEBUG_RESIDUAL_NAN:
            global _debug_residual_call_count
            _debug_residual_call_count += 1
            call_count = _debug_residual_call_count
            _check_tensor_nan_residual(x_last_f32, "inv_x_last_norm_input", call_count)
            _check_tensor_nan_residual(Δx_f32, "inv_Δx_norm_input", call_count)

        # Multiplication in float32
        multiplier = self._std_tendency[self._prog_idx].float() * norm_mul[self._prog_idx].float()
        Δx_scaled = Δx_f32 * multiplier

        if _DEBUG_RESIDUAL_NAN:
            _check_tensor_nan_residual(Δx_scaled, "inv_Δx_after_mul", call_count)

        result = x_last_f32 + Δx_scaled

        if _DEBUG_RESIDUAL_NAN:
            _check_tensor_nan_residual(result, "inv_result", call_count)

        # Cast back to original dtype
        return result.to(original_dtype)

    def inverse_transform_physical_from_normalized(
        self,
        x_last_norm: torch.Tensor,
        Δx_norm: torch.Tensor,
        norm_mul: torch.Tensor,
        norm_add: torch.Tensor,
        in_place: bool = True,
    ) -> torch.Tensor:
        """Reconstruct next state in physical space from normalized inputs.

        Performs computation in float32 for numerical stability.
        """
        original_dtype = Δx_norm.dtype

        # Cast to float32 for numerical stability
        Δx_f32 = Δx_norm.float()
        x_last_f32 = x_last_norm.float()
        norm_mul_f32 = norm_mul[self._prog_idx].float()
        norm_add_f32 = norm_add[self._prog_idx].float()

        Δx_phys = Δx_f32 * self._std_tendency[self._prog_idx].float()
        x_last_phys = (x_last_f32 - norm_add_f32) / norm_mul_f32
        result = x_last_phys + Δx_phys
        return result.to(original_dtype)
