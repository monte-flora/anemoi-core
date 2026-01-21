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
from typing import Optional

import numpy as np
import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import BasePreprocessor

LOGGER = logging.getLogger(__name__)


class InputNormalizer(BasePreprocessor):
    """Normalizes input data with a configurable method."""

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        """Initialize the normalizer.

        Parameters
        ----------
        config : DotDict
            configuration object of the processor
        data_indices : IndexCollection
            Data indices for input and output variables
        statistics : dict
            Data statistics dictionary
        """
        super().__init__(config, data_indices, statistics)

        name_to_index_training_input = self.data_indices.data.input.name_to_index

        # Store for diagnostic access
        self._name_to_index = name_to_index_training_input
        self._index_to_name = {v: k for k, v in name_to_index_training_input.items()}
        self._statistics = statistics

        minimum = statistics["minimum"]
        maximum = statistics["maximum"]
        mean = statistics["mean"]
        stdev = statistics["stdev"]

        # Validate index alignment during initialization
        LOGGER.info("=" * 80)
        LOGGER.info("NORMALIZER INIT: Index Alignment Audit")
        LOGGER.info("=" * 80)
        LOGGER.info("Number of variables in name_to_index: %d", len(name_to_index_training_input))
        LOGGER.info("Statistics array sizes: mean=%d, stdev=%d, min=%d, max=%d",
                   mean.size, stdev.size, minimum.size, maximum.size)

        # Check for size mismatch
        if len(name_to_index_training_input) != mean.size:
            LOGGER.error(
                "INDEX ALIGNMENT ERROR: name_to_index has %d entries but statistics has %d entries!",
                len(name_to_index_training_input), mean.size
            )

        # Log first 20 variable mappings for verification
        sorted_vars = sorted(name_to_index_training_input.items(), key=lambda x: x[1])
        LOGGER.info("First 20 variable index mappings:")
        for name, idx in sorted_vars[:20]:
            if idx < mean.size:
                LOGGER.info("  %3d: %-30s mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
                           idx, name, mean[idx], stdev[idx], minimum[idx], maximum[idx])
            else:
                LOGGER.error("  %3d: %-30s INDEX OUT OF BOUNDS!", idx, name)

        # Log any variables with suspicious statistics (very small std or extreme values)
        suspicious_vars = []
        for name, idx in name_to_index_training_input.items():
            if idx >= mean.size:
                suspicious_vars.append((name, idx, "INDEX_OUT_OF_BOUNDS", None, None))
            elif stdev[idx] < 1e-8:
                suspicious_vars.append((name, idx, "ZERO_STD", stdev[idx], mean[idx]))
            elif abs(mean[idx]) > 1e10 or abs(stdev[idx]) > 1e10:
                suspicious_vars.append((name, idx, "EXTREME_STATS", stdev[idx], mean[idx]))

        if suspicious_vars:
            LOGGER.warning("Found %d variables with suspicious statistics:", len(suspicious_vars))
            for name, idx, reason, std_val, mean_val in suspicious_vars[:20]:
                LOGGER.warning("  %s (idx=%d): %s, std=%.4e, mean=%.4e",
                              name, idx, reason, std_val or 0, mean_val or 0)
        LOGGER.info("=" * 80)

        # Optionally reuse statistic of one variable for another variable
        statistics_remap = {}
        for remap, source in self.remap.items():
            idx_src, idx_remap = name_to_index_training_input[source], name_to_index_training_input[remap]
            statistics_remap[idx_remap] = (minimum[idx_src], maximum[idx_src], mean[idx_src], stdev[idx_src])

        # Two-step to avoid overwriting the original statistics in the loop (this reduces dependence on order)
        for idx, new_stats in statistics_remap.items():
            LOGGER.info("Statistics remapping happened!")
            minimum[idx], maximum[idx], mean[idx], stdev[idx] = new_stats

        self._validate_normalization_inputs(name_to_index_training_input, minimum, maximum, mean, stdev)

        _norm_add = np.zeros((minimum.size,), dtype=np.float32)
        _norm_mul = np.ones((minimum.size,), dtype=np.float32)

        for name, i in name_to_index_training_input.items():
            method = self.methods.get(name, self.default)

            if method == "mean-std":
                LOGGER.info(f"Normalizing: {name} is mean-std-normalised. {stdev[i]=:.5f} {mean[i]=:.5f}")
                if stdev[i] < (mean[i] * 1e-6):
                    warnings.warn(f"Normalizing: the field seems to have only one value {mean[i]}")
                    
                if stdev[i] < 0.00000001:
                    LOGGER.info(f"CAUTION CAUTION {name} {stdev[i]:.6f} has stdev of 0!!")
                    
                _norm_mul[i] = 1 / stdev[i]
                _norm_add[i] = -mean[i] / stdev[i]

            elif method == "std":
                LOGGER.info(f"Normalizing: {name} is std-normalised.")
                if stdev[i] < (mean[i] * 1e-6):
                    warnings.warn(f"Normalizing: the field seems to have only one value {mean[i]}")
                _norm_mul[i] = 1 / stdev[i]
                _norm_add[i] = 0

            elif method == "min-max":
                LOGGER.info(f"Normalizing: {name} is min-max-normalised to [0, 1].")
                x = maximum[i] - minimum[i]
                if x < 1e-9:
                    warnings.warn(f"Normalizing: the field {name} seems to have only one value {maximum[i]}.")
                _norm_mul[i] = 1 / x
                _norm_add[i] = -minimum[i] / x

            elif method == "max":
                LOGGER.info(f"Normalizing: {name} is max-normalised to [0, 1]. max={maximum[i]}")
                _norm_mul[i] = 1 / maximum[i]

            elif method == "none":
                LOGGER.info(f"Normalizing: {name} is not normalized.")

            else:
                raise ValueError[f"Unknown normalisation method for {name}: {method}"]

        # register buffer - this will ensure they get copied to the correct device(s)
        self.register_buffer("_norm_mul", torch.from_numpy(_norm_mul), persistent=True)
        self.register_buffer("_norm_add", torch.from_numpy(_norm_add), persistent=True)
        self.register_buffer("_input_idx", data_indices.data.input.full, persistent=True)
        self.register_buffer("_output_idx", self.data_indices.data.output.full, persistent=True)

        # Log any problematic normalization coefficients
        inf_mask = np.isinf(_norm_mul)
        large_mask = np.abs(_norm_mul) > 1e6
        if inf_mask.any():
            inf_indices = np.where(inf_mask)[0].tolist()
            inf_names = [name for name, idx in name_to_index_training_input.items() if idx in inf_indices]
            LOGGER.error(
                "NORMALIZER INIT: Found %d variables with inf _norm_mul (1/stdev): indices=%s, names=%s",
                inf_mask.sum(), inf_indices[:10], inf_names[:10]
            )
        if large_mask.any():
            large_indices = np.where(large_mask)[0].tolist()
            large_names = [name for name, idx in name_to_index_training_input.items() if idx in large_indices]
            large_vals = _norm_mul[large_mask][:10].tolist()
            LOGGER.warning(
                "NORMALIZER INIT: Found %d variables with large _norm_mul (>1e6): indices=%s, names=%s, vals=%s",
                large_mask.sum(), large_indices[:10], large_names[:10], large_vals
            )

    def _validate_normalization_inputs(self, name_to_index_training_input: dict, minimum, maximum, mean, stdev):
        assert len(self.methods) == sum(len(v) for v in self.method_config.values()), (
            f"Error parsing methods in InputNormalizer methods ({len(self.methods)}) "
            f"and entries in config ({sum(len(v) for v in self.method_config)}) do not match."
        )

        # Check that all sizes align
        n = minimum.size
        assert maximum.size == n, (maximum.size, n)
        assert mean.size == n, (mean.size, n)
        assert stdev.size == n, (stdev.size, n)

        # Check for typos in method config
        assert isinstance(self.methods, dict)
        for name, method in self.methods.items():
            assert name in name_to_index_training_input, f"{name} is not a valid variable name"
            assert method in [
                "mean-std",
                "std",
                # "robust",
                "min-max",
                "max",
                "none",
            ], f"{method} is not a valid normalisation method"

    def transform(
        self, x: torch.Tensor, in_place: bool = True, data_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Normalizes an input tensor x of shape [..., nvars].

        Normalization is performed in float32 for numerical stability with bfloat16,
        then cast back to the original dtype.

        The default usecase either assume the full batch tensor or the full input tensor.
        A dataindex is based on the full data can be supplied to choose which variables to normalise.

        Parameters
        ----------
        x : torch.Tensor
            Data to normalize
        in_place : bool, optional
            Normalize in-place, by default True (ignored when dtype conversion needed)
        data_index : Optional[torch.Tensor], optional
            Normalize only the specified indices, by default None

        Returns
        -------
        torch.Tensor
            Normalized tensor
        """
        original_dtype = x.dtype

        # Perform computation in float32 for numerical stability
        # This is especially important for variables with large norm_mul (small stdev)
        x_f32 = x.float()

        if data_index is not None:
            x_f32 = x_f32 * self._norm_mul[data_index].float() + self._norm_add[data_index].float()
        elif x_f32.shape[-1] == len(self._input_idx):
            x_f32 = x_f32 * self._norm_mul[self._input_idx].float() + self._norm_add[self._input_idx].float()
        else:
            x_f32 = x_f32 * self._norm_mul.float() + self._norm_add.float()

        # Cast back to original dtype
        result = x_f32.to(original_dtype)

        # If in_place was requested and dtypes match, copy back
        if in_place and original_dtype == torch.float32:
            x.copy_(result)
            return x

        return result

    def inverse_transform(
        self, x: torch.Tensor, in_place: bool = True, data_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Denormalizes an input tensor x of shape [..., nvars | nvars_pred].

        Denormalization is performed in float32 for numerical stability with bfloat16,
        then cast back to the original dtype.

        The default usecase either assume the full batch tensor or the full output tensor.
        A dataindex is based on the full data can be supplied to choose which variables to denormalise.

        Parameters
        ----------
        x : torch.Tensor
            Data to denormalize
        in_place : bool, optional
            Denormalize in-place, by default True (ignored when dtype conversion needed)
        data_index : Optional[torch.Tensor], optional
            Denormalize only the specified indices, by default None

        Returns
        -------
        torch.Tensor
            Denormalized data
        """
        original_dtype = x.dtype

        # Perform computation in float32 for numerical stability
        x_f32 = x.float()

        # Denormalize dynamic or full tensors
        # input and predicted tensors have different shapes
        # hence, we mask out the forcing indices
        if data_index is not None:
            x_f32 = (x_f32 - self._norm_add[data_index].float()) / self._norm_mul[data_index].float()
        elif x_f32.shape[-1] == len(self._output_idx):
            x_f32 = (x_f32 - self._norm_add[self._output_idx].float()) / self._norm_mul[self._output_idx].float()
        else:
            x_f32 = (x_f32 - self._norm_add.float()) / self._norm_mul.float()

        # Cast back to original dtype
        result = x_f32.to(original_dtype)

        # If in_place was requested and dtypes match, copy back
        if in_place and original_dtype == torch.float32:
            x.copy_(result)
            return x

        return result
