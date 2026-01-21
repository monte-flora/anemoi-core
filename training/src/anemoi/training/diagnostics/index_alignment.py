# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Diagnostic utilities for detecting index misalignment in normalization pipeline.

This module provides comprehensive logging and analysis tools to diagnose
extreme normalized values that may result from index mismatches between:
- Data tensor variable indices
- Normalization statistics arrays
- Loss computation variable groups

Key Features:
- Per-variable diagnostic reporting
- Index alignment auditing across pipeline stages
- Detection of suspicious normalized values (|value| > 10)
- Physical bounds checking for meteorological variables

Usage:
    from anemoi.training.diagnostics.index_alignment import (
        IndexAlignmentAuditor,
        FeatureDiagnosticReport,
        log_normalization_diagnostics,
    )
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)

# Thresholds for flagging suspicious values
NORMALIZED_EXTREME_THRESHOLD = 10.0  # |normalized| > 10 is suspicious
NEAR_ZERO_STD_THRESHOLD = 1e-8  # std < 1e-8 causes division instability

# Physical bounds for common meteorological variables (used for sanity checking)
PHYSICAL_BOUNDS = {
    # Temperature variables (Kelvin)
    "t": (150.0, 350.0),
    "t2m": (180.0, 340.0),
    "skt": (180.0, 350.0),
    # Specific humidity (kg/kg)
    "q": (0.0, 0.05),
    "qv": (0.0, 0.05),
    # Relative humidity (0-1 or 0-100 depending on dataset)
    "r": (0.0, 110.0),
    # Geopotential (m^2/s^2)
    "z": (-5000.0, 100000.0),
    # Wind components (m/s)
    "u": (-150.0, 150.0),
    "v": (-150.0, 150.0),
    "u10": (-60.0, 60.0),
    "v10": (-60.0, 60.0),
    # Surface pressure (Pa)
    "sp": (40000.0, 110000.0),
    "msl": (85000.0, 110000.0),
    # Precipitation (m or mm depending on dataset)
    "tp": (0.0, 0.5),
    "cp": (0.0, 0.3),
    # Cloud cover (0-1)
    "tcc": (0.0, 1.0),
    "lcc": (0.0, 1.0),
    "mcc": (0.0, 1.0),
    "hcc": (0.0, 1.0),
}


@dataclass
class FeatureDiagnostic:
    """Diagnostic information for a single feature/variable."""

    name: str
    index: int
    # Raw (unnormalized) statistics
    raw_min: float = float('nan')
    raw_max: float = float('nan')
    raw_mean: float = float('nan')
    raw_std: float = float('nan')
    # Normalization parameters being applied
    norm_mean: float = float('nan')
    norm_std: float = float('nan')
    norm_min: float = float('nan')
    norm_max: float = float('nan')
    norm_method: str = "unknown"
    # Resulting normalized statistics
    normalized_min: float = float('nan')
    normalized_max: float = float('nan')
    normalized_mean: float = float('nan')
    # Flags
    suspicious: bool = False
    suspicion_reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "index": self.index,
            "raw_min": self.raw_min,
            "raw_max": self.raw_max,
            "raw_mean": self.raw_mean,
            "raw_std": self.raw_std,
            "norm_mean": self.norm_mean,
            "norm_std": self.norm_std,
            "norm_method": self.norm_method,
            "normalized_min": self.normalized_min,
            "normalized_max": self.normalized_max,
            "normalized_mean": self.normalized_mean,
            "suspicious": self.suspicious,
            "reasons": self.suspicion_reasons,
        }


class FeatureDiagnosticReport:
    """Generates and stores diagnostic reports for all features."""

    def __init__(self, name_to_index: dict[str, int]):
        self.name_to_index = name_to_index
        self.index_to_name = {v: k for k, v in name_to_index.items()}
        self.diagnostics: dict[str, FeatureDiagnostic] = {}

        # Initialize empty diagnostics for all features
        for name, idx in name_to_index.items():
            self.diagnostics[name] = FeatureDiagnostic(name=name, index=idx)

    def update_raw_stats(
        self,
        data: torch.Tensor,
        dim: tuple[int, ...] = (0, 1, 2),
    ) -> None:
        """Update raw data statistics from a data tensor.

        Parameters
        ----------
        data : torch.Tensor
            Data tensor with variables in the last dimension.
        dim : tuple[int, ...]
            Dimensions to reduce over (default: batch, ensemble, grid).
        """
        n_vars = data.shape[-1]

        for name, diag in self.diagnostics.items():
            idx = diag.index
            if idx >= n_vars:
                continue

            var_data = data[..., idx]
            valid_data = var_data[~torch.isnan(var_data) & ~torch.isinf(var_data)]

            if valid_data.numel() > 0:
                diag.raw_min = valid_data.min().item()
                diag.raw_max = valid_data.max().item()
                diag.raw_mean = valid_data.mean().item()
                diag.raw_std = valid_data.std().item()

    def update_norm_params(
        self,
        statistics: dict,
        methods: dict[str, str],
        default_method: str = "mean-std",
    ) -> None:
        """Update normalization parameters from statistics dictionary.

        Parameters
        ----------
        statistics : dict
            Dictionary containing 'mean', 'stdev', 'minimum', 'maximum' arrays.
        methods : dict[str, str]
            Dictionary mapping variable names to normalization methods.
        default_method : str
            Default normalization method if not specified.
        """
        mean = statistics.get("mean", np.array([]))
        stdev = statistics.get("stdev", np.array([]))
        minimum = statistics.get("minimum", np.array([]))
        maximum = statistics.get("maximum", np.array([]))

        for name, diag in self.diagnostics.items():
            idx = diag.index

            # Get normalization method
            diag.norm_method = methods.get(name, default_method)

            # Get normalization parameters
            if idx < len(mean):
                diag.norm_mean = float(mean[idx])
            if idx < len(stdev):
                diag.norm_std = float(stdev[idx])
            if idx < len(minimum):
                diag.norm_min = float(minimum[idx])
            if idx < len(maximum):
                diag.norm_max = float(maximum[idx])

    def update_normalized_stats(
        self,
        normalized_data: torch.Tensor,
        dim: tuple[int, ...] = (0, 1, 2),
    ) -> None:
        """Update normalized data statistics.

        Parameters
        ----------
        normalized_data : torch.Tensor
            Normalized data tensor with variables in the last dimension.
        dim : tuple[int, ...]
            Dimensions to reduce over.
        """
        n_vars = normalized_data.shape[-1]

        for name, diag in self.diagnostics.items():
            idx = diag.index
            if idx >= n_vars:
                continue

            var_data = normalized_data[..., idx]
            valid_data = var_data[~torch.isnan(var_data) & ~torch.isinf(var_data)]

            if valid_data.numel() > 0:
                diag.normalized_min = valid_data.min().item()
                diag.normalized_max = valid_data.max().item()
                diag.normalized_mean = valid_data.mean().item()

    def flag_suspicious(self) -> list[FeatureDiagnostic]:
        """Flag suspicious features and return the list."""
        suspicious = []

        for name, diag in self.diagnostics.items():
            diag.suspicious = False
            diag.suspicion_reasons = []

            # Check for extreme normalized values
            if abs(diag.normalized_min) > NORMALIZED_EXTREME_THRESHOLD or \
               abs(diag.normalized_max) > NORMALIZED_EXTREME_THRESHOLD:
                diag.suspicious = True
                diag.suspicion_reasons.append(
                    f"Extreme normalized values: [{diag.normalized_min:.2f}, {diag.normalized_max:.2f}]"
                )

            # Check for near-zero std (division instability)
            if diag.norm_std < NEAR_ZERO_STD_THRESHOLD and diag.norm_method in ("mean-std", "std"):
                diag.suspicious = True
                diag.suspicion_reasons.append(
                    f"Near-zero std: {diag.norm_std:.2e}"
                )

            # Check for raw values outside expected physical bounds
            base_name = name.split("_")[0] if "_" in name else name
            if base_name in PHYSICAL_BOUNDS:
                phys_min, phys_max = PHYSICAL_BOUNDS[base_name]
                if diag.raw_min < phys_min * 0.5 or diag.raw_max > phys_max * 2.0:
                    diag.suspicious = True
                    diag.suspicion_reasons.append(
                        f"Raw values outside physical bounds: [{diag.raw_min:.2f}, {diag.raw_max:.2f}] "
                        f"vs expected [{phys_min:.2f}, {phys_max:.2f}]"
                    )

            # Check for potential index mismatch: raw stats don't match norm params
            if diag.norm_method == "mean-std" and not np.isnan(diag.raw_mean) and not np.isnan(diag.norm_mean):
                # If the raw mean is very different from the norm mean used, might be a mismatch
                mean_diff = abs(diag.raw_mean - diag.norm_mean)
                if diag.raw_std > 0:
                    rel_diff = mean_diff / max(diag.raw_std, 1e-10)
                    if rel_diff > 5.0:  # Raw mean differs by >5 std from norm mean
                        diag.suspicious = True
                        diag.suspicion_reasons.append(
                            f"Potential index mismatch: raw_mean={diag.raw_mean:.4f} vs norm_mean={diag.norm_mean:.4f} "
                            f"(diff={mean_diff:.4f}, {rel_diff:.1f} std away)"
                        )

            if diag.suspicious:
                suspicious.append(diag)

        return suspicious

    def generate_report(self) -> str:
        """Generate a formatted diagnostic report."""
        lines = []
        lines.append("=" * 140)
        lines.append("FEATURE DIAGNOSTIC REPORT")
        lines.append("=" * 140)
        lines.append(
            f"{'Feature Name':<25} {'Idx':>4} {'Raw Min':>12} {'Raw Max':>12} "
            f"{'Norm Mean':>12} {'Norm Std':>12} {'Norm Min':>12} {'Norm Max':>12} {'Suspicious?':>11}"
        )
        lines.append("-" * 140)

        # Sort by index
        sorted_diags = sorted(self.diagnostics.values(), key=lambda d: d.index)

        for diag in sorted_diags:
            flag = "YES" if diag.suspicious else ""
            lines.append(
                f"{diag.name:<25} {diag.index:>4} "
                f"{diag.raw_min:>12.4f} {diag.raw_max:>12.4f} "
                f"{diag.norm_mean:>12.4f} {diag.norm_std:>12.4f} "
                f"{diag.normalized_min:>12.4f} {diag.normalized_max:>12.4f} "
                f"{flag:>11}"
            )

        lines.append("=" * 140)

        # Add suspicious features summary
        suspicious = [d for d in sorted_diags if d.suspicious]
        if suspicious:
            lines.append("\nSUSPICIOUS FEATURES SUMMARY:")
            lines.append("-" * 100)
            for diag in suspicious:
                lines.append(f"\n{diag.name} (index {diag.index}):")
                for reason in diag.suspicion_reasons:
                    lines.append(f"  - {reason}")

        return "\n".join(lines)


class IndexAlignmentAuditor:
    """Audits index alignment through the data pipeline."""

    def __init__(self):
        self.dataset_name_to_index: Optional[dict[str, int]] = None
        self.normalizer_name_to_index: Optional[dict[str, int]] = None
        self.loss_variable_groups: Optional[dict[str, list[int]]] = None
        self.statistics_shape: Optional[tuple] = None
        self.data_shape: Optional[tuple] = None
        self.misalignments: list[str] = []

    def register_dataset(
        self,
        name_to_index: dict[str, int],
        data_shape: tuple,
    ) -> None:
        """Register dataset index mapping."""
        self.dataset_name_to_index = name_to_index.copy()
        self.data_shape = data_shape
        LOGGER.info(
            "IndexAlignmentAuditor: Registered dataset with %d variables, data shape %s",
            len(name_to_index), data_shape
        )

    def register_normalizer(
        self,
        name_to_index: dict[str, int],
        statistics: dict,
    ) -> None:
        """Register normalizer index mapping and statistics."""
        self.normalizer_name_to_index = name_to_index.copy()

        # Get statistics shapes
        mean_shape = statistics.get("mean", np.array([])).shape
        self.statistics_shape = mean_shape

        LOGGER.info(
            "IndexAlignmentAuditor: Registered normalizer with %d variables, stats shape %s",
            len(name_to_index), mean_shape
        )

    def register_loss(
        self,
        variable_groups: dict[str, list[int]],
    ) -> None:
        """Register loss variable groups."""
        self.loss_variable_groups = variable_groups.copy()
        n_vars = sum(len(v) for v in variable_groups.values())
        LOGGER.info(
            "IndexAlignmentAuditor: Registered loss with %d groups, %d total variables",
            len(variable_groups), n_vars
        )

    def audit(self) -> list[str]:
        """Perform alignment audit and return list of issues found."""
        self.misalignments = []

        # Check 1: Dataset vs Normalizer name_to_index
        if self.dataset_name_to_index and self.normalizer_name_to_index:
            ds_vars = set(self.dataset_name_to_index.keys())
            norm_vars = set(self.normalizer_name_to_index.keys())

            if ds_vars != norm_vars:
                only_ds = ds_vars - norm_vars
                only_norm = norm_vars - ds_vars
                if only_ds:
                    self.misalignments.append(
                        f"Variables only in dataset: {sorted(only_ds)[:10]}..."
                    )
                if only_norm:
                    self.misalignments.append(
                        f"Variables only in normalizer: {sorted(only_norm)[:10]}..."
                    )

            # Check index consistency
            for name in ds_vars & norm_vars:
                ds_idx = self.dataset_name_to_index[name]
                norm_idx = self.normalizer_name_to_index[name]
                if ds_idx != norm_idx:
                    self.misalignments.append(
                        f"Index mismatch for '{name}': dataset={ds_idx}, normalizer={norm_idx}"
                    )

        # Check 2: Statistics shape vs data shape
        if self.data_shape and self.statistics_shape:
            n_data_vars = self.data_shape[-1] if self.data_shape else 0
            n_stat_vars = self.statistics_shape[0] if self.statistics_shape else 0
            if n_data_vars != n_stat_vars:
                self.misalignments.append(
                    f"Variable count mismatch: data has {n_data_vars} vars, "
                    f"statistics has {n_stat_vars} entries"
                )

        # Log results
        if self.misalignments:
            LOGGER.error(
                "INDEX ALIGNMENT AUDIT FAILED: Found %d issues",
                len(self.misalignments)
            )
            for issue in self.misalignments:
                LOGGER.error("  - %s", issue)
        else:
            LOGGER.info("INDEX ALIGNMENT AUDIT PASSED: No misalignments detected")

        return self.misalignments


# Global auditor instance for cross-module coordination
_global_auditor: Optional[IndexAlignmentAuditor] = None


def get_global_auditor() -> IndexAlignmentAuditor:
    """Get or create the global index alignment auditor."""
    global _global_auditor
    if _global_auditor is None:
        _global_auditor = IndexAlignmentAuditor()
    return _global_auditor


def log_normalization_diagnostics(
    name: str,
    index: int,
    raw_value: float,
    norm_mean: float,
    norm_std: float,
    normalized_value: float,
    threshold: float = NORMALIZED_EXTREME_THRESHOLD,
) -> None:
    """Log diagnostic information for a single normalization operation.

    Parameters
    ----------
    name : str
        Variable name.
    index : int
        Variable index.
    raw_value : float
        Raw (unnormalized) value.
    norm_mean : float
        Mean used for normalization.
    norm_std : float
        Standard deviation used for normalization.
    normalized_value : float
        Resulting normalized value.
    threshold : float
        Threshold for flagging extreme values.
    """
    if abs(normalized_value) > threshold:
        LOGGER.warning(
            "EXTREME NORMALIZED: var=%s (idx=%d), raw=%.4f, norm_mean=%.4f, "
            "norm_std=%.4f, normalized=%.4f",
            name, index, raw_value, norm_mean, norm_std, normalized_value
        )


def log_batch_normalization_summary(
    raw_data: torch.Tensor,
    normalized_data: torch.Tensor,
    name_to_index: dict[str, int],
    statistics: dict,
    methods: dict[str, str],
    batch_id: int,
    threshold: float = NORMALIZED_EXTREME_THRESHOLD,
) -> None:
    """Log a summary of normalization for a batch.

    Parameters
    ----------
    raw_data : torch.Tensor
        Raw data tensor before normalization.
    normalized_data : torch.Tensor
        Data tensor after normalization.
    name_to_index : dict[str, int]
        Mapping from variable names to indices.
    statistics : dict
        Normalization statistics dictionary.
    methods : dict[str, str]
        Normalization methods per variable.
    batch_id : int
        Batch identifier for logging.
    threshold : float
        Threshold for flagging extreme values.
    """
    mean = statistics.get("mean", np.array([]))
    stdev = statistics.get("stdev", np.array([]))

    n_vars = normalized_data.shape[-1]
    extreme_vars = []

    for name, idx in name_to_index.items():
        if idx >= n_vars:
            continue

        var_norm = normalized_data[..., idx]
        valid_norm = var_norm[~torch.isnan(var_norm) & ~torch.isinf(var_norm)]

        if valid_norm.numel() == 0:
            continue

        norm_min = valid_norm.min().item()
        norm_max = valid_norm.max().item()

        if abs(norm_min) > threshold or abs(norm_max) > threshold:
            var_raw = raw_data[..., idx]
            valid_raw = var_raw[~torch.isnan(var_raw) & ~torch.isinf(var_raw)]

            raw_min = valid_raw.min().item() if valid_raw.numel() > 0 else float('nan')
            raw_max = valid_raw.max().item() if valid_raw.numel() > 0 else float('nan')
            raw_mean = valid_raw.mean().item() if valid_raw.numel() > 0 else float('nan')

            stat_mean = mean[idx] if idx < len(mean) else float('nan')
            stat_std = stdev[idx] if idx < len(stdev) else float('nan')

            extreme_vars.append({
                "name": name,
                "index": idx,
                "raw_range": (raw_min, raw_max),
                "raw_mean": raw_mean,
                "stat_mean": stat_mean,
                "stat_std": stat_std,
                "norm_range": (norm_min, norm_max),
            })

    if extreme_vars:
        LOGGER.warning(
            "BATCH %d: Found %d variables with extreme normalized values (>%.1f):",
            batch_id, len(extreme_vars), threshold
        )
        for ev in extreme_vars[:20]:  # Limit output
            LOGGER.warning(
                "  %s (idx=%d): raw=[%.4f, %.4f] (mean=%.4f), "
                "stats: mean=%.4f, std=%.4f -> normalized=[%.4f, %.4f]",
                ev["name"], ev["index"],
                ev["raw_range"][0], ev["raw_range"][1], ev["raw_mean"],
                ev["stat_mean"], ev["stat_std"],
                ev["norm_range"][0], ev["norm_range"][1]
            )


def check_variable_order_consistency(
    dataset_name_to_index: dict[str, int],
    statistics_order: list[str],
) -> list[str]:
    """Check if the variable order in dataset matches statistics order.

    Parameters
    ----------
    dataset_name_to_index : dict[str, int]
        Mapping from variable names to indices in the dataset.
    statistics_order : list[str]
        Order of variables in the statistics file.

    Returns
    -------
    list[str]
        List of mismatched variables.
    """
    mismatches = []

    for name, ds_idx in dataset_name_to_index.items():
        if name in statistics_order:
            stat_idx = statistics_order.index(name)
            if ds_idx != stat_idx:
                mismatches.append(
                    f"{name}: dataset_idx={ds_idx}, statistics_idx={stat_idx}"
                )

    if mismatches:
        LOGGER.error(
            "VARIABLE ORDER MISMATCH: Found %d variables with inconsistent indices:",
            len(mismatches)
        )
        for m in mismatches[:20]:
            LOGGER.error("  - %s", m)

    return mismatches
