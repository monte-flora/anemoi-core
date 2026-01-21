#!/usr/bin/env python
# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Standalone script to diagnose index alignment issues in Anemoi training pipeline.

This script helps identify the root cause of extreme normalized values by:
1. Loading dataset and statistics
2. Comparing variable order between data and statistics
3. Generating a detailed feature diagnostic report
4. Identifying potential index mismatches

Usage:
    python run_index_alignment_audit.py --config path/to/config.yaml
    python run_index_alignment_audit.py --dataset path/to/dataset.zarr --stats path/to/stats.zarr

Example output:
    ================================================================================
    FEATURE DIAGNOSTIC REPORT
    ================================================================================
    Feature Name              Index    Raw Min    Raw Max    Norm Mean    Norm Std    Normalized Min    Normalized Max    Suspicious?
    --------------------------------------------------------------------------------
    t2m                           0     250.0000   320.0000    287.5000     15.2000         -2.4671           2.1382
    u10                           1     -30.0000    30.0000      0.1200      5.3400         -5.6554           5.5917
    ...
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def load_dataset(dataset_path: str):
    """Load an Anemoi dataset."""
    try:
        from anemoi.datasets import open_dataset
        return open_dataset(dataset_path)
    except ImportError:
        LOGGER.error("anemoi-datasets not installed. Please install it first.")
        sys.exit(1)


def generate_feature_report(
    data_sample: np.ndarray,
    statistics: dict,
    name_to_index: dict,
    output_file: str = None,
) -> str:
    """Generate a comprehensive feature diagnostic report.

    Parameters
    ----------
    data_sample : np.ndarray
        Sample of data with shape (..., n_variables).
    statistics : dict
        Statistics dictionary with 'mean', 'stdev', 'minimum', 'maximum'.
    name_to_index : dict
        Mapping from variable names to indices.
    output_file : str, optional
        Path to save the report. If None, prints to stdout.

    Returns
    -------
    str
        The formatted report.
    """
    mean = statistics.get("mean", np.array([]))
    stdev = statistics.get("stdev", np.array([]))
    minimum = statistics.get("minimum", np.array([]))
    maximum = statistics.get("maximum", np.array([]))

    index_to_name = {v: k for k, v in name_to_index.items()}
    n_vars = data_sample.shape[-1]

    # Threshold for flagging suspicious values
    NORM_EXTREME_THRESHOLD = 10.0
    NEAR_ZERO_STD = 1e-8

    lines = []
    lines.append("=" * 150)
    lines.append("FEATURE DIAGNOSTIC REPORT")
    lines.append("=" * 150)
    lines.append(
        f"{'Feature Name':<30} {'Idx':>5} "
        f"{'Raw Min':>14} {'Raw Max':>14} {'Raw Mean':>14} "
        f"{'Stat Mean':>14} {'Stat Std':>14} "
        f"{'Norm Min':>14} {'Norm Max':>14} {'Suspicious?'}"
    )
    lines.append("-" * 150)

    suspicious_features = []

    for idx in range(n_vars):
        var_name = index_to_name.get(idx, f"unknown_idx_{idx}")

        # Raw data statistics
        var_data = data_sample[..., idx].flatten()
        valid_data = var_data[~np.isnan(var_data) & ~np.isinf(var_data)]

        if len(valid_data) > 0:
            raw_min = np.min(valid_data)
            raw_max = np.max(valid_data)
            raw_mean = np.mean(valid_data)
        else:
            raw_min = raw_max = raw_mean = np.nan

        # Statistics for this index
        stat_mean = mean[idx] if idx < len(mean) else np.nan
        stat_std = stdev[idx] if idx < len(stdev) else np.nan

        # Compute normalized values
        if not np.isnan(stat_mean) and not np.isnan(stat_std) and stat_std > NEAR_ZERO_STD:
            norm_min = (raw_min - stat_mean) / stat_std
            norm_max = (raw_max - stat_mean) / stat_std
        else:
            norm_min = norm_max = np.nan

        # Check for suspicious conditions
        reasons = []

        # 1. Extreme normalized values
        if abs(norm_min) > NORM_EXTREME_THRESHOLD or abs(norm_max) > NORM_EXTREME_THRESHOLD:
            reasons.append("EXTREME_NORM")

        # 2. Near-zero std
        if stat_std < NEAR_ZERO_STD:
            reasons.append("ZERO_STD")

        # 3. Potential index mismatch: raw mean very different from stat mean
        if not np.isnan(raw_mean) and not np.isnan(stat_mean) and stat_std > NEAR_ZERO_STD:
            mean_diff = abs(raw_mean - stat_mean) / stat_std
            if mean_diff > 5.0:
                reasons.append(f"INDEX_MISMATCH?({mean_diff:.1f}σ)")

        # 4. Check physical bounds for known variables
        base_name = var_name.split("_")[0] if "_" in var_name else var_name
        physical_bounds = {
            "t": (150, 350), "t2m": (180, 340), "skt": (180, 350),
            "q": (0, 0.05), "qv": (0, 0.05),
            "u": (-150, 150), "v": (-150, 150),
            "u10": (-60, 60), "v10": (-60, 60),
            "sp": (40000, 110000), "msl": (85000, 110000),
            "tp": (0, 0.5), "cp": (0, 0.3),
        }
        if base_name in physical_bounds:
            phys_min, phys_max = physical_bounds[base_name]
            if raw_min < phys_min * 0.5 or raw_max > phys_max * 2:
                reasons.append("OUT_OF_BOUNDS")

        suspicious_str = ", ".join(reasons) if reasons else ""

        lines.append(
            f"{var_name:<30} {idx:>5} "
            f"{raw_min:>14.4f} {raw_max:>14.4f} {raw_mean:>14.4f} "
            f"{stat_mean:>14.4f} {stat_std:>14.4f} "
            f"{norm_min:>14.4f} {norm_max:>14.4f} {suspicious_str}"
        )

        if reasons:
            suspicious_features.append({
                "name": var_name,
                "index": idx,
                "raw_range": (raw_min, raw_max),
                "raw_mean": raw_mean,
                "stat_mean": stat_mean,
                "stat_std": stat_std,
                "norm_range": (norm_min, norm_max),
                "reasons": reasons,
            })

    lines.append("=" * 150)

    # Summary of suspicious features
    if suspicious_features:
        lines.append("")
        lines.append("=" * 100)
        lines.append(f"SUSPICIOUS FEATURES SUMMARY ({len(suspicious_features)} found)")
        lines.append("=" * 100)

        for feat in suspicious_features:
            lines.append("")
            lines.append(f"{feat['name']} (index {feat['index']}):")
            for reason in feat["reasons"]:
                lines.append(f"  - {reason}")
            lines.append(f"    Raw range: [{feat['raw_range'][0]:.4f}, {feat['raw_range'][1]:.4f}]")
            lines.append(f"    Raw mean: {feat['raw_mean']:.4f}")
            lines.append(f"    Stat mean: {feat['stat_mean']:.4f}, Stat std: {feat['stat_std']:.4f}")
            lines.append(f"    Normalized range: [{feat['norm_range'][0]:.4f}, {feat['norm_range'][1]:.4f}]")

    # Hypothesis about root cause
    lines.append("")
    lines.append("=" * 100)
    lines.append("ROOT CAUSE HYPOTHESIS")
    lines.append("=" * 100)

    index_mismatch_features = [f for f in suspicious_features if any("INDEX_MISMATCH" in r for r in f["reasons"])]
    extreme_features = [f for f in suspicious_features if "EXTREME_NORM" in f["reasons"]]
    zero_std_features = [f for f in suspicious_features if "ZERO_STD" in f["reasons"]]

    if index_mismatch_features:
        lines.append("")
        lines.append("LIKELY INDEX MISMATCH DETECTED!")
        lines.append(f"  {len(index_mismatch_features)} features have raw means that differ significantly")
        lines.append("  from the statistics means being applied to them.")
        lines.append("")
        lines.append("  Possible causes:")
        lines.append("  1. Variable order in dataset differs from statistics file")
        lines.append("  2. Variables were reordered or renamed without updating statistics")
        lines.append("  3. Atmospheric levels are ordered differently (ascending vs descending)")
        lines.append("")
        lines.append("  RECOMMENDED FIX:")
        lines.append("  - Compare the variable order in your dataset's name_to_index")
        lines.append("    with the order in your statistics file")
        lines.append("  - Check if 'reorder' is needed in your dataloader config")
    elif extreme_features and not index_mismatch_features:
        lines.append("")
        lines.append("EXTREME VALUES DETECTED (but not obviously index mismatch)")
        lines.append(f"  {len(extreme_features)} features have normalized values > {NORM_EXTREME_THRESHOLD}")
        lines.append("")
        lines.append("  Possible causes:")
        lines.append("  1. Statistics were computed on a different data distribution")
        lines.append("  2. Outliers in the current data that weren't in statistics")
        lines.append("  3. Physical units mismatch (e.g., m vs mm for precipitation)")
    elif zero_std_features:
        lines.append("")
        lines.append("ZERO STD FEATURES DETECTED")
        lines.append(f"  {len(zero_std_features)} features have near-zero standard deviation")
        lines.append("  This causes division instability in normalization.")
    else:
        lines.append("")
        lines.append("No obvious issues detected in the sampled data.")
        lines.append("Consider running with more samples or checking specific time periods.")

    report = "\n".join(lines)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        LOGGER.info("Report saved to %s", output_file)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose index alignment issues in Anemoi training pipeline"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to the Anemoi dataset (zarr format)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to training config YAML (alternative to --dataset)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of samples to analyze (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the diagnostic report",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.dataset and not args.config:
        parser.error("Either --dataset or --config must be provided")

    # Load dataset
    if args.dataset:
        LOGGER.info("Loading dataset from %s", args.dataset)
        dataset = load_dataset(args.dataset)
    else:
        LOGGER.info("Loading config from %s", args.config)
        # TODO: Implement config loading
        LOGGER.error("Config loading not yet implemented. Please use --dataset.")
        sys.exit(1)

    # Get name_to_index and statistics
    name_to_index = dataset.name_to_index
    statistics = dataset.statistics

    LOGGER.info("Dataset has %d variables", len(name_to_index))
    LOGGER.info("Statistics keys: %s", list(statistics.keys()))

    # Sample some data
    LOGGER.info("Loading %d samples for analysis...", args.n_samples)

    # Get data shape
    n_times = len(dataset)
    n_vars = len(name_to_index)

    # Sample evenly spaced time indices
    sample_indices = np.linspace(0, n_times - 1, args.n_samples, dtype=int)

    # Load samples
    samples = []
    for idx in sample_indices:
        try:
            sample = dataset[int(idx)]
            samples.append(sample)
        except Exception as e:
            LOGGER.warning("Failed to load sample %d: %s", idx, e)

    if not samples:
        LOGGER.error("Failed to load any samples!")
        sys.exit(1)

    # Stack samples
    data_array = np.stack(samples, axis=0)
    LOGGER.info("Loaded data shape: %s", data_array.shape)

    # Generate report
    report = generate_feature_report(
        data_sample=data_array,
        statistics=statistics,
        name_to_index=name_to_index,
        output_file=args.output,
    )

    if not args.output:
        print(report)


if __name__ == "__main__":
    main()
