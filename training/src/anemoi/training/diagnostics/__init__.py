# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Diagnostic utilities for Anemoi training pipeline."""

from anemoi.training.diagnostics.index_alignment import (
    FeatureDiagnostic,
    FeatureDiagnosticReport,
    IndexAlignmentAuditor,
    check_variable_order_consistency,
    get_global_auditor,
    log_batch_normalization_summary,
    log_normalization_diagnostics,
    NORMALIZED_EXTREME_THRESHOLD,
    NEAR_ZERO_STD_THRESHOLD,
    PHYSICAL_BOUNDS,
)

__all__ = [
    "FeatureDiagnostic",
    "FeatureDiagnosticReport",
    "IndexAlignmentAuditor",
    "check_variable_order_consistency",
    "get_global_auditor",
    "log_batch_normalization_summary",
    "log_normalization_diagnostics",
    "NORMALIZED_EXTREME_THRESHOLD",
    "NEAR_ZERO_STD_THRESHOLD",
    "PHYSICAL_BOUNDS",
]
