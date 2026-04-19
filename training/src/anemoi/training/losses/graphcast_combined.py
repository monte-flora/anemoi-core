# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""GraphCast-aware combined loss.

Thin subclass of ``CombinedLoss`` that additionally routes
``set_data_indices`` to every child loss. This is required when any of the
children inherits from ``GraphCastBaseLoss`` (e.g. GraphCastMSELoss,
GraphCastMSHLoss, SpatialGradientLoss) because those losses build their
per-variable-group slices in ``set_data_indices`` — without the routing they
crash at forward time with ``'NoneType' object has no attribute 'data'``
coming out of ``_build_variable_groups``.

All other behaviour (forward, add_scaler, update_scaler, __init__ semantics)
is inherited unchanged from ``CombinedLoss`` so future upstream changes to
``combined.py`` propagate here automatically.

Also inheriting from ``CombinedLoss`` means downstream diagnostics
(``losses/utils.py:print_variable_scaling``) correctly recurse into child
losses via the ``isinstance(loss, CombinedLoss)`` branch.
"""

from __future__ import annotations

from anemoi.training.losses.combined import CombinedLoss


class GraphCastCombinedLoss(CombinedLoss):
    """CombinedLoss that forwards ``set_data_indices`` to every child.

    Interchangeable drop-in for ``CombinedLoss``. Use this target whenever
    any constituent loss is a ``GraphCastBaseLoss`` subclass (per-variable
    group reduction needs ``data_indices`` at init time).
    """

    def set_data_indices(self, data_indices) -> None:
        """Route data_indices to every child that supports it.

        ``_apply_scalers`` (in losses/loss.py) calls this on the top-level
        GraphCastCombinedLoss once scalers + data_indices are ready. We
        forward to children so their per-variable-group machinery
        (group_slices, variable_groups) can be built just as it would be
        for a non-combined GraphCast loss.
        """
        if data_indices is None:
            return
        for child in self.losses:
            if hasattr(child, "set_data_indices"):
                child.set_data_indices(data_indices)
