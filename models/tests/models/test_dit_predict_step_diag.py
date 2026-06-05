# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
"""Regression tests for AnemoiDiTModel.predict_step index-space handling.

These guard the two diagnostic-mode inference bugs fixed for v39 (the first
checkpoint with prognostic + diagnostic outputs together):

  1. ``x`` is in MODEL-INPUT space (prognostic+forcing, diagnostics/dropped
     vars excluded), so the previous-state prognostic slice must be taken with
     ``data_indices.model.input.prognostic`` -- NOT ``data.input.prognostic``
     (data space). The two diverge as soon as any output-only/dropped variable
     precedes a prognostic in data space; the data-space index then selects the
     wrong channels (or runs off the end).

  2. Diagnostics are OUTPUT-only, so their normalizer coefficients live at the
     ``data.output.diagnostic`` positions. ``data.input.diagnostic`` is empty,
     and using it silently leaves diagnostics in NORMALIZED units.

The inference diagnostic path was entirely unexercised before v39, so these
build a minimal stand-in (no DiT, no real normalizers) and drive predict_step
directly. Backward compat (no diagnostics) is covered by the third test.
"""
from types import SimpleNamespace

import torch

from anemoi.models.models.dit_wrapper import AnemoiDiTModel


def _di(*, model_in_prog, model_out_prog, model_out_diag, model_out_full,
        data_in_prog, data_in_diag, data_out_prog, data_out_diag) -> SimpleNamespace:
    """Minimal stand-in for IndexCollection: only attrs predict_step reads."""
    return SimpleNamespace(
        model=SimpleNamespace(
            input=SimpleNamespace(prognostic=torch.tensor(model_in_prog, dtype=torch.long)),
            output=SimpleNamespace(
                prognostic=torch.tensor(model_out_prog, dtype=torch.long),
                diagnostic=torch.tensor(model_out_diag, dtype=torch.long),
                full=torch.tensor(model_out_full, dtype=torch.long),
            ),
        ),
        data=SimpleNamespace(
            input=SimpleNamespace(
                prognostic=torch.tensor(data_in_prog, dtype=torch.long),
                diagnostic=torch.tensor(data_in_diag, dtype=torch.long),
            ),
            output=SimpleNamespace(
                prognostic=torch.tensor(data_out_prog, dtype=torch.long),
                diagnostic=torch.tensor(data_out_diag, dtype=torch.long),
            ),
        ),
    )


def _bare_model(model_output, norm_mul, norm_add, output_mode="residual"):
    """AnemoiDiTModel with __init__ skipped; only predict_step deps stubbed."""
    m = AnemoiDiTModel.__new__(AnemoiDiTModel)
    m.output_mode = output_mode
    # forward returns the canned model output regardless of x.
    m.forward = lambda x, **kw: model_output
    # normalizer buffers are read via this helper; stub it.
    m._get_normalizer_buffers = lambda pp: (norm_mul, norm_add)
    return m


class _IdentityResidualNorm:
    """Returns the previous-state prognostic slice verbatim, so the test can
    assert WHICH channels of x were selected (the index-space check)."""

    def inverse_transform_physical_from_normalized(self, x_last_prog, delta, mul, add):
        return x_last_prog


def test_predict_step_indexes_x_in_model_input_space() -> None:
    # Data space (4 vars): 0=diag_z (output-only), 1=a (prog), 2=b (prog), 3=forcing.
    # Model-input tensor x therefore holds [a, b, forcing] -> width 3.
    #   model.input.prognostic = [0, 1]  (a, b inside the 3-wide input tensor)  <- correct
    #   data.input.prognostic  = [1, 2]  (a, b in 4-wide data space)            <- the old bug
    di = _di(
        model_in_prog=[0, 1], model_out_prog=[0, 1], model_out_diag=[2],
        model_out_full=[0, 1, 2],
        data_in_prog=[1, 2], data_in_diag=[],
        data_out_prog=[1, 2], data_out_diag=[0],
    )
    # batch is model-input width (3). Marker values: a=1, b=2, forcing=9.
    batch = torch.zeros(1, 1, 2, 3)  # (B, T, G, V_in)
    batch[..., 0] = 1.0
    batch[..., 1] = 2.0
    batch[..., 2] = 9.0
    # model output (B, E, G, V_out=3): prog deltas ignored by the identity stub;
    # diag channel (idx 2) carries a normalized value to denormalize.
    model_output = torch.zeros(1, 1, 2, 3)
    model_output[..., 2] = 30.0
    norm_mul = torch.tensor([2.0, 1.0, 1.0, 1.0])  # data-space (4 wide); diag_z @0 -> *2
    norm_add = torch.tensor([10.0, 0.0, 0.0, 0.0])  # diag_z @0 -> +10

    m = _bare_model(model_output, norm_mul, norm_add)
    y = m.predict_step(
        batch, pre_processors=lambda x, in_place=True: x, post_processors=None,
        residual_normalizer=_IdentityResidualNorm(), data_indices=di, multi_step=1,
    )  # (B, G, V_out)

    # Prognostic outputs == x channels [0,1] (a=1, b=2) -> proves MODEL-INPUT
    # indexing. The data-space index [1,2] would have selected (b=2, forcing=9).
    assert torch.allclose(y[..., 0], torch.full((1, 2), 1.0))
    assert torch.allclose(y[..., 1], torch.full((1, 2), 2.0))


def test_predict_step_denormalizes_diagnostics_via_data_output_index() -> None:
    di = _di(
        model_in_prog=[0, 1], model_out_prog=[0, 1], model_out_diag=[2],
        model_out_full=[0, 1, 2],
        data_in_prog=[1, 2], data_in_diag=[],  # diagnostics are NOT inputs (empty)
        data_out_prog=[1, 2], data_out_diag=[0],  # diag_z normalizer coeffs live @ data idx 0
    )
    batch = torch.zeros(1, 1, 2, 3)
    model_output = torch.zeros(1, 1, 2, 3)
    model_output[..., 2] = 30.0  # diag normalized value
    norm_mul = torch.tensor([2.0, 1.0, 1.0, 1.0])
    norm_add = torch.tensor([10.0, 0.0, 0.0, 0.0])

    m = _bare_model(model_output, norm_mul, norm_add)
    y = m.predict_step(
        batch, pre_processors=lambda x, in_place=True: x, post_processors=None,
        residual_normalizer=_IdentityResidualNorm(), data_indices=di, multi_step=1,
    )
    # Physical diag = (30 - 10) / 2 = 10.0. The bug (empty data.input.diagnostic)
    # would have left it at the normalized 30.0.
    assert torch.allclose(y[..., 2], torch.full((1, 2), 10.0))


def test_predict_step_no_diagnostics_is_backward_compatible() -> None:
    # No diagnostics: model.input.prognostic == data.input.prognostic, output has
    # no diagnostic channels. Must behave exactly as the pre-v39 path.
    di = _di(
        model_in_prog=[0, 1], model_out_prog=[0, 1], model_out_diag=[],
        model_out_full=[0, 1],
        data_in_prog=[0, 1], data_in_diag=[],
        data_out_prog=[0, 1], data_out_diag=[],
    )
    batch = torch.zeros(1, 1, 2, 2)
    batch[..., 0] = 5.0
    batch[..., 1] = 7.0
    model_output = torch.zeros(1, 1, 2, 2)
    norm_mul = torch.ones(2)
    norm_add = torch.zeros(2)

    m = _bare_model(model_output, norm_mul, norm_add)
    y = m.predict_step(
        batch, pre_processors=lambda x, in_place=True: x, post_processors=None,
        residual_normalizer=_IdentityResidualNorm(), data_indices=di, multi_step=1,
    )
    assert y.shape == (1, 2, 2)  # (B, G, V_out) with no diagnostic channels
    assert torch.allclose(y[..., 0], torch.full((1, 2), 5.0))
    assert torch.allclose(y[..., 1], torch.full((1, 2), 7.0))
