# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
"""Unit tests for name-mapped channel transfer in transfer_learning_loading.

Covers the new behaviour (name-map added/removed input/output channels by variable
name) AND backward compatibility (no variable change -> unchanged; non-variable-count
or multi-dim mismatch -> still dropped/re-init; normalization buffers never transferred;
no data_indices -> historical drop behaviour).
"""
from types import SimpleNamespace

import torch

from anemoi.training.utils.checkpoint import _di_ordered_names
from anemoi.training.utils.checkpoint import _name_map_channels
from anemoi.training.utils.checkpoint import remap_state_dict_for_transfer


def _di(name_to_index: dict, input_full: list[int], output_full: list[int]) -> SimpleNamespace:
    """Minimal stand-in for IndexCollection: only the attributes the code reads."""
    return SimpleNamespace(
        name_to_index=name_to_index,
        data=SimpleNamespace(
            input=SimpleNamespace(full=input_full),
            output=SimpleNamespace(full=output_full),
        ),
    )


def _chan_marked(out_ch: int, in_ch: int) -> torch.Tensor:
    """Conv-like weight [out_ch, in_ch, 1, 1] where input-channel c holds the value (c+1)."""
    w = torch.zeros(out_ch, in_ch, 1, 1)
    for c in range(in_ch):
        w[:, c] = c + 1
    return w


# --------------------------------------------------------------------------- helpers


def test_di_ordered_names() -> None:
    di = _di({"a": 0, "b": 1, "c": 2, "d": 3}, input_full=[0, 2, 3], output_full=[1, 2])
    assert _di_ordered_names(di, "input") == ["a", "c", "d"]
    assert _di_ordered_names(di, "output") == ["b", "c"]


def test_name_map_channels_copies_shared_zeros_new_and_reorders() -> None:
    # old inputs [a,b,c]; new inputs [a,x,b,c] (x inserted -> b,c shift right)
    src = _chan_marked(out_ch=2, in_ch=3)  # channel a=1, b=2, c=3
    tgt = torch.full((2, 4, 1, 1), -99.0)  # sentinel so we know zeros come from the map
    out = _name_map_channels(src, tgt, dim=1, new_names=["a", "x", "b", "c"], old_idx={"a": 0, "b": 1, "c": 2})
    assert out.shape == tgt.shape
    assert torch.equal(out[:, 0], src[:, 0])  # a
    assert torch.equal(out[:, 2], src[:, 1])  # b -> new pos 2
    assert torch.equal(out[:, 3], src[:, 2])  # c -> new pos 3
    assert torch.count_nonzero(out[:, 1]) == 0  # x (new) zeroed


# --------------------------------------------------------------------------- remap: new behaviour


def test_remap_add_input_channel_is_name_mapped() -> None:
    old_di = _di({"a": 0, "b": 1, "c": 2}, [0, 1, 2], [0, 1])
    new_di = _di({"a": 0, "x": 1, "b": 2, "c": 3}, [0, 1, 2, 3], [0, 2])  # outputs a,b unchanged
    src = {"tok.weight": _chan_marked(2, 3), "trunk.weight": torch.randn(8, 8)}
    model = {"tok.weight": torch.full((2, 4, 1, 1), -99.0), "trunk.weight": torch.randn(8, 8)}
    sd, n = remap_state_dict_for_transfer(src, model, old_di, new_di)
    assert n == 1
    assert torch.equal(sd["tok.weight"][:, 0], src["tok.weight"][:, 0])  # a copied
    assert torch.equal(sd["tok.weight"][:, 2], src["tok.weight"][:, 1])  # b reordered
    assert torch.count_nonzero(sd["tok.weight"][:, 1]) == 0  # new x zeroed
    assert torch.equal(sd["trunk.weight"], src["trunk.weight"])  # shape-identical passthrough


def test_remap_add_output_channel_is_name_mapped() -> None:
    old_di = _di({"a": 0, "b": 1, "c": 2}, [0, 1, 2], [0, 1])  # outputs a,b
    new_di = _di({"a": 0, "b": 1, "c": 2, "e": 3}, [0, 1, 2], [0, 1, 3])  # outputs a,b,e
    src = {"head.weight": torch.stack([torch.full((4,), 1.0), torch.full((4,), 2.0)])}  # [2,4]
    model = {"head.weight": torch.full((3, 4), -99.0)}
    sd, n = remap_state_dict_for_transfer(src, model, old_di, new_di)
    assert n == 1
    assert torch.equal(sd["head.weight"][0], src["head.weight"][0])  # a
    assert torch.equal(sd["head.weight"][1], src["head.weight"][1])  # b
    assert torch.count_nonzero(sd["head.weight"][2]) == 0  # new e zeroed


# --------------------------------------------------------------------------- remap: backward compatibility


def test_remap_no_variable_change_is_passthrough() -> None:
    di = _di({"a": 0, "b": 1, "c": 2}, [0, 1, 2], [0, 1])
    src = {"tok.weight": _chan_marked(2, 3), "trunk.weight": torch.randn(8, 8)}
    model = {"tok.weight": torch.zeros(2, 3, 1, 1), "trunk.weight": torch.zeros(8, 8)}
    sd, n = remap_state_dict_for_transfer(src, model, di, di)
    assert n == 0
    assert set(sd) == set(src)
    assert torch.equal(sd["tok.weight"], src["tok.weight"])
    assert torch.equal(sd["trunk.weight"], src["trunk.weight"])


def test_remap_multidim_mismatch_is_dropped() -> None:
    old_di = _di({"a": 0, "b": 1, "c": 2}, [0, 1, 2], [0, 1])
    new_di = _di({"a": 0, "x": 1, "b": 2, "c": 3}, [0, 1, 2, 3], [0, 2])
    src = {"blk.weight": torch.randn(5, 5)}
    model = {"blk.weight": torch.randn(6, 6)}  # two dims differ
    sd, n = remap_state_dict_for_transfer(src, model, old_di, new_di)
    assert n == 0
    assert "blk.weight" not in sd  # dropped -> model keeps its fresh init (historical behaviour)


def test_remap_nonvariable_dim_mismatch_is_dropped() -> None:
    old_di = _di({"a": 0, "b": 1, "c": 2}, [0, 1, 2], [0, 1])  # in=3 out=2
    new_di = _di({"a": 0, "x": 1, "b": 2, "c": 3}, [0, 1, 2, 3], [0, 2])  # in=4 out=2
    src = {"other.weight": torch.randn(7, 8)}  # 7 matches neither in(3/4) nor out(2/2)
    model = {"other.weight": torch.randn(9, 8)}
    sd, n = remap_state_dict_for_transfer(src, model, old_di, new_di)
    assert n == 0
    assert "other.weight" not in sd


def test_remap_normalizer_buffer_never_transferred() -> None:
    old_di = _di({"a": 0, "b": 1, "c": 2}, [0, 1, 2], [0, 1])
    new_di = _di({"a": 0, "x": 1, "b": 2, "c": 3}, [0, 1, 2, 3], [0, 2])
    # length matches an input count, but it's a normalization buffer -> must NOT be mapped
    src = {"model.residual_normalizer._std_tendency": torch.arange(3.0)}
    model = {"model.residual_normalizer._std_tendency": torch.zeros(4)}
    sd, n = remap_state_dict_for_transfer(src, model, old_di, new_di)
    assert n == 0
    assert "model.residual_normalizer._std_tendency" not in sd  # dropped -> new model's buffer kept


def test_remap_without_data_indices_falls_back_to_drop() -> None:
    # historical behaviour when data_indices is unavailable: mismatches are dropped
    src = {"tok.weight": _chan_marked(2, 3)}
    model = {"tok.weight": torch.zeros(2, 4, 1, 1)}
    sd, n = remap_state_dict_for_transfer(src, model, None, None)
    assert n == 0
    assert "tok.weight" not in sd
