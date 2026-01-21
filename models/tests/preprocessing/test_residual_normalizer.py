# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing.normalizer import InputNormalizer
from anemoi.models.preprocessing.residual_normalizer import ResidualNormalizer


def _build_normalizers():
    config = DictConfig(
        {
            "data": {
                "normalizer": {"default": "mean-std"},
                "forcing": [],
                "diagnostic": [],
            },
        },
    )
    name_to_index = {"a": 0, "b": 1, "c": 2}
    statistics = {
        "mean": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "stdev": np.array([0.5, 0.25, 2.0], dtype=np.float32),
        "minimum": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "maximum": np.array([10.0, 10.0, 10.0], dtype=np.float32),
    }
    statistics_tendencies = {"stdev": np.array([0.2, 0.4, 0.8], dtype=np.float32)}

    data_indices = IndexCollection(config=config, name_to_index=name_to_index)
    input_normalizer = InputNormalizer(config=config.data.normalizer, data_indices=data_indices, statistics=statistics)
    residual_normalizer = ResidualNormalizer(data_indices=data_indices, statistics_tendencies=statistics_tendencies)
    return input_normalizer, residual_normalizer, data_indices


def test_residual_normalizer_normalized_space_matches_physical():
    input_normalizer, residual_normalizer, data_indices = _build_normalizers()

    torch.manual_seed(7)
    x_last_phys = torch.randn(2, 4, 3)
    y_true_phys = x_last_phys + torch.randn(2, 4, 3)

    x_last_norm = input_normalizer.transform(x_last_phys, in_place=False)
    y_true_norm = input_normalizer.transform(y_true_phys, in_place=False)

    Δx_norm_from_phys = residual_normalizer.transform(x_last_phys, y_true_phys, in_place=False)
    Δx_norm_from_norm = residual_normalizer.transform_from_normalized(
        x_last_norm,
        y_true_norm,
        input_normalizer._norm_mul,
        in_place=False,
    )

    assert torch.allclose(Δx_norm_from_phys, Δx_norm_from_norm, atol=1e-6, rtol=1e-6)

    y_pred_norm = residual_normalizer.inverse_transform_to_normalized(
        x_last_norm,
        Δx_norm_from_phys,
        input_normalizer._norm_mul,
        in_place=False,
    )
    y_true_norm_expected = input_normalizer.transform(y_true_phys, in_place=False)
    assert torch.allclose(y_pred_norm, y_true_norm_expected, atol=1e-6, rtol=1e-6)

    y_pred_phys = residual_normalizer.inverse_transform_physical_from_normalized(
        x_last_norm,
        Δx_norm_from_phys,
        input_normalizer._norm_mul,
        input_normalizer._norm_add,
        in_place=False,
    )
    assert torch.allclose(y_pred_phys, y_true_phys, atol=1e-6, rtol=1e-6)
