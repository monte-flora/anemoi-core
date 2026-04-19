# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Spatial gradient loss for 2D Cartesian LAM grids.

Penalises the MSE between the finite-difference spatial-gradient maps of the
prediction vs. the target:

    G_loss = mean [ (d_x pred - d_x truth)^2 + (d_y pred - d_y truth)^2 ]

Simple forward differences with replicate padding are used so the gradient
maps have the same (H, W) as the input — faster and less smoothing than Sobel.
Complements MSE (pointwise) and MSH (spectral amplitude) in FastNet's recipe.
"""

import logging

import torch
import torch.nn.functional as F

from anemoi.training.losses.base import GraphCastBaseLoss
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


def _forward_diff_2d(field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward differences along H and W with replicate padding.

    Parameters
    ----------
    field : torch.Tensor
        Shape (..., H, W, V).

    Returns
    -------
    (dx, dy) : tuple[torch.Tensor, torch.Tensor]
        Both of shape (..., H, W, V).
    """
    # dx: difference along H (axis -3). Pad one row at the end of H.
    dx_raw = field[..., 1:, :, :] - field[..., :-1, :, :]  # (..., H-1, W, V)
    # dy: difference along W (axis -2). Pad one col at the end of W.
    dy_raw = field[..., :, 1:, :] - field[..., :, :-1, :]  # (..., H, W-1, V)

    # Replicate-pad one slice on the trailing edge of H / W so shape == (H, W).
    # F.pad on (..., H, W, V) with pad=(left_V, right_V, left_W, right_W, left_H, right_H)
    # pad only H-trailing for dx, only W-trailing for dy.
    dx = F.pad(dx_raw, (0, 0, 0, 0, 0, 1), mode="replicate")
    dy = F.pad(dy_raw, (0, 0, 0, 1, 0, 0), mode="replicate")
    return dx, dy


def spatial_gradient_squared_error(
    predicted_output: torch.Tensor,
    real_output: torch.Tensor,
    dims: tuple[int, int],
) -> torch.Tensor:
    r"""Per-cell squared error between forward-difference gradient maps.

    Returns a tensor of shape (..., x_dim, y_dim, variable) equal to
    ``(dx_pred - dx_truth)^2 + (dy_pred - dy_truth)^2``.
    """
    x_dim, y_dim = dims
    assert x_dim * y_dim == real_output.shape[TensorDim.GRID], (
        "The product of dims must match the spatial dims of the output. "
        "Please use x_dim and y_dim such that field_shape=(x_dim, y_dim)."
    )
    dims_total = (
        *real_output.shape[: TensorDim.GRID],
        x_dim,
        y_dim,
        real_output.shape[TensorDim.VARIABLE],
    )
    pred_2d = predicted_output.reshape(dims_total)
    real_2d = real_output.reshape(dims_total)

    dx_pred, dy_pred = _forward_diff_2d(pred_2d)
    dx_real, dy_real = _forward_diff_2d(real_2d)

    return (dx_pred - dx_real) ** 2 + (dy_pred - dy_real) ** 2


class SpatialGradientLoss(GraphCastBaseLoss):
    r"""Spatial gradient loss for 2D Cartesian LAM grids.

    Computes the mean squared error between forward-difference spatial
    gradient maps of prediction and target. Part of the FastNet recipe
    (MSE + MSH + SpatialGradient) — MSE supplies the pointwise signal,
    MSH the amplitude spectrum, and this loss the local derivative
    (edge/sharpness) signal.

    Parameters
    ----------
    x_dim : int
        X dimension of the 2D grid (must satisfy x_dim * y_dim == grid size).
    y_dim : int
        Y dimension of the 2D grid.
    ignore_nans : bool, optional
        Use nan-safe reductions in the parent class, by default False.
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        ignore_nans: bool = False,
    ) -> None:
        super().__init__(ignore_nans)
        LOGGER.warning(
            "SpatialGradientLoss can only be used with data on 2D grids.",
        )
        self.x_dim = x_dim
        self.y_dim = y_dim

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sq_err = spatial_gradient_squared_error(pred, target, dims=(self.x_dim, self.y_dim))
        return sq_err.reshape(pred.shape)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        **kwargs,  # noqa: ARG002
    ) -> torch.Tensor:
        result = super().forward(pred, target, squash, scaler_indices=scaler_indices, without_scalers=without_scalers)
        return torch.mean(result)


# Optional alias — Sobel is a common name in the literature even though we
# use plain forward differences (faster, less smoothing).
SobelLoss = SpatialGradientLoss
