# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import numpy as np
import torch
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.nodes.attributes.base_attributes import BaseNodeAttribute

LOGGER = logging.getLogger(__name__)


class GridIndexPosition(BaseNodeAttribute):
    """Grid-index (i, j) position of each node on a template NWP grid.

    Given a reference Anemoi zarr whose ``latitudes``/``longitudes`` describe
    a single 2-D (``H``, ``W``) patch in row-major order, this attribute
    assigns every node a fractional (i, j) coordinate in ``[0, H] × [0, W]``:

    * **Data nodes** coincide with template cells, so the result is
      effectively integer ``(k // W, k % W)``.
    * **Hidden / icosahedron mesh nodes** lie at arbitrary (lat, lon)
      inside the patch. We locate each such node by KNN-averaging the
      (i, j) of its ``n_neighbours`` nearest template cells in lat/lon,
      weighted by inverse distance — giving a smooth inverse of the
      template's (i, j) → (lat, lon) map.

    The resulting coordinate is **identical across patches** regardless of
    where on Earth the patch sits, because ``(i, j)`` is the cell identity
    rather than a physical distance. Edge attributes built from ``Δi``,
    ``Δj`` are therefore trivially location-invariant.

    Parameters
    ----------
    field_shape : list[int]
        ``[H, W]`` of the template NWP grid.
    reference_dataset : str
        Path to an anemoi zarr store whose ``latitudes``/``longitudes``
        define the (i, j) ↔ (lat, lon) mapping. Typically the same zarr
        used by the graph's data nodes.
    n_neighbours : int, default 4
        How many nearest template cells to average when interpolating a
        node's (i, j). 1 → nearest-neighbour (integer output for data
        nodes, coarse for hidden); 4 → bilinear-ish.
    """

    def __init__(
        self,
        field_shape: list[int] | tuple[int, int],
        reference_dataset: str,
        n_neighbours: int = 4,
        norm: str | None = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__(norm=norm, dtype=dtype)
        H, W = field_shape
        self.H = int(H)
        self.W = int(W)
        self.reference_dataset = reference_dataset
        self.n_neighbours = int(n_neighbours)

    @staticmethod
    def _wrap_pi(lon_rad: np.ndarray) -> np.ndarray:
        """Wrap radians to the interval [-π, π]."""
        return (lon_rad + np.pi) % (2.0 * np.pi) - np.pi

    def _load_template(self) -> tuple[np.ndarray, np.ndarray]:
        import zarr  # local import to keep anemoi.graphs lean

        z = zarr.open(self.reference_dataset, mode="r")
        tpl_lat_deg = np.asarray(z["latitudes"][:])
        tpl_lon_deg = np.asarray(z["longitudes"][:])
        n_expected = self.H * self.W
        assert tpl_lat_deg.shape[0] == n_expected, (
            f"reference_dataset has {tpl_lat_deg.shape[0]} cells, "
            f"expected H*W = {self.H}*{self.W} = {n_expected}"
        )
        tpl_lat_rad = np.radians(tpl_lat_deg)
        tpl_lon_rad = self._wrap_pi(np.radians(tpl_lon_deg))
        return tpl_lat_rad, tpl_lon_rad

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        try:
            from scipy.spatial import cKDTree
        except ImportError as exc:
            raise ImportError(
                "GridIndexPosition requires scipy (scipy.spatial.cKDTree). "
                "Install scipy to use this node attribute."
            ) from exc

        tpl_lat_rad, tpl_lon_rad = self._load_template()
        tpl_pts = np.stack([tpl_lat_rad, tpl_lon_rad], axis=-1)

        node_lat = nodes.x[:, 0].detach().cpu().numpy().astype(np.float64)
        node_lon = self._wrap_pi(nodes.x[:, 1].detach().cpu().numpy().astype(np.float64))
        nod_pts = np.stack([node_lat, node_lon], axis=-1)

        tree = cKDTree(tpl_pts)
        k = min(self.n_neighbours, tpl_pts.shape[0])
        if k == 1:
            _, nn_idx = tree.query(nod_pts, k=1)
            tpl_i = (nn_idx // self.W).astype(np.float64)
            tpl_j = (nn_idx % self.W).astype(np.float64)
        else:
            dists, nn_idx = tree.query(nod_pts, k=k)  # [N, k]
            weights = 1.0 / np.maximum(dists, 1e-10)
            weights /= weights.sum(axis=1, keepdims=True)
            tpl_i = ((nn_idx // self.W) * weights).sum(axis=1)
            tpl_j = ((nn_idx % self.W) * weights).sum(axis=1)

        grid_ij = np.stack([tpl_i, tpl_j], axis=-1).astype(np.float32)
        LOGGER.info(
            "GridIndexPosition: %d nodes → (i, j) ∈ [%.2f, %.2f] × [%.2f, %.2f] "
            "(template %dx%d)",
            grid_ij.shape[0],
            grid_ij[:, 0].min(),
            grid_ij[:, 0].max(),
            grid_ij[:, 1].min(),
            grid_ij[:, 1].max(),
            self.H,
            self.W,
        )
        return torch.from_numpy(grid_ij)
