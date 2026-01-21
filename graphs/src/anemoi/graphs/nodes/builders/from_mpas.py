# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import numpy as np
import torch

from anemoi.graphs.generate.masks import KNNAreaMaskBuilder
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder

LOGGER = logging.getLogger(__name__)


class MPASCoarseNodes(BaseNodeBuilder):
    """Nodes built from a coarsened MPAS Voronoi mesh.

    The coarsening aggregates neighboring MPAS cells until a target area is reached.
    Coordinates are derived from area-weighted cell centroids.
    """

    def __init__(
        self,
        mpas_mesh_path: str,
        name: str,
        target_spacing_km: float,
        max_cells: int | None = None,
        seed: int = 0,
    ) -> None:
        self.mpas_mesh_path = mpas_mesh_path
        self.target_spacing_km = target_spacing_km
        self.max_cells = max_cells
        self.seed = seed
        super().__init__(name)
        self.hidden_attributes = BaseNodeBuilder.hidden_attributes | {
            "mpas_mesh_path",
            "target_spacing_km",
            "max_cells",
            "seed",
            "mpas_coarse_cells_on_cell",
            "mpas_coarse_n_edges",
            "mpas_fine_to_coarse",
        }

    def get_coordinates(self) -> torch.Tensor:
        import xarray as xr

        coords_rad, coarse_cells_on_cell, coarse_n_edges, fine_to_coarse = self._build_coarse_mesh_data(xr)
        self.mpas_coarse_cells_on_cell = coarse_cells_on_cell
        self.mpas_coarse_n_edges = coarse_n_edges
        self.mpas_fine_to_coarse = fine_to_coarse
        return torch.from_numpy(coords_rad)

    def _build_coarse_mesh_data(self, xr_module) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mesh_path = Path(self.mpas_mesh_path)
        assert mesh_path.exists(), f"{self.__class__.__name__}.mpas_mesh_path does not exist: {mesh_path}"

        with xr_module.open_dataset(mesh_path) as ds:
            if self.max_cells is not None:
                ds = ds.isel(nCells=slice(0, self.max_cells))

            for var_name in ("cellsOnCell", "areaCell", "xCell", "yCell", "zCell"):
                assert var_name in ds, f"Variable '{var_name}' not found in {mesh_path}"

            cells_on_cell = ds["cellsOnCell"].values.astype(np.int64)
            area_cell = ds["areaCell"].values.astype(np.float64)
            x_cell = ds["xCell"].values.astype(np.float64)
            y_cell = ds["yCell"].values.astype(np.float64)
            z_cell = ds["zCell"].values.astype(np.float64)

        neighbors = self._build_neighbors(cells_on_cell)
        coarse_xyz, coarse_cells_on_cell, coarse_n_edges, fine_to_coarse = self._coarsen_cells(
            area_cell, x_cell, y_cell, z_cell, neighbors
        )
        coarse_lat, coarse_lon = self._latlon_from_xyz(coarse_xyz[:, 0], coarse_xyz[:, 1], coarse_xyz[:, 2])
        coords_rad = np.stack([coarse_lat, coarse_lon], axis=-1)
        return coords_rad, coarse_cells_on_cell, coarse_n_edges, fine_to_coarse

    @staticmethod
    def _build_neighbors(cells_on_cell: np.ndarray) -> list[np.ndarray]:
        neighbors = []
        for row in cells_on_cell:
            row = row[row > 0] - 1
            neighbors.append(row.astype(np.int64))
        return neighbors

    def _coarsen_cells(
        self,
        area_cell: np.ndarray,
        x_cell: np.ndarray,
        y_cell: np.ndarray,
        z_cell: np.ndarray,
        neighbors: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Coarsen cells using BFS aggregation until target area is reached.

        The algorithm processes cells in random order. For each unassigned cell,
        it grows a cluster by BFS, stopping when the accumulated area reaches
        target_area. Cells are only marked as belonging to a cluster when they
        are actually processed (popped from queue), not when they are discovered.
        This ensures clusters don't exceed target_area by too much.
        """
        n_cells = area_cell.size
        target_area = (self.target_spacing_km * 1000.0) ** 2

        coarse_id = np.full(n_cells, -1, dtype=np.int64)
        coarse_xyz: list[list[float]] = []
        coarse_area: list[float] = []

        rng = np.random.default_rng(self.seed)
        order = np.arange(n_cells)
        rng.shuffle(order)

        current_id = 0
        for idx in order:
            if coarse_id[idx] != -1:
                continue

            total_area = 0.0
            sum_x = 0.0
            sum_y = 0.0
            sum_z = 0.0

            # Use a set to track cells we've added to queue (to avoid duplicates)
            # but only mark coarse_id when actually processing
            q = deque([idx])
            in_queue = {idx}

            while q and total_area < target_area:
                cell = q.popleft()

                # Skip if already assigned to another cluster (shouldn't happen with in_queue check)
                if coarse_id[cell] != -1:
                    continue

                # NOW mark as belonging to this cluster (when processing, not when discovering)
                coarse_id[cell] = current_id
                total_area += area_cell[cell]
                sum_x += x_cell[cell] * area_cell[cell]
                sum_y += y_cell[cell] * area_cell[cell]
                sum_z += z_cell[cell] * area_cell[cell]

                # Add unvisited neighbors to queue (but don't mark them yet)
                for nb in neighbors[cell]:
                    if nb < n_cells and coarse_id[nb] == -1 and nb not in in_queue:
                        q.append(nb)
                        in_queue.add(nb)

            coarse_xyz.append([sum_x / total_area, sum_y / total_area, sum_z / total_area])
            coarse_area.append(total_area)
            current_id += 1

        coarse_xyz = np.asarray(coarse_xyz, dtype=np.float64)
        coarse_cells_on_cell, coarse_n_edges = self._build_coarse_adjacency(coarse_id, neighbors, coarse_xyz.shape[0])
        return coarse_xyz, coarse_cells_on_cell, coarse_n_edges, coarse_id.astype(np.int64)

    @staticmethod
    def _build_coarse_adjacency(
        fine_to_coarse: np.ndarray,
        neighbors: list[np.ndarray],
        n_coarse: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        edge_set: set[tuple[int, int]] = set()
        for cell, coarse_id in enumerate(fine_to_coarse):
            for nb in neighbors[cell]:
                if nb >= fine_to_coarse.size:
                    continue
                nb_coarse = fine_to_coarse[nb]
                if coarse_id != nb_coarse:
                    edge_set.add((int(coarse_id), int(nb_coarse)))

        adj = [[] for _ in range(n_coarse)]
        for src, dst in edge_set:
            adj[src].append(dst)

        max_degree = max((len(x) for x in adj), default=0)
        coarse_cells_on_cell = np.full((n_coarse, max_degree), -1, dtype=np.int64)
        coarse_n_edges = np.zeros(n_coarse, dtype=np.int64)
        for i, nbs in enumerate(adj):
            coarse_n_edges[i] = len(nbs)
            if nbs:
                coarse_cells_on_cell[i, : len(nbs)] = nbs

        return coarse_cells_on_cell, coarse_n_edges

    @staticmethod
    def _latlon_from_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = np.sqrt(x * x + y * y + z * z)
        lat = np.arcsin(z / r)
        lon = np.arctan2(y, x)
        return lat, lon


class LimitedAreaMPASNodes(MPASCoarseNodes):
    """MPAS coarse nodes limited to an area of interest."""

    def __init__(
        self,
        mpas_mesh_path: str,
        name: str,
        target_spacing_km: float,
        reference_node_name: str,
        mask_attr_name: str | None = None,
        margin_radius_km: float = 100.0,
        max_cells: int | None = None,
        seed: int = 0,
    ) -> None:
        self.reference_node_name = reference_node_name
        self.mask_attr_name = mask_attr_name
        self.margin_radius_km = margin_radius_km
        super().__init__(mpas_mesh_path, name, target_spacing_km, max_cells=max_cells, seed=seed)
        self.area_mask_builder = KNNAreaMaskBuilder(reference_node_name, margin_radius_km, mask_attr_name)
        self.hidden_attributes = self.hidden_attributes | {
            "reference_node_name",
            "mask_attr_name",
            "margin_radius_km",
        }

    def register_nodes(self, graph):
        self.area_mask_builder.fit(graph)
        return super().register_nodes(graph)

    def get_coordinates(self) -> torch.Tensor:
        import xarray as xr

        coords_rad, coarse_cells_on_cell, coarse_n_edges, fine_to_coarse = self._build_coarse_mesh_data(xr)
        area_mask = self.area_mask_builder.get_mask(coords_rad)

        if area_mask.sum() == 0:
            raise ValueError(
                f"{self.__class__.__name__} produced an empty mask for {self.reference_node_name}."
            )

        LOGGER.info(
            "Limiting MPAS coarse nodes to %d of %d nodes within %.2f km.",
            int(area_mask.sum()),
            int(area_mask.size),
            float(self.margin_radius_km),
        )

        masked_coords, masked_cells_on_cell, masked_n_edges, masked_fine_to_coarse = self._apply_area_mask(
            coords_rad, coarse_cells_on_cell, coarse_n_edges, fine_to_coarse, area_mask
        )

        self.mpas_coarse_cells_on_cell = masked_cells_on_cell
        self.mpas_coarse_n_edges = masked_n_edges
        self.mpas_fine_to_coarse = masked_fine_to_coarse
        return torch.from_numpy(masked_coords)

    @staticmethod
    def _apply_area_mask(
        coords_rad: np.ndarray,
        coarse_cells_on_cell: np.ndarray,
        coarse_n_edges: np.ndarray,
        fine_to_coarse: np.ndarray,
        area_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mask_idx = np.flatnonzero(area_mask)
        old_to_new = np.full(coords_rad.shape[0], -1, dtype=np.int64)
        old_to_new[mask_idx] = np.arange(mask_idx.size, dtype=np.int64)

        edge_set: set[tuple[int, int]] = set()
        for old_src in mask_idx:
            new_src = old_to_new[old_src]
            for nb in coarse_cells_on_cell[old_src]:
                if nb < 0:
                    continue
                new_nb = old_to_new[nb]
                if new_nb >= 0:
                    edge_set.add((int(new_src), int(new_nb)))

        n_new = mask_idx.size
        adj = [[] for _ in range(n_new)]
        for src, dst in edge_set:
            adj[src].append(dst)

        max_degree = max((len(x) for x in adj), default=0)
        masked_cells_on_cell = np.full((n_new, max_degree), -1, dtype=np.int64)
        masked_n_edges = np.zeros(n_new, dtype=np.int64)
        for i, nbs in enumerate(adj):
            masked_n_edges[i] = len(nbs)
            if nbs:
                masked_cells_on_cell[i, : len(nbs)] = nbs

        masked_coords = coords_rad[mask_idx]

        masked_fine_to_coarse = np.full_like(fine_to_coarse, -1)
        for idx, old_id in enumerate(fine_to_coarse):
            new_id = old_to_new[old_id]
            if new_id >= 0:
                masked_fine_to_coarse[idx] = new_id

        return masked_coords, masked_cells_on_cell, masked_n_edges, masked_fine_to_coarse
