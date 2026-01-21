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
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.edges.builders.base import BaseEdgeBuilder

LOGGER = logging.getLogger(__name__)


class MPASTopologicalEdges(BaseEdgeBuilder):
    """Edges based on MPAS mesh topology.

    If mpas_mesh_path is provided, edges are built from cellsOnCell in the MPAS mesh file.
    Otherwise, adjacency is taken from MPASCoarseNodes hidden attributes.
    """

    def __init__(
        self,
        source_name: str,
        target_name: str,
        mpas_mesh_path: str | None = None,
        max_cells: int | None = None,
        source_mask_attr_name: str | None = None,
        target_mask_attr_name: str | None = None,
    ) -> None:
        self.mpas_mesh_path = mpas_mesh_path
        self.max_cells = max_cells
        super().__init__(source_name, target_name, source_mask_attr_name, target_mask_attr_name)

    def compute_edge_index(self, source_nodes: NodeStorage, target_nodes: NodeStorage) -> torch.Tensor:
        if self.source_name != self.target_name:
            raise ValueError(f"{self.__class__.__name__} requires source_name == target_name.")

        if self.mpas_mesh_path is not None:
            adjacency = self._read_mpas_cells_on_cell()
        else:
            if "_mpas_coarse_cells_on_cell" not in source_nodes:
                raise ValueError(
                    f"{self.__class__.__name__} requires mpas_mesh_path or MPASCoarseNodes with "
                    f"stored adjacency for '{self.source_name}'."
                )
            adjacency = source_nodes["_mpas_coarse_cells_on_cell"]
            if isinstance(adjacency, torch.Tensor):
                adjacency = adjacency.cpu().numpy()
            else:
                adjacency = np.asarray(adjacency)

        edge_index = self._edge_index_from_adjacency(adjacency)
        return edge_index

    def _read_mpas_cells_on_cell(self) -> np.ndarray:
        import xarray as xr

        mesh_path = Path(self.mpas_mesh_path)
        assert mesh_path.exists(), f"{self.__class__.__name__}.mpas_mesh_path does not exist: {mesh_path}"

        with xr.open_dataset(mesh_path) as ds:
            if self.max_cells is not None:
                ds = ds.isel(nCells=slice(0, self.max_cells))

            if "cellsOnCell" not in ds:
                raise ValueError(f"Variable 'cellsOnCell' not found in {mesh_path}")

            cells_on_cell = ds["cellsOnCell"].values.astype(np.int64)

        adjacency = cells_on_cell - 1
        adjacency[adjacency < 0] = -1
        return adjacency

    @staticmethod
    def _edge_index_from_adjacency(adjacency: np.ndarray) -> torch.Tensor:
        sources: list[int] = []
        targets: list[int] = []
        num_nodes = adjacency.shape[0]

        for src in range(num_nodes):
            for dst in adjacency[src]:
                if dst < 0 or dst >= num_nodes:
                    continue
                sources.append(src)
                targets.append(int(dst))

        if not sources:
            return torch.empty((2, 0), dtype=torch.int64)

        return torch.tensor([sources, targets], dtype=torch.int64)
