# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import importlib.util
from pathlib import Path

import numpy as np
import torch
import pytest
import xarray as xr
from torch_geometric.data import HeteroData

from anemoi.graphs.edges import MPASTopologicalEdges
from anemoi.graphs.nodes import MPASCoarseNodes


def _write_netcdf(ds: xr.Dataset, path: Path) -> None:
    if importlib.util.find_spec("scipy") is not None:
        ds.to_netcdf(path, engine="scipy")
        return
    if importlib.util.find_spec("netCDF4") is not None:
        ds.to_netcdf(path, engine="netcdf4")
        return
    pytest.skip("scipy or netCDF4 is required to write netCDF for this test.")


def _make_mpas_dataset() -> xr.Dataset:
    n_cells = 4
    max_edges = 2

    lat_deg = np.zeros(n_cells, dtype=np.float64)
    lon_deg = np.arange(n_cells, dtype=np.float64)
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)

    x_cell = np.cos(lat_rad) * np.cos(lon_rad)
    y_cell = np.cos(lat_rad) * np.sin(lon_rad)
    z_cell = np.sin(lat_rad)

    cells_on_cell = np.array(
        [
            [2, 0],
            [1, 3],
            [2, 4],
            [3, 0],
        ],
        dtype=np.int64,
    )

    return xr.Dataset(
        data_vars={
            "cellsOnCell": (("nCells", "maxEdges"), cells_on_cell),
            "areaCell": (("nCells",), np.ones(n_cells, dtype=np.float64)),
            "xCell": (("nCells",), x_cell),
            "yCell": (("nCells",), y_cell),
            "zCell": (("nCells",), z_cell),
        },
        coords={"nCells": np.arange(n_cells), "maxEdges": np.arange(max_edges)},
    )


def _edge_set(edge_index) -> set[tuple[int, int]]:
    return {(int(src), int(dst)) for src, dst in edge_index.T}


def test_mpas_topological_edges_from_coarse_nodes(tmp_path: Path) -> None:
    ds = _make_mpas_dataset()
    mesh_path = tmp_path / "mpas_mesh.nc"
    _write_netcdf(ds, mesh_path)

    graph = HeteroData()
    node_builder = MPASCoarseNodes(
        mpas_mesh_path=str(mesh_path),
        name="coarse",
        target_spacing_km=0.002,
        seed=0,
    )
    graph = node_builder.update_graph(graph, {})

    edge_builder = MPASTopologicalEdges(source_name="coarse", target_name="coarse")
    graph = edge_builder.update_graph(graph, {})

    adjacency = graph["coarse"]["_mpas_coarse_cells_on_cell"]
    if hasattr(adjacency, "numpy"):
        adjacency = adjacency.numpy()
    expected_edges = _edge_set(MPASTopologicalEdges._edge_index_from_adjacency(adjacency))
    actual_edges = _edge_set(graph["coarse", "to", "coarse"].edge_index)

    assert actual_edges == expected_edges


def test_mpas_topological_edges_from_mesh(tmp_path: Path) -> None:
    ds = _make_mpas_dataset()
    mesh_path = tmp_path / "mpas_mesh.nc"
    _write_netcdf(ds, mesh_path)

    graph = HeteroData()
    graph["data"].x = torch.zeros((4, 2), dtype=torch.float32)
    graph["data"].node_type = "LatLonNodes"

    edge_builder = MPASTopologicalEdges(source_name="data", target_name="data", mpas_mesh_path=str(mesh_path))
    graph = edge_builder.update_graph(graph, {})

    expected_adjacency = ds["cellsOnCell"].values.astype(np.int64) - 1
    expected_adjacency[expected_adjacency < 0] = -1
    expected_edges = _edge_set(MPASTopologicalEdges._edge_index_from_adjacency(expected_adjacency))
    actual_edges = _edge_set(graph["data", "to", "data"].edge_index)

    assert actual_edges == expected_edges
