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
import pytest
import xarray as xr
from torch_geometric.data import HeteroData

from anemoi.graphs.nodes import LatLonNodes
from anemoi.graphs.nodes import LimitedAreaMPASNodes


def _write_netcdf(ds: xr.Dataset, path: Path) -> None:
    if importlib.util.find_spec("scipy") is not None:
        ds.to_netcdf(path, engine="scipy")
        return
    if importlib.util.find_spec("netCDF4") is not None:
        ds.to_netcdf(path, engine="netcdf4")
        return
    pytest.skip("scipy or netCDF4 is required to write netCDF for this test.")


def _make_isolated_mpas_dataset() -> xr.Dataset:
    n_cells = 4
    max_edges = 1

    lat_deg = np.zeros(n_cells, dtype=np.float64)
    lon_deg = np.arange(n_cells, dtype=np.float64)
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)

    x_cell = np.cos(lat_rad) * np.cos(lon_rad)
    y_cell = np.cos(lat_rad) * np.sin(lon_rad)
    z_cell = np.sin(lat_rad)

    cells_on_cell = np.zeros((n_cells, max_edges), dtype=np.int64)

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


def test_limited_area_mpas_nodes_masking(tmp_path: Path) -> None:
    ds = _make_isolated_mpas_dataset()
    mesh_path = tmp_path / "mpas_mesh.nc"
    _write_netcdf(ds, mesh_path)

    graph = HeteroData()
    graph = LatLonNodes(latitudes=[0.0], longitudes=[0.0], name="data").register_nodes(graph)

    node_builder = LimitedAreaMPASNodes(
        mpas_mesh_path=str(mesh_path),
        name="hidden",
        target_spacing_km=0.1,
        reference_node_name="data",
        margin_radius_km=10.0,
        seed=0,
    )

    graph = node_builder.update_graph(graph, {})

    assert graph["hidden"].x.shape[0] == 1
    assert graph["hidden"].node_type == "LimitedAreaMPASNodes"
    assert graph["hidden"]["_mpas_coarse_cells_on_cell"].shape[0] == 1
