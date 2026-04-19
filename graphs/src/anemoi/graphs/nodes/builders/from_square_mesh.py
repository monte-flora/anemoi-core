"""Node builder for Cartesian rectangular triangular meshes.

Creates uniform square meshes at multiple resolutions for LAM domains,
providing perfectly uniform encoder neighborhoods (unlike icosahedral meshes
which have position-dependent neighborhood sizes).

The finest mesh defines the graph nodes; coarser meshes provide multi-scale
processor edges via BallTree nearest-neighbor mapping.
"""

import logging

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.generate.square_mesh import (
    create_rectangular_mesh_hierarchy,
    mesh_vertices_to_latlon,
)
from anemoi.graphs.generate.utils import get_coordinates_ordering
from anemoi.graphs.nodes.builders.from_refined_icosahedron import LimitedAreaIcosahedralNodes

LOGGER = logging.getLogger(__name__)


class LimitedAreaSquareNodes(LimitedAreaIcosahedralNodes):
    """Nodes from a Cartesian rectangular triangular mesh for LAM domains.

    Creates multiple independent RectangularTriMesh instances at doubling
    grid spacings (e.g., dx = [40, 20, 10, 5] grid points). The finest mesh
    defines the graph nodes; coarser meshes are stored for multi-scale edge
    building.

    This provides perfectly uniform spacing and identical encoder neighborhoods
    for every mesh node, unlike icosahedral meshes which have 3.5x variation
    in neighborhood sizes on projected LAM domains.

    Parameters
    ----------
    resolution : int
        Number of doubling levels. E.g., 4 produces dx_values = [40, 20, 10, 5]
        when dx_finest=5.
    reference_node_name : str
        Name of the data grid nodes (e.g., "data") for extracting domain bounds.
    nx_grid : int
        Data grid x-dimension (longitude axis).
    ny_grid : int
        Data grid y-dimension (latitude axis).
    dx_finest : float
        Target finest mesh spacing in data grid points (default: 5).
    offset : int
        Corner inset in grid points (default: 2).
    margin_radius_km : float
        Passed to parent for area masking (default: 100.0).
    mask_attr_name : str or None
        Passed to parent for optional mask attribute (default: None).
    """

    multi_scale_edge_cls: str = "anemoi.graphs.generate.multi_scale_edges.SquareNodesEdgeBuilder"

    def __init__(
        self,
        resolution: int,
        reference_node_name: str,
        name: str,
        nx_grid: int,
        ny_grid: int,
        dx_finest: float = 5,
        offset: int = 2,
        margin_radius_km: float = 100.0,
        mask_attr_name: str | None = None,
    ) -> None:
        self.nx_grid = nx_grid
        self.ny_grid = ny_grid
        self.offset = offset
        self.dx_finest = dx_finest
        self._resolution = resolution

        super().__init__(
            resolution=resolution,
            reference_node_name=reference_node_name,
            name=name,
            mask_attr_name=mask_attr_name,
            margin_radius_km=margin_radius_km,
        )

        # Override resolutions to be 0-indexed list matching mesh_hierarchy indices
        self.resolutions = list(range(resolution))

        # Add mesh_hierarchy and domain bounds to hidden attributes
        self.hidden_attributes = self.hidden_attributes | {"mesh_hierarchy", "lat_range", "lon_range"}

    def register_nodes(self, graph: HeteroData) -> None:
        """Register nodes, extracting domain bounds from reference grid first."""
        self.area_mask_builder.fit(graph)

        # Extract full domain lat/lon bounds from reference data nodes.
        # Uses min/max to work with both regular and unstructured grids.
        # The mesh covers [offset, nx-offset-1] in abstract grid-point space,
        # which gets mapped to the full [lat_min, lat_max] x [lon_min, lon_max]
        # range via mesh_vertices_to_latlon normalization.
        ref_coords_rad = graph[self.area_mask_builder.reference_node_name].x.cpu().numpy()
        self.lat_range = (float(ref_coords_rad[:, 0].min()), float(ref_coords_rad[:, 0].max()))
        self.lon_range = (float(ref_coords_rad[:, 1].min()), float(ref_coords_rad[:, 1].max()))

        LOGGER.info(
            "LimitedAreaSquareNodes domain bounds: lat=[%.4f, %.4f] lon=[%.4f, %.4f] rad "
            "(offset=%d, grid %dx%d, %d data nodes)",
            self.lat_range[0], self.lat_range[1], self.lon_range[0], self.lon_range[1],
            self.offset, self.nx_grid, self.ny_grid, len(ref_coords_rad),
        )

        # Now call grandparent register_nodes (skipping LimitedAreaIcosahedralNodes
        # which would call area_mask_builder.fit again)
        from anemoi.graphs.nodes.builders.base import BaseNodeBuilder
        return BaseNodeBuilder.register_nodes(self, graph)

    def create_nodes(self) -> tuple[nx.DiGraph, np.ndarray, list[int]]:
        """Create graph nodes from finest-resolution rectangular mesh.

        Returns
        -------
        nx_graph : nx.DiGraph
            Graph with nodes having 'hcoords_rad' attribute.
        coords_rad : np.ndarray of shape (N, 2)
            Node coordinates as [lat, lon] in radians.
        node_ordering : list of int
            Indices sorting nodes by latitude then longitude.
        """
        # Build full mesh hierarchy (coarsest → finest)
        self.mesh_hierarchy = create_rectangular_mesh_hierarchy(
            ny=self.ny_grid,
            nx_=self.nx_grid,
            resolution=self._resolution,
            dx_finest=self.dx_finest,
            offset=self.offset,
        )

        # Finest mesh is last in hierarchy
        finest_mesh = self.mesh_hierarchy[-1]

        LOGGER.info(
            "Square mesh hierarchy: %d levels, dx_finest=%d",
            len(self.mesh_hierarchy), self.dx_finest,
        )
        for i, mesh in enumerate(self.mesh_hierarchy):
            dx_actual = (mesh.vertices[:, 0].max() - mesh.vertices[:, 0].min()) / max(1, len(np.unique(mesh.vertices[:, 0])) - 1)
            LOGGER.info(
                "  Level %d: %d vertices, %d faces, ~%.1f pt spacing",
                i, len(mesh.vertices), len(mesh.faces), dx_actual,
            )

        # Convert finest-level vertices to lat/lon radians
        coords_rad = mesh_vertices_to_latlon(
            finest_mesh.vertices, self.lat_range, self.lon_range
        )

        # Get coordinate ordering (sort by lat then lon, same as icosahedral)
        node_ordering = get_coordinates_ordering(coords_rad)

        # Apply area mask if margin is tight
        if self.area_mask_builder is not None:
            area_mask = self.area_mask_builder.get_mask(coords_rad)
            node_ordering = node_ordering[area_mask[node_ordering]]

        LOGGER.info(
            "LimitedAreaSquareNodes: %d nodes after area masking (of %d total)",
            len(node_ordering), len(coords_rad),
        )

        # Create nx.DiGraph with hcoords_rad attributes
        nx_graph = nx.DiGraph()
        for i, coords in enumerate(coords_rad[node_ordering]):
            node_id = node_ordering[i]
            nx_graph.add_node(node_id, hcoords_rad=coords)

        return nx_graph, coords_rad, list(node_ordering)
