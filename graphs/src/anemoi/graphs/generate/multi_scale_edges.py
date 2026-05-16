# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from abc import ABC
from abc import abstractmethod

from torch_geometric.data.storage import NodeStorage


class BaseIcosahedronEdgeStrategy(ABC):
    """Abstract base class for different edge-building strategies."""

    @abstractmethod
    def add_edges(self, nodes: NodeStorage, x_hops: int, scale_resolutions: list[int]) -> NodeStorage: ...


class TriNodesEdgeBuilder(BaseIcosahedronEdgeStrategy):
    """Edge builder for TriNodes and LimitedAreaTriNodes."""

    def add_edges(self, nodes: NodeStorage, x_hops: int, scale_resolutions: list[int]) -> NodeStorage:
        from anemoi.graphs.generate import tri_icosahedron

        nodes["_nx_graph"] = tri_icosahedron.add_edges_to_nx_graph(
            nodes["_nx_graph"],
            resolutions=scale_resolutions,
            x_hops=x_hops,
            area_mask_builder=nodes.get("_area_mask_builder", None),
        )
        return nodes


class HexNodesEdgeBuilder(BaseIcosahedronEdgeStrategy):
    """Edge builder for HexNodes and LimitedAreaHexNodes."""

    def add_edges(self, nodes: NodeStorage, x_hops: int, scale_resolutions: list[int]) -> NodeStorage:
        from anemoi.graphs.generate import hex_icosahedron

        nodes["_nx_graph"] = hex_icosahedron.add_edges_to_nx_graph(
            nodes["_nx_graph"],
            resolutions=scale_resolutions,
            x_hops=x_hops,
        )
        return nodes


class StretchedTriNodesEdgeBuilder(BaseIcosahedronEdgeStrategy):
    """Edge builder for StretchedTriNodes."""

    def add_edges(self, nodes: NodeStorage, x_hops: int, scale_resolutions: list[int]) -> NodeStorage:
        from anemoi.graphs.generate import tri_icosahedron
        from anemoi.graphs.generate.masks import KNNAreaMaskBuilder

        all_points_mask_builder = KNNAreaMaskBuilder("all_nodes", 1.0)
        all_points_mask_builder.fit_coords(nodes.x.numpy())

        nodes["_nx_graph"] = tri_icosahedron.add_edges_to_nx_graph(
            nodes["_nx_graph"],
            resolutions=scale_resolutions,
            x_hops=x_hops,
            area_mask_builder=all_points_mask_builder,
        )
        return nodes


class SquareNodesEdgeBuilder(BaseIcosahedronEdgeStrategy):
    """Edge builder for LimitedAreaSquareNodes.

    Unlike the icosphere strategy (which rebuilds a global ``trimesh.icosphere``
    at every resolution), this one reuses the pre-computed ``mesh_hierarchy``
    stored on the node builder. Because ``create_rectangular_mesh_hierarchy``
    enforces coarse cell counts that are integer divisors of the finest
    level, every coarse mesh vertex coincides with a fine mesh vertex — so
    the BallTree lookup from coarse vertex to fine hidden-node is exact and
    coarse edges are straight lines through fine vertices.
    """

    def add_edges(self, nodes: NodeStorage, x_hops: int, scale_resolutions: list[int]) -> NodeStorage:
        import numpy as np
        from sklearn.neighbors import BallTree

        from anemoi.graphs.generate.square_mesh import (
            get_neighbours_from_faces,
            mesh_vertices_to_latlon,
        )

        hierarchy = nodes["_mesh_hierarchy"]
        lat_range = nodes["_lat_range"]
        lon_range = nodes["_lon_range"]
        graph = nodes["_nx_graph"]
        area_mask_builder = nodes.get("_area_mask_builder", None)

        sorted_ids = sorted(graph.nodes)
        hidden_coords = np.stack([graph.nodes[nid]["hcoords_rad"] for nid in sorted_ids])
        sorted_to_id = {k: nid for k, nid in enumerate(sorted_ids)}

        # Mean spacing of the finest mesh -> tolerance for "coarse vertex
        # coincides with a fine hidden node". Coarse vertices outside the patch
        # get snapped far to the nearest boundary hidden node; those must be
        # excluded so coarse edges don't collapse onto the perimeter.
        fine_mesh = hierarchy[-1]
        x_unique = np.unique(fine_mesh.vertices[:, 0])
        y_unique = np.unique(fine_mesh.vertices[:, 1])
        dx_fine = (x_unique[-1] - x_unique[0]) / max(1, len(x_unique) - 1)
        dy_fine = (y_unique[-1] - y_unique[0]) / max(1, len(y_unique) - 1)
        # Convert fine spacing from grid-point units to radians via the stored
        # (lat, lon) mapping. Take the larger of the two axis spacings.
        lon_per_x = (lon_range[1] - lon_range[0]) / max(1.0, x_unique[-1] - x_unique[0])
        lat_per_y = (lat_range[1] - lat_range[0]) / max(1.0, y_unique[-1] - y_unique[0])
        snap_tol = 0.25 * max(dx_fine * lon_per_x, dy_fine * lat_per_y)

        tree = BallTree(hidden_coords, metric="haversine")

        for r in scale_resolutions:
            if r < 0 or r >= len(hierarchy):
                continue
            mesh = hierarchy[r]
            mesh_latlon = mesh_vertices_to_latlon(mesh.vertices, lat_range, lon_range)

            # Drop coarse vertices that fall outside the patch.
            if area_mask_builder is not None:
                in_patch = area_mask_builder.get_mask(mesh_latlon)
            else:
                in_patch = np.ones(len(mesh.vertices), dtype=bool)

            dists, idx = tree.query(mesh_latlon, k=1)
            mesh_to_hidden = idx[:, 0]
            near_hidden = dists[:, 0] <= snap_tol

            valid_mask = in_patch & near_hidden
            valid_mesh_vertices = np.flatnonzero(valid_mask).tolist()

            neighbours = get_neighbours_from_faces(
                mesh.faces, x_hops=x_hops, valid_nodes=valid_mesh_vertices,
            )
            for u_mesh, v_set in neighbours.items():
                u_hidden = sorted_to_id[int(mesh_to_hidden[u_mesh])]
                for v_mesh in v_set:
                    v_hidden = sorted_to_id[int(mesh_to_hidden[v_mesh])]
                    if u_hidden != v_hidden:
                        graph.add_edge(u_hidden, v_hidden)
                        graph.add_edge(v_hidden, u_hidden)

        nodes["_nx_graph"] = graph
        return nodes
