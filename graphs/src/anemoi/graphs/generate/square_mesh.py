"""Cartesian rectangular triangular mesh generation for LAM domains.

Adapted from JAX/GraphCast's square_mesh.py. Creates uniform grids with cell
centers and 4 isotropic triangles per cell in Cartesian grid-point space, then
maps vertices to lat/lon for use in Anemoi.

Multi-scale hierarchy uses linspace (coverage-first) so every resolution level
spans the full domain. Cell counts at the finest level are rounded to the
nearest multiple of 2^(resolution-1) so coarse levels have integer cell counts
and coarse gridlines nest exactly into fine gridlines.

Key functions:
  - rectangular_tri_mesh: builds a single mesh from linspace coordinates
  - create_rectangular_mesh_hierarchy: builds nested meshes at multiple levels
  - mesh_vertices_to_latlon: converts Cartesian vertices to lat/lon radians
  - get_neighbours_from_faces: builds neighbor dict from face connectivity
  - faces_to_edges / make_edges_bi_directional: face-to-edge conversion
"""

from typing import NamedTuple

import networkx as nx
import numpy as np


class TriangularMesh(NamedTuple):
    """A triangular mesh defined by vertices and triangular faces.

    Attributes
    ----------
    vertices : np.ndarray of shape (num_vertices, 2)
        Vertex positions in Cartesian grid-point space (x, y).
    faces : np.ndarray of shape (num_faces, 3)
        Integer indices into vertices defining each triangle.
    """

    vertices: np.ndarray
    faces: np.ndarray


def rectangular_tri_mesh(ncells_x: int, ncells_y: int,
                         x_min: float, x_max: float,
                         y_min: float, y_max: float,
                         include_cell_centers: bool = True) -> TriangularMesh:
    """Create a rectangular triangular mesh using linspace coordinates.

    Creates a uniform grid via linspace (guaranteeing coverage of [x_min, x_max]
    and [y_min, y_max]) and builds the triangulation.

    Two triangulation modes:

    * ``include_cell_centers=True`` (default): adds a vertex at each cell
      center and connects it to the 4 corners, producing 4 isotropic triangles
      per cell (8 edges per cell before dedup). Vertices live at both corner
      and half-step positions.
    * ``include_cell_centers=False``: splits each cell with a single
      bottom-left-to-top-right diagonal, producing 2 triangles per cell
      (5 edges per cell before dedup). Every vertex coincides exactly with
      a linspace corner — so when used with ``create_rectangular_mesh_hierarchy``
      and factor-2 doubling, **every coarse-level vertex is a vertex of every
      finer level** and coarse edges are straight unions of collinear fine
      edges (perfect multi-scale nesting, no off-grid spoke vertices).

    Parameters
    ----------
    ncells_x, ncells_y : int
        Number of cells in each direction.
    x_min, x_max, y_min, y_max : float
        Domain bounds in grid-point units.
    include_cell_centers : bool
        See mode description above.
    """
    x_coords = np.linspace(x_min, x_max, ncells_x + 1, dtype=np.float64)
    y_coords = np.linspace(y_min, y_max, ncells_y + 1, dtype=np.float64)

    n_x = len(x_coords)  # ncells_x + 1

    # Create 2D grid: vertices[j, i] at position (x_coords[i], y_coords[j])
    xx, yy = np.meshgrid(x_coords, y_coords)
    grid_vertices = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    n_grid = len(grid_vertices)

    jj, ii = np.meshgrid(np.arange(ncells_y), np.arange(ncells_x), indexing="ij")
    jj = jj.ravel()
    ii = ii.ravel()

    bl = jj * n_x + ii            # bottom-left
    br = jj * n_x + ii + 1        # bottom-right
    tl = (jj + 1) * n_x + ii      # top-left
    tr = (jj + 1) * n_x + ii + 1  # top-right

    if include_cell_centers:
        cx = (x_coords[:-1] + x_coords[1:]) / 2.0
        cy = (y_coords[:-1] + y_coords[1:]) / 2.0
        cxx, cyy = np.meshgrid(cx, cy)
        centers = np.stack([cxx.ravel(), cyy.ravel()], axis=-1)
        vertices = np.concatenate([grid_vertices, centers], axis=0)
        c = n_grid + jj * ncells_x + ii
        faces = np.stack([
            np.stack([bl, br, c], axis=-1),  # bottom
            np.stack([br, tr, c], axis=-1),  # right
            np.stack([tr, tl, c], axis=-1),  # top
            np.stack([tl, bl, c], axis=-1),  # left
        ], axis=1).reshape(-1, 3)
    else:
        # 2 triangles per cell, split by the BL -> TR diagonal (CCW).
        vertices = grid_vertices
        faces = np.stack([
            np.stack([bl, br, tr], axis=-1),  # lower-right triangle
            np.stack([bl, tr, tl], axis=-1),  # upper-left triangle
        ], axis=1).reshape(-1, 3)

    _ensure_ccw(vertices, faces)
    return TriangularMesh(vertices=vertices, faces=faces)


def _ensure_ccw(vertices: np.ndarray, faces: np.ndarray) -> None:
    """Verify all faces have counter-clockwise orientation (positive signed area)."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    if np.any(cross <= 0):
        n_bad = int(np.sum(cross <= 0))
        raise ValueError(f"{n_bad} faces have non-CCW orientation (of {len(faces)} total).")


def create_rectangular_mesh_hierarchy(ny: int, nx_: int, resolution: int,
                                      dx_finest: float, offset: int = 2,
                                      include_cell_centers: bool = True) -> list[TriangularMesh]:
    """Create nested mesh hierarchy with full domain coverage at every level.

    Uses linspace (coverage-first) so every level spans the full domain
    [offset, nx-offset-1] x [offset, ny-offset-1]. Cell counts at the finest
    level are rounded to the nearest multiple of 2^(resolution-1), so coarse
    levels have integer cell counts and coarse gridlines nest exactly into
    fine gridlines.

    Parameters
    ----------
    ny : int
        Data grid y-dimension.
    nx_ : int
        Data grid x-dimension.
    resolution : int
        Number of doubling levels (e.g., 4 → levels 0-3).
    dx_finest : float
        Target finest mesh spacing in data grid points (e.g., 4.6 for ~17km).
    offset : int
        Corner inset in grid points (default 2).

    Returns
    -------
    list of TriangularMesh
        Meshes ordered from coarsest (level 0) to finest (level resolution-1).
    """
    x_min = float(offset)
    x_max = float(nx_ - offset - 1)
    y_min = float(offset)
    y_max = float(ny - offset - 1)

    domain_x = x_max - x_min
    domain_y = y_max - y_min

    # Compute finest cell counts, rounded to nearest multiple of 2^(resolution-1)
    factor = 2 ** (resolution - 1)
    ncells_x_fine = max(factor, round(domain_x / dx_finest / factor) * factor)
    ncells_y_fine = max(factor, round(domain_y / dx_finest / factor) * factor)

    meshes = []
    for level in range(resolution):
        divisor = 2 ** (resolution - 1 - level)
        ncells_x = ncells_x_fine // divisor
        ncells_y = ncells_y_fine // divisor

        mesh = rectangular_tri_mesh(ncells_x, ncells_y, x_min, x_max, y_min, y_max,
                                    include_cell_centers=include_cell_centers)
        meshes.append(mesh)

    return meshes


def faces_to_edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert triangular faces to directed edge pairs.

    Parameters
    ----------
    faces : np.ndarray of shape (num_faces, 3)
        Triangular face indices.

    Returns
    -------
    senders : np.ndarray of shape (num_edges,)
    receivers : np.ndarray of shape (num_edges,)
        Directed edges from each triangle edge.
    """
    senders = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    receivers = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])

    edges = np.stack([senders, receivers], axis=-1)
    edges = np.unique(edges, axis=0)

    return edges[:, 0], edges[:, 1]


def make_edges_bi_directional(senders: np.ndarray, receivers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Make edge pairs bidirectional by adding reverse edges.

    Parameters
    ----------
    senders : np.ndarray
    receivers : np.ndarray

    Returns
    -------
    senders : np.ndarray
    receivers : np.ndarray
        Bidirectional edge pairs (deduplicated).
    """
    all_senders = np.concatenate([senders, receivers])
    all_receivers = np.concatenate([receivers, senders])

    edges = np.stack([all_senders, all_receivers], axis=-1)
    edges = np.unique(edges, axis=0)

    return edges[:, 0], edges[:, 1]


def mesh_vertices_to_latlon(vertices: np.ndarray, lat_range: tuple[float, float],
                            lon_range: tuple[float, float]) -> np.ndarray:
    """Convert Cartesian grid-point vertices to lat/lon in radians.

    Maps x-axis to longitude, y-axis to latitude via independent per-axis
    linear normalization.

    Parameters
    ----------
    vertices : np.ndarray of shape (N, 2)
        Vertex positions in Cartesian grid-point space. Column 0 = x (lon axis),
        column 1 = y (lat axis).
    lat_range : tuple of (lat_min_rad, lat_max_rad)
        Latitude bounds in radians.
    lon_range : tuple of (lon_min_rad, lon_max_rad)
        Longitude bounds in radians.

    Returns
    -------
    np.ndarray of shape (N, 2)
        Coordinates as [lat, lon] in radians (Anemoi convention).
    """
    x = vertices[:, 0]
    y = vertices[:, 1]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_norm = (x - x_min) / (x_max - x_min) if x_max > x_min else np.zeros_like(x)
    y_norm = (y - y_min) / (y_max - y_min) if y_max > y_min else np.zeros_like(y)

    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range

    lon = lon_min + (lon_max - lon_min) * x_norm
    lat = lat_min + (lat_max - lat_min) * y_norm

    return np.stack([lat, lon], axis=-1)


def get_neighbours_from_faces(faces: np.ndarray, x_hops: int,
                              valid_nodes: list[int] | None = None) -> dict[int, set[int]]:
    """Build neighbor dict from face connectivity, expanded by x_hops.

    Parameters
    ----------
    faces : np.ndarray of shape (num_faces, 3)
        Triangular face indices.
    x_hops : int
        Number of hops to expand neighborhoods.
    valid_nodes : list of int, optional
        Restrict to these node indices. If None, use all nodes in faces.

    Returns
    -------
    dict mapping node_idx -> set of neighbor indices (excluding self).
    """
    edges = set()
    for f in faces:
        for i in range(3):
            edges.add((int(f[i]), int(f[(i + 1) % 3])))
            edges.add((int(f[(i + 1) % 3]), int(f[i])))

    graph = nx.from_edgelist(list(edges))

    if valid_nodes is None:
        valid_nodes = list(graph.nodes)

    return {
        i: set(nx.ego_graph(graph, i, radius=x_hops, center=False))
        for i in valid_nodes
        if i in graph
    }
