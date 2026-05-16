# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import ABC
from abc import abstractmethod

import torch
from torch_geometric.data.storage import NodeStorage
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import PairTensor
from torch_geometric.typing import Size
from torch_geometric.utils import scatter

from anemoi.graphs.edges.directional import compute_directions
from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian
from anemoi.graphs.normalise import NormaliserMixin
from anemoi.graphs.utils import NodesAxis
from anemoi.graphs.utils import get_distributed_device
from anemoi.graphs.utils import haversine_distance

LOGGER = logging.getLogger(__name__)


class BaseEdgeAttributeBuilder(MessagePassing, NormaliserMixin, ABC):
    """Base class for edge attribute builders."""

    node_attr_name: str = None
    norm_by_group: bool = False

    def __init__(self, norm: str | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.norm = norm
        self.dtype = dtype
        self.device = get_distributed_device()
        if self.node_attr_name is None:
            error_msg = f"Class {self.__class__.__name__} must define 'node_attr_name' either as a class attribute or in __init__"
            raise TypeError(error_msg)

    def subset_node_information(self, source_nodes: NodeStorage, target_nodes: NodeStorage) -> PairTensor:
        if self.node_attr_name in source_nodes:
            source_nodes_data = source_nodes[self.node_attr_name].to(self.device)
        else:
            source_nodes_data = None
            LOGGER.warning("The attribute %s is not in the source nodes.", self.node_attr_name)

        if self.node_attr_name in target_nodes:
            target_nodes_data = target_nodes[self.node_attr_name].to(self.device)
        else:
            target_nodes_data = None
            LOGGER.warning("The attribute %s is not in the target nodes.", self.node_attr_name)

        return source_nodes_data, target_nodes_data

    def forward(self, x: tuple[NodeStorage, NodeStorage], edge_index: Adj, size: Size = None) -> torch.Tensor:
        x = self.subset_node_information(*x)
        return self.propagate(edge_index.to(self.device), x=x, size=size)

    @abstractmethod
    def compute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor: ...

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        edge_features = self.compute(x_i, x_j)

        if edge_features.ndim == 1:
            edge_features = edge_features.unsqueeze(-1)

        return edge_features

    def aggregate(self, edge_features: torch.Tensor, index: torch.Tensor, ptr=None, dim_size=None) -> torch.Tensor:
        return self.normalise(edge_features, index, dim_size)


class BasePositionalBuilder(BaseEdgeAttributeBuilder, ABC):
    node_attr_name: str = "x"
    _idx_lat: int = 0
    _idx_lon: int = 1


class EdgeLength(BasePositionalBuilder):
    """Computes edge length for bipartite graphs."""

    def compute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        edge_length = haversine_distance(x_i, x_j)
        return edge_length


class EdgeDirection(BasePositionalBuilder):
    """Computes edge direction for bipartite graphs."""

    def compute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        edge_dirs = compute_directions(source_coords=x_j, target_coords=x_i)
        return edge_dirs


class DirectionalHarmonics(EdgeDirection):
    """Computes directional harmonics from edge directions.

    Builds directional harmonics [sin(mψ), cos(mψ)]_{m=1..order} from per-edge
    2D directions (dx, dy). Returns shape [N, 2*order].

    Attributes
    ----------
    order : int
        The maximum order of harmonics to compute.
    norm : str | None
        Normalisation method. Options: None, "l1", "l2", "unit-max", "unit-range", "unit-std".

    Methods
    -------
    compute(x_i, x_j)
        Compute directional harmonics from edge directions.
    """

    def __init__(self, order: int = 3, norm: str | None = None, dtype: str = "float32") -> None:
        self.order = order
        super().__init__(norm=norm, dtype=dtype)

    def compute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        # Get the 2D direction vectors [dx, dy]
        edge_dirs = compute_directions(source_coords=x_j, target_coords=x_i)

        # Compute the angle ψ from the direction vectors
        psi = torch.atan2(edge_dirs[:, 1], edge_dirs[:, 0])  # atan2(dy, dx)

        # Build harmonics: [sin(ψ), cos(ψ), sin(2ψ), cos(2ψ), ..., sin(order*ψ), cos(order*ψ)]
        harmonics = []
        for m in range(1, self.order + 1):
            harmonics.append(torch.sin(m * psi))
            harmonics.append(torch.cos(m * psi))

        # Stack into shape [N, 2*order]
        return torch.stack(harmonics, dim=1)


class Azimuth(BasePositionalBuilder):
    """Compute the azimuth of the edge.

    Attributes
    ----------
    norm : str | None
        Normalisation method. Options: None, "l1", "l2", "unit-max", "unit-range", "unit-std".
    invert : bool
        Whether to invert the edge lengths, i.e. 1 - edge_length. Defaults to False.

    Methods
    -------
    compute(x_i, x_j)
        Compute edge lengths attributes.

    References
    ----------
    - https://www.movable-type.co.uk/scripts/latlong.html
    """

    def compute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        # Forward bearing. x_i, x_j must be radians.
        a11 = torch.cos(x_i[:, self._idx_lat]) * torch.sin(x_j[:, self._idx_lat])
        a12 = (
            torch.sin(x_i[:, self._idx_lat])
            * torch.cos(x_j[:, self._idx_lat])
            * torch.cos(x_j[..., self._idx_lon] - x_i[..., self._idx_lon])
        )
        a1 = a11 - a12
        a2 = torch.sin(x_j[..., self._idx_lon] - x_i[..., self._idx_lon]) * torch.cos(x_j[:, self._idx_lat])
        edge_dirs = torch.atan2(a2, a1)

        return edge_dirs


class BaseBooleanEdgeAttributeBuilder(BaseEdgeAttributeBuilder, ABC):
    """Base class for boolean edge attributes."""

    def __init__(self) -> None:
        super().__init__(norm=None, dtype="bool")


class BaseEdgeAttributeFromNodeBuilder(BaseBooleanEdgeAttributeBuilder, ABC):
    """Base class for propagating an attribute from the nodes to the edges."""

    nodes_axis: NodesAxis | None = None

    def __init__(self, node_attr_name: str) -> None:
        self.node_attr_name = node_attr_name
        super().__init__()
        if self.nodes_axis is None:
            raise AttributeError(f"{self.__class__.__name__} class must set 'nodes_axis' attribute.")

    def compute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        node_attr = (x_j, x_i)[self.nodes_axis.value]
        assert (
            node_attr is not None
        ), f"The node attribute specified for {self.node_attr_name} cannot be found in the nodes."
        return node_attr


class AttributeFromSourceNode(BaseEdgeAttributeFromNodeBuilder):
    """Copy an attribute of the source node to the edge."""

    nodes_axis = NodesAxis.SOURCE


class AttributeFromTargetNode(BaseEdgeAttributeFromNodeBuilder):
    """Copy an attribute of the target node to the edge."""

    nodes_axis = NodesAxis.TARGET


class RadialBasisFeatures(EdgeLength):
    """Radial basis features from edge distances using Gaussian RBFs.

    Computes Gaussian radial basis function features from normalized great-circle distances:
    phi_r = [exp(-((α - c)/σ)²) for c in centers], where α = r_ij / r_scale.

    Provides RBF features via per-node adaptive scaling.
    By default, each destination node's edges are normalized by that node's maximum edge length.
    RBF features are normalized per target node per RBF center: within each RBF center,
    all edges pointing to the same target node have values that sum to 1 (L1 norm).

    Parameters
    ----------
    r_scale : float | None, optional
        Global scale factor for normalizing distances. Default is None.
        If None: Use per-node adaptive scaling (max edge length per destination node).
        If float: Use global scale for all nodes.
    centers : list of float, optional
        RBF center positions along normalized distance axis [0, 1].
        Default is [0.0, 0.25, 0.5, 0.75, 1.0].
    sigma : float, optional
        Width (standard deviation) of Gaussian RBF functions. Default is 0.2.
        Controls how localized each basis function is around its center.
    epsilon : float, optional
        Small constant to avoid division by zero. Default is 1e-10.
    dtype : str, optional
        Data type for computations. Default is "float32".

    Note
    ----
    RBF features are normalized per target node per RBF center.
    Within each RBF center, all edges to the same target node sum to 1.

    Methods
    -------
    compute(x_i, x_j)
        Compute raw edge distances (RBF computation happens in aggregate).
    aggregate(edge_features, index, ptr, dim_size)
        Compute RBF features with adaptive scaling and per-target-node normalization.

    Examples
    --------
    # Default: per-node adaptive scaling with grouped normalization
    rbf = RadialBasisFeatures()

    # To use global scale
    rbf_global = RadialBasisFeatures(r_scale=1.0)

    # Custom RBF centers and width
    rbf_custom = RadialBasisFeatures(centers=[0.0, 0.33, 0.67, 1.0], sigma=0.15)

    Notes
    -----
    - Closer edges → higher values at low-distance centers (0.0, 0.25)
    - Farther edges → higher values at high-distance centers (0.75, 1.0)
    """

    norm_by_group: bool = True  # normalise the RBF features per destination node

    def __init__(
        self,
        r_scale: float | None = None,
        centers: list[float] | None = None,
        sigma: float = 0.2,
        norm: str = "l1",
        epsilon: float = 1e-10,
        dtype: str = "float32",
    ) -> None:
        self.epsilon = epsilon
        self.r_scale = r_scale

        if self.r_scale is not None and self.r_scale < self.epsilon:
            LOGGER.warning(
                "r_scale (%f) is too small (< epsilon=%f). Clamping to epsilon to avoid division by zero.",
                self.r_scale,
                self.epsilon,
            )
            self.r_scale = self.epsilon

        self.centers = centers if centers is not None else [0.0, 0.25, 0.5, 0.75, 1.0]

        # Normalize centers if using global scaling
        if self.r_scale is not None:
            self.centers = [c / self.r_scale for c in self.centers]

        # Check that centers are in the range [0, 1]
        assert all(
            0.0 <= c <= 1.0 for c in self.centers
        ), f"RBF centers must be in range [0, 1] (or [0, r_scale] if r_scale is set). Got centers: {centers}, r_scale: {r_scale}"

        self.sigma = sigma
        super().__init__(norm=norm, dtype=dtype)

    def aggregate(self, edge_features: torch.Tensor, index: torch.Tensor, ptr=None, dim_size=None) -> torch.Tensor:
        """Aggregate edge features with per-node scaling and per-target-node normalization.

        Parameters
        ----------
        edge_features : torch.Tensor
            Raw edge distances, shape [num_edges] or [num_edges, 1]
        index : torch.Tensor
            Destination node index for each edge
        ptr : optional
            CSR pointer (not used)
        dim_size : int, optional
            Number of destination nodes

        Returns
        -------
        torch.Tensor
            RBF features, shape [num_edges, num_centers].
            Normalized per target node per RBF center .
        """
        # Ensure edge_features is 1D
        if edge_features.ndim == 2:
            edge_features = edge_features.squeeze(-1)

        # Compute scale factor per destination node
        if self.r_scale is None:
            # Per-node max edge length scaling
            max_dists = scatter(edge_features, index.long(), dim=0, dim_size=dim_size, reduce="max")

            # Clamp to epsilon to avoid division by zero
            max_dists = torch.clamp(max_dists, min=self.epsilon)

            # Broadcast to each edge
            scales = max_dists[index]
            alpha = edge_features / scales  # Normalized distance [0, 1]
        else:
            # Global scaling
            scales = torch.full_like(edge_features, self.r_scale)
            alpha = edge_features / scales  # Scaled distance [0, max_edge/r_scale]

        # Compute Gaussian RBF for each center
        rbf_features = []
        for center in self.centers:
            rbf = torch.exp(-(((alpha - center) / self.sigma) ** 2))
            rbf_features.append(rbf)

        rbf_features = torch.stack(rbf_features, dim=1)

        # Within each RBF center, normalise edges to the same target node
        rbf_features = self.normalise(rbf_features, index, dim_size)

        return rbf_features


class GaussianDistanceWeights(EdgeLength):
    """Gaussian distance weights."""

    norm_by_group: bool = True  # normalise the gaussian weights by target node

    def __init__(self, sigma: float = 1.0, norm: str = "l1", **kwargs) -> None:
        self.sigma = sigma
        super().__init__(norm=norm)

    def compute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        dists = super().compute(x_i, x_j)
        gaussian_weights = torch.exp(-(dists**2) / (2 * self.sigma**2))
        return gaussian_weights


class EdgeRelativePosition3D(BasePositionalBuilder):
    """Computes 4D edge features matching JAX/GraphCast convention.

    Returns [||rel_pos||, rel_x, rel_y, rel_z] where the relative position
    is computed in a receiver-local coordinate frame. The rotation places
    the receiver at the equator/prime meridian (1, 0, 0) on a unit sphere,
    matching JAX's ``get_bipartite_relative_position_in_receiver_local_coordinates``
    with ``relative_latitude_local_coordinates=True`` and
    ``relative_longitude_local_coordinates=True``.

    With ``norm="unit-max"``, all 4 features are divided by ``max(||rel_pos||)``
    across all edges, giving distance in [0, 1] and positions in [-1, 1].
    """

    def compute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        # Convert lat/lon (radians) to 3D Cartesian on unit sphere
        sender_xyz = latlon_rad_to_cartesian(x_j, 1.0)
        receiver_xyz = latlon_rad_to_cartesian(x_i, 1.0)

        lat = x_i[:, 0]
        lon = x_i[:, 1]
        cos_lat = torch.cos(lat)
        sin_lat = torch.sin(lat)
        cos_lon = torch.cos(lon)
        sin_lon = torch.sin(lon)

        def _apply_rotation(pos: torch.Tensor) -> torch.Tensor:
            x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
            rx = cos_lat * cos_lon * x + cos_lat * sin_lon * y + sin_lat * z
            ry = -sin_lon * x + cos_lon * y
            rz = -sin_lat * cos_lon * x - sin_lat * sin_lon * y + cos_lat * z
            return torch.stack([rx, ry, rz], dim=-1)

        sender_rotated = _apply_rotation(sender_xyz)
        receiver_rotated = _apply_rotation(receiver_xyz)

        rel_pos = sender_rotated - receiver_rotated
        distance = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)

        return torch.cat([distance, rel_pos], dim=-1)


class EdgeTangentPlanePosition(BasePositionalBuilder):
    """Location-invariant edge features in the receiver's local east/north tangent plane.

    For each edge (sender → receiver), returns a 3-D feature vector

        [distance_km, dx_east_km, dy_north_km]

    computed in the tangent plane at the receiver's lat/lon using the
    standard small-angle formulas

        dy_north = R * (lat_s - lat_r)
        dx_east  = R * cos(lat_r) * unwrap(lon_s - lon_r)
        distance = sqrt(dx_east^2 + dy_north^2)

    where ``R = 6371 km`` and ``unwrap`` maps Δlon to ``[-π, π]`` so
    antimeridian crossings do not blow up.

    Why this matters for template / patch graphs
    --------------------------------------------
    For patches sampled from a fixed regular km-spaced grid (e.g. HRRR 4 km
    LCC), two neighbouring cells are separated by the same local east/north
    offset regardless of where on Earth the patch sits. That means the
    (distance, dx_east, dy_north) computed here is *identical* across all
    patches for the same cell-index pair — so a graph built once from any
    template patch can be reused at any other location without retraining.

    This differs from :class:`EdgeRelativePosition3D`, which rotates into a
    receiver-local 3-D frame; that frame is approximately invariant for
    small patches but has O((L/R)^2) curvature error (~1% for L=1000 km).

    With ``norm="unit-max"``, all three features are divided by
    ``max(distance)`` across edges, giving distance in ``[0, 1]`` and
    positions in ``[-1, 1]``.
    """

    EARTH_RADIUS_KM: float = 6371.0

    def compute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        # x_i: receiver, x_j: sender; both [N, 2] in radians as (lat, lon).
        lat_r = x_i[:, self._idx_lat]
        lon_r = x_i[:, self._idx_lon]
        lat_s = x_j[:, self._idx_lat]
        lon_s = x_j[:, self._idx_lon]

        dlat = lat_s - lat_r
        two_pi = 2.0 * torch.pi
        dlon = torch.remainder(lon_s - lon_r + torch.pi, two_pi) - torch.pi

        dy_north = self.EARTH_RADIUS_KM * dlat
        dx_east = self.EARTH_RADIUS_KM * torch.cos(lat_r) * dlon
        distance = torch.sqrt(dx_east * dx_east + dy_north * dy_north)

        return torch.stack([distance, dx_east, dy_north], dim=-1)


class EdgeGridIndexPosition(BaseEdgeAttributeBuilder):
    """Location-invariant edge features in NWP grid-index coordinates.

    For each edge (sender → receiver), returns a 3-D feature vector

        [distance_pix, di, dj]

    where ``(i, j)`` is the integer/fractional row/column index of a node
    on the underlying NWP grid (``H × W``), not a physical distance. The
    sender/receiver ``(i, j)`` is provided by
    :class:`anemoi.graphs.nodes.attributes.GridIndexPosition`:

    * Data nodes inherit their own cell's ``(i, j)`` exactly.
    * Hidden / icosahedron mesh nodes get a bilinear-style ``(i, j)``
      via inverse distance weighting against the template lat/lon.

    Because ``(i, j)`` is the cell identity — not an Earth-surface offset
    — the resulting edge features are **bit-identical across patches**
    for patches slicing the same source grid at different CONUS locations.
    This is the cleanest representation for a template-graph approach
    where one graph is reused at every patch location.

    Parameters
    ----------
    norm : str | None, default "unit-max"
        Normalisation applied via ``NormaliserMixin`` from the empirical
        statistics of *this* graph's edge features. Convenient for a
        single-graph workflow but the divisor moves between graphs of
        different size — e.g. on an LCC-projected dataset the empirical
        max edge length in (i, j) cells grows with the latitude range
        the graph spans. A checkpoint trained on small patches and
        deployed on a larger inference domain therefore sees a shifted
        feature distribution. Pin ``divisor`` instead in that case.
    divisor : float | None, default None
        If set, divide all three channels (distance, di, dj) by this
        fixed value and ignore ``norm``. The divisor lives in the same
        cell-units as :meth:`compute` returns. For an encoder edge set
        with an 18 km physical cutoff and a nominally ~4 km/cell dataset
        the natural divisor is ``18.0 / 4.0 = 4.5``. Tying the divisor to
        the recipe rather than the empirical max of the graph makes edge
        features bit-identical across graphs of different sizes (same
        recipe → same divisor → same input distribution at training and
        at inference).
    """

    node_attr_name: str = "grid_ij"

    def __init__(
        self,
        norm: str | None = "unit-max",
        divisor: float | None = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__(norm=norm, dtype=dtype)
        if divisor is not None and float(divisor) <= 0:
            raise ValueError(f"divisor must be positive, got {divisor}")
        self.divisor = float(divisor) if divisor is not None else None
        if self.divisor is not None and self.norm not in (None, "unit-max"):
            LOGGER.info(
                "EdgeGridIndexPosition: explicit divisor=%g overrides norm=%r",
                self.divisor, self.norm,
            )

    def compute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        # x_i: receiver (target), x_j: sender (source); both [N, 2] as (i, j)
        di = x_j[:, 0] - x_i[:, 0]
        dj = x_j[:, 1] - x_i[:, 1]
        distance = torch.sqrt(di * di + dj * dj)
        return torch.stack([distance, di, dj], dim=-1)

    def aggregate(self, edge_features, index, ptr=None, dim_size=None):
        if self.divisor is not None:
            return edge_features / self.divisor
        return self.normalise(edge_features, index, dim_size)
