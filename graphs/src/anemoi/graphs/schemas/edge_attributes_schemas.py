# (C) Copyright 2024-2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from enum import Enum
from typing import Literal

from pydantic import Field

from anemoi.graphs.schemas.normalise import ImplementedNormalisationSchema
from anemoi.utils.schemas import BaseModel


class ImplementedEdgeAttributeSchema(str, Enum):
    edge_length = "anemoi.graphs.edges.attributes.EdgeLength"
    edge_dirs = "anemoi.graphs.edges.attributes.EdgeDirection"
    directional_harmonics = "anemoi.graphs.edges.attributes.DirectionalHarmonics"
    azimuth = "anemoi.graphs.edges.attributes.Azimuth"
    gaussian_weights = "anemoi.graphs.edges.attributes.GaussianDistanceWeights"
    radial_basis_features = "anemoi.graphs.edges.attributes.RadialBasisFeatures"
    edge_relative_position_3d = "anemoi.graphs.edges.attributes.EdgeRelativePosition3D"
    edge_tangent_plane_position = "anemoi.graphs.edges.attributes.EdgeTangentPlanePosition"
    edge_grid_index_position = "anemoi.graphs.edges.attributes.EdgeGridIndexPosition"


class BaseEdgeAttributeSchema(BaseModel):
    target_: ImplementedEdgeAttributeSchema = Field(..., alias="_target_")
    "Edge attribute builder object from anemoi.graphs.edges.attributes"
    norm: ImplementedNormalisationSchema = Field(example="unit-std")
    "Normalisation method applied to the edge attribute."


class EdgeAttributeFromNodeSchema(BaseModel):
    target_: Literal[
        "anemoi.graphs.edges.attributes.AttributeFromSourceNode",
        "anemoi.graphs.edges.attributes.AttributeFromTargetNode",
    ] = Field(..., alias="_target_")
    "Edge attributes from node attribute"
    norm: ImplementedNormalisationSchema = Field(example="unit-std")
    "Normalisation method applied to the edge attribute."


class DirectionalHarmonicsSchema(BaseModel):
    target_: Literal["anemoi.graphs.edges.attributes.DirectionalHarmonics"] = Field(..., alias="_target_")
    "Directional harmonics from edge directions"
    order: int = Field(default=3, description="Maximum order of harmonics to compute")
    norm: ImplementedNormalisationSchema | None = Field(default=None, description="Normalization method")
    dtype: str = Field(default="float32", description="Data type for computations")


class RadialBasisFeaturesSchema(BaseModel):
    target_: Literal["anemoi.graphs.edges.attributes.RadialBasisFeatures"] = Field(..., alias="_target_")
    "Radial basis function features from edge distances"
    r_scale: float | None = Field(default=None, description="Global scale factor (None for adaptive per-node scaling)")
    centers: list[float] | None = Field(default=None, description="RBF center positions [0, 1]")
    sigma: float = Field(default=0.2, description="Width of Gaussian RBF functions")
    epsilon: float = Field(default=1e-10, description="Small constant to avoid division by zero")
    dtype: str = Field(default="float32", description="Data type for computations")


class EdgeRelativePosition3DSchema(BaseModel):
    target_: Literal["anemoi.graphs.edges.attributes.EdgeRelativePosition3D"] = Field(..., alias="_target_")
    "4D relative position features [||rel_pos||, rel_x, rel_y, rel_z] in receiver-local frame"
    norm: ImplementedNormalisationSchema = Field(example="unit-max")
    "Normalisation method applied across all 4 feature components."


class EdgeTangentPlanePositionSchema(BaseModel):
    target_: Literal["anemoi.graphs.edges.attributes.EdgeTangentPlanePosition"] = Field(..., alias="_target_")
    "3D location-invariant edge features [distance_km, dx_east_km, dy_north_km] in receiver's tangent plane"
    norm: ImplementedNormalisationSchema = Field(example="unit-max")
    "Normalisation method applied across all 3 feature components."


class EdgeGridIndexPositionSchema(BaseModel):
    target_: Literal["anemoi.graphs.edges.attributes.EdgeGridIndexPosition"] = Field(..., alias="_target_")
    "3D grid-index edge features [distance_pix, di, dj] — bit-invariant across patches"
    norm: ImplementedNormalisationSchema | None = Field(default="unit-max", example="unit-max")
    "Normalisation method (empirical, per-graph). Ignored if ``divisor`` is set."
    divisor: float | None = Field(default=None, example=4.5)
    "Fixed divisor in cell-units (overrides ``norm``). Pin for cross-graph stability."


EdgeAttributeSchema = (
    BaseEdgeAttributeSchema
    | EdgeAttributeFromNodeSchema
    | DirectionalHarmonicsSchema
    | RadialBasisFeaturesSchema
    | EdgeRelativePosition3DSchema
    | EdgeTangentPlanePositionSchema
    | EdgeGridIndexPositionSchema
)
