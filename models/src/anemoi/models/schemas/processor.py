# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from typing import Any
from typing import Literal

from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import NonNegativeInt
from pydantic import model_validator

from .common_components import GNNModelComponent
from .common_components import PointWiseModelComponent
from .common_components import TransformerModelComponent


class GNNProcessorSchema(GNNModelComponent):
    target_: Literal["anemoi.models.layers.processor.GNNProcessor"] = Field(..., alias="_target_")
    "GNN Processor object from anemoi.models.layers.processor."
    num_layers: NonNegativeInt = Field(example=16)
    "Number of layers of GNN processor. Default to 16."
    num_chunks: NonNegativeInt = Field(example=2)
    "Number of chunks to divide the layer into. Default to 2."


class GraphInteractionNetProcessorSchema(GNNModelComponent):
    """GraphCast-style GNN Processor with correct Interaction Network residual pattern.

    This implements the Interaction Network architecture (Battaglia et al. 2018)
    where residuals are applied AFTER edge and node updates, not inside the
    message function. This prevents gradient collapse in deep networks.
    """

    target_: Literal["anemoi.models.layers.processor.GraphInteractionNetProcessor"] = Field(..., alias="_target_")
    "GraphInteractionNet Processor object from anemoi.models.layers.processor."
    num_layers: NonNegativeInt = Field(example=16)
    "Number of layers of GraphInteractionNet processor. Default to 16."
    num_chunks: NonNegativeInt = Field(example=2)
    "Number of chunks to divide the layer into. Default to 2."
    aggr_reduce: Literal["sum", "mean", "max", "min", "add"] = Field(default="sum")
    "Aggregation op for incoming edge deltas at each processor block. 'sum' = GraphCast default (preserves sparse extremes); 'mean' = low-pass across neighbors (for the sum→mean inverse test)."


class GraphTransformerProcessorSchema(TransformerModelComponent):
    target_: Literal["anemoi.models.layers.processor.GraphTransformerProcessor"] = Field(..., alias="_target_")
    "Graph transformer processor object from anemoi.models.layers.processor."
    trainable_size: NonNegativeInt = Field(example=8)
    "Size of trainable parameters vector. Default to 8."
    sub_graph_edge_attributes: list[str] = Field(example=["edge_length", "edge_dir"])
    "Edge attributes to consider in the processor features. Default [edge_length, endge_dirs]."
    num_layers: NonNegativeInt = Field(example=16)
    "Number of layers of Graph Transformer processor. Default to 16."
    num_chunks: NonNegativeInt = Field(example=2)
    "Number of chunks to divide the layer into. Default to 2."
    qk_norm: bool = Field(example=False)
    "Normalize the query and key vectors. Default to False."

    @model_validator(mode="after")
    def check_valid_extras(self) -> Any:
        # This is a check to allow backwards compatibilty of the configs, as the extra fields are not required.
        allowed_extras = {"graph_attention_backend": str, "edge_pre_mlp": bool}
        extras = getattr(self, "__pydantic_extra__", {}) or {}
        for extra_field, value in extras.items():
            if extra_field not in allowed_extras:
                msg = f"Extra field '{extra_field}' is not allowed. Allowed fields are: {list(allowed_extras.keys())}."
                raise ValueError(msg)
            if not isinstance(value, allowed_extras[extra_field]):
                msg = f"Extra field '{extra_field}' must be of type {allowed_extras[extra_field].__name__}."
                raise TypeError(msg)

        return self


class TransformerProcessorSchema(TransformerModelComponent):
    target_: Literal["anemoi.models.layers.processor.TransformerProcessor"] = Field(..., alias="_target_")
    "Transformer processor object from anemoi.models.layers.processor."
    num_layers: NonNegativeInt = Field(example=16)
    "Number of layers of Transformer processor. Default to 16."
    num_chunks: NonNegativeInt = Field(example=2)
    "Number of chunks to divide the layer into. Default to 2."
    window_size: NonNegativeInt = Field(example=512)
    "Attention window size along the longitude axis. Default to 512."
    dropout_p: NonNegativeFloat = Field(example=0.0)
    "Dropout probability used for multi-head self attention, default 0.0"
    attention_implementation: str = Field(example="flash_attention")
    "Attention implementation to use. Default to 'flash_attention'."
    qk_norm: bool = Field(example=False)
    "Normalize the query and key vectors. Default to False."
    softcap: NonNegativeFloat = Field(example=0.0)
    "Softcap value for attention. Default to 0.0."
    use_alibi_slopes: bool = Field(example=False)
    "Use alibi slopes for attention implementation. Default to False."

    @model_validator(mode="after")
    def check_valid_extras(self) -> Any:
        # Check for valid extra fields related to MultiHeadSelfAttention and MultiHeadCrossAttention
        # This is a check to allow backwards compatibilty of the configs, as the extra fields are not required.
        allowed_extras = {"use_rotary_embeddings": bool}
        extras = getattr(self, "__pydantic_extra__", {}) or {}
        for extra_field, value in extras.items():
            if extra_field not in allowed_extras:
                msg = f"Extra field '{extra_field}' is not allowed. Allowed fields are: {list(allowed_extras.keys())}."
                raise ValueError(msg)
            if not isinstance(value, allowed_extras[extra_field]):
                msg = f"Extra field '{extra_field}' must be of type {allowed_extras[extra_field].__name__}."
                raise TypeError(msg)

        return self


class BandedTransformerProcessorSchema(TransformerModelComponent):
    target_: Literal["anemoi.models.layers.processor.BandedTransformerProcessor"] = Field(..., alias="_target_")
    "Banded Transformer processor with graph-aware sparse attention."
    num_layers: NonNegativeInt = Field(example=16)
    "Number of layers of Banded Transformer processor. Default to 16."
    num_chunks: NonNegativeInt = Field(example=2)
    "Number of chunks to divide the layer into. Default to 2."
    window_size: NonNegativeInt = Field(example=128)
    "Attention window size in permuted sequence space. Should be >= k-hop bandwidth after RCM. Default to 128."
    dropout_p: NonNegativeFloat = Field(example=0.0)
    "Dropout probability used for multi-head self attention, default 0.0"
    attention_implementation: str = Field(example="flash_attention")
    "Attention implementation to use. Default to 'flash_attention'."
    qk_norm: bool = Field(example=False)
    "Normalize the query and key vectors. Default to False."
    softcap: NonNegativeFloat = Field(example=0.0)
    "Softcap value for attention. Default to 0.0."
    use_alibi_slopes: bool = Field(example=False)
    "Use alibi slopes for attention implementation. Default to False."
    sub_graph_edge_attributes: list[str] = Field(example=["edge_length", "edge_dirs"])
    "Edge attributes used for graph topology (required for RCM permutation computation)."

    @model_validator(mode="after")
    def check_valid_extras(self) -> Any:
        allowed_extras = {"use_rotary_embeddings": bool, "trainable_size": int}
        extras = getattr(self, "__pydantic_extra__", {}) or {}
        for extra_field, value in extras.items():
            if extra_field not in allowed_extras:
                msg = f"Extra field '{extra_field}' is not allowed. Allowed fields are: {list(allowed_extras.keys())}."
                raise ValueError(msg)
            if not isinstance(value, allowed_extras[extra_field]):
                msg = f"Extra field '{extra_field}' must be of type {allowed_extras[extra_field].__name__}."
                raise TypeError(msg)

        return self


class PointWiseMLPProcessorSchema(PointWiseModelComponent):
    target_: Literal["anemoi.models.layers.processor.PointWiseMLPProcessor"] = Field(..., alias="_target_")
    "Transformer processor object from anemoi.models.layers.processor."
    num_layers: NonNegativeInt = Field(example=16)
    "Number of layers of Transformer processor."
    num_channels: NonNegativeInt = Field(example=128)
    "Number of channels."
    dropout_p: NonNegativeFloat = Field(example=0.1)
    "Dropout probability, default 0.0"
