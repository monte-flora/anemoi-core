# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from abc import ABC
from typing import Optional

from torch import Tensor
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import offload_wrapper
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.khop_edges import sort_edges_1hop_sharding
from anemoi.models.distributed.shapes import change_channels_in_shape
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.layers.attention import compute_banded_permutation
from anemoi.models.layers.block import BandedTransformerProcessorBlock
from anemoi.models.layers.block import GraphConvProcessorBlock
from anemoi.models.layers.block import GraphInteractionNetProcessorBlock
from anemoi.models.layers.block import GraphTransformerProcessorBlock
from anemoi.models.layers.block import PointWiseMLPProcessorBlock
from anemoi.models.layers.block import TransformerProcessorBlock
from anemoi.models.layers.graph import TrainableTensor
from anemoi.models.layers.mapper import GraphEdgeMixin
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict


class BaseProcessor(nn.Module, ABC):
    """Base Processor."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_channels: int,
        num_chunks: int,
        cpu_offload: bool = False,
        layer_kernels: DotDict,
        **kwargs,
    ) -> None:
        """Initialize BaseProcessor.

        Parameters
        ----------
        num_layers : int
            Number of processor layers.
        num_channels : int
            Number of channels, i.e. feature dimension of the processor state.
        num_chunks: int
            Number of chunks of the processor. The num_chunks and num_layers, defines how many layers are grouped together for checkpointing, i.e. chunk_size = num_layers/ num_chunks.
        cpu_offload : bool
            Whether to offload processing to CPU, by default False
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml
        **kwargs : dict
            Additional keyword arguments
        """
        super().__init__()

        self.num_layers = num_layers
        self.num_chunks = num_chunks
        self.chunk_size = num_layers // num_chunks
        self.num_channels = num_channels

        self.layer_factory = load_layer_kernels(layer_kernels)

        self._has_dropout = kwargs.get("dropout_p", 0.0) > 0 if "dropout_p" in kwargs else False

        assert (
            num_layers % num_chunks == 0
        ), f"Number of processor layers ({num_layers}) has to be divisible by the number of processor chunks ({num_chunks})."

    def offload_layers(self, cpu_offload):
        if cpu_offload:
            self.proc = nn.ModuleList([offload_wrapper(x) for x in self.proc])

    def build_layers(self, layer_class, *layer_args, **layer_kwargs) -> None:
        """Build Layers."""
        self.proc = nn.ModuleList(
            [
                layer_class(
                    *layer_args,
                    **layer_kwargs,
                )
                for _ in range(self.num_layers)
            ],
        )

    def run_layer_chunk(self, chunk_start: int, data: tuple, *args, **kwargs) -> tuple:
        for layer_id in range(chunk_start, chunk_start + self.chunk_size):
            data = self.proc[layer_id](*data, *args, **kwargs)

        return data

    def run_layers(self, data: tuple, *args, **kwargs) -> tuple:
        """Run Layers with checkpoints around chunks."""
        for chunk_start in range(0, self.num_layers, self.chunk_size):
            data = checkpoint(self.run_layer_chunk, chunk_start, data, *args, **kwargs, use_reentrant=False)

        return data

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Example forward pass."""

        if (model_comm_group := kwargs.get("model_comm_group", None)) is not None:
            assert (
                model_comm_group.size() == 1 or not self._has_dropout
            ), f"Dropout is not supported when model is sharded across {model_comm_group.size()} GPUs"

        x = self.run_layers((x,), *args, **kwargs)
        return x


class PointWiseMLPProcessor(BaseProcessor):
    """Point-wise MLP Processor."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_channels: int,
        num_chunks: int,
        mlp_hidden_ratio: int,
        cpu_offload: bool = False,
        dropout_p: float = 0.0,
        layer_kernels: DotDict,
        **kwargs,
    ):
        super().__init__(
            num_layers=num_layers,
            num_channels=num_channels,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            layer_kernels=layer_kernels,
            dropout_p=dropout_p,
        )

        self.build_layers(
            PointWiseMLPProcessorBlock,
            num_channels=num_channels,
            hidden_dim=(mlp_hidden_ratio * num_channels),
            layer_kernels=self.layer_factory,
            dropout_p=dropout_p,
        )

        self.offload_layers(cpu_offload)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        shard_shapes: list[list[int]],
        model_comm_group: Optional[ProcessGroup] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        shape_nodes = change_channels_in_shape(shard_shapes, self.num_channels)
        if model_comm_group:
            assert (
                model_comm_group.size() == 1 or batch_size == 1
            ), f"Only batch size of 1 is supported when model is sharded accross {model_comm_group.size()} GPUs"

        (x,) = self.run_layers((x,), shape_nodes, batch_size, model_comm_group, **kwargs)

        return x


class TransformerProcessor(BaseProcessor):
    """Transformer Processor."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_channels: int,
        num_chunks: int,
        num_heads: int,
        mlp_hidden_ratio: int,
        qk_norm=False,
        dropout_p: float = 0.0,
        attention_implementation: str = "flash_attention",
        softcap: float = 0,
        use_alibi_slopes: bool = False,
        window_size: Optional[int] = None,
        cpu_offload: bool = False,
        layer_kernels: DotDict,
        **kwargs,
    ) -> None:
        """Initialize TransformerProcessor.

        Parameters
        ----------
        num_layers : int
            Number of layers
        num_channels : int
            Number of channels
        num_chunks: int
            Number of chunks in processor
        num_heads: int
            Number of heads in transformer
        mlp_hidden_ratio: int
            Ratio of mlp hidden dimension to embedding dimension
        qk_norm: bool, optional
            Normalize query and key, by default False
        dropout_p: float, optional
            Dropout probability used for multi-head self attention, default 0.1
        attention_implementation: str
            A predefined string which selects which underlying attention
            implementation, by default "flash_attention"
        softcap : float, optional
            Anything > 0 activates softcapping flash attention, by default 0
        use_alibi_slopes : bool
            Use aLiBI option, only used for flash attention, by default False
        window_size: int, optional
            1/2 size of shifted window for attention computation, by default None
        cpu_offload : bool
            Whether to offload processing to CPU, by default False
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml
        """
        super().__init__(
            num_layers=num_layers,
            num_channels=num_channels,
            window_size=window_size,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            layer_kernels=layer_kernels,
            dropout_p=dropout_p,
        )

        self.build_layers(
            TransformerProcessorBlock,
            num_channels=num_channels,
            hidden_dim=(mlp_hidden_ratio * num_channels),
            num_heads=num_heads,
            qk_norm=qk_norm,
            window_size=window_size,
            layer_kernels=self.layer_factory,
            dropout_p=dropout_p,
            attention_implementation=attention_implementation,
            softcap=softcap,
            use_alibi_slopes=use_alibi_slopes,
        )

        self.offload_layers(cpu_offload)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        shard_shapes: list[list[int]],
        model_comm_group: Optional[ProcessGroup] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        shape_nodes = change_channels_in_shape(shard_shapes, self.num_channels)
        if model_comm_group:
            assert (
                model_comm_group.size() == 1 or batch_size == 1
            ), "Only batch size of 1 is supported when model is sharded accross GPUs"

        (x,) = self.run_layers((x,), shape_nodes, batch_size, model_comm_group=model_comm_group, **kwargs)

        return x


class BandedTransformerProcessor(GraphEdgeMixin, BaseProcessor):
    """Banded Transformer Processor with graph-aware sparse attention.

    This processor uses the Reverse Cuthill-McKee (RCM) algorithm to reorder
    graph nodes so that neighbors are adjacent in sequence space. Combined with
    windowed flash attention, this approximates k-hop graph attention efficiently.

    The permutation is computed once during initialization based on the graph
    topology and reused for all forward passes.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_channels: int,
        num_chunks: int,
        num_heads: int,
        mlp_hidden_ratio: int,
        src_grid_size: int,
        dst_grid_size: int,
        sub_graph: HeteroData,
        sub_graph_edge_attributes: list[str],
        trainable_size: int = 8,
        qk_norm: bool = False,
        dropout_p: float = 0.0,
        attention_implementation: str = "flash_attention",
        softcap: float = 0,
        use_alibi_slopes: bool = False,
        window_size: Optional[int] = None,
        cpu_offload: bool = False,
        layer_kernels: DotDict,
        **kwargs,
    ) -> None:
        """Initialize BandedTransformerProcessor.

        Parameters
        ----------
        num_layers : int
            Number of layers
        num_channels : int
            Number of channels
        num_chunks : int
            Number of chunks in processor
        num_heads : int
            Number of heads in transformer
        mlp_hidden_ratio : int
            Ratio of mlp hidden dimension to embedding dimension
        src_grid_size : int
            Source grid size (number of nodes)
        dst_grid_size : int
            Destination grid size (number of nodes)
        sub_graph : HeteroData
            Graph containing edge topology for computing RCM permutation
        sub_graph_edge_attributes : list[str]
            Edge attributes (not used but kept for API consistency)
        trainable_size : int
            Size of trainable tensor (not used but kept for API consistency)
        qk_norm : bool, optional
            Normalize query and key, by default False
        dropout_p : float, optional
            Dropout probability, by default 0.0
        attention_implementation : str
            Attention implementation ("flash_attention" or "scaled_dot_product_attention")
        softcap : float, optional
            Softcapping for flash attention, by default 0
        use_alibi_slopes : bool
            Use ALiBi slopes, by default False
        window_size : int, optional
            Window size for windowed attention. Should be >= k-hop bandwidth after RCM.
        cpu_offload : bool
            Whether to offload processing to CPU, by default False
        layer_kernels : DotDict
            Layer kernel implementations
        """
        super().__init__(
            num_layers=num_layers,
            num_channels=num_channels,
            window_size=window_size,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            layer_kernels=layer_kernels,
            dropout_p=dropout_p,
        )

        # Register edges from graph (we only need edge_index for permutation)
        self._register_edges(sub_graph, sub_graph_edge_attributes, src_grid_size, dst_grid_size, trainable_size)

        # Compute RCM permutation once based on graph topology
        # edge_index_base is (2, num_edges) tensor
        self.num_nodes = src_grid_size  # For processor, src == dst
        perm, inv_perm = compute_banded_permutation(self.edge_index_base, self.num_nodes)

        # Register as buffers so they're saved with the model and moved to correct device
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv_perm)

        self.build_layers(
            BandedTransformerProcessorBlock,
            num_channels=num_channels,
            hidden_dim=(mlp_hidden_ratio * num_channels),
            num_heads=num_heads,
            qk_norm=qk_norm,
            window_size=window_size,
            layer_kernels=self.layer_factory,
            dropout_p=dropout_p,
            attention_implementation=attention_implementation,
            softcap=softcap,
            use_alibi_slopes=use_alibi_slopes,
        )

        self.offload_layers(cpu_offload)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        shard_shapes: list[list[int]],
        model_comm_group: Optional[ProcessGroup] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        shape_nodes = change_channels_in_shape(shard_shapes, self.num_channels)
        if model_comm_group:
            assert (
                model_comm_group.size() == 1 or batch_size == 1
            ), "Only batch size of 1 is supported when model is sharded across GPUs"

        # Pass permutation tensors to layers
        (x,) = self.run_layers(
            (x,),
            shape_nodes,
            batch_size,
            model_comm_group=model_comm_group,
            perm=self.perm,
            inv_perm=self.inv_perm,
            **kwargs,
        )

        return x


class GNNProcessor(GraphEdgeMixin, BaseProcessor):
    """GNN Processor."""

    def __init__(
        self,
        *,
        num_channels: int,
        num_layers: int,
        num_chunks: int,
        mlp_extra_layers: int,
        trainable_size: int,
        src_grid_size: int,
        dst_grid_size: int,
        sub_graph: HeteroData,
        sub_graph_edge_attributes: list[str],
        cpu_offload: bool = False,
        layer_kernels: DotDict,
        **kwargs,
    ) -> None:
        """Initialize GNNProcessor.

        Parameters
        ----------
        num_layers : int
            Number of layers
        num_channels : int
            Number of channels
        num_chunks: int
            Number of chunks in processor
        mlp_extra_layers : int, optional
            Number of extra layers in MLP
        trainable_size : int
            Size of trainable tensor
        src_grid_size : int
            Source grid size
        dst_grid_size : int
            Destination grid size
        sub_graph : HeteroData
            Graph for sub graph in GNN
        sub_graph_edge_attributes : list[str]
            Sub graph edge attributes
        cpu_offload : bool
            Whether to offload processing to CPU, by default False
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml

        """
        super().__init__(
            num_channels=num_channels,
            num_layers=num_layers,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            mlp_extra_layers=mlp_extra_layers,
            layer_kernels=layer_kernels,
        )

        self._register_edges(sub_graph, sub_graph_edge_attributes, src_grid_size, dst_grid_size, trainable_size)

        self.trainable = TrainableTensor(trainable_size=trainable_size, tensor_size=self.edge_attr.shape[0])

        kwargs = {
            "mlp_extra_layers": mlp_extra_layers,
            "layer_kernels": self.layer_factory,
            "edge_dim": None,
        }

        self.build_layers(
            GraphConvProcessorBlock,
            in_channels=num_channels,
            out_channels=num_channels,
            num_chunks=1,
            **kwargs,
        )

        kwargs["edge_dim"] = self.edge_dim  # Edge dim for first layer
        self.proc[0] = GraphConvProcessorBlock(
            in_channels=num_channels,
            out_channels=num_channels,
            num_chunks=1,
            **kwargs,
        )

        self.offload_layers(cpu_offload)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        shard_shapes: list[list[int]],
        model_comm_group: Optional[ProcessGroup] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        shape_nodes = change_channels_in_shape(shard_shapes, self.num_channels)
        edge_attr = self.trainable(self.edge_attr, batch_size)
        edge_index = self._expand_edges(self.edge_index_base, self.edge_inc, batch_size)
        target_nodes = sum(x[0] for x in shape_nodes)
        edge_attr, edge_index, shapes_edge_attr, shapes_edge_idx = sort_edges_1hop_sharding(
            target_nodes,
            edge_attr,
            edge_index,
            model_comm_group,
        )
        edge_index = shard_tensor(edge_index, 1, shapes_edge_idx, model_comm_group)
        edge_attr = shard_tensor(edge_attr, 0, shapes_edge_attr, model_comm_group)

        x, edge_attr = self.run_layers(
            (x, edge_attr), edge_index, (shape_nodes, shape_nodes), model_comm_group, **kwargs
        )

        return x


class GraphInteractionNetProcessor(GraphEdgeMixin, BaseProcessor):
    """GNN Processor implementing the correct GraphCast/Interaction Network pattern.

    This follows the Interaction Network architecture (Battaglia et al. 2018)
    as used in GraphCast (Lam et al. 2022):

    The correct order is:
        1. edge_delta = EdgeMLP([sender, receiver, edge])  (NO residual inside)
        2. aggregated = Σ(edge_delta)  (sum of DELTAS only)
        3. node_delta = NodeMLP([node, aggregated])
        4. new_edge = edge + edge_delta  (residual AFTER)
        5. new_node = node + node_delta  (residual AFTER)

    This differs from the original GNNProcessor/GraphConvProcessorBlock which applies
    edge residual INSIDE the message function, causing the node MLP to receive
    Σ(delta + old_edge) instead of just Σ(delta). This incorrect pattern can cause
    gradient collapse in deep networks (16+ layers).
    """

    def __init__(
        self,
        *,
        num_channels: int,
        num_layers: int,
        num_chunks: int,
        mlp_extra_layers: int,
        trainable_size: int,
        src_grid_size: int,
        dst_grid_size: int,
        sub_graph: HeteroData,
        sub_graph_edge_attributes: list[str],
        cpu_offload: bool = False,
        layer_kernels: DotDict,
        aggr_reduce: str = "sum",
        **kwargs,
    ) -> None:
        """Initialize GraphInteractionNetProcessor.

        Parameters
        ----------
        num_layers : int
            Number of layers
        num_channels : int
            Number of channels
        num_chunks: int
            Number of chunks in processor
        mlp_extra_layers : int, optional
            Number of extra layers in MLP
        trainable_size : int
            Size of trainable tensor
        src_grid_size : int
            Source grid size
        dst_grid_size : int
            Destination grid size
        sub_graph : HeteroData
            Graph for sub graph in GNN
        sub_graph_edge_attributes : list[str]
            Sub graph edge attributes
        cpu_offload : bool
            Whether to offload processing to CPU, by default False
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml
        aggr_reduce : str, optional
            Aggregation op over incoming edge deltas at each processor block.
            Default "sum" (GraphCast convention). Use "mean" for the sum→mean
            inverse test (low-pass across neighbors).

        """
        super().__init__(
            num_channels=num_channels,
            num_layers=num_layers,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            mlp_extra_layers=mlp_extra_layers,
            layer_kernels=layer_kernels,
        )

        self._register_edges(sub_graph, sub_graph_edge_attributes, src_grid_size, dst_grid_size, trainable_size)

        self.trainable = TrainableTensor(trainable_size=trainable_size, tensor_size=self.edge_attr.shape[0])

        kwargs = {
            "mlp_extra_layers": mlp_extra_layers,
            "layer_kernels": self.layer_factory,
            "edge_dim": None,
            "aggr_reduce": aggr_reduce,
        }

        # Use the new GraphInteractionNetProcessorBlock with correct residual order
        self.build_layers(
            GraphInteractionNetProcessorBlock,
            in_channels=num_channels,
            out_channels=num_channels,
            num_chunks=1,
            **kwargs,
        )

        kwargs["edge_dim"] = self.edge_dim  # Edge dim for first layer
        self.proc[0] = GraphInteractionNetProcessorBlock(
            in_channels=num_channels,
            out_channels=num_channels,
            num_chunks=1,
            **kwargs,
        )

        self.offload_layers(cpu_offload)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        shard_shapes: list[list[int]],
        model_comm_group: Optional[ProcessGroup] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        shape_nodes = change_channels_in_shape(shard_shapes, self.num_channels)
        edge_attr = self.trainable(self.edge_attr, batch_size)
        edge_index = self._expand_edges(self.edge_index_base, self.edge_inc, batch_size)
        target_nodes = sum(x[0] for x in shape_nodes)
        edge_attr, edge_index, shapes_edge_attr, shapes_edge_idx = sort_edges_1hop_sharding(
            target_nodes,
            edge_attr,
            edge_index,
            model_comm_group,
        )
        edge_index = shard_tensor(edge_index, 1, shapes_edge_idx, model_comm_group)
        edge_attr = shard_tensor(edge_attr, 0, shapes_edge_attr, model_comm_group)

        x, edge_attr = self.run_layers(
            (x, edge_attr), edge_index, (shape_nodes, shape_nodes), model_comm_group, **kwargs
        )

        return x


class GraphTransformerProcessor(GraphEdgeMixin, BaseProcessor):
    """Processor."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_channels: int,
        num_chunks: int,
        num_heads: int,
        mlp_hidden_ratio: int,
        trainable_size: int,
        src_grid_size: int,
        dst_grid_size: int,
        sub_graph: HeteroData,
        sub_graph_edge_attributes: list[str],
        qk_norm: bool = False,
        cpu_offload: bool = False,
        layer_kernels: DotDict,
        graph_attention_backend: str = "triton",
        edge_pre_mlp: bool = False,
        **kwargs,
    ) -> None:
        """Initialize GraphTransformerProcessor.

        Parameters
        ----------
        num_layers : int
            Number of layers
        num_channels : int
            Number of channels
        num_chunks: int
            Number of chunks in processor
        num_heads: int
            Number of heads in transformer
        mlp_hidden_ratio: int
            Ratio of mlp hidden dimension to embedding dimension
        trainable_size : int
            Size of trainable tensor
        src_grid_size : int
            Source grid size
        dst_grid_size : int
            Destination grid size
        sub_graph : HeteroData
            Graph for sub graph in GNN
        sub_graph_edge_attributes : list[str]
            Sub graph edge attributes
        qk_norm: bool, optional
            Normalize query and key, by default False
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml
        graph_attention_backend: str, by default "triton"
            Backend to use for graph transformer conv, options are "triton" and "pyg"
        edge_pre_mlp: bool, by default False
            Allow for edge feature mixing
        """
        super().__init__(
            num_channels=num_channels,
            num_layers=num_layers,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            layer_kernels=layer_kernels,
        )

        self._register_edges(sub_graph, sub_graph_edge_attributes, src_grid_size, dst_grid_size, trainable_size)

        self.trainable = TrainableTensor(trainable_size=trainable_size, tensor_size=self.edge_attr.shape[0])

        self.build_layers(
            GraphTransformerProcessorBlock,
            in_channels=num_channels,
            hidden_dim=(mlp_hidden_ratio * num_channels),
            out_channels=num_channels,
            num_heads=num_heads,
            edge_dim=self.edge_dim,
            layer_kernels=self.layer_factory,
            qk_norm=qk_norm,
            graph_attention_backend=graph_attention_backend,
            edge_pre_mlp=edge_pre_mlp,
        )

        self.offload_layers(cpu_offload)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        shard_shapes: list[list[int]],
        model_comm_group: Optional[ProcessGroup] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        size = sum(x[0] for x in shard_shapes)

        shape_nodes = change_channels_in_shape(shard_shapes, self.num_channels)
        edge_attr = self.trainable(self.edge_attr, batch_size)

        edge_index = self._expand_edges(self.edge_index_base, self.edge_inc, batch_size)

        shapes_edge_attr = get_shard_shapes(edge_attr, 0, model_comm_group)
        edge_attr = shard_tensor(edge_attr, 0, shapes_edge_attr, model_comm_group)

        x, edge_attr = self.run_layers(
            data=(x, edge_attr),
            edge_index=edge_index,
            shapes=(shape_nodes, shape_nodes, shapes_edge_attr),
            batch_size=batch_size,
            size=size,
            model_comm_group=model_comm_group,
            **kwargs,
        )

        return x


def _sincos_2d_pos_embed_natten(
    h: int, w: int, embed_dim: int, device, dtype=None
) -> Tensor:
    """Standard 2-D sine-cosine positional embedding (ViT/Atlas style).

    Returns a tensor of shape ``(h*w, embed_dim)``. ``embed_dim`` must be
    divisible by 4. The formula is analytical so any ``(h, w)`` works
    without retraining or interpolation — re-instantiate the processor at
    a different ``hidden_field_shape`` and the embedding regenerates.

    Adapted from ``decoder_dit_wrapper._sincos_2d_pos_embed``; inlined here
    to keep ``layers/processor`` independent of the ``models/`` subpackage.
    """
    import torch as _torch

    if embed_dim % 4 != 0:
        raise ValueError(
            f"_sincos_2d_pos_embed_natten embed_dim must be divisible by 4, got {embed_dim}"
        )
    dtype = dtype or _torch.float32
    grid_y = _torch.arange(h, device=device, dtype=dtype)
    grid_x = _torch.arange(w, device=device, dtype=dtype)
    yy, xx = _torch.meshgrid(grid_y, grid_x, indexing="ij")  # (h, w)
    half = embed_dim // 2
    omega = _torch.arange(half // 2, device=device, dtype=dtype)
    omega = 1.0 / (10000 ** (omega / (half // 2)))
    pe_y = _torch.cat([
        _torch.sin(yy.unsqueeze(-1) * omega),
        _torch.cos(yy.unsqueeze(-1) * omega),
    ], dim=-1)
    pe_x = _torch.cat([
        _torch.sin(xx.unsqueeze(-1) * omega),
        _torch.cos(xx.unsqueeze(-1) * omega),
    ], dim=-1)
    pe = _torch.cat([pe_y, pe_x], dim=-1)  # (h, w, embed_dim)
    return pe.reshape(h * w, embed_dim)


class NATTEN2DProcessorBlock(nn.Module):
    """Pre-LN ViT-style block with NATTEN 2D neighborhood self-attention.

    Used by :class:`NATTEN2DProcessor`. No adaLN modulation — we have no
    conditioning input for a deterministic forecaster. Pure pre-LN block:

        x = x + NATTEN(LN(x), latent_hw=(H, W))
        x = x + MLP(LN(x))

    The block expects ``x`` of shape ``(B, N, C)`` with ``N == H * W``
    (row-major flattened image), where ``(H, W)`` is the hidden-grid
    shape baked in at ``__init__``.
    """

    def __init__(
        self,
        *,
        num_channels: int,
        hidden_dim: int,
        num_heads: int,
        attn_kernel: int,
        latent_hw: tuple[int, int],
        layer_kernels: DotDict,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        from physicsnemo.nn.module.dit_layers import Natten2DSelfAttention

        self.latent_hw = (int(latent_hw[0]), int(latent_hw[1]))
        self.layer_norm_attention = layer_kernels.LayerNorm(normalized_shape=num_channels)
        self.layer_norm_mlp = layer_kernels.LayerNorm(normalized_shape=num_channels)

        self.attention = Natten2DSelfAttention(
            hidden_size=num_channels,
            num_heads=num_heads,
            attn_kernel=attn_kernel,
            qk_norm=qk_norm,
            norm_layer="torch",
        )

        self.mlp = nn.Sequential(
            layer_kernels.Linear(num_channels, hidden_dim),
            layer_kernels.Activation(),
            layer_kernels.Linear(hidden_dim, num_channels),
        )

    def forward(self, x: Tensor, *args, **kwargs) -> tuple[Tensor]:
        # x: (B, N, C), N == H*W. The trailing positional / keyword args are
        # ignored — BaseProcessor.run_layers passes shard_shapes/batch_size/
        # model_comm_group through, but NATTEN handles its own batching and
        # does not yet support sharding (single-GPU only).
        x = x + self.attention(self.layer_norm_attention(x), latent_hw=self.latent_hw)
        x = x + self.mlp(self.layer_norm_mlp(x))
        return (x,)


class NATTEN2DProcessor(BaseProcessor):
    """Processor that applies stacked NATTEN 2D neighborhood self-attention
    on a regular 2D hidden grid.

    Used as a drop-in replacement for :class:`GraphInteractionNetProcessor`
    in the GraphCast-style encoder→processor→decoder stack when the hidden
    mesh is a regular ``H_hidden × W_hidden`` grid (e.g. built by
    :class:`anemoi.graphs.nodes.LimitedAreaSquareNodes`). NATTEN provides
    O(N·k²) local attention on the 2D grid; no hidden↔hidden graph edges
    are consumed.

    Constraint: ``N_hidden == hidden_field_shape[0] * hidden_field_shape[1]``.
    Bump ``margin_radius_km`` in the graph YAML if the area-mask drops
    boundary nodes and breaks this rectangular invariant.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_channels: int,
        num_chunks: int,
        num_heads: int,
        attn_kernel: int,
        hidden_field_shape,
        mlp_hidden_ratio: float = 4.0,
        qk_norm: bool = True,
        cpu_offload: bool = False,
        layer_kernels: DotDict,
        **kwargs,
    ) -> None:
        super().__init__(
            num_layers=num_layers,
            num_channels=num_channels,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            layer_kernels=layer_kernels,
        )

        h, w = int(hidden_field_shape[0]), int(hidden_field_shape[1])
        self.latent_hw = (h, w)

        # If the framework passed src_grid_size (hidden node count), check
        # it matches H*W. Emit a warning if not — the inference-time graph
        # swap (external_graph.py) rebuilds the model with the training
        # config's hidden_field_shape, so the mismatch is expected when
        # inferring at full-CONUS on a 62x62-trained ckpt. The post-load
        # `Predictor._resize_natten_for_inference_graph` hook then
        # regenerates pos_embed + latent_hw at the correct size. The hard
        # constraint is enforced in forward() against the actual latent_hw
        # at runtime, which is the source of truth post-resize.
        if "src_grid_size" in kwargs:
            expected = h * w
            actual = int(kwargs["src_grid_size"])
            if actual != expected:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "NATTEN2DProcessor: hidden node count (%d) != "
                    "hidden_field_shape product (%d*%d=%d). This is OK at "
                    "inference if Predictor._resize_natten_for_inference_graph "
                    "will fix it later via inference_hidden_field_shape; "
                    "otherwise the rectangular H*W assumption is broken at "
                    "training and you should bump margin_radius_km in the "
                    "graph YAML.",
                    actual, h, w, expected,
                )

        hidden_dim = int(mlp_hidden_ratio * num_channels)
        self.build_layers(
            NATTEN2DProcessorBlock,
            num_channels=num_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            attn_kernel=attn_kernel,
            latent_hw=self.latent_hw,
            qk_norm=qk_norm,
            layer_kernels=self.layer_factory,
        )

        # Sinusoidal 2D positional embedding for the hidden grid. Analytical
        # (no learned params), so re-instantiating with a different
        # hidden_field_shape — e.g. full-CONUS inference at a larger hidden
        # mesh — regenerates correctly without retraining or interpolation.
        # Registered as a non-persistent buffer so it doesn't bloat ckpts
        # (regenerated on load from the same formula).
        pe = _sincos_2d_pos_embed_natten(h, w, num_channels, device="cpu")
        self.register_buffer("pos_embed", pe.unsqueeze(0), persistent=False)  # (1, N, C)

        self.offload_layers(cpu_offload)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        shard_shapes: list[list[int]],
        model_comm_group: Optional[ProcessGroup] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        if model_comm_group is not None and model_comm_group.size() > 1:
            raise NotImplementedError(
                "NATTEN2DProcessor does not yet support sharding across multiple GPUs."
            )

        # The anemoi encoder feeds processors a flat (B*N_hidden, C) tensor
        # (PyG message-passing convention). NATTEN expects (B, N, C). Reshape
        # in, run blocks, reshape out — the decoder consumes flat (B*N, C).
        B = int(batch_size)
        N = x.shape[0] // B
        H, W = self.latent_hw
        assert N == H * W, (
            f"NATTEN2DProcessor: per-sample hidden node count {N} != H*W={H*W}. "
            f"x.shape={tuple(x.shape)}, batch_size={B}, latent_hw=({H},{W})."
        )
        C = x.shape[-1]
        x = x.reshape(B, N, C)

        x = x + self.pos_embed.to(dtype=x.dtype)
        (x,) = self.run_layers((x,))

        x = x.reshape(B * N, C)
        return x
