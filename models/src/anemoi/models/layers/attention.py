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
import math
from typing import Any
from typing import Optional
from typing import Tuple

import einops
import numpy as np
import torch
from packaging import version
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.typing import PairTensor

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.graph import sync_tensor
from anemoi.models.distributed.transformer import shard_heads
from anemoi.models.distributed.transformer import shard_sequence
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class MultiHeadSelfAttention(nn.Module):
    """Multi Head Self Attention Pytorch Layer

    allows for three different attention implementations:
    - scaled dot product attention, see https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    - flash attention, see https://github.com/Dao-AILab/flash-attention

    The config parameter "model.processor.attention_implementation" is used to control which attention implementation is used.

    "scaled_dot_product_attention" (SDPA)
        SDPA is a pytorch function, so it is easiest to use but the least performant.
        It runs on CPUs and GPUs.

    "flash_attention"
        Flash attention is optimised for efficient usage of the GPUs memory hierarchy. It loads smaller chunks
        into fast local memory, and fuses attention into a single kernel to reduce the passes through memory.
        It runs on Nvidia Ampere (e.g. A100) GPUs or newer and AMD MI200 GPUs or newer. Check the GitHub for
        the full requirements.
        You have to install flash attention yourself. If you are running on an x86 system, there are prebuilt
        wheels available on the GitHub repo. On an aarch64 system, you have to build flash attention from source.
    """

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        layer_kernels: DotDict,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        is_causal: bool = False,
        window_size: Optional[int] = None,
        dropout_p: float = 0.0,
        attention_implementation: str = "flash_attention",
        softcap: Optional[float] = None,
        use_alibi_slopes: bool = False,
        use_rotary_embeddings: bool = False,
    ):
        """Initialize MultiHeadSelfAttention.

        For the flash attention implementation, two additional parameters are available: softcap, use_alibi_slopes

        softcap: Softcapping prevents the logits from growing excessively large

        use_alibi_slopes: Adds bias of `(-alibi_slope * |i + seqlen_k - seqlen_q - j|)` to the attention score of
        query i and key j, where alibi_slope is calculated using get_alibi_slopes

        Parameters
        ----------
        num_heads : int
            number of heads
        embed_dim : int
            embedding dimension
        qkv_bias : bool, optional
            bias for querys, keys and values, by default False
        qk_norm : bool, optional
            normalize q and k, by default False
        is_causal : bool, optional
            apply causal attention mask, by default False
        window_size : Optional[int], optional
            window_size, by default None
        dropout_p : float, optional
            dropout probability, by default 0.0
        attention_implementation: str
            A predefined string which selects which underlying attention
            implementation, by default "flash_attention"
        softcap : float, optional
            Anything > 0 activates softcapping attention, by default None
        use_alibi_slopes : bool, optional
            Adds bias
        """
        super().__init__()

        assert (
            embed_dim % num_heads == 0
        ), f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})"

        self.attention_implementation = attention_implementation
        self.use_alibi_slopes = use_alibi_slopes

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads  # q k v
        self.window_size = window_size
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.qk_norm = qk_norm
        self.softcap = softcap
        self.use_rotary_embeddings = use_rotary_embeddings

        self.set_attention_function()

        if self.use_alibi_slopes:
            self.alibi_slopes = get_alibi_slopes(num_heads)
            assert self.alibi_slopes.shape[0] == num_heads, "Error: Number of alibi_slopes must match number of heads"
        else:
            self.alibi_slopes = None

        linear = layer_kernels.Linear
        self.lin_q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.lin_k = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.lin_v = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)

        self.projection = linear(embed_dim, embed_dim, bias=True)

        if self.qk_norm:
            self.q_norm = layer_kernels["QueryNorm"](self.head_dim)
            self.k_norm = layer_kernels["KeyNorm"](self.head_dim)

    def set_attention_function(self):
        attn_funcs = {
            "flash_attention": FlashAttentionWrapper,
            "scaled_dot_product_attention": SDPAAttentionWrapper,
        }
        assert (
            self.attention_implementation in attn_funcs
        ), f"{self.attention_implementation} not supported. \
              Please change model.processor.attention_implementation to one of: {attn_funcs.keys()}"

        # initalise the attn func here
        if self.attention_implementation == "flash_attention":
            self.attention = attn_funcs[self.attention_implementation](
                use_rotary_embeddings=self.use_rotary_embeddings, head_dim=self.head_dim
            )
        else:
            self.attention = attn_funcs[self.attention_implementation]()

    def attention_computation(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        shapes: list,
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> Tensor:
        if model_comm_group:
            assert (
                model_comm_group.size() == 1 or batch_size == 1
            ), "Only batch size of 1 is supported when model is sharded accross GPUs"

        query, key, value = (
            einops.rearrange(
                t,
                "(batch grid) (heads vars) -> batch heads grid vars",
                batch=batch_size,
                heads=self.num_heads,
            )
            for t in (query, key, value)
        )

        query = shard_heads(query, shapes=shapes, mgroup=model_comm_group)
        key = shard_heads(key, shapes=shapes, mgroup=model_comm_group)
        value = shard_heads(value, shapes=shapes, mgroup=model_comm_group)
        dropout_p = self.dropout_p if self.training else 0.0

        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)

        out = self.attention(
            query,
            key,
            value,
            batch_size,
            causal=False,
            window_size=self.window_size,
            dropout_p=dropout_p,
            softcap=self.softcap,
            alibi_slopes=self.alibi_slopes,
        )

        out = shard_sequence(out, shapes=shapes, mgroup=model_comm_group)
        out = einops.rearrange(out, "batch heads grid vars -> (batch grid) (heads vars)")

        out = self.projection(out)

        return out

    def forward(
        self, x: Tensor, shapes: list, batch_size: int, model_comm_group: Optional[ProcessGroup] = None
    ) -> Tensor:

        query = self.lin_q(x)
        key = self.lin_k(x)
        value = self.lin_v(x)

        return self.attention_computation(query, key, value, shapes, batch_size, model_comm_group)

def compute_banded_permutation(edge_index: Tensor, num_nodes: int) -> Tuple[Tensor, Tensor]:
    """Compute a banded permutation using reverse Cuthill-McKee ordering.

    This helps make graph adjacency local in sequence space so windowed
    flash attention can approximate k-hop neighborhoods efficiently.
    """
    try:
        import scipy.sparse as sp
    except ImportError as exc:
        raise ImportError(
            "scipy is required to compute a banded permutation; "
            "precompute perm/inv_perm and pass them in if scipy is unavailable."
        ) from exc

    edge_index_cpu = edge_index.detach().cpu()
    rows = edge_index_cpu[0].numpy()
    cols = edge_index_cpu[1].numpy()
    rows_sym = np.concatenate([rows, cols])
    cols_sym = np.concatenate([cols, rows])
    data = np.ones(rows_sym.shape[0], dtype=np.bool_)
    adj = sp.csr_matrix((data, (rows_sym, cols_sym)), shape=(num_nodes, num_nodes))
    perm = sp.csgraph.reverse_cuthill_mckee(adj, symmetric_mode=True)
    perm = torch.as_tensor(perm.copy(), dtype=torch.long)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(num_nodes, dtype=torch.long)
    return perm, inv_perm


class BandedGraphSelfAttention(nn.Module):
    """Self-attention with a banded graph permutation and windowed attention."""

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        layer_kernels: DotDict,
        window_size: Optional[int] = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        is_causal: bool = False,
        dropout_p: float = 0.0,
        attention_implementation: str = "flash_attention",
        softcap: Optional[float] = None,
        use_alibi_slopes: bool = False,
        use_rotary_embeddings: bool = False,
    ):
        super().__init__()
        self.attention = MultiHeadSelfAttention(
            num_heads=num_heads,
            embed_dim=embed_dim,
            layer_kernels=layer_kernels,
            window_size=window_size,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            is_causal=is_causal,
            dropout_p=dropout_p,
            attention_implementation=attention_implementation,
            softcap=softcap,
            use_alibi_slopes=use_alibi_slopes,
            use_rotary_embeddings=use_rotary_embeddings,
        )
        self._perm = None
        self._inv_perm = None

    def set_permutation(self, perm: Tensor, inv_perm: Tensor) -> None:
        self._perm = perm
        self._inv_perm = inv_perm

    def forward(
        self,
        x: Tensor,
        shapes: list,
        batch_size: int,
        *,
        perm: Optional[Tensor] = None,
        inv_perm: Optional[Tensor] = None,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> Tensor:
        perm = perm if perm is not None else self._perm
        inv_perm = inv_perm if inv_perm is not None else self._inv_perm
        if perm is None or inv_perm is None:
            raise ValueError("perm and inv_perm must be provided or set via set_permutation")

        num_nodes = perm.numel()

        if perm.device != x.device:
            perm = perm.to(device=x.device)
        if inv_perm.device != x.device:
            inv_perm = inv_perm.to(device=x.device)

        # Handle distributed training: sync (gather) tensor from all GPUs before permutation
        is_distributed = model_comm_group is not None and model_comm_group.size() > 1
        if is_distributed:
            # Gather all shards to get full tensor on each GPU
            x = sync_tensor(x, dim=0, shapes=shapes, mgroup=model_comm_group)
            # Compute full shapes for attention after sync
            full_shapes = [[batch_size * num_nodes, x.shape[-1]]]
        else:
            full_shapes = shapes

        # Verify shape after potential sync
        expected = batch_size * num_nodes
        if x.shape[0] != expected:
            raise ValueError(f"Expected x.shape[0] == {expected}, got {x.shape[0]}")

        # Apply RCM permutation to reorder nodes
        x = x.view(batch_size, num_nodes, -1)
        x = x.index_select(1, perm)
        x = x.reshape(batch_size * num_nodes, -1)

        # Run attention (with window_size, this approximates k-hop attention)
        # Note: Don't pass model_comm_group here since we've already synced
        out = self.attention(x, full_shapes, batch_size, model_comm_group=None)

        # Apply inverse permutation to restore original node order
        out = out.view(batch_size, num_nodes, -1)
        out = out.index_select(1, inv_perm)
        out = out.reshape(batch_size * num_nodes, -1)

        # Shard output back to original distribution
        if is_distributed:
            out = shard_tensor(out, dim=0, shapes=shapes, mgroup=model_comm_group, gather_in_backward=True)

        return out
    
    

class SDPAAttentionWrapper(nn.Module):
    """Wrapper for Pytorch scaled dot product attention
    To use this attention implementation: model.processor.attention_implementation='scaled_dot_product_attention'
    """

    def __init__(self):
        super().__init__()

        from torch.nn.functional import scaled_dot_product_attention

        self.attention = scaled_dot_product_attention
        self.mask = None
        self.window_size = None
        LOGGER.info("Using scaled_dot_product_attention.")

    def update_mask(self, seq_len, window_size: int, device: str):

        self.mask = (
            torch.abs(
                torch.arange(seq_len, device=device).unsqueeze(0) - torch.arange(seq_len, device=device).unsqueeze(1)
            )
            <= window_size
        )

    def forward(
        self,
        query,
        key,
        value,
        batch_size: int,
        causal=False,
        window_size=None,
        dropout_p=0.0,
        softcap=None,
        alibi_slopes=None,
    ):
        if softcap is not None:
            NotImplementedError(
                "Softcap not supported by Pytorchs SDPA. please switch to flash attention or disable softcap."
            )
        if alibi_slopes is not None:
            NotImplementedError(
                "Alibi slopes not supported by Pytorchs SDPA. please switch to flash attention v2 or disable alibi slopes."
            )

        sequence_len = query.shape[-2]

        if window_size is not None and (self.mask is None or tuple(self.mask.shape) != (sequence_len, sequence_len)):
            self.update_mask(sequence_len, window_size=window_size, device=query.device)

        out = self.attention(
            query,
            key,
            value,
            attn_mask=self.mask,
            is_causal=causal,
            dropout_p=dropout_p,
        )

        return out


class FlashAttentionWrapper(nn.Module):
    """Wrapper for Flash attention.

    Either flash attn v2 or flash attn v3 (optimised for hoppers and newer), based on
    what is installed.
    flash attention v3 does not support rotary embeddings or alibi slopes. To use these
    features, you should downgrade to flash attention v2.

    """

    def __init__(self, use_rotary_embeddings: bool = False, head_dim: int = None):
        super().__init__()

        flash_attn, self.use_flash_attn_v3 = self._import_flash_attn()

        flash_attn_version = version.parse(flash_attn.__version__)
        self._init_rotary_embeddings(use_rotary_embeddings, head_dim, flash_attn_version)

        self.attention = flash_attn.flash_attn_func

    def _init_rotary_embeddings(self, use_rotary_embeddings: bool, head_dim: int, flash_attn_version) -> None:
        """Enables rotary embeddings if flash attention version is between 2.6.0 and 3."""
        self.use_rotary_embeddings = False
        if use_rotary_embeddings:
            if flash_attn_version >= version.parse("3"):
                raise RuntimeError("Rotary Embeddings not supported with flash attention v3")
            elif flash_attn_version <= version.parse("2.6"):
                raise RuntimeError("Rotary Embeddings not supported with flash attention v2 < v2.6.0")

            from flash_attn.layers.rotary import RotaryEmbedding

            self.use_rotary_embeddings = True
            self.rotary_emb = RotaryEmbedding(dim=head_dim)

    def _import_flash_attn(self) -> (Any, bool):
        """imports either flash attention v2 or v3.

        returns:
            flash attention module
            use_flash_attention_v3 (bool)
        """
        use_flash_attn_v3 = False

        # to detect which flash-attn interface we're using we try import them
        # Since each import is semantically different we use this to
        # distringuish flash attention versions
        try:
            # first try import flash attn v2
            import flash_attn

        except ImportError as e_v2:

            # failed importing flash attn v2,
            # try import flash attn v3
            try:
                import flash_attn_interface as flash_attn

            except ImportError as e_v3:
                # print both errors if both fail
                raise ImportError(f"Error importing flash-attn v2: {e_v2}\nError importing flash-attn v2: {e_v3}")
            else:
                LOGGER.info("Using flash attention v3")
                use_flash_attn_v3 = True
        else:
            LOGGER.info("Using flash attention v2")
        return flash_attn, use_flash_attn_v3

    def forward(
        self,
        query,
        key,
        value,
        batch_size: int,
        causal: bool = False,
        window_size: int = None,
        dropout_p: float = 0.0,
        softcap: Optional[float] = None,
        alibi_slopes: torch.Tensor = None,
    ):
        query, key, value = (
            einops.rearrange(t, "batch heads grid vars -> batch grid heads vars") for t in (query, key, value)
        )

        if alibi_slopes is not None and self.use_flash_attn_v3:
            NotImplementedError(
                "Alibi slopes is currently not supported by flash attention v3. please switch to flash attention v2 or disable alibi slopes."
            )

        alibi_slopes = alibi_slopes.repeat(batch_size, 1).to(query.device) if alibi_slopes is not None else None

        if self.use_rotary_embeddings:
            key = key.unsqueeze(-3)
            value = value.unsqueeze(-3)
            keyvalue = torch.cat((key, value), dim=-3)
            query, keyvalue = self.rotary_emb(
                query, keyvalue, max_seqlen=max(keyvalue.shape[1], query.shape[1])
            )  # assumption seq const
            key = keyvalue[:, :, 0, ...]
            value = keyvalue[:, :, 1, ...]

        if self.use_flash_attn_v3:
            out = self.attention(
                query,
                key,
                value,
                causal=False,
                window_size=(window_size, window_size) if window_size is not None else (-1, -1),
                softcap=softcap,
            )[
                0
            ]  # fav3 returns a tuple with '(out, softmax_lse)'. here we drop to 'out'
        else:
            out = self.attention(
                query,
                key,
                value,
                causal=False,
                window_size=(window_size, window_size) if window_size is not None else (-1, -1),
                dropout_p=dropout_p,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
            )
        out = einops.rearrange(out, "batch grid heads vars -> batch heads grid vars")
        return out


class MultiHeadCrossAttention(MultiHeadSelfAttention):
    """Multi Head Cross Attention Pytorch Layer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self, x: PairTensor, shapes: list, batch_size: int, model_comm_group: Optional[ProcessGroup] = None
    ) -> Tensor:
        query = self.lin_q(x[1])
        key = self.lin_k(x[0])
        value = self.lin_v(x[0])

        return self.attention_computation(query, key, value, shapes, batch_size, model_comm_group)


def get_alibi_slopes(num_heads: int) -> Tensor:
    """Calculates linearly decreasing slopes for alibi attention.

    Parameters
    ----------
    num_heads : int
        number of attention heads

    Returns
    -------
    Tensor
        aLiBi slopes
    """
    n = 2 ** math.floor(math.log2(num_heads))
    slope_0 = 2 ** (-8 / n)
    alibi_slopes = torch.pow(slope_0, torch.arange(1, 1 + n))
    if n < num_heads:
        slope_hat_0 = 2 ** (-4 / n)
        alibi_slopes_hat = torch.pow(slope_hat_0, torch.arange(1, 1 + 2 * (num_heads - n), 2))
        alibi_slopes = torch.cat([alibi_slopes, alibi_slopes_hat])
    return alibi_slopes
