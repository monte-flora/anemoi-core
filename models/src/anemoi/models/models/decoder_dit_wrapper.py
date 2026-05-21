"""Decoder model for Atlas-style latent-rollout architecture.

The decoder takes a LATENT residual ``r_t`` and the FULL-RESOLUTION current
state ``x_t`` (plus forcings) and outputs the FULL-RESOLUTION residual
``delta_t`` such that ``x_{t+1} = x_t + delta_t``.

Atlas paper §2.3 (Kossaifi et al., NVIDIA, Jan 2026):

  * 4x4 strided conv tokenizes ``x_t`` (250x250) -> 63x63 token grid with
    embedding dim ``e_x``.
  * 1x1 conv on ``r_t`` (already 63x63) -> embedding dim ``e_r``.
  * Concatenate along channel dim into a single 63x63 field with
    ``e_x + e_r`` channels.
  * Sine-cosine 2-D positional embedding added.
  * Stack of ``depth`` DiT blocks with LOCAL attention (NATTEN, small
    kernel ~7-9). Atlas: "the task of the decoder is spatially local, we
    have found local attention to be incredibly effective while being
    an order of magnitude faster than its global counterpart."
  * Final linear projection back to 250x250 x ``out_channels`` (the full-
    resolution residual delta).

The decoder is trained DETERMINISTICALLY with an L1 loss on the residual
(Atlas §3.1: "minimizing the l1-norm between its output and the residual
field x_1 - x_0"). It is trained ONCE, separately from the predictive
model, and frozen for all subsequent probabilistic experiments.

Current status: SCAFFOLD. The architectural skeleton + forward signature
are real and produce shapes-correct output. The DiT internals are
placeholder conv layers; full physicsnemo DiT integration is Phase 2.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from anemoi.models.layers.bilinear_encoder import bilinear_upsample, resize_pos_embed

LOGGER = logging.getLogger(__name__)


def _sincos_2d_pos_embed(
    h: int, w: int, embed_dim: int, device, dtype=torch.float32
) -> torch.Tensor:
    """Standard 2-D sine-cosine positional embedding (ViT/Atlas style).

    Output shape: ``(h*w, embed_dim)``. embed_dim must be divisible by 4.
    """
    if embed_dim % 4 != 0:
        raise ValueError(f"_sincos_2d_pos_embed embed_dim must be % 4 == 0, got {embed_dim}")
    grid_y = torch.arange(h, device=device, dtype=dtype)
    grid_x = torch.arange(w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")  # (h, w)
    # half the channels for y, half for x
    half = embed_dim // 2
    omega = torch.arange(half // 2, device=device, dtype=dtype)
    omega = 1.0 / (10000 ** (omega / (half // 2)))
    pe_y = torch.cat([
        torch.sin(yy.unsqueeze(-1) * omega),
        torch.cos(yy.unsqueeze(-1) * omega),
    ], dim=-1)  # (h, w, half)
    pe_x = torch.cat([
        torch.sin(xx.unsqueeze(-1) * omega),
        torch.cos(xx.unsqueeze(-1) * omega),
    ], dim=-1)  # (h, w, half)
    pe = torch.cat([pe_y, pe_x], dim=-1)  # (h, w, embed_dim)
    return pe.reshape(h * w, embed_dim)


class AnemoiDecoderDiTModel(nn.Module):
    """DiT-based decoder: (latent residual + full-res state) -> full-res residual.

    See module docstring for the architectural recipe (Atlas §2.3).

    Parameters
    ----------
    full_res_shape : tuple[int, int]
        Spatial extent of the full-resolution grid (e.g., (250, 250) for the
        graf-conus-patches dataset).
    latent_shape : tuple[int, int]
        Spatial extent of the latent grid (e.g., (63, 63)).
    in_channels_xt : int
        Channels in the full-resolution input ``x_t`` (prognostic + forcings).
    in_channels_r : int
        Channels in the latent residual ``r_t`` (typically just prognostics).
    out_channels : int
        Channels in the predicted full-resolution residual ``delta_t``
        (typically the same as the number of prognostic variables).
    hidden_size : int
        Embedding dimension inside the DiT blocks.
    depth : int
        Number of DiT blocks.
    num_heads : int
        Number of attention heads. Must divide ``hidden_size``.
    attn_kernel : int
        NATTEN local-attention kernel size. Atlas uses 3; for our finer
        resolution we suggest 7-9.
    embed_split : float
        Fraction of ``hidden_size`` allocated to the x_t branch; the rest
        goes to the r_t branch. Default 0.5 (equal split).
    """

    def __init__(
        self,
        *,
        # Either provide model_config (anemoi-standard path; pulls fields
        # from model_config.model.decoder) OR pass the explicit kwargs
        # directly (unit-test path, isolated construction).
        model_config=None,
        data_indices=None,
        statistics=None,
        graph_data=None,
        full_res_shape: tuple[int, int] | None = None,
        latent_shape: tuple[int, int] | None = None,
        in_channels_xt: int | None = None,
        in_channels_r: int | None = None,
        out_channels: int | None = None,
        hidden_size: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        attn_kernel: int = 9,
        embed_split: float = 0.5,
        **_extra_kwargs,  # accept anemoi's _convert_, etc.
    ) -> None:
        super().__init__()

        # Anemoi-standard path: extract config from model_config.model.decoder.
        if model_config is not None:
            from anemoi.utils.config import DotDict
            cfg = DotDict(model_config).model.model.decoder
            full_res_shape = tuple(cfg.full_res_shape)
            latent_shape = tuple(cfg.latent_shape)
            in_channels_xt = int(cfg.in_channels_xt)
            in_channels_r = int(cfg.in_channels_r)
            out_channels = int(cfg.out_channels)
            hidden_size = int(getattr(cfg, "hidden_size", hidden_size))
            depth = int(getattr(cfg, "depth", depth))
            num_heads = int(getattr(cfg, "num_heads", num_heads))
            attn_kernel = int(getattr(cfg, "attn_kernel", attn_kernel))
            embed_split = float(getattr(cfg, "embed_split", embed_split))

        if any(v is None for v in (full_res_shape, latent_shape, in_channels_xt,
                                   in_channels_r, out_channels)):
            error = (
                "AnemoiDecoderDiTModel: must provide either model_config or all "
                "of full_res_shape / latent_shape / in_channels_xt / in_channels_r / out_channels."
            )
            raise ValueError(error)

        self.full_res_shape = tuple(full_res_shape)
        self.latent_shape = tuple(latent_shape)
        self.hidden_size = hidden_size
        self.depth = depth

        # Channel split between the two input branches.
        e_x = int(hidden_size * embed_split)
        e_r = hidden_size - e_x
        self.e_x = e_x
        self.e_r = e_r

        # Strided conv tokenizer for x_t (full-res -> latent token grid).
        # Atlas uses non-overlapping 4x4 patching. When ``full_res_shape`` is
        # not exactly divisible by stride (e.g., 250 / 4 = 62.5, but we want
        # 63 to match the existing DiT token grid), we pad spatially before
        # convolving. Padding is computed once at init.
        stride_h = math.ceil(self.full_res_shape[0] / self.latent_shape[0])
        stride_w = math.ceil(self.full_res_shape[1] / self.latent_shape[1])
        # Total padded extent to produce exactly latent_shape tokens at stride.
        padded_h = stride_h * self.latent_shape[0]
        padded_w = stride_w * self.latent_shape[1]
        pad_h = padded_h - self.full_res_shape[0]
        pad_w = padded_w - self.full_res_shape[1]
        # Pad on the high side (right/bottom) — same convention as the
        # existing DiT wrapper's _pad_to_patch_size.
        self._xt_pad = (0, pad_w, 0, pad_h)  # F.pad: (W_left, W_right, H_top, H_bot)
        self.xt_tokenizer = nn.Conv2d(
            in_channels_xt, e_x,
            kernel_size=(stride_h, stride_w),
            stride=(stride_h, stride_w),
        )

        # 1x1 conv on r_t (already at latent resolution).
        self.r_tokenizer = nn.Conv2d(in_channels_r, e_r, kernel_size=1)

        # Cached positional embedding for the latent token grid.
        # Buffer rather than parameter (sine-cosine is fixed).
        pos = _sincos_2d_pos_embed(
            self.latent_shape[0], self.latent_shape[1],
            hidden_size, device="cpu",
        )
        self.register_buffer("pos_embed", pos.unsqueeze(0))  # (1, h*w, D)

        # SCAFFOLD placeholder: use conv blocks where Atlas would use NATTEN-DiT.
        # Phase 2 will replace this with the physicsnemo DiT stack.
        # Each block: LayerNorm -> Conv2d(k=attn_kernel) -> GELU -> Conv2d(k=1).
        self.blocks = nn.ModuleList()
        pad = attn_kernel // 2
        for _ in range(depth):
            self.blocks.append(
                nn.Sequential(
                    nn.GroupNorm(num_groups=8, num_channels=hidden_size),
                    nn.Conv2d(hidden_size, hidden_size, kernel_size=attn_kernel, padding=pad),
                    nn.GELU(),
                    nn.Conv2d(hidden_size, hidden_size, kernel_size=1),
                )
            )

        # Project per-token back to (stride_h * stride_w * out_channels) so a
        # PixelShuffle-style upsample reconstructs the full-resolution residual.
        self.final = nn.Conv2d(
            hidden_size, out_channels * stride_h * stride_w, kernel_size=1,
        )
        self.pixel_shuffle = nn.PixelShuffle(stride_h)  # assumes square stride
        self.stride = (stride_h, stride_w)
        self.out_channels = out_channels

        # Atlas-style zero-init of the final layer so the decoder starts as
        # an identity (delta = 0) and learns to add detail.
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(
        self,
        r_t: torch.Tensor,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        """Decode a latent residual back to a full-resolution residual.

        Parameters
        ----------
        r_t : torch.Tensor
            Latent residual, shape ``(B, in_channels_r, h_lat, w_lat)``.
        x_t : torch.Tensor
            Full-resolution current state + forcings, shape
            ``(B, in_channels_xt, H, W)``.

        Returns
        -------
        torch.Tensor
            Full-resolution predicted residual ``delta_t``, shape
            ``(B, out_channels, H, W)``. Sized to be added to a prognostic-
            only ``x_t`` slice.
        """
        # Capture the target output shape from the input x_t. We crop back
        # to this at the end so transfer to a different field_shape at
        # inference works without changing self.full_res_shape.
        x_t_target_shape = (int(x_t.shape[-2]), int(x_t.shape[-1]))

        # Tokenize both inputs to a stride-aligned latent grid. Compute the
        # padding dynamically from x_t's actual shape so transfer to a
        # different field_shape at inference works (the init-time
        # self._xt_pad was sized for the configured full_res_shape only).
        stride_h, stride_w = self.stride
        H_in, W_in = x_t_target_shape
        pad_h = (stride_h - H_in % stride_h) % stride_h
        pad_w = (stride_w - W_in % stride_w) % stride_w
        if pad_h > 0 or pad_w > 0:
            x_t = F.pad(x_t, (0, pad_w, 0, pad_h), mode="replicate")
        tok_x = self.xt_tokenizer(x_t)
        tok_r = self.r_tokenizer(r_t)
        tok = torch.cat([tok_x, tok_r], dim=1)  # (B, e_x + e_r, h_lat, w_lat)

        # Add positional embedding. If the token grid (h, w) differs from
        # the latent_shape configured at construction (e.g. inference on a
        # larger full_res grid than training), bicubic-interpolate the
        # registered pos_embed buffer to the new token grid first. This is
        # the ViT/DEiT/Atlas transfer-learning convention.
        B, C, h, w = tok.shape
        cfg_h, cfg_w = self.latent_shape
        if (h, w) != (cfg_h, cfg_w):
            pos_3d = resize_pos_embed(
                self.pos_embed, old_shape=(cfg_h, cfg_w), new_shape=(h, w),
            )
        else:
            pos_3d = self.pos_embed
        pos = pos_3d.to(dtype=tok.dtype).reshape(1, h, w, C).permute(0, 3, 1, 2)
        tok = tok + pos

        # Run through (placeholder) blocks with residual connections.
        for blk in self.blocks:
            tok = tok + blk(tok)

        # Project + PixelShuffle back to full-res.
        out = self.final(tok)  # (B, out_channels * stride^2, h_lat, w_lat)
        out = self.pixel_shuffle(out)  # (B, out_channels, h_lat*stride, w_lat*stride)

        # Crop to match the input x_t's spatial shape. We capture
        # x_t_target_shape from the input (NOT self.full_res_shape) so
        # transfer-learning to a different field_shape at inference works.
        # The stride-aligned padding only added pixels along the high-side
        # edges, so cropping back to (H_in, W_in) recovers the original.
        out = out[..., : x_t_target_shape[0], : x_t_target_shape[1]]
        return out


class IdentityBilinearDecoder(nn.Module):
    """Decoder placeholder that just upsamples the latent residual.

    Used for scaffold validation: an ``AnemoiAtlasModel`` with this decoder
    is equivalent to "bilinear-encode -> bilinear-decode" round-trip, which
    is the low-pass-filtered version of the input. Useful for end-to-end
    pipeline tests before the real decoder is trained.
    """

    def __init__(self, full_res_shape: tuple[int, int]) -> None:
        super().__init__()
        self.full_res_shape = tuple(full_res_shape)

    def forward(self, r_t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        # Ignore x_t; just upsample r_t to full-res via bilinear.
        del x_t
        return bilinear_upsample(r_t, target_shape=self.full_res_shape)
