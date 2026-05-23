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
from anemoi.models.models.flexible_dit import FlexibleDiT

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


class AnemoiDecoderDiTModelV2(nn.Module):
    """v2: three-path decoder addressing the latent-bottleneck pixelation.

    v1 (AnemoiDecoderDiTModel) has a single information path: a strided
    Conv2d tokenizes x_t (250x250 -> 63x63) before the DiT blocks, so any
    spatial scale finer than the latent grid is destroyed before it can
    reach the output. Result: error fields show ~4-pixel blocky residuals
    along sharp gradients (storm cells, fronts), and the trained decoder
    is only marginally better than naive bilinear upsample of r_lat.

    v2 adds two paths so the output can carry high-frequency content:

      Path 1 (bilinear baseline):
        r_bilinear = F.interpolate(r_lat, full_res, mode="bilinear")
        Guarantees output >= naive bilinear at init. The combiner only
        learns the correction on top.

      Path 2 (existing latent DiT, with bilinear-downsampled x_t):
        x_t_lat = F.interpolate(x_t, latent_shape, mode="bilinear")
        tok_x = xt_proj(x_t_lat)      # 1x1 Conv channel projection
        tok_r = r_tokenizer(r_t)
        out_latent = DiT(concat(tok_x, tok_r))
        out_full = pixel_shuffle(final(out_latent))
        Note: bilinear downsample matches r_lat's coordinate system exactly
        (vs v1's learned strided Conv2d which doesn't), removing a learning
        burden — the model doesn't have to reconcile two different
        latent representations of x_t.

      Path 3 (NEW full-res x_t skip):
        x_t_feat = small_convnet(x_t)
        Preserves high-frequency context from x_t that survives to the
        output without going through the latent bottleneck. This is what
        addresses the pixelation in error fields.

      Combiner: combiner(concat(out_full, x_t_feat)) -> correction
        Zero-init final layer so the correction starts at 0 and the
        model output starts at r_bilinear (naive baseline).

      Output: delta_pred = r_bilinear + correction

    Why each piece is load-bearing:
      * Without bilinear baseline, the decoder has to relearn smooth
        low-freq upsampling from scratch.
      * Without x_t skip, the decoder has no path to high-freq content
        and can't beat naive bilinear's pixelation floor (the latent grid
        resolution).
      * Without zero-init combiner, the model output at init is random
        and the loss landscape starts far from the bilinear basin —
        observed in v30a v1 as failure to converge to a useful state.

    See [project_v30_variant_b_design] and
    [reference_predictability_cutoffs] memories for the architectural
    context.
    """

    def __init__(
        self,
        *,
        # Either provide model_config OR explicit kwargs (matches v1 pattern).
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
        # New v2 knobs:
        x_t_skip_channels: int = 64,
        x_t_skip_kernel: int = 5,
        combiner_hidden: int = 64,
        # Noise injection (FGN-style) — captures aleatoric uncertainty at
        # storm scales (e.g., convective initiation where the LATENT r_lat
        # constrains the synoptic pattern but the precise storm location/
        # intensity is stochastic). Set to 0 for the Atlas-faithful
        # deterministic decoder. Set to 32+ to match the v30b predictive's
        # noise_vector_dim and let both models contribute to ensemble spread.
        noise_vector_dim: int = 0,
        noise_hidden: int = 64,
        **_extra_kwargs,
    ) -> None:
        super().__init__()

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
            x_t_skip_channels = int(getattr(cfg, "x_t_skip_channels", x_t_skip_channels))
            x_t_skip_kernel = int(getattr(cfg, "x_t_skip_kernel", x_t_skip_kernel))
            combiner_hidden = int(getattr(cfg, "combiner_hidden", combiner_hidden))
            noise_vector_dim = int(getattr(cfg, "noise_vector_dim", noise_vector_dim))
            noise_hidden = int(getattr(cfg, "noise_hidden", noise_hidden))

        if any(v is None for v in (full_res_shape, latent_shape, in_channels_xt,
                                   in_channels_r, out_channels)):
            error = (
                "AnemoiDecoderDiTModelV2: must provide either model_config or all "
                "of full_res_shape / latent_shape / in_channels_xt / in_channels_r / out_channels."
            )
            raise ValueError(error)

        # Standard Anemoi init kwargs: data_indices / statistics / graph_data
        # are passed by AnemoiModelInterface to every model class regardless
        # of whether the class uses them. Handle each explicitly:
        #
        # * data_indices: USED for sanity checks — catches in_channels_xt /
        #   in_channels_r / out_channels drift between config and data
        #   before it surfaces as a shape mismatch in forward.
        # * statistics: Variant B normalization keeps stats in
        #   AnemoiModelInterface.pre_processors; the model itself doesn't
        #   need them. Accept and ignore.
        # * graph_data: GNN-only. FlexibleDiT + Conv2d skip paths are
        #   grid-native — no graph. Accept and ignore.
        if data_indices is not None:
            n_full_data = len(data_indices.data.input.full)
            n_prog_data = len(data_indices.data.input.prognostic)
            if n_full_data != in_channels_xt:
                error = (
                    f"AnemoiDecoderDiTModelV2: config in_channels_xt={in_channels_xt} "
                    f"but data_indices.data.input.full has {n_full_data} entries. "
                    f"Edit data/vars or model.decoder.in_channels_xt."
                )
                raise ValueError(error)
            if n_prog_data != in_channels_r:
                error = (
                    f"AnemoiDecoderDiTModelV2: config in_channels_r={in_channels_r} "
                    f"but data_indices.data.input.prognostic has {n_prog_data} "
                    f"entries (latent residual matches prognostic count). "
                    f"Edit data/vars or model.decoder.in_channels_r."
                )
                raise ValueError(error)
            if n_prog_data != out_channels:
                error = (
                    f"AnemoiDecoderDiTModelV2: config out_channels={out_channels} "
                    f"but data_indices.data.input.prognostic has {n_prog_data} "
                    f"entries (full-res residual is prognostic-only). "
                    f"Edit data/vars or model.decoder.out_channels."
                )
                raise ValueError(error)
        del statistics, graph_data  # accepted for Anemoi compat; not used

        self.full_res_shape = tuple(full_res_shape)
        self.latent_shape = tuple(latent_shape)
        self.hidden_size = hidden_size
        self.depth = depth
        self.out_channels = out_channels
        # Expose the expected input-channel count so the task class can
        # validate batch shapes without poking into the internal FlexibleDiT.
        self.in_channels_xt = int(in_channels_xt)
        self.in_channels_r = int(in_channels_r)

        # v2 UPGRADED FROM PLACEHOLDER (2026-05-22):
        # Replaced the v1 SCAFFOLD conv blocks (GroupNorm + Conv2d + GELU +
        # Conv2d, NOT a DiT) with a real physicsnemo FlexibleDiT.
        #
        # FlexibleDiT does its own tokenize → DiT-blocks → detokenize chain
        # at full-resolution input. We feed it:
        #     concat([r_bilinear, x_t], dim=1)   # (B, out_channels + in_channels_xt, H, W)
        # It tokenizes via 4×4 strided Conv2d (patch_size=4) → 63×63 tokens,
        # runs `depth` DiT blocks with NATTEN local attention (Atlas's
        # decoder recipe: "the task of the decoder is spatially local"),
        # and uses pixel_shuffle detokenizer to upsample back to full-res.
        #
        # The decoder is deterministic (Atlas-faithful) — noise stays in
        # v30b. So conditioning_embedder='zero' (no adaLN modulation).
        stride_h = math.ceil(self.full_res_shape[0] / self.latent_shape[0])
        stride_w = math.ceil(self.full_res_shape[1] / self.latent_shape[1])
        if stride_h != stride_w:
            error = (
                f"AnemoiDecoderDiTModelV2: requires square pixel-shuffle stride, "
                f"got {stride_h}x{stride_w} from full_res={full_res_shape}/latent={latent_shape}."
            )
            raise ValueError(error)
        self.stride = (stride_h, stride_w)

        # Pad full_res_shape up to a multiple of stride so the tokenizer's
        # learnable pos_embed has the right token count (mirrors v17 dit_wrapper).
        pad_h = (stride_h - self.full_res_shape[0] % stride_h) % stride_h
        pad_w = (stride_w - self.full_res_shape[1] % stride_w) % stride_w
        padded_full_res = (self.full_res_shape[0] + pad_h, self.full_res_shape[1] + pad_w)

        # FlexibleDiT input channels: bilinear-upsampled r_t (out_channels) +
        # x_t with forcings (in_channels_xt).
        dit_in_channels = out_channels + in_channels_xt

        self.dit = FlexibleDiT(
            input_size=padded_full_res,
            in_channels=dit_in_channels,
            out_channels=out_channels,
            patch_size=stride_h,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            attention_backend="natten2d",
            # qk_norm=True adds LayerNorm to Q and K before the score
            # product (Dehghani et al. 2023 "Scaling Vision Transformers").
            # Bounds attention magnitudes regardless of input scale —
            # eliminates bf16 softmax overflow that crashed v30a-v2's
            # first 100K attempt at step 8840 with NaN loss. Adds <1% of
            # params (LayerNorm(head_dim=64) × depth × 2 = ~8K params).
            attn_kwargs={"attn_kernel": attn_kernel, "qk_norm": True},
            # NOTE: 'zero' conditioning is INCOMPATIBLE with the pixel_shuffle
            # detokenizer's ProjLayer adaLN (it expects a real (B, D)
            # conditioning vector). Use 'dit' with t=0 at forward time —
            # produces a fixed but valid conditioning vector. Matches the
            # convention v17's dit_natten.yaml documents:
            # "conditioning_embedder: dit  # unconditional (t=0 at forward
            #  time); 'zero' incompatible with ProjLayer".
            conditioning_embedder="dit",
            condition_dim=0,
            force_tokenization_fp32=False,
            detokenizer_type="pixel_shuffle",  # cross-patch refinement convs avoid the
                                                # per-patch independence pixelation
        )
        # embed_split is unused with FlexibleDiT but kept in the schema for
        # backwards compat; just consume the variable to silence linters.
        _ = embed_split

        # v2 NEW: full-res x_t skip path. Two-layer ConvNet, no downsampling.
        sk_pad = x_t_skip_kernel // 2
        self.x_t_skip = nn.Sequential(
            nn.Conv2d(in_channels_xt, x_t_skip_channels,
                      kernel_size=x_t_skip_kernel, padding=sk_pad),
            nn.GELU(),
            nn.Conv2d(x_t_skip_channels, x_t_skip_channels,
                      kernel_size=3, padding=1),
            nn.GELU(),
        )

        # v2 NEW: noise injection (FGN-style). If noise_vector_dim > 0, the
        # forward signature accepts a noise vector of shape (B, noise_vector_dim).
        # A small MLP maps it to (noise_hidden,) per-sample channel modulation
        # that gets broadcast to (B, noise_hidden, H, W) and concatenated into
        # the combiner input. This lets the decoder express aleatoric
        # uncertainty at scales finer than r_lat resolves (storm initiation
        # within a synoptic-scale convective pattern).
        self.noise_vector_dim = int(noise_vector_dim)
        self.noise_hidden = int(noise_hidden) if self.noise_vector_dim > 0 else 0
        if self.noise_vector_dim > 0:
            self.noise_encoder = nn.Sequential(
                nn.Linear(self.noise_vector_dim, self.noise_hidden),
                nn.GELU(),
                nn.Linear(self.noise_hidden, self.noise_hidden),
            )

        # v2 NEW: combiner. Sees the full-res DiT output (out_channels)
        # concatenated with the x_t skip features (x_t_skip_channels)
        # and (if enabled) the broadcasted noise features (noise_hidden).
        # Outputs a correction tensor (out_channels, H, W).
        combiner_in_ch = out_channels + x_t_skip_channels + self.noise_hidden
        self.combiner = nn.Sequential(
            nn.Conv2d(combiner_in_ch, combiner_hidden,
                      kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(combiner_hidden, out_channels, kernel_size=1),
        )

        # Zero-init the LAST conv of the combiner so the correction starts at 0.
        # FlexibleDiT's detokenizer already has adaLN-Zero internally (its
        # `adaptive_modulation` zeros out the final scale/shift), so out_full
        # ≈ 0 at init regardless. The combiner zero-init guarantees
        # correction = 0 → output starts EXACTLY at the bilinear baseline.
        nn.init.zeros_(self.combiner[-1].weight)
        nn.init.zeros_(self.combiner[-1].bias)

    def forward(
        self,
        r_t: torch.Tensor,
        x_t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode latent residual + full-res state -> full-res residual.

        Three-path forward (see class docstring).

        Parameters
        ----------
        r_t : torch.Tensor
            ``(B, in_channels_r, h_lat, w_lat)`` latent residual.
        x_t : torch.Tensor
            ``(B, in_channels_xt, H, W)`` full-resolution state + forcings.
        noise : Optional[torch.Tensor]
            ``(B, noise_vector_dim)`` FGN-style noise vector. Only used if
            ``noise_vector_dim > 0`` (default 0 = deterministic Atlas-faithful
            decoder). If None and noise_vector_dim > 0, samples N(0, I).

        Returns
        -------
        torch.Tensor
            ``(B, out_channels, H, W)`` matching the input ``x_t``'s spatial shape.
        """
        H, W = int(x_t.shape[-2]), int(x_t.shape[-1])

        # Path 1: bilinear baseline (the floor we never want to drop below).
        r_bilinear = F.interpolate(
            r_t, size=(H, W), mode="bilinear", align_corners=False,
        )

        # Path 2: real DiT (FlexibleDiT with NATTEN local attention).
        # Concatenate r_bilinear (already at full-res) with x_t and feed to
        # FlexibleDiT. It will tokenize via 4×4 strided Conv2d → 63×63 tokens,
        # run NATTEN-DiT blocks (Atlas decoder recipe), then pixel_shuffle
        # detokenize back to full-res. Pad input to a multiple of stride if
        # needed; FlexibleDiT crops back via its detokenizer convention.
        s_h, s_w = self.stride
        pad_h = (s_h - H % s_h) % s_h
        pad_w = (s_w - W % s_w) % s_w
        dit_in = torch.cat([r_bilinear, x_t], dim=1)  # (B, out + xt_in, H, W)
        if pad_h or pad_w:
            dit_in = F.pad(dit_in, (0, pad_w, 0, pad_h), mode="replicate")

        # FlexibleDiT requires (x, t[, condition]). decoder is deterministic
        # (conditioning_embedder='zero') — pass t=zeros, condition=None.
        B_in = dit_in.shape[0]
        t0 = torch.zeros(B_in, device=dit_in.device, dtype=dit_in.dtype)
        out_full = self.dit(dit_in, t0, condition=None)   # (B, out_channels, padded H, padded W)
        out_full = out_full[..., :H, :W]                  # crop back

        # Path 3: full-res x_t skip.
        x_t_feat = self.x_t_skip(x_t)        # (B, skip_ch, H, W)

        # Build combiner input. If noise injection enabled, broadcast a
        # per-sample noise feature map and concat. Default config (no noise)
        # leaves the decoder deterministic — stochasticity lives in v30b.
        combiner_inputs = [out_full, x_t_feat]
        if self.noise_vector_dim > 0:
            B = r_t.shape[0]
            if noise is None:
                noise = torch.randn(
                    (B, self.noise_vector_dim),
                    device=r_t.device, dtype=r_t.dtype,
                )
            elif noise.shape != (B, self.noise_vector_dim):
                error = (
                    f"AnemoiDecoderDiTModelV2: noise shape mismatch — got "
                    f"{tuple(noise.shape)}, expected ({B}, {self.noise_vector_dim})."
                )
                raise ValueError(error)
            noise_feat = self.noise_encoder(noise)                # (B, noise_hidden)
            noise_feat = noise_feat.view(B, -1, 1, 1).expand(-1, -1, H, W)
            combiner_inputs.append(noise_feat)
        elif noise is not None:
            # User passed noise but model wasn't configured to use it; warn
            # rather than silently ignore.
            LOGGER.warning(
                "AnemoiDecoderDiTModelV2: received noise but noise_vector_dim=0; "
                "ignoring. Set noise_vector_dim > 0 in the config to enable noise.",
            )

        correction = self.combiner(torch.cat(combiner_inputs, dim=1))
        return r_bilinear + correction


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
