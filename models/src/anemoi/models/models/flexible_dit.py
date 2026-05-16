"""Domain-size-independent DiT for transfer learning (train on patch, infer on CONUS).

The stock physicsnemo DiT hardcodes spatial dimensions at init time:
  - latent_hw for NATTEN attention (dit.py:233-237)
  - h_patches/w_patches for the detokenizer reshape (dit_layers.py:1054-1055)

FlexibleDiT overrides forward() to compute these dynamically from the actual
input tensor shape, enabling inference at arbitrary resolutions without
re-initializing the model.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from physicsnemo.models.dit.dit import DiT
from physicsnemo.nn.module.dit_layers import (
    PatchEmbed2DTokenizer,
    ProjReshape2DDetokenizer,
)


class FlexiblePatchEmbed2DTokenizer(PatchEmbed2DTokenizer):
    """PatchEmbed2DTokenizer that interpolates learnable pos_embed to runtime grid.

    Standard ViT trick (Dosovitskiy et al. 2020) for evaluating a model
    trained at one (h_patches, w_patches) at any other spatial resolution.
    The learned positional embedding is reshaped to its training-time
    spatial grid, bicubic-interpolated to the runtime grid, and flattened
    back to a (1, L_runtime, D) tensor before being added to the tokens.

    When the runtime grid matches training, this is a no-op identical to
    the base PatchEmbed2DTokenizer (passes the original pos_embed through).
    """

    def forward(self, x):
        x_emb = self.x_embedder(x)             # (B, D, Hp, Wp)
        B, D, Hp, Wp = x_emb.shape
        tokens = x_emb.flatten(2).transpose(1, 2)  # (B, L, D)
        if isinstance(self.pos_embed, nn.Parameter):
            train_h, train_w = self.h_patches, self.w_patches
            if (Hp, Wp) != (train_h, train_w):
                # (1, train_h*train_w, D) -> (1, D, train_h, train_w)
                pe = self.pos_embed.transpose(1, 2).reshape(1, D, train_h, train_w)
                # Bicubic-interpolate to runtime grid
                pe = F.interpolate(pe, size=(Hp, Wp), mode="bicubic", align_corners=False)
                # back to (1, Hp*Wp, D)
                pe = pe.flatten(2).transpose(1, 2)
                tokens = tokens + pe
            else:
                tokens = tokens + self.pos_embed
        else:
            tokens = tokens + self.pos_embed
        return tokens


class _AdaLNModulation2D(nn.Module):
    """LayerNorm + adaLN (scale, shift) on a 4-D feature map.

    Used by the PixelShuffle detokenizer to inject conditioning before the
    upsample head, matching DiT's adaLN modulation pattern.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 2 * channels, bias=True),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, feat: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # feat: (B, C, H, W); c: (B, C)
        scale, shift = self.modulation(c).chunk(2, dim=1)
        feat_n = self.norm(feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return feat_n * (1 + scale[..., None, None]) + shift[..., None, None]


class FlexiblePixelShuffleDetokenizer(nn.Module):
    """Sub-pixel-conv detokenizer; replaces the per-token Linear+reshape head.

    Architecture (Tier-A1 from the literature survey, the standard ViT
    pixel-shuffle decoder pattern that fixes patch-grid artefacts):

        tokens (B, L, hidden) -> reshape (B, hidden, h_patches, w_patches)
            -> adaLN modulation
            -> Conv2d 1x1 (hidden -> refine_channels * patch_size**2)
            -> PixelShuffle(patch_size)         # spatial up-sample
            -> Conv2d 3x3, GELU, Conv2d 3x3     # cross-patch refinement

    Each output cell is a function of multiple tokens via the post-shuffle
    3x3 convs, so the per-patch independence that drives the patch-grid
    pixelation in the standard ProjReshape2DDetokenizer is broken.

    Avoids the ConvTranspose2d k=2s checkerboard pattern (Odena et al.,
    Distill 2016) by using PixelShuffle for the upsample.

    Spatial-dim agnostic via h_patches / w_patches forward kwargs, matching
    FlexibleProjReshape2DDetokenizer.
    """

    def __init__(
        self,
        input_size: tuple,
        patch_size,
        out_channels: int,
        hidden_size: int,
        refine_channels: int = 128,
        conv_kernel: int = 3,
        n_conv_layers: int = 2,
        layernorm_backend: str = "torch",  # accepted for API parity, unused
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = (
            patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
        )
        ph, pw = self.patch_size
        if ph != pw:
            raise NotImplementedError("PixelShuffle requires square patch_size.")
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.refine_channels = refine_channels

        self.h_patches = self.input_size[0] // ph
        self.w_patches = self.input_size[1] // pw

        self.adaln = _AdaLNModulation2D(hidden_size)
        self.proj = nn.Conv2d(hidden_size, refine_channels * ph * pw, kernel_size=1)
        self.shuffle = nn.PixelShuffle(ph)
        # Build conv refinement stack with configurable kernel + depth
        layers: list[nn.Module] = [nn.GELU()]
        pad = conv_kernel // 2
        for _ in range(n_conv_layers - 1):
            layers.append(
                nn.Conv2d(refine_channels, refine_channels,
                          kernel_size=conv_kernel, padding=pad)
            )
            layers.append(nn.GELU())
        layers.append(
            nn.Conv2d(refine_channels, out_channels,
                      kernel_size=conv_kernel, padding=pad)
        )
        self.refine = nn.Sequential(*layers)
        # Zero-init the final conv so the head starts producing zeros — keeps
        # warm-start training stable when the rest of the model is loaded
        # from a v10c-style checkpoint that hasn't seen this head.
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def initialize_weights(self):
        """Match the parent ProjReshape2DDetokenizer API."""
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)
        # adaln modulation already zero-init'd in _AdaLNModulation2D.__init__

    def forward(
        self,
        x_tokens: Float[torch.Tensor, "batch sequence hidden_size"],
        c: Float[torch.Tensor, "batch hidden_size"],
        h_patches: Optional[int] = None,
        w_patches: Optional[int] = None,
    ) -> Float[torch.Tensor, "batch out_channels height width"]:
        hp = h_patches if h_patches is not None else self.h_patches
        wp = w_patches if w_patches is not None else self.w_patches
        B, L, D = x_tokens.shape
        feat = x_tokens.transpose(1, 2).reshape(B, D, hp, wp)
        feat = self.adaln(feat, c)
        feat = self.proj(feat)
        feat = self.shuffle(feat)  # (B, refine_channels, hp*ph, wp*pw)
        feat = self.refine(feat)
        return feat


class FlexibleConvTransposeDetokenizer(nn.Module):
    """Direct ConvTranspose2d detokenizer with kernel > stride for cross-patch overlap.

    Tier-A1 candidate: kernel=12, stride=4 (k=3s) avoids the Odena k=2s
    checkerboard while giving 3x overlap between adjacent output windows.
    Each output cell is a learned blend of up to 12 token contributions in
    each axis (effective receptive field ~50 km, ~3x patch wavelength).
    """

    def __init__(
        self,
        input_size: tuple,
        patch_size,
        out_channels: int,
        hidden_size: int,
        refine_channels: int = 128,
        kernel_size: int = 12,
        layernorm_backend: str = "torch",
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = (
            patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
        )
        ph, pw = self.patch_size
        if ph != pw:
            raise NotImplementedError("ConvTranspose detokenizer requires square patch_size.")
        self.out_channels = out_channels
        self.hidden_size = hidden_size

        self.h_patches = self.input_size[0] // ph
        self.w_patches = self.input_size[1] // pw

        self.adaln = _AdaLNModulation2D(hidden_size)
        self.proj = nn.Conv2d(hidden_size, refine_channels, kernel_size=1)
        # ConvTranspose with kernel>stride: padding = (kernel - stride) // 2 to keep
        # output spatial dim = stride * input_dim. For k=12, s=4: padding = 4.
        padding = (kernel_size - ph) // 2
        self.deconv = nn.ConvTranspose2d(
            refine_channels, out_channels,
            kernel_size=kernel_size, stride=ph, padding=padding,
        )
        # Zero-init final deconv for warm-start stability
        nn.init.zeros_(self.deconv.weight)
        nn.init.zeros_(self.deconv.bias)

    def initialize_weights(self):
        nn.init.zeros_(self.deconv.weight)
        nn.init.zeros_(self.deconv.bias)

    def forward(
        self,
        x_tokens: Float[torch.Tensor, "batch sequence hidden_size"],
        c: Float[torch.Tensor, "batch hidden_size"],
        h_patches: Optional[int] = None,
        w_patches: Optional[int] = None,
    ) -> Float[torch.Tensor, "batch out_channels height width"]:
        hp = h_patches if h_patches is not None else self.h_patches
        wp = w_patches if w_patches is not None else self.w_patches
        B, L, D = x_tokens.shape
        feat = x_tokens.transpose(1, 2).reshape(B, D, hp, wp)
        feat = self.adaln(feat, c)
        feat = self.proj(feat)
        return self.deconv(feat)


class FlexibleBilinearConvDetokenizer(nn.Module):
    """Bilinear upsample + learned conv stack.

    Tier-A1 candidate (c). Cheaper than ConvTranspose: bilinear interpolation
    is non-learnable but smooth across patches; the post-upsample 3x3 convs
    (with the learned features carried through) recover detail.
    """

    def __init__(
        self,
        input_size: tuple,
        patch_size,
        out_channels: int,
        hidden_size: int,
        refine_channels: int = 128,
        conv_kernel: int = 3,
        n_layers: int = 2,
        layernorm_backend: str = "torch",
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = (
            patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
        )
        ph, pw = self.patch_size
        if ph != pw:
            raise NotImplementedError("Bilinear detokenizer requires square patch_size.")
        self.out_channels = out_channels
        self.hidden_size = hidden_size

        self.h_patches = self.input_size[0] // ph
        self.w_patches = self.input_size[1] // pw
        self.scale = ph

        self.adaln = _AdaLNModulation2D(hidden_size)
        self.proj = nn.Conv2d(hidden_size, refine_channels, kernel_size=1)
        layers: list[nn.Module] = [nn.GELU()]
        for i in range(n_layers - 1):
            layers.append(
                nn.Conv2d(refine_channels, refine_channels,
                          kernel_size=conv_kernel, padding=conv_kernel // 2)
            )
            layers.append(nn.GELU())
        layers.append(
            nn.Conv2d(refine_channels, out_channels,
                      kernel_size=conv_kernel, padding=conv_kernel // 2)
        )
        self.refine = nn.Sequential(*layers)
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def initialize_weights(self):
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(
        self,
        x_tokens: Float[torch.Tensor, "batch sequence hidden_size"],
        c: Float[torch.Tensor, "batch hidden_size"],
        h_patches: Optional[int] = None,
        w_patches: Optional[int] = None,
    ) -> Float[torch.Tensor, "batch out_channels height width"]:
        hp = h_patches if h_patches is not None else self.h_patches
        wp = w_patches if w_patches is not None else self.w_patches
        B, L, D = x_tokens.shape
        feat = x_tokens.transpose(1, 2).reshape(B, D, hp, wp)
        feat = self.adaln(feat, c)
        feat = self.proj(feat)
        feat = F.interpolate(feat, scale_factor=self.scale, mode="bilinear",
                             align_corners=False)
        return self.refine(feat)


class FlexibleHierarchicalDetokenizer(nn.Module):
    """Two-stage hierarchical upsample: PixelShuffle 2x -> conv -> PixelShuffle 2x -> conv.

    Tier-A3 candidate. Each stage halves the upsample factor, with conv layers
    between stages mixing across the (now finer) spatial grid. Inspired by
    SegFormer / Aurora multi-scale decoder pattern.

    For patch_size=4: stage1 upsamples 2x (h_patches -> 2*h_patches), stage2
    upsamples another 2x (final = 4*h_patches = full spatial dim).
    """

    def __init__(
        self,
        input_size: tuple,
        patch_size,
        out_channels: int,
        hidden_size: int,
        refine_channels: int = 128,
        layernorm_backend: str = "torch",
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = (
            patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
        )
        ph, pw = self.patch_size
        if ph != 4 or pw != 4:
            raise NotImplementedError("Hierarchical 2-stage detokenizer requires patch_size=4.")
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.h_patches = self.input_size[0] // ph
        self.w_patches = self.input_size[1] // pw

        self.adaln = _AdaLNModulation2D(hidden_size)
        # Stage 1: hidden -> refine*4 channels (for 2x PixelShuffle)
        self.proj1 = nn.Conv2d(hidden_size, refine_channels * 4, kernel_size=1)
        self.shuffle1 = nn.PixelShuffle(2)
        self.refine1 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(refine_channels, refine_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        # Stage 2: refine -> refine*4 channels (for another 2x PixelShuffle) -> out
        self.proj2 = nn.Conv2d(refine_channels, refine_channels * 4, kernel_size=1)
        self.shuffle2 = nn.PixelShuffle(2)
        self.refine2 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(refine_channels, refine_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(refine_channels, out_channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.refine2[-1].weight)
        nn.init.zeros_(self.refine2[-1].bias)

    def initialize_weights(self):
        nn.init.zeros_(self.refine2[-1].weight)
        nn.init.zeros_(self.refine2[-1].bias)

    def forward(
        self,
        x_tokens: Float[torch.Tensor, "batch sequence hidden_size"],
        c: Float[torch.Tensor, "batch hidden_size"],
        h_patches: Optional[int] = None,
        w_patches: Optional[int] = None,
    ) -> Float[torch.Tensor, "batch out_channels height width"]:
        hp = h_patches if h_patches is not None else self.h_patches
        wp = w_patches if w_patches is not None else self.w_patches
        B, L, D = x_tokens.shape
        feat = x_tokens.transpose(1, 2).reshape(B, D, hp, wp)
        feat = self.adaln(feat, c)
        feat = self.proj1(feat)
        feat = self.shuffle1(feat)              # (B, refine, hp*2, wp*2)
        feat = self.refine1(feat)
        feat = self.proj2(feat)
        feat = self.shuffle2(feat)              # (B, refine, hp*4, wp*4)
        return self.refine2(feat)


class FlexibleProjReshape2DDetokenizer(ProjReshape2DDetokenizer):
    """Detokenizer that infers spatial dims from (h_patches, w_patches) at runtime.

    The base ProjReshape2DDetokenizer uses self.h_patches and self.w_patches
    (set at init from input_size). This subclass accepts them as forward kwargs
    so the same weights work for any spatial resolution.
    """

    def forward(
        self,
        x_tokens: Float[torch.Tensor, "batch sequence hidden_size"],
        c: Float[torch.Tensor, "batch cond_dim"],
        h_patches: Optional[int] = None,
        w_patches: Optional[int] = None,
    ) -> Float[torch.Tensor, "batch out_channels height width"]:
        hp = h_patches if h_patches is not None else self.h_patches
        wp = w_patches if w_patches is not None else self.w_patches

        # Project tokens to per-patch pixel embeddings
        x = self.proj_layer(x_tokens, c)  # (B, L, p0*p1*C_out)

        # Reshape back to image
        x = x.reshape(
            x.shape[0],
            hp,
            wp,
            self.patch_size[0],
            self.patch_size[1],
            self.out_channels,
        )
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(
            x.shape[0],
            self.out_channels,
            hp * self.patch_size[0],
            wp * self.patch_size[1],
        )
        return x


class FlexibleDiT(DiT):
    """DiT with dynamic spatial dimensions for domain-size-independent inference.

    Overrides forward() to:
    1. Compute latent_hw from the actual input tensor (not stored at init)
    2. Pass dynamic h_patches/w_patches to the detokenizer
    3. Swap the detokenizer to FlexibleProjReshape2DDetokenizer at init

    All other behavior (tokenizer, conditioning, blocks) is inherited from DiT.
    """

    @staticmethod
    def _modulation_fn(x, scale, shift):
        """Picklable replacement for lambda modulation in DiTBlock and ProjLayer."""
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def __init__(self, *args, detokenizer_type: str = "linear_reshape", **kwargs):
        """Build a FlexibleDiT.

        Parameters
        ----------
        detokenizer_type : str
            "linear_reshape" (default): the original ProjReshape2DDetokenizer
                (per-token Linear -> reshape, no cross-patch blending).
                Mathematically equivalent to a strided conv-transpose with
                kernel == stride == patch_size; structurally pixelates.
            "pixel_shuffle": FlexiblePixelShuffleDetokenizer
                (1x1 conv -> PixelShuffle -> 3x3 conv x2). Each output cell
                is a function of multiple tokens via the post-shuffle 3x3
                convs, breaking the per-patch independence.
        """
        super().__init__(*args, **kwargs)

        # Swap the tokenizer for the FlexiblePatchEmbed2DTokenizer if it has a
        # learnable pos_embed. This makes the model resolution-agnostic at
        # inference: a checkpoint trained at e.g. 250x250 patches can be
        # evaluated on full-CONUS 992x1524 by bicubic-interpolating the
        # learned pos_embed to the new token grid. Standard ViT trick.
        if (
            isinstance(self.tokenizer, PatchEmbed2DTokenizer)
            and isinstance(self.tokenizer.pos_embed, nn.Parameter)
            and not isinstance(self.tokenizer, FlexiblePatchEmbed2DTokenizer)
        ):
            orig_tok = self.tokenizer
            flex_tok = FlexiblePatchEmbed2DTokenizer(
                input_size=orig_tok.input_size,
                patch_size=orig_tok.patch_size,
                in_channels=orig_tok.in_channels,
                hidden_size=orig_tok.hidden_size,
                pos_embed="learnable",
            )
            flex_tok.load_state_dict(orig_tok.state_dict())
            self.tokenizer = flex_tok

        # Replace the detokenizer with the flexible version.
        if isinstance(self.detokenizer, ProjReshape2DDetokenizer):
            orig = self.detokenizer
            common = dict(
                input_size=orig.input_size,
                patch_size=orig.patch_size,
                out_channels=orig.out_channels,
                hidden_size=orig.hidden_size,
            )
            if detokenizer_type == "linear_reshape":
                flexible = FlexibleProjReshape2DDetokenizer(**common)
                flexible.load_state_dict(orig.state_dict())
                self.detokenizer = flexible
            elif detokenizer_type == "pixel_shuffle":
                # Backward-compat alias: 3x3 conv x 2.
                self.detokenizer = FlexiblePixelShuffleDetokenizer(
                    **common, conv_kernel=3, n_conv_layers=2,
                )
            elif detokenizer_type == "pixel_shuffle_3x3x2":
                self.detokenizer = FlexiblePixelShuffleDetokenizer(
                    **common, conv_kernel=3, n_conv_layers=2,
                )
            elif detokenizer_type == "pixel_shuffle_5x5x2":
                self.detokenizer = FlexiblePixelShuffleDetokenizer(
                    **common, conv_kernel=5, n_conv_layers=2,
                )
            elif detokenizer_type == "pixel_shuffle_7x7x1":
                self.detokenizer = FlexiblePixelShuffleDetokenizer(
                    **common, conv_kernel=7, n_conv_layers=1,
                )
            elif detokenizer_type == "conv_transpose_k12_s4":
                self.detokenizer = FlexibleConvTransposeDetokenizer(
                    **common, kernel_size=12,
                )
            elif detokenizer_type == "bilinear_3x3x2":
                self.detokenizer = FlexibleBilinearConvDetokenizer(
                    **common, conv_kernel=3, n_layers=2,
                )
            elif detokenizer_type == "hierarchical_2stage":
                self.detokenizer = FlexibleHierarchicalDetokenizer(**common)
            else:
                raise ValueError(
                    f"Unknown detokenizer_type={detokenizer_type!r}. Expected one of "
                    "linear_reshape, pixel_shuffle, pixel_shuffle_3x3x2, "
                    "pixel_shuffle_5x5x2, pixel_shuffle_7x7x1, "
                    "conv_transpose_k12_s4, bilinear_3x3x2, hierarchical_2stage."
                )

        # Replace unpicklable lambdas with a proper static method so torch.save works.
        # Must come AFTER detokenizer swap so the new proj_layer is patched (only
        # applies to the linear_reshape head; pixel_shuffle has no proj_layer).
        for block in self.blocks:
            block.modulation = self._modulation_fn
        if hasattr(self.detokenizer, 'proj_layer'):
            self.detokenizer.proj_layer.modulation = self._modulation_fn

    def forward(
        self,
        x: Float[torch.Tensor, "batch in_channels *spatial_dims"],
        t: Float[torch.Tensor, " batch"],
        condition: Optional[Float[torch.Tensor, "batch condition_dim"]] = None,
        p_dropout: Optional[float | Float[torch.Tensor, " batch"]] = None,
        attn_kwargs: Dict[str, Any] = {},
        tokenizer_kwargs: Dict[str, Any] = {},
    ) -> Float[torch.Tensor, "batch out_channels *spatial_dims"]:
        # Compute dynamic latent_hw from actual input shape
        H, W = x.shape[-2], x.shape[-1]
        ps_h, ps_w = self.patch_size
        h_patches = H // ps_h
        w_patches = W // ps_w
        dynamic_latent_hw = (h_patches, w_patches)

        # Tokenize: (B, C, H, W) -> (B, L, D)
        if self.force_tokenization_fp32:
            dtype = x.dtype
            x = x.to(torch.float32)
            with torch.autocast(device_type="cuda", enabled=False):
                x = self.tokenizer(x, **tokenizer_kwargs)
            x = x.to(dtype)
        else:
            x = self.tokenizer(x, **tokenizer_kwargs)

        # Compute conditioning embedding
        c = self.conditioning_embedder(t, condition=condition)  # (B, D) or (B, 0)

        # Override latent_hw with dynamic value (only relevant for NATTEN backend)
        if self.attn_kwargs_forward:
            # NATTEN backend: override stored latent_hw with dynamic value
            merged_attn_kwargs = {**self.attn_kwargs_forward, "latent_hw": dynamic_latent_hw, **attn_kwargs}
        else:
            # timm / transformer_engine: no latent_hw needed
            merged_attn_kwargs = {**attn_kwargs}

        for block in self.blocks:
            x = block(
                x,
                c,
                p_dropout=p_dropout,
                attn_kwargs=merged_attn_kwargs,
            )  # (B, L, D)

        # De-tokenize with dynamic spatial dims
        if self.force_tokenization_fp32:
            dtype = x.dtype
            x = x.to(torch.float32)
            c_fp32 = c.to(torch.float32)  # detokenizer's adaptive_modulation Linear is fp32 under this flag
            with torch.autocast(device_type="cuda", enabled=False):
                x = self.detokenizer(x, c_fp32, h_patches=h_patches, w_patches=w_patches)
            x = x.to(dtype)
        else:
            x = self.detokenizer(x, c, h_patches=h_patches, w_patches=w_patches)

        return x
