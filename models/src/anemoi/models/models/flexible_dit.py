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


class OverlappingPatchEmbed2DTokenizer(FlexiblePatchEmbed2DTokenizer):
    """Tokenizer with **overlapping** input patches via kernel_size > stride.

    Standard `PatchEmbed2DTokenizer` uses `Conv2d(kernel=patch_size, stride=patch_size)`,
    so each output token encodes a disjoint patch_size × patch_size block of
    input cells. This commits the input field to a hard 16 km (patch_size=4 at
    4 km grid) partition, and any downstream pixelation in the output is a
    structural consequence of that partition — no decoder can fully erase it.

    This subclass replaces the inner `Conv2d` with a wider-kernel version
    (`kernel_size > patch_size`) while keeping `stride = patch_size` so the
    token grid size is unchanged. Adjacent tokens then **share input cells**:
    with patch_size=4, kernel_size=8 each token sees an 8×8 cell region with
    its 4-cell-wide grid spacing, so neighbors share 4 of every 8 cells (50%
    overlap per axis). This structurally breaks per-token spatial
    independence at the source.

    Parameters
    ----------
    kernel_size : int
        Receptive field of each token, in input cells. Must be ≥ patch_size,
        and (kernel_size − patch_size) must be even so symmetric padding can
        keep the output grid size identical to the non-overlapping case.

    Notes
    -----
    Padding is set to `(kernel_size − patch_size) // 2` on each side so the
    output spatial size matches `input // patch_size`. For input 252×252,
    patch_size=4, kernel_size=8 → padding=2 → output 63×63 tokens (same as
    stock). At the boundary 2-cell reflection padding from `nn.ZeroPad2d` is
    *not* applied here (PatchEmbed2D already handles boundary residuals via
    the upstream `_pad_to_patch_size` reflect-pad in dit_wrapper); the inner
    conv padding is zero on the field, which is acceptable because the
    LAM-interior loss masks the boundary cells anyway.
    """

    def __init__(
        self,
        *,
        kernel_size: int,
        input_size,
        patch_size,
        in_channels: int,
        hidden_size: int,
        pos_embed: str = "learnable",
    ):
        super().__init__(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            pos_embed=pos_embed,
        )
        ph, pw = self.patch_size if isinstance(self.patch_size, tuple) else (
            self.patch_size, self.patch_size
        )
        if ph != pw:
            raise NotImplementedError("Overlapping tokenizer requires square patches.")
        if kernel_size < ph or (kernel_size - ph) % 2 != 0:
            raise ValueError(
                f"kernel_size={kernel_size} must be >= patch_size={ph} and "
                f"(kernel_size - patch_size) must be even for symmetric padding."
            )

        padding = (kernel_size - ph) // 2
        # Replace the inner Conv2d. Initialize from Kaiming-normal because the
        # kernel shape changed and we can't load the old (k=patch_size) weights.
        new_proj = nn.Conv2d(
            self.in_channels, self.hidden_size,
            kernel_size=kernel_size, stride=ph, padding=padding,
        )
        nn.init.kaiming_normal_(new_proj.weight, mode="fan_out", nonlinearity="linear")
        nn.init.zeros_(new_proj.bias)
        self.x_embedder.proj = new_proj
        self.tokenizer_kernel_size = kernel_size


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


class FlexibleHierarchicalResizeConvDetokenizer(nn.Module):
    """Two-stage hierarchical upsample, bilinear-resize variant of
    FlexibleHierarchicalDetokenizer (no PixelShuffle).

    Identical structure to FlexibleHierarchicalDetokenizer EXCEPT the two 2x
    PixelShuffle upsamples are replaced by bilinear ``F.interpolate``. This
    removes the sub-pixel-convolution checkerboard artifact (Odena et al. 2016):
    PixelShuffle's adjacent output cells come from independently-learned
    sub-kernels and can carry systematic offsets -> periodic checkerboard at the
    upsample stride. Bilinear resize is smooth across cells, so the only
    small-scale structure is what the post-resize 3x3 refine convs add.

    Layer names match FlexibleHierarchicalDetokenizer (adaln, proj1, refine1,
    proj2, refine2) so adaln/refine1/refine2 weights transfer verbatim from a
    trained hierarchical_2stage checkpoint; only proj1/proj2 change shape
    (refine_channels instead of refine_channels*4, since bilinear keeps the
    channel count while PixelShuffle divides it by 4). The natural warm-start for
    proj1/proj2 is the mean over the 4 PixelShuffle sub-kernel groups of the
    source checkpoint — that is exactly the checkerboard-free projection.
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
            raise NotImplementedError(
                "Hierarchical 2-stage resize-conv detokenizer requires patch_size=4."
            )
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.h_patches = self.input_size[0] // ph
        self.w_patches = self.input_size[1] // pw

        self.adaln = _AdaLNModulation2D(hidden_size)
        # Stage 1: hidden -> refine channels, then bilinear 2x (no *4 for shuffle).
        self.proj1 = nn.Conv2d(hidden_size, refine_channels, kernel_size=1)
        self.refine1 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(refine_channels, refine_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        # Stage 2: refine -> refine channels, bilinear 2x, refine -> out.
        self.proj2 = nn.Conv2d(refine_channels, refine_channels, kernel_size=1)
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
        feat = F.interpolate(feat, scale_factor=2, mode="bilinear", align_corners=False)
        feat = self.refine1(feat)
        feat = self.proj2(feat)
        feat = F.interpolate(feat, scale_factor=2, mode="bilinear", align_corners=False)
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
        """Picklable replacement for lambda modulation in DiTBlock and ProjLayer.

        Supports both 2-D (B, D) and 3-D (B, L, D) conditioning. The 2-D form is
        FGN-style global noise / time embedding (one modulation per sample
        broadcast across tokens); the 3-D form is AIFS-style per-token noise.
        """
        if scale.ndim == 3:
            # Per-token: scale/shift already (B, L, D); direct multiply.
            return x * (1 + scale) + shift
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    @staticmethod
    def _block_forward_per_token(
        block,
        x,
        c,
        attn_kwargs: Optional[Dict[str, Any]] = None,
        p_dropout: Optional[float] = None,
    ):
        """Mirror physicsnemo DiTBlock.forward but for 3-D per-token conditioning.

        Two differences vs the upstream forward (which assumes 2-D c):
        1. ``adaptive_modulation(c).chunk(6, dim=1)`` would split the L dim when
           c is (B, L, D); use ``dim=-1`` instead.
        2. ``attention_gate.unsqueeze(1)`` would inject an extra dim when gate
           is already (B, L, D); skip the unsqueeze.

        block.modulation is shared with the 2-D path via ``_modulation_fn``,
        which is ndim-aware.
        """
        (
            attention_shift,
            attention_scale,
            attention_gate,
            mlp_shift,
            mlp_scale,
            mlp_gate,
        ) = block.adaptive_modulation(c).chunk(6, dim=-1)

        # Attention block (modulated norm -> attention -> gated residual)
        modulated_attn_input = block.modulation(
            block.pre_attention_norm(x), attention_scale, attention_shift,
        )
        if block.interdrop is not None:
            modulated_attn_input = block.interdrop(modulated_attn_input, p_dropout)
        elif p_dropout is not None:
            raise ValueError(
                "p_dropout passed to DiTBlock but intermediate_dropout is disabled"
            )

        attention_output = block.attention(
            modulated_attn_input, **(attn_kwargs or {}),
        )
        # gate is (B, L, D); DropPath broadcasts (B, 1, 1) across (L, D). No unsqueeze.
        x = torch.addcmul(x, block.drop_path(attention_gate), attention_output)

        # MLP block
        modulated_mlp_input = block.modulation(
            block.pre_mlp_norm(x), mlp_scale, mlp_shift,
        )
        mlp_output = block.linear(modulated_mlp_input)
        x = torch.addcmul(x, block.drop_path(mlp_gate), mlp_output)
        return x

    def __init__(
        self,
        *args,
        detokenizer_type: str = "linear_reshape",
        tokenizer_kernel_size: Optional[int] = None,
        **kwargs,
    ):
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
            tok_kwargs = dict(
                input_size=orig_tok.input_size,
                patch_size=orig_tok.patch_size,
                in_channels=orig_tok.in_channels,
                hidden_size=orig_tok.hidden_size,
                pos_embed="learnable",
            )
            ph = (
                orig_tok.patch_size[0] if isinstance(orig_tok.patch_size, tuple)
                else orig_tok.patch_size
            )
            if tokenizer_kernel_size is not None and tokenizer_kernel_size > ph:
                # Overlapping tokenizer: kernel > stride. Cannot load old weights
                # (kernel shape changed), so this path requires from-scratch
                # training. Pos_embed and other buffers transfer cleanly via
                # state_dict load below, since the token grid size is unchanged.
                flex_tok = OverlappingPatchEmbed2DTokenizer(
                    kernel_size=tokenizer_kernel_size, **tok_kwargs,
                )
                # Load only the pos_embed (kernel weights have different shape).
                with torch.no_grad():
                    flex_tok.pos_embed.copy_(orig_tok.pos_embed)
            else:
                flex_tok = FlexiblePatchEmbed2DTokenizer(**tok_kwargs)
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
            elif detokenizer_type == "hierarchical_2stage_resizeconv":
                self.detokenizer = FlexibleHierarchicalResizeConvDetokenizer(**common)
            else:
                raise ValueError(
                    f"Unknown detokenizer_type={detokenizer_type!r}. Expected one of "
                    "linear_reshape, pixel_shuffle, pixel_shuffle_3x3x2, "
                    "pixel_shuffle_5x5x2, pixel_shuffle_7x7x1, "
                    "conv_transpose_k12_s4, bilinear_3x3x2, hierarchical_2stage, "
                    "hierarchical_2stage_resizeconv."
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

        # Compute conditioning embedding.
        # The conditioning_embedder may be a _PassthroughConditionEmbedder
        # (AnemoiDiTModel swaps it in when noise conditioning is enabled), in
        # which case ``c`` inherits the shape of ``condition`` — either (B, D)
        # for FGN-style global noise or (B, L, D) for AIFS-style per-token noise.
        c = self.conditioning_embedder(t, condition=condition)

        # Override latent_hw with dynamic value (only relevant for NATTEN backend)
        if self.attn_kwargs_forward:
            # NATTEN backend: override stored latent_hw with dynamic value
            merged_attn_kwargs = {**self.attn_kwargs_forward, "latent_hw": dynamic_latent_hw, **attn_kwargs}
        else:
            # timm / transformer_engine: no latent_hw needed
            merged_attn_kwargs = {**attn_kwargs}

        # Dispatch on c.ndim: 3-D = per-token (AIFS), 2-D = global (FGN / time emb).
        if c.ndim == 3:
            for block in self.blocks:
                x = self._block_forward_per_token(
                    block, x, c,
                    attn_kwargs=merged_attn_kwargs,
                    p_dropout=p_dropout,
                )
            # Detokenizer's adaptive_modulation expects (B, D). Mean-pool the
            # per-token conditioning over the token axis so the detokenizer
            # path stays unchanged. AIFS only conditions the processor on
            # per-token noise; the detokenizer sees a global summary.
            c_for_detok = c.mean(dim=1)
        else:
            for block in self.blocks:
                x = block(
                    x,
                    c,
                    p_dropout=p_dropout,
                    attn_kwargs=merged_attn_kwargs,
                )  # (B, L, D)
            c_for_detok = c

        # De-tokenize with dynamic spatial dims
        if self.force_tokenization_fp32:
            dtype = x.dtype
            x = x.to(torch.float32)
            c_fp32 = c_for_detok.to(torch.float32)  # detokenizer's adaptive_modulation Linear is fp32 under this flag
            with torch.autocast(device_type="cuda", enabled=False):
                x = self.detokenizer(x, c_fp32, h_patches=h_patches, w_patches=w_patches)
            x = x.to(dtype)
        else:
            x = self.detokenizer(x, c_for_detok, h_patches=h_patches, w_patches=w_patches)

        return x
