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
from jaxtyping import Float

from physicsnemo.models.dit.dit import DiT
from physicsnemo.nn.module.dit_layers import ProjReshape2DDetokenizer


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Replace the detokenizer with the flexible version.
        # Copy the weights from the original (already initialized by DiT.__init__).
        if isinstance(self.detokenizer, ProjReshape2DDetokenizer):
            orig = self.detokenizer
            flexible = FlexibleProjReshape2DDetokenizer(
                input_size=orig.input_size,
                patch_size=orig.patch_size,
                out_channels=orig.out_channels,
                hidden_size=orig.hidden_size,
            )
            # Transfer the initialized weights
            flexible.load_state_dict(orig.state_dict())
            self.detokenizer = flexible

        # Replace unpicklable lambdas with a proper static method so torch.save works.
        # Must come AFTER detokenizer swap so the new proj_layer is patched.
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
            with torch.autocast(device_type="cuda", enabled=False):
                x = self.detokenizer(x, c, h_patches=h_patches, w_patches=w_patches)
            x = x.to(dtype)
        else:
            x = self.detokenizer(x, c, h_patches=h_patches, w_patches=w_patches)

        return x
