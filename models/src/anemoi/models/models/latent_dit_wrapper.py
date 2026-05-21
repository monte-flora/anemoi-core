"""Predictive model for Atlas-style latent-rollout architecture.

Operates entirely in LATENT space. Takes the latent encoding of the current
state ``z_t`` and one previous state ``z_{t-1}``, plus a noise vector
``xi``, and outputs a latent residual ``r_t`` which is the predicted change
``B(x_{t+1} - x_t)`` in latent space.

Atlas paper §2.3 + §2.4 (Kossaifi et al., NVIDIA, Jan 2026):

  * Two input history states (z_0, z_{-1}). Both at latent resolution.
  * Concatenated along channel dim (Atlas patchifies them with a 2x3
    strided conv; we use patch_size=1 — no further compression).
  * Sine-cosine positional embedding added.
  * Stack of ``depth`` DiT blocks with GLOBAL attention. Atlas: "we use
    global attention as we find that it significantly improves the
    stability of the model compared to local attention."
  * Noise vector ``xi`` projected into the DiT block conditioning channel
    via a single learned linear layer (adaLN-Zero modulation in Phase 3).
  * Final linear projection back to (latent_h, latent_w, out_channels).

FGN vs Atlas noise-conditioning regimes (controlled by ``noise_vector_dim``
and the loss recipe used by the training task):

  ============  ===========================  ===========================
                FGN (Alet et al. 2026)        Atlas (Kossaifi et al. 2026)
  ============  ===========================  ===========================
  Noise dim     LOW (32) - explicitly         HIGH (e.g. 256+) - "full
                restricted to constrain        expressive stochastic
                stochastic variance and        conditioning"
                promote coherent global
                structures
  CRPS variant  Mixture of biased +           Standard fair-CRPS
                fair (to handle low-N
                ensemble degeneracy)
  Spectral CRPS Not used                      USED in latent space as
                                              variance stabilizer
  Argument      Low-d noise prevents          Spectral CRPS removes the
                ensemble collapse             need for low-d noise
  ============  ===========================  ===========================

From Atlas §2.4.3: "we find that spectral CRPS regularization not only
stabilizes training with high-dimensional noise with the standard
two-sample CRPS estimator but also allows full expressive stochastic
conditioning without biased-fair mixtures or low-dimensional noise
constraints."

Note: at full-resolution (v25/v28) we found spectral CRPS catastrophic
(broadband amplification, exploited the |FFT| phase-invariance). At
LATENT resolution, the high-k content the model could exploit is
already removed by the bilinear downsample, so spectral CRPS may behave
very differently. Atlas's empirical claim is exactly this. Both regimes
should be ablated.

Current status: SCAFFOLD. Architectural skeleton + forward signature
real and shape-correct. DiT internals are placeholder conv+attention
blocks; full physicsnemo DiT with global attention is Phase 3.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from anemoi.models.layers.bilinear_encoder import resize_pos_embed
from anemoi.models.models.decoder_dit_wrapper import _sincos_2d_pos_embed

LOGGER = logging.getLogger(__name__)


class _PlaceholderGlobalAttentionBlock(nn.Module):
    """Stand-in for a real DiT block with global attention.

    Implements the structural pattern (LayerNorm -> attention -> residual ->
    LayerNorm -> MLP -> residual) but uses a single global self-attention
    via the standard ``nn.MultiheadAttention`` so the skeleton is end-to-
    end runnable. Phase 3 replaces with physicsnemo DiT + adaLN-Zero
    modulation by the noise-vector conditioning ``c``.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, N, D). c: (B, D) conditioning vector (ignored in scaffold)."""
        del c  # placeholder; real adaLN-Zero modulation in Phase 3
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class AnemoiLatentDiTModel(nn.Module):
    """Latent predictive model: (z_t, z_{t-1}, xi) -> r_t.

    Parameters
    ----------
    latent_shape : tuple[int, int]
        Spatial extent of the latent grid (e.g., (63, 63)).
    in_channels : int
        Prognostic channels in each latent state (typically 105). Forcings
        are passed separately via ``forcings_channels`` and concatenated
        only at the tokenizer input — the model output is sized to
        ``out_channels`` (prognostic-only).
    out_channels : int
        Channels in the predicted latent residual (typically same as
        ``in_channels``).
    forcings_channels : int, optional
        Number of forcing channels concatenated to each history state at
        tokenize time. 0 disables forcings (Atlas's literal recipe).
        When >0, the caller passes ``forcings_curr`` / ``forcings_prev``
        to ``forward`` already at the latent resolution.
    hidden_size : int
        DiT block hidden dimension.
    depth : int
        Number of DiT blocks.
    num_heads : int
        Attention heads. Must divide ``hidden_size``.
    history_len : int
        Number of history states to concat. 2 = Atlas-style (z_0, z_{-1}).
    noise_vector_dim : int
        Dimension of the FGN noise vector xi. 0 disables FGN noise. Atlas
        uses 32.
    """

    def __init__(
        self,
        *,
        model_config=None,
        data_indices=None,
        statistics=None,
        graph_data=None,
        latent_shape: tuple[int, int] | None = None,
        in_channels: int | None = None,
        out_channels: int | None = None,
        forcings_channels: int = 0,
        hidden_size: int = 512,
        depth: int = 16,
        num_heads: int = 8,
        history_len: int = 2,
        noise_vector_dim: int = 32,
        **_extra_kwargs,
    ) -> None:
        super().__init__()

        # Anemoi-standard path: pull fields from model_config.model.latent.
        if model_config is not None:
            from anemoi.utils.config import DotDict
            cfg = DotDict(model_config).model.model.latent
            latent_shape = tuple(cfg.latent_shape)
            in_channels = int(cfg.in_channels)
            out_channels = int(cfg.out_channels)
            forcings_channels = int(getattr(cfg, "forcings_channels", forcings_channels))
            hidden_size = int(getattr(cfg, "hidden_size", hidden_size))
            depth = int(getattr(cfg, "depth", depth))
            num_heads = int(getattr(cfg, "num_heads", num_heads))
            history_len = int(getattr(cfg, "history_len", history_len))
            noise_vector_dim = int(getattr(cfg, "noise_vector_dim", noise_vector_dim))

        if any(v is None for v in (latent_shape, in_channels, out_channels)):
            error = (
                "AnemoiLatentDiTModel: must provide either model_config or all "
                "of latent_shape / in_channels / out_channels."
            )
            raise ValueError(error)

        self.latent_shape = tuple(latent_shape)
        self.history_len = history_len
        self.noise_vector_dim = noise_vector_dim
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.forcings_channels = forcings_channels

        # Tokenize each history latent state (prognostic + forcings) via a
        # 1x1 conv (patch_size=1 inside the latent — our locked decision).
        # Per-history channel count = in_channels + forcings_channels.
        per_step_in = in_channels + forcings_channels
        total_in = per_step_in * history_len
        self.tokenize = nn.Conv2d(total_in, hidden_size, kernel_size=1)

        # FGN noise-vector projection. Adds to the per-block conditioning c.
        if noise_vector_dim > 0:
            self.noise_proj = nn.Linear(noise_vector_dim, hidden_size)
        else:
            self.noise_proj = None

        # Sine-cosine positional embedding for the (h_lat * w_lat) tokens.
        pos = _sincos_2d_pos_embed(
            self.latent_shape[0], self.latent_shape[1],
            hidden_size, device="cpu",
        )
        self.register_buffer("pos_embed", pos.unsqueeze(0))  # (1, h*w, D)

        # SCAFFOLD: placeholder global-attention blocks. Phase 3 replaces
        # with physicsnemo DiT with proper adaLN-Zero modulation by c.
        self.blocks = nn.ModuleList([
            _PlaceholderGlobalAttentionBlock(hidden_size, num_heads)
            for _ in range(depth)
        ])

        # Output projection back to latent residual channels.
        self.final_norm = nn.LayerNorm(hidden_size)
        self.final_proj = nn.Linear(hidden_size, out_channels)

        # Atlas zero-init of final layer so the predictive model starts as
        # an identity (r = 0).
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(
        self,
        z_curr: torch.Tensor,
        z_prev: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        *,
        forcings_curr: Optional[torch.Tensor] = None,
        forcings_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict latent residual r_t.

        Parameters
        ----------
        z_curr : torch.Tensor
            Current latent state, shape ``(B, in_channels, h_lat, w_lat)``.
        z_prev : torch.Tensor, optional
            Previous latent state (history). Required if ``history_len=2``.
            Shape ``(B, in_channels, h_lat, w_lat)``.
        noise : torch.Tensor, optional
            FGN noise vector, shape ``(B, noise_vector_dim)`` (or
            ``(B, E, noise_vector_dim)`` for ensemble of E samples). If
            not provided and ``noise_vector_dim > 0``, sampled internally.
        forcings_curr, forcings_prev : torch.Tensor, optional
            Latent-resolution forcings at the same timestep as
            ``z_curr`` / ``z_prev``. Required when ``forcings_channels>0``.
            Shape ``(B, forcings_channels, h_lat, w_lat)`` each. Used to
            condition the predictive model on time-invariant geography
            (HGT, land/sea) and time-varying solar/temporal forcings.

        Returns
        -------
        torch.Tensor
            Predicted latent residual, shape ``(B, out_channels, h_lat, w_lat)``
            (or ``(B, E, out_channels, h_lat, w_lat)`` if an ensemble of
            noise samples is provided). ``out_channels`` is prognostic-only;
            forcings are conditioning, not predicted.
        """
        B = z_curr.shape[0]
        # Use the ACTUAL spatial shape of z_curr, not the configured one —
        # at inference the input latent grid may be larger than at training
        # (e.g. full-CONUS at preserved 16-km resolution vs patches at 63x63).
        # pos_embed is bicubic-resized below to match.
        h, w = int(z_curr.shape[-2]), int(z_curr.shape[-1])
        cfg_h, cfg_w = self.latent_shape

        # Validate forcings presence vs. config.
        if self.forcings_channels > 0:
            if forcings_curr is None:
                error = (
                    f"AnemoiLatentDiTModel was configured with "
                    f"forcings_channels={self.forcings_channels} but forcings_curr is None."
                )
                raise ValueError(error)
            if self.history_len == 2 and forcings_prev is None:
                forcings_prev = forcings_curr  # cold-start mirror, matching z_prev fallback

        # Per-step input: concat [prognostic_latent, forcings_latent] along channel.
        def _per_step(z, f):
            if self.forcings_channels > 0:
                return torch.cat([z, f], dim=1)
            return z

        # Stack history along channel dim.
        if self.history_len == 2:
            if z_prev is None:
                # Cold-start: use z_curr for both (model will see zero
                # tendency at t=0 — acceptable for first AR step).
                z_prev = z_curr
            tok_in = torch.cat(
                [_per_step(z_curr, forcings_curr), _per_step(z_prev, forcings_prev)],
                dim=1,
            )
        else:
            tok_in = _per_step(z_curr, forcings_curr)

        # Tokenize: (B, C*H, h, w) -> (B, D, h, w)
        tok = self.tokenize(tok_in)

        # Flatten spatial -> sequence: (B, D, h, w) -> (B, h*w, D)
        tok = tok.permute(0, 2, 3, 1).reshape(B, h * w, self.hidden_size)
        # If input latent grid differs from the configured shape (e.g.
        # transfer to full-CONUS at a different latent resolution),
        # bicubic-resize the registered pos_embed buffer to the new grid.
        # Sine-cosine pos embeds make this essentially lossless; same path
        # also handles learnable pos embeds (future configs).
        if (h, w) != (cfg_h, cfg_w):
            pos = resize_pos_embed(
                self.pos_embed, old_shape=(cfg_h, cfg_w), new_shape=(h, w),
            )
        else:
            pos = self.pos_embed
        tok = tok + pos.to(dtype=tok.dtype)

        # FGN noise conditioning vector c. Always a single conditioning
        # tensor per (batch, [ensemble_member]). For now we collapse any
        # ensemble dim into batch — Phase 3 will route through the DiT's
        # block-conditioning channel.
        if self.noise_proj is not None:
            if noise is None:
                noise = torch.randn(B, self.noise_vector_dim, device=z_curr.device, dtype=z_curr.dtype)
            if noise.dim() == 3:
                # ensemble dim — fold into batch
                B_orig, E, _ = noise.shape
                noise = noise.reshape(B_orig * E, -1)
                # Need to also tile tok across ensemble: (B, ...) -> (B*E, ...)
                tok = tok.unsqueeze(1).expand(B_orig, E, *tok.shape[1:]).reshape(B_orig * E, *tok.shape[1:])
                B_out = B_orig
                E_out = E
            else:
                B_out = B
                E_out = None
            c = self.noise_proj(noise)  # (B*E, D)
        else:
            c = None
            B_out = B
            E_out = None

        # DiT blocks (placeholder global attention; ignores c for now).
        for blk in self.blocks:
            tok = blk(tok, c)

        # Project + un-flatten back to spatial.
        tok = self.final_norm(tok)
        out = self.final_proj(tok)  # (..., out_channels)
        out = out.reshape(-1, h, w, self.out_channels).permute(0, 3, 1, 2).contiguous()

        if E_out is not None:
            out = out.reshape(B_out, E_out, self.out_channels, h, w)
        return out
