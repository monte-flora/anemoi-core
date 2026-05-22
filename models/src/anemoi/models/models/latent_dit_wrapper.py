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

Status (2026-05-22): UPGRADED FROM SCAFFOLD. The block stack now uses
physicsnemo's FlexibleDiT with attention_backend='timm' (SDPA → flash-
attn under the hood for bf16-mixed) and conditioning_embedder='dit'
which threads the noise vector through every block via real adaLN-Zero
modulation. The earlier `_PlaceholderGlobalAttentionBlock` chain (which
silently discarded the noise input — `del c`) is gone. The old class is
kept temporarily as a fallback for unit-test isolation but is NOT used
by `AnemoiLatentDiTModel`.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from anemoi.models.layers.bilinear_encoder import resize_pos_embed
from anemoi.models.models.decoder_dit_wrapper import _sincos_2d_pos_embed
from anemoi.models.models.flexible_dit import FlexibleDiT

LOGGER = logging.getLogger(__name__)


class _PlaceholderGlobalAttentionBlock(nn.Module):
    """Stand-in for a real DiT block with global attention.

    Implements the structural pattern (LayerNorm -> attention -> residual ->
    LayerNorm -> MLP -> residual). The attention uses
    ``F.scaled_dot_product_attention`` (SDPA), which auto-dispatches to
    PyTorch's flash-attention or memory-efficient backends when the input
    shape + dtype permit (it does for our bf16 / 3969-token / depth-16
    config). Empirically 2-4x faster than the old ``nn.MultiheadAttention``
    path on bf16-mixed training. Same numerics in the limit (SDPA is the
    fused kernel that nn.MultiheadAttention falls back to anyway, just
    without the Python-side overhead).

    Phase 3 replaces with physicsnemo DiT + adaLN-Zero modulation by the
    noise-vector conditioning ``c``.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            error = (
                f"_PlaceholderGlobalAttentionBlock: hidden_size {hidden_size} "
                f"must be divisible by num_heads {num_heads}."
            )
            raise ValueError(error)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.norm1 = nn.LayerNorm(hidden_size)
        # Single learned qkv projection feeding SDPA — same param count and
        # init scale as nn.MultiheadAttention but without the Python overhead.
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
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
        B, N, D = x.shape
        h = self.norm1(x)
        # qkv: (B, N, 3*D) -> split to (B, num_heads, N, head_dim) for each.
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # SDPA auto-dispatches to flash-attn on supported (dtype, shape) combos.
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        attn_out = self.proj(attn_out)
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

        # Standard Anemoi init kwargs: data_indices / statistics / graph_data
        # are passed by AnemoiModelInterface to every model class regardless
        # of whether the class uses them. Handle each explicitly:
        #
        # * data_indices: USED for sanity checks below — catches the case
        #   where in_channels / forcings_channels in the config drift away
        #   from what the data actually has, before the shape mismatch
        #   surfaces 16 DiT blocks deep in a forward pass.
        # * statistics: Variant B normalization keeps stats in
        #   AnemoiModelInterface.pre_processors; the model itself doesn't
        #   need them. Accept and ignore.
        # * graph_data: GNN-only (encoder/processor/decoder graphs).
        #   FlexibleDiT is grid-native — no graph. Accept and ignore.
        if data_indices is not None:
            n_prog_data = len(data_indices.data.input.prognostic)
            n_forcings_data = len(data_indices.data.input.forcing)
            if n_prog_data != in_channels:
                error = (
                    f"AnemoiLatentDiTModel: config in_channels={in_channels} "
                    f"but data_indices.data.input.prognostic has {n_prog_data} "
                    f"entries. Edit data/vars or model.latent.in_channels."
                )
                raise ValueError(error)
            if forcings_channels > 0 and n_forcings_data != forcings_channels:
                error = (
                    f"AnemoiLatentDiTModel: config forcings_channels="
                    f"{forcings_channels} but data_indices.data.input.forcing "
                    f"has {n_forcings_data} entries. Edit data/vars or "
                    f"model.latent.forcings_channels."
                )
                raise ValueError(error)
        del statistics, graph_data  # accepted for Anemoi compat; not used

        self.latent_shape = tuple(latent_shape)
        self.history_len = history_len
        self.noise_vector_dim = noise_vector_dim
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.forcings_channels = forcings_channels

        # Per-history input channels (prognostic + forcings).
        per_step_in = in_channels + forcings_channels
        total_in = per_step_in * history_len
        self._total_in = total_in

        # Use the real physicsnemo FlexibleDiT for the block stack. The
        # latent input is already at the target resolution, so patch_size=1
        # makes the tokenizer a 1×1 channel projection (no further spatial
        # compression).
        #
        # attention_backend='timm' uses PyTorch SDPA, which auto-dispatches
        # to flash-attn on bf16-mixed inputs and arbitrary spatial shapes.
        # 'natten2d' would be local-window attention — Atlas explicitly
        # chooses global for the predictive ("significantly improves the
        # stability of the model compared to local attention").
        #
        # conditioning_embedder='dit' enables adaLN-Zero modulation by the
        # noise vector via the DiTConditionEmbedder MLP. The earlier
        # placeholder ignored the noise entirely — fixed here.
        self.dit = FlexibleDiT(
            input_size=self.latent_shape,
            in_channels=total_in,
            out_channels=out_channels,
            patch_size=1,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            attention_backend="timm",
            conditioning_embedder="dit",
            condition_dim=noise_vector_dim if noise_vector_dim > 0 else 0,
            force_tokenization_fp32=False,
            detokenizer_type="linear_reshape",
        )

        # Atlas zero-init of the FlexibleDiT detokenizer's projection so
        # the predictive model starts as identity (r = 0). FlexibleDiT
        # already zero-inits its adaLN modulation and final projection;
        # we additionally zero its detokenizer's final linear if present.
        for name, module in self.dit.named_modules():
            if isinstance(module, nn.Linear) and name.endswith("detokenizer.proj"):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

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

        # Stack history along channel dim (still spatial, NOT flattened).
        if self.history_len == 2:
            if z_prev is None:
                z_prev = z_curr
            x_2d = torch.cat(
                [_per_step(z_curr, forcings_curr), _per_step(z_prev, forcings_prev)],
                dim=1,
            )
        else:
            x_2d = _per_step(z_curr, forcings_curr)

        # Handle FGN noise vector (B[, E], noise_vector_dim).
        if self.noise_vector_dim > 0:
            if noise is None:
                noise = torch.randn(
                    B, self.noise_vector_dim,
                    device=z_curr.device, dtype=z_curr.dtype,
                )
            if noise.dim() == 3:
                # Ensemble: (B, E, noise_dim) — fold E into batch dim and
                # tile x_2d to match so each ensemble member gets the same
                # inputs but a distinct noise sample.
                B_orig, E, _ = noise.shape
                noise = noise.reshape(B_orig * E, -1)
                x_2d = (
                    x_2d.unsqueeze(1)
                    .expand(B_orig, E, *x_2d.shape[1:])
                    .reshape(B_orig * E, *x_2d.shape[1:])
                )
                B_out = B_orig
                E_out = E
            else:
                B_out = B
                E_out = None
        else:
            noise = None
            B_out = B
            E_out = None

        # FlexibleDiT.forward requires (x, t[, condition]). For our use
        # case t is unused (no diffusion timestep); pass zeros. The
        # noise vector is routed via `condition` and threaded through
        # every block's adaLN-Zero modulation by DiTConditionEmbedder.
        B_eff = x_2d.shape[0]
        t = torch.zeros(B_eff, device=x_2d.device, dtype=x_2d.dtype)
        out = self.dit(x_2d, t, condition=noise)  # (B_eff, out_channels, H, W)

        if E_out is not None:
            out = out.reshape(B_out, E_out, *out.shape[1:])
        return out
