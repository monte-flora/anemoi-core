"""Anemoi-compatible wrapper for the PhysicsNeMo DiT with NATTEN.

Supports two modes:
  - deterministic: MSE training with bias-only AdaLN (conditioning_embedder="zero")
  - probabilistic: EDM diffusion training with noise-level conditioning (conditioning_embedder="dit")

The wrapper handles:
  - Reshaping from Anemoi's flat grid (B, T, E, H*W, V) to 2D spatial (B, C, H, W)
  - Dynamic padding for patch_size divisibility
  - Residual prediction and denormalization in predict_step
  - The same forward/predict_step interface as AnemoiResidualModelEncProcDec
"""

import logging
from typing import Optional

import einops
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup

from anemoi.models.distributed.graph import gather_tensor, shard_tensor
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.layers.bounding import build_boundings
from anemoi.models.models.flexible_dit import FlexibleDiT
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class _PassthroughConditionEmbedder(nn.Module):
    """Conditioning embedder that returns the ``condition`` kwarg unchanged.

    Used when we want to inject a pre-encoded conditioning vector into
    every adaLN block — e.g. an FGN noise-vector encoding — without
    going through a timestep-based sinusoidal+MLP transformation.

    Matches the ``conditioning_embedder(t, condition=...)`` signature
    that ``FlexibleDiT.forward`` calls at line ~674.
    """

    def forward(self, t: Tensor, condition: Optional[Tensor] = None) -> Tensor:
        if condition is None:
            # Defensive: return zeros sized to the timestep tensor so the
            # downstream blocks don't blow up. In practice we always call
            # this via ``forward_with_noise`` which provides a condition.
            return torch.zeros(t.shape[0], 1, device=t.device, dtype=t.dtype)
        return condition


def _swap_activation(module: nn.Module, old_cls: type, new_cls: type) -> int:
    """Walk a module tree and replace every instance of ``old_cls`` with
    ``new_cls``. Returns the number of swaps performed.

    Used to retrofit activation choices (e.g., GELU\u2192SiLU) onto the physicsnemo
    DiT whose internal MLPs have the activation hardcoded.
    """
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, old_cls):
            setattr(module, name, new_cls())
            count += 1
        else:
            count += _swap_activation(child, old_cls, new_cls)
    return count


def _gaussian_kernel_2d(kernel_size: int, sigma: float) -> torch.Tensor:
    """Return a normalized 2-D Gaussian kernel of shape (ks, ks)."""
    half = (kernel_size - 1) / 2.0
    ax = torch.arange(kernel_size, dtype=torch.float32) - half
    gy = torch.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    k2 = torch.outer(gy, gy)
    return k2 / k2.sum()


class _DepthwiseGaussianLPF(nn.Module):
    """Fixed depth-wise 2-D Gaussian low-pass filter applied per-channel.

    Implemented as a reflect-padded Conv2d with non-learnable weights
    (registered as a buffer). Used right after the DiT detokenizer to
    enforce a physical Nyquist rolloff and kill >Nyquist content coming
    from the tile-level projection.
    """

    def __init__(self, channels: int, kernel_size: int = 5, sigma: float = 0.7):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.channels = channels
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        k = _gaussian_kernel_2d(kernel_size, sigma)
        # Shape needed for groupwise conv: (out_ch=channels, in_ch/groups=1, kH, kW)
        w = k.view(1, 1, kernel_size, kernel_size).expand(channels, 1, kernel_size, kernel_size).clone()
        self.register_buffer("weight", w)

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
        return F.conv2d(x, self.weight, bias=None, groups=self.channels)


def _gaussian_init_first_conv(seq: nn.Sequential, sigma: float = 0.7) -> int:
    """Overwrite the first `nn.Conv2d` weight in `seq` with a channel-wise
    2-D Gaussian kernel (broadcast across in/out channels with unit sum).

    Returns the kernel size of the initialized conv for logging. Leaves
    bias zero-init. Keeps the overall conv expressive (other convs in the
    block still use Kaiming) but gives the refinement a smooth starting
    point rather than the default Kaiming-random kernel.
    """
    for m in seq.modules():
        if isinstance(m, nn.Conv2d):
            ks = m.kernel_size[0]
            kern = _gaussian_kernel_2d(ks, sigma)  # (ks, ks)
            # Each (out_c, in_c) pair takes the same normalized kernel.
            w = kern.view(1, 1, ks, ks).expand_as(m.weight).clone()
            with torch.no_grad():
                m.weight.copy_(w)
                if m.bias is not None:
                    m.bias.zero_()
            return int(ks)
    return 0


class AnemoiDiTModel(nn.Module):
    """DiT model wrapped for the Anemoi training/inference pipeline.

    Replaces the encoder-processor-decoder GNN with a single DiT that operates
    on 2D spatial grids. Transfer learning (train on patch, infer on CONUS)
    is enabled by NATTEN neighborhood attention + disabled absolute positional
    embeddings.
    """

    def __init__(
        self,
        *,
        model_config,
        data_indices: dict,
        statistics: dict,
        graph_data=None,
        **kwargs,
    ) -> None:
        super().__init__()

        config = DotDict(model_config)
        # dit config is at config.model.model.dit (outer model = schema, inner model = target)
        dit_cfg = config.model.model.dit

        # Grid shape for the 2D reshape (from zarr attrs: field_shape=[ny, nx])
        self.field_shape = tuple(dit_cfg.field_shape)
        self.multi_step = config.training.multistep_input
        self.mode = getattr(dit_cfg, "mode", "deterministic")

        # Variable counts from data_indices
        self.num_input_channels = len(data_indices.model.input)
        self.num_output_channels = len(data_indices.model.output)

        # Diffusion conditioning subset (probabilistic mode only). Default
        # "full" = condition on the whole input history (forecaster behaviour;
        # byte-identical for existing probabilistic DiT checkpoints). "forcing"
        # = condition on forcings only (thermalizer DENOISER — conditioning on
        # the clean prognostic state to denoise itself would be trivial).
        # "none" = unconditional. in_channels is sized to the actual
        # conditioning so we never allocate dead input-projection weights.
        self.condition_on = str(getattr(dit_cfg, "condition_on", "full"))
        if self.condition_on == "full":
            cond_idx = torch.arange(self.num_input_channels, dtype=torch.long)
        elif self.condition_on == "forcing":
            cond_idx = torch.as_tensor(list(data_indices.model.input.forcing), dtype=torch.long)
        elif self.condition_on == "none":
            cond_idx = torch.zeros(0, dtype=torch.long)
        else:
            raise ValueError(f"diffusion.condition_on must be full|forcing|none, got {self.condition_on!r}")
        self.register_buffer("_cond_idx", cond_idx, persistent=False)
        n_cond = int(cond_idx.numel())

        # For probabilistic: input = [x_history(cond subset), y_noised] on channels
        if self.mode == "probabilistic":
            in_channels = self.multi_step * n_cond + self.num_output_channels
        else:
            in_channels = self.multi_step * self.num_input_channels

        # Convert OmegaConf objects to plain dicts for physicsnemo
        tokenizer_kwargs = dict(getattr(dit_cfg, "tokenizer_kwargs", {}))
        attn_kwargs = dict(getattr(dit_cfg, "attn_kwargs", {}))
        conditioning_embedder_kwargs = dict(getattr(dit_cfg, "conditioning_embedder_kwargs", {}))

        # Compute padded field_shape so the tokenizer's pos_embed (sized
        # to input_size // patch_size in PatchEmbed2DTokenizer.__init__)
        # matches the token count produced AFTER _pad_to_patch_size at
        # forward time. Without this, pos_embed=learnable raises a shape
        # mismatch whenever field_shape is not divisible by patch_size
        # (e.g. 250 % 4 = 2 → input is padded to 252 → 63² tokens but
        # pos_embed is sized for 62² = 3844 tokens).
        ps_h = ps_w = int(dit_cfg.patch_size)
        _pad_h = (ps_h - self.field_shape[0] % ps_h) % ps_h
        _pad_w = (ps_w - self.field_shape[1] % ps_w) % ps_w
        padded_field_shape = (self.field_shape[0] + _pad_h, self.field_shape[1] + _pad_w)

        LOGGER.info(
            f"Initializing FlexibleDiT: mode={self.mode}, in_channels={in_channels}, "
            f"out_channels={self.num_output_channels}, field_shape={self.field_shape}, "
            f"padded_field_shape={padded_field_shape}, "
            f"patch_size={dit_cfg.patch_size}, hidden_size={dit_cfg.hidden_size}, "
            f"depth={dit_cfg.depth}, num_heads={dit_cfg.num_heads}, "
            f"attention_backend={dit_cfg.attention_backend}"
        )

        # Dropout knobs — exposed as part of the U-Cast (Cachay et al, 2026)
        # MC-dropout CRPS recipe. All default to 0.0 to preserve byte-identical
        # behaviour for v17 / v22 / v23 / v24 checkpoints.
        attn_drop_rate = float(getattr(dit_cfg, "attn_drop_rate", 0.0))
        proj_drop_rate = float(getattr(dit_cfg, "proj_drop_rate", 0.0))
        drop_path_rate = float(getattr(dit_cfg, "drop_path_rate", 0.0))
        if attn_drop_rate or proj_drop_rate or drop_path_rate:
            LOGGER.info(
                "DiT dropout knobs: attn_drop=%.3f, proj_drop=%.3f, drop_path=%.3f",
                attn_drop_rate, proj_drop_rate, drop_path_rate,
            )

        # Plumbing: physicsnemo's DiT accepts attn_drop_rate/proj_drop_rate
        # only via ``block_kwargs`` (kwargs forwarded to DiTBlock.__init__),
        # and ``drop_path_rate`` via ``drop_path_rates: list[float]`` (one
        # per block, depth-many). Merge with any user-supplied block_kwargs.
        depth = int(dit_cfg.depth)
        block_kwargs_in = {}
        if attn_drop_rate:
            block_kwargs_in["attn_drop_rate"] = attn_drop_rate
        if proj_drop_rate:
            block_kwargs_in["proj_drop_rate"] = proj_drop_rate
        drop_path_rates = [drop_path_rate] * depth if drop_path_rate else None

        self.dit = FlexibleDiT(
            input_size=padded_field_shape,
            in_channels=in_channels,
            out_channels=self.num_output_channels,
            patch_size=int(dit_cfg.patch_size),
            hidden_size=int(dit_cfg.hidden_size),
            depth=depth,
            num_heads=int(dit_cfg.num_heads),
            mlp_ratio=float(getattr(dit_cfg, "mlp_ratio", 4.0)),
            attention_backend=str(dit_cfg.attention_backend),
            conditioning_embedder=str(getattr(dit_cfg, "conditioning_embedder", "zero")),
            condition_dim=getattr(dit_cfg, "condition_dim", None),
            tokenizer_kwargs=tokenizer_kwargs,
            attn_kwargs=attn_kwargs,
            conditioning_embedder_kwargs=conditioning_embedder_kwargs,
            force_tokenization_fp32=bool(getattr(dit_cfg, "force_tokenization_fp32", True)),
            detokenizer_type=str(getattr(dit_cfg, "detokenizer_type", "linear_reshape")),
            tokenizer_kernel_size=getattr(dit_cfg, "tokenizer_kernel_size", None),
            tokenizer_anti_aliased=bool(getattr(dit_cfg, "tokenizer_anti_aliased", False)),
            block_kwargs=block_kwargs_in,
            drop_path_rates=drop_path_rates,
        )

        # Architecture summary at instantiation time. Printed once per rank
        # construction; surfaces the actual detokenizer class so config-vs-
        # instantiation mismatches (Hydra override silently dropped, schema
        # not plumbed, etc.) are caught immediately rather than after a
        # full training run with a wrong head.
        def _fmt_params(n):
            return f"{n / 1e6:>7.2f}M" if n >= 1e5 else f"{n:>9d}"

        configured_detok = str(getattr(dit_cfg, "detokenizer_type", "linear_reshape"))
        actual_detok = type(self.dit.detokenizer).__name__
        LOGGER.info("=" * 78)
        LOGGER.info("DiT model summary")
        LOGGER.info("-" * 78)
        LOGGER.info("  configured detokenizer_type:  %s", configured_detok)
        LOGGER.info("  instantiated detokenizer cls: %s", actual_detok)
        if configured_detok in ("pixel_shuffle", "linear_reshape"):
            pass
        elif (
            (configured_detok.startswith("pixel_shuffle") and "PixelShuffle" not in actual_detok)
            or (configured_detok.startswith("conv_transpose") and "ConvTranspose" not in actual_detok)
            or (configured_detok.startswith("bilinear") and "Bilinear" not in actual_detok)
            or (configured_detok.startswith("hierarchical") and "Hierarchical" not in actual_detok)
        ):
            LOGGER.warning(
                "  configured detokenizer_type=%r but instantiated %s — config likely not "
                "plumbed through. Check dit_wrapper.py and FlexibleDiT.__init__ dispatch.",
                configured_detok, actual_detok,
            )
        total_dit = 0
        for name, sub in self.dit.named_children():
            n = sum(p.numel() for p in sub.parameters())
            total_dit += n
            LOGGER.info(
                "  dit.%-25s %s params  (%s)",
                name + ":", _fmt_params(n), type(sub).__name__,
            )
        LOGGER.info("  dit total: %s params", _fmt_params(total_dit))
        LOGGER.info("=" * 78)

        # Gradient (activation) checkpointing on the DiT blocks: recompute in
        # backward instead of storing activations. Required for full-CONUS
        # (992x1524) training on 40 GB GPUs; ~30-40% step-time cost. Default
        # False preserves existing behaviour/throughput.
        self.dit.gradient_checkpointing = bool(getattr(dit_cfg, "gradient_checkpointing", False))
        if self.dit.gradient_checkpointing:
            LOGGER.info("DiT gradient checkpointing ENABLED (per-block recompute in backward)")

        # Optional post-init activation swap: set dit_cfg.activation to one of
        # {"gelu", "silu", "relu"}. Replaces every nn.GELU inside the DiT
        # transformer (including physicsnemo-internal MLPs) with the chosen
        # activation. Default is "gelu" (no-op) to preserve existing behaviour
        # and old checkpoints.
        act_name = str(getattr(dit_cfg, "activation", "gelu")).lower()
        if act_name != "gelu":
            act_cls = {"silu": nn.SiLU, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}.get(act_name)
            if act_cls is None:
                raise ValueError(f"Unknown DiT activation '{act_name}'")
            n_replaced = _swap_activation(self.dit, nn.GELU, act_cls)
            LOGGER.info("DiT activation swapped: %d nn.GELU -> nn.%s", n_replaced, act_cls.__name__)
        # conv_refinement activation is configured below and can be separate

        # Optional conv refinement after DiT detokenizer to smooth patch-boundary
        # artifacts and enforce spatial coherence. Each block is 3x3 conv + GELU +
        # 3x3 conv with a residual skip; output is (dit_out + refinement).
        # Enable with conv_refinement_blocks: int > 0 in the model config.
        n_refine = int(getattr(dit_cfg, "conv_refinement_blocks", 0))
        refine_kernel = int(getattr(dit_cfg, "conv_refinement_kernel", 3))
        refine_hidden = int(getattr(dit_cfg, "conv_refinement_hidden", 0)) or self.num_output_channels
        raw_refine_act = getattr(dit_cfg, "conv_refinement_activation", None)
        refine_act_name = str(raw_refine_act).lower() if raw_refine_act is not None else act_name
        refine_act_cls = {"gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU,
                          "leaky_relu": nn.LeakyReLU}.get(refine_act_name, nn.GELU)
        if n_refine > 0:
            layers = []
            c_in = self.num_output_channels
            for i in range(n_refine):
                c_mid = refine_hidden
                c_out = self.num_output_channels if i == n_refine - 1 else refine_hidden
                layers.append(nn.Conv2d(c_in, c_mid, kernel_size=refine_kernel,
                                       padding=refine_kernel // 2, padding_mode="reflect"))
                layers.append(refine_act_cls())
                layers.append(nn.Conv2d(c_mid, c_out, kernel_size=refine_kernel,
                                       padding=refine_kernel // 2, padding_mode="reflect"))
                c_in = c_out
            self.conv_refinement = nn.Sequential(*layers)
            # Optional smooth-prior initialization: seed the FIRST conv with
            # a normalized 2-D Gaussian kernel so the refinement starts as a
            # smoothing filter rather than the default Kaiming-random kernel.
            refine_init = str(getattr(dit_cfg, "conv_refinement_init", "default")).lower()
            if refine_init in ("gaussian", "gaussian_lowpass"):
                gi_sigma = float(getattr(dit_cfg, "conv_refinement_init_sigma", 0.7))
                ks_init = _gaussian_init_first_conv(self.conv_refinement, sigma=gi_sigma)
                LOGGER.info("DiT conv refinement: gaussian-init first conv (k=%d, \u03c3=%.2f)",
                            ks_init, gi_sigma)
            LOGGER.info("DiT conv refinement enabled: %d blocks, kernel=%d, hidden=%d, act=%s",
                        n_refine, refine_kernel, refine_hidden, refine_act_cls.__name__)
        else:
            self.conv_refinement = None

        # ---- Fixed Gaussian LPF right after detokenize (anti-aliasing) ----
        # Enable by setting `detokenizer_lowpass_sigma` to a positive float
        # (typical 0.5\u20131.0). Zero disables. This applies a depth-wise, fixed
        # 2-D Gaussian blur to the detokenized output at full resolution and
        # is non-learnable \u2014 a physically-motivated Nyquist rolloff filter.
        lowpass_sigma = float(getattr(dit_cfg, "detokenizer_lowpass_sigma", 0.0))
        lowpass_k = int(getattr(dit_cfg, "detokenizer_lowpass_kernel", 5))
        if lowpass_sigma > 0.0:
            self.detokenizer_lowpass = _DepthwiseGaussianLPF(
                channels=self.num_output_channels,
                kernel_size=lowpass_k,
                sigma=lowpass_sigma,
            )
            LOGGER.info("DiT detokenizer Gaussian LPF enabled: k=%d, \u03c3=%.2f",
                        lowpass_k, lowpass_sigma)
        else:
            self.detokenizer_lowpass = None

        # Store data indices for predict_step
        self.data_indices = data_indices
        self._internal_input_idx = data_indices.model.input.prognostic
        self._internal_output_idx = data_indices.model.output.prognostic

        # output_mode controls what the model's forward returns + how boundings
        # are applied (see _forward_deterministic):
        #   "residual" (default, back-compat): forward returns whatever the DiT
        #       backbone produces. Interpreted as a normalised residual at the
        #       task level by GraphResidualForecaster (which adds it to the
        #       previous state after physical-space reconstruction). Boundings
        #       are NOT applied (clipping a residual to >=0 would force
        #       monotonic increase, which is wrong).
        #   "state": forward returns predicted state in normalised space. An
        #       internal skip adds the input prognostic state to the DiT output
        #       (mirroring AnemoiModelEncProcDec._assemble_output), making the
        #       DiT effectively predict a delta-from-input. Boundings ARE then
        #       applied (e.g. ReluBounding on apcp/qv enforces x_phys >= 0 in
        #       normalised space — works directly if the variable is std-norm,
        #       otherwise use NormalizedReluBounding). Task should be the
        #       default GraphForecaster.
        self.output_mode = str(getattr(dit_cfg, "output_mode", "residual")).lower()
        if self.output_mode not in ("residual", "state"):
            raise ValueError(
                f"AnemoiDiTModel: output_mode must be 'residual' or 'state', got {self.output_mode!r}."
            )
        LOGGER.info("AnemoiDiTModel: output_mode = %s", self.output_mode)

        # AIFS-CRPS reference-field truncation (eq. 1): x_{t+1} = U(D(x_t)) + f(x_t).
        # 0/None = off. Factor 2 removes <4dx from the carried reference state.
        self.reference_truncation = int(getattr(dit_cfg, "reference_truncation", 0) or 0)
        if self.reference_truncation:
            LOGGER.info("AnemoiDiTModel: reference_truncation factor = %d", self.reference_truncation)
        self.reference_truncation_exclude = list(getattr(dit_cfg, "reference_truncation_exclude", ["pressure_*", "t2m", "skintemp", "snowh"]) or [])

        # Boundings (e.g., ReLU for precipitation). Applied only in state mode;
        # see _forward_deterministic.
        self.boundings = build_boundings(config, data_indices, statistics)
        if self.boundings and self.output_mode == "residual":
            LOGGER.warning(
                "AnemoiDiTModel: %d bounding(s) configured but output_mode='residual' — "
                "boundings will NOT be applied (clipping residuals is incorrect). "
                "Set output_mode='state' to enable bounding.",
                len(self.boundings),
            )

        # Diffusion parameters (probabilistic mode)
        if self.mode == "probabilistic":
            self.sigma_data = float(getattr(dit_cfg, "sigma_data", 1.0))
            self.sigma_max = float(getattr(dit_cfg, "sigma_max", 100.0))
            self.sigma_min = float(getattr(dit_cfg, "sigma_min", 0.02))
            self.rho = float(getattr(dit_cfg, "rho", 7.0))
            self.inference_defaults = dict(getattr(dit_cfg, "inference_defaults", {}))

        # FGN-style noise-vector conditioning (CRPS ensemble training). When
        # ``noise_vector_dim`` is set, we add a small Linear that maps the
        # per-member noise vector to the DiT hidden_size and swap the
        # conditioning_embedder for a passthrough so the encoded noise is what
        # actually reaches the adaLN layers in every block + the detokenizer.
        noise_vector_dim = getattr(dit_cfg, "noise_vector_dim", None)
        self.noise_vector_dim = (
            int(noise_vector_dim) if noise_vector_dim is not None else None
        )
        self.noise_encoder_type = str(getattr(dit_cfg, "noise_encoder_type", "none")).lower()
        if self.noise_vector_dim is not None and self.noise_encoder_type != "none":
            hidden_size = int(dit_cfg.hidden_size)
            if self.noise_encoder_type == "matmul":
                # FGN-faithful: single Linear (no activation). Initialise with
                # small std so the warm-started deterministic features dominate
                # at step 0 and noise contribution grows during FT.
                self.noise_encoder = nn.Linear(self.noise_vector_dim, hidden_size)
                nn.init.normal_(self.noise_encoder.weight, std=0.02)
                nn.init.zeros_(self.noise_encoder.bias)
            elif self.noise_encoder_type == "fourier_mlp":
                # GenCast-style for ablation: Fourier embedding + 2-layer MLP.
                from anemoi.models.layers.diffusion import SinusoidalEmbeddings
                self.noise_encoder = nn.Sequential(
                    SinusoidalEmbeddings(noise_channels=self.noise_vector_dim),
                    nn.Linear(self.noise_vector_dim, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
            else:
                raise ValueError(
                    f"noise_encoder_type must be 'matmul', 'fourier_mlp', or "
                    f"'none'; got {self.noise_encoder_type!r}."
                )

            # Swap the FlexibleDiT conditioning_embedder for a passthrough so
            # the encoded noise we hand to ``self.dit(x, t, condition=...)``
            # reaches every adaLN unchanged. The original embedder was either
            # "zero" (returns zeros — would discard our noise) or "dit" /
            # "edm" (would re-embed the timestep — also wrong for FGN).
            self.dit.conditioning_embedder = _PassthroughConditionEmbedder()
            LOGGER.info(
                "AnemoiDiTModel: noise-vector conditioning enabled  "
                "(dim=%d, encoder=%s, swapped conditioning_embedder to passthrough)",
                self.noise_vector_dim, self.noise_encoder_type,
            )
        else:
            self.noise_encoder = None

        # AIFS-style per-grid-point noise conditioning (NoiseConditioning).
        # Mutually exclusive with the FGN noise-vector path above; the schema
        # validator catches the both-set case but we re-check defensively in
        # case a caller bypasses pydantic.
        noise_injector_cfg = getattr(dit_cfg, "noise_injector", None)
        if noise_injector_cfg is not None:
            if self.noise_encoder is not None:
                raise ValueError(
                    "AnemoiDiTModel received both `noise_vector_dim` (FGN-style) "
                    "and `noise_injector` (AIFS-style). They are mutually exclusive. "
                    "Pick one in DiTConfigSchema."
                )
            # Instantiate the NoiseConditioning / NoiseInjector layer. _recursive_
            # is False so nested ``_target_`` keys (e.g. inside layer_kernels) are
            # not pre-instantiated by Hydra before NoiseConditioning sees them.
            self.noise_injector = hydra.utils.instantiate(noise_injector_cfg, _recursive_=False)
            # Project per-grid-point noise (noise_channels_dim → hidden_size) so
            # each token's noise lands directly in the adaLN input space. Small-std
            # init keeps the warm-started deterministic features dominant at FT
            # step 0; ensemble spread grows as noise_to_hidden trains.
            hidden_size = int(dit_cfg.hidden_size)
            nc = int(self.noise_injector.noise_channels)
            self.noise_to_hidden = nn.Linear(nc, hidden_size)
            nn.init.normal_(self.noise_to_hidden.weight, std=0.02)
            nn.init.zeros_(self.noise_to_hidden.bias)
            # Swap the FlexibleDiT conditioning_embedder for a passthrough so
            # the per-token noise we pass via ``condition=...`` reaches every
            # adaLN unchanged. FlexibleDiT detects c.ndim == 3 and routes
            # through the per-token block forward.
            self.dit.conditioning_embedder = _PassthroughConditionEmbedder()
            LOGGER.info(
                "AnemoiDiTModel: AIFS-style noise conditioning enabled  "
                "(layer=%s, noise_channels=%d, hidden_size=%d, "
                "swapped conditioning_embedder to passthrough)",
                type(self.noise_injector).__name__, nc, hidden_size,
            )
        else:
            self.noise_injector = None
            self.noise_to_hidden = None

    # ------------------------------------------------------------------
    # Padding
    # ------------------------------------------------------------------

    def _pad_to_patch_size(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        """Pad spatial dims so H and W are divisible by patch_size."""
        ps_h, ps_w = self.dit.patch_size
        _, _, H, W = x.shape
        pad_h = (ps_h - H % ps_h) % ps_h
        pad_w = (ps_w - W % ps_w) % ps_w
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, (pad_h, pad_w)

    # ------------------------------------------------------------------
    # Forward: deterministic mode
    # ------------------------------------------------------------------

    def _forward_deterministic(
        self,
        x: Tensor,
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        **kwargs,
    ) -> Tensor:
        B, T, E, G, V = x.shape
        H, W = self.field_shape
        input_dtype = x.dtype

        # Reshape: (B, T, E, H*W, V) -> (B*E, T*V, H, W)
        x_2d = einops.rearrange(x, "b t e (h w) v -> (b e) (t v) h w", h=H, w=W)

        # Pad for patch_size
        x_2d, (pad_h, pad_w) = self._pad_to_patch_size(x_2d)

        # Forward through DiT with zero timestep
        t = torch.zeros(x_2d.shape[0], device=x_2d.device, dtype=x_2d.dtype)
        y_2d = self.dit(x_2d, t)  # (B*E, V_out, H_padded, W_padded)

        # Optional Gaussian LPF (anti-aliasing) \u2014 applied in-line (not as a
        # residual) because it models a physical rolloff rather than a
        # correction. Safe on a cold-start checkpoint: \u03c3=0 (default) is a
        # pure bypass via self.detokenizer_lowpass is None.
        if getattr(self, "detokenizer_lowpass", None) is not None:
            y_2d = self.detokenizer_lowpass(y_2d)

        # Optional conv refinement to smooth patch-boundary artifacts (residual).
        # hasattr guard for backward compatibility with pre-refinement checkpoints.
        if getattr(self, "conv_refinement", None) is not None:
            y_2d = y_2d + self.conv_refinement(y_2d)

        # Crop padding
        if pad_h > 0 or pad_w > 0:
            y_2d = y_2d[:, :, :H, :W]

        # Reshape back and cast to input dtype (matches GNN's _assemble_output pattern)
        y = einops.rearrange(y_2d, "(b e) v h w -> b e (h w) v", b=B, e=E).to(dtype=input_dtype).clone()

        if getattr(self, "output_mode", "residual") == "state":
            # State-space skip: add the input prognostic state to the DiT output
            # so the model effectively predicts a delta-from-input, and the
            # return value is the predicted state in normalised space. This
            # mirrors AnemoiModelEncProcDec._assemble_output line 105:
            #   x_out[..., output_prog_idx] += x_skip[..., input_prog_idx]
            # x has shape (B, T, E, G, V_in); take the last input timestep.
            x_last = x[:, -1, ...]
            y[..., self._internal_output_idx] = (
                y[..., self._internal_output_idx]
                + x_last[..., self._internal_input_idx]
            )
            # Apply boundings (e.g. ReluBounding on apcp/comp_refl/qv) in
            # normalised state space.
            for bounding in self.boundings:
                y = bounding(y)
        # NB: residual mode does NOT apply boundings (see __init__ warning).

        return y

    # ------------------------------------------------------------------
    # Forward: noise-vector conditioning (FGN-style ensemble training)
    # ------------------------------------------------------------------

    def forward_with_noise(
        self,
        x: Tensor,
        noise_vec: Tensor,
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        **kwargs,
    ) -> Tensor:
        """Forward with per-(batch, member) noise-vector conditioning.

        Mirrors ``_forward_deterministic`` but threads a noise vector
        ``z ~ N(0, I)^{noise_vector_dim}`` (one per ensemble member)
        through every block's adaLN modulation via the FlexibleDiT
        ``condition`` kwarg.

        Parameters
        ----------
        x : Tensor
            Input window, shape ``(B, T, E, G, V)``. ``E`` is the
            ensemble dim — typically replicated from a single
            realisation, with diversity coming entirely from
            ``noise_vec``.
        noise_vec : Tensor
            Per-(batch, member) noise, shape ``(B, E, noise_vector_dim)``.

        Returns
        -------
        Tensor
            Model output, shape ``(B, E, G, V_out)``.

        Notes
        -----
        Uses the same ``(b e) → flat batch`` einops rearrange as the
        deterministic path so the DiT processes each ensemble member as
        an independent batch row. ``noise_vec`` is folded the same way
        so member ``j`` of batch ``i`` ends up at the same flat index in
        both tensors.
        """
        if self.noise_encoder is None:
            raise RuntimeError(
                "forward_with_noise() called but noise_vector_dim was not "
                "configured at init. Set DiTConfigSchema.noise_vector_dim "
                "(e.g. 32) and noise_encoder_type='matmul'."
            )

        B, T, E, G, V = x.shape
        if noise_vec.shape != (B, E, self.noise_vector_dim):
            raise ValueError(
                f"noise_vec has shape {tuple(noise_vec.shape)}; expected "
                f"({B}, {E}, {self.noise_vector_dim})."
            )

        H, W = self.field_shape
        input_dtype = x.dtype

        # Reshape: (B, T, E, H*W, V) -> (B*E, T*V, H, W). Same as
        # _forward_deterministic.
        x_2d = einops.rearrange(x, "b t e (h w) v -> (b e) (t v) h w", h=H, w=W)
        x_2d, (pad_h, pad_w) = self._pad_to_patch_size(x_2d)

        # Fold noise the same way: (B, E, D) -> (B*E, D). Member j of batch i
        # lands at flat index i*E + j, matching x_2d's fold ordering.
        noise_fold = einops.rearrange(noise_vec, "b e d -> (b e) d")
        # Match the DiT input dtype for the encoded condition.
        noise_enc = self.noise_encoder(noise_fold.to(self.noise_encoder.weight.dtype))
        noise_enc = noise_enc.to(x_2d.dtype)

        # Forward through DiT. The passthrough conditioning_embedder returns
        # `condition=noise_enc` unchanged, so this goes to every block's adaLN.
        t = torch.zeros(x_2d.shape[0], device=x_2d.device, dtype=x_2d.dtype)
        y_2d = self.dit(x_2d, t, condition=noise_enc)

        # Same post-processing as _forward_deterministic.
        if getattr(self, "detokenizer_lowpass", None) is not None:
            y_2d = self.detokenizer_lowpass(y_2d)
        if getattr(self, "conv_refinement", None) is not None:
            y_2d = y_2d + self.conv_refinement(y_2d)
        if pad_h > 0 or pad_w > 0:
            y_2d = y_2d[:, :, :H, :W]

        # Reshape back (B*E, V_out, H, W) -> (B, E, G, V_out).
        y = einops.rearrange(
            y_2d, "(b e) v h w -> b e (h w) v", b=B, e=E,
        ).to(dtype=input_dtype).clone()

        # state-mode persistence skip — same as _forward_deterministic for
        # parity with non-CRPS predict_step (kept for back-compat).
        if getattr(self, "output_mode", "residual") == "state":
            x_last = x[:, -1, ...]
            y[..., self._internal_output_idx] = (
                y[..., self._internal_output_idx]
                + x_last[..., self._internal_input_idx]
            )
            for bounding in self.boundings:
                y = bounding(y)

        return y

    # ------------------------------------------------------------------
    # Forward: AIFS-style per-grid-point noise conditioning
    # ------------------------------------------------------------------

    def forward_with_spatial_noise(
        self,
        x: Tensor,
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        **kwargs,
    ) -> Tensor:
        """Forward with AIFS-style per-grid-point noise conditioning.

        Mirrors ``_forward_deterministic`` but draws a fresh
        ``(B, E, L_pad, noise_channels)`` noise tensor per call via
        ``self.noise_injector`` (a ``NoiseConditioning`` layer with an MLP+LN
        noise embedder), projects it to ``hidden_size`` via
        ``self.noise_to_hidden``, and threads it through every block's adaLN
        modulation as a 3-D conditioning tensor. FlexibleDiT detects
        ``c.ndim == 3`` and routes through ``_block_forward_per_token``.

        Parameters
        ----------
        x : Tensor
            Input window, shape ``(B, T, E, G, V)``. ``E`` is the ensemble
            dim; member diversity comes from the freshly-sampled per-token
            noise (one independent sample per (batch, member, token)).

        Returns
        -------
        Tensor
            Model output, shape ``(B, E, G, V_out)``.
        """
        if self.noise_injector is None or self.noise_to_hidden is None:
            raise RuntimeError(
                "forward_with_spatial_noise() called but noise_injector was not "
                "configured. Set dit.noise_injector in the model config."
            )

        B, T, E, G, V = x.shape
        H, W = self.field_shape
        input_dtype = x.dtype

        # Same fold as the deterministic / FGN paths: each ensemble member
        # becomes its own flat batch row so the DiT processes it independently.
        x_2d = einops.rearrange(x, "b t e (h w) v -> (b e) (t v) h w", h=H, w=W)
        x_2d, (pad_h, pad_w) = self._pad_to_patch_size(x_2d)

        # Token-grid geometry after padding.
        ps_h, ps_w = self.dit.patch_size
        h_patches = x_2d.shape[-2] // ps_h
        w_patches = x_2d.shape[-1] // ps_w
        L_pad = h_patches * w_patches

        # Sample noise and embed it.
        # NoiseConditioning.forward signature:
        #   (x, batch_size, ensemble_size, grid_size, shard_shapes_ref,
        #    noise_dtype=fp32, model_comm_group=None) -> (x_unchanged, noise)
        # The returned ``noise`` is shape ``(B*E*L_pad, noise_channels)`` after
        # the internal MLP+LN. ``shard_shapes_ref`` must be a list of lists
        # (each ending with a channels dim) because the layer internally calls
        # ``change_channels_in_shape``; we pass a trivial single-shard layout
        # since we run without a noise_projector and on a single model_comm_group.
        _, noise_flat = self.noise_injector(
            x_2d,
            batch_size=B,
            ensemble_size=E,
            grid_size=L_pad,
            shard_shapes_ref=[[B * E * L_pad, 1]],
            noise_dtype=torch.float32,
            model_comm_group=model_comm_group,
        )
        # (B*E*L_pad, noise_channels) -> (B*E, L_pad, noise_channels)
        noise_per_token = einops.rearrange(
            noise_flat, "(bse l) c -> bse l c", bse=B * E, l=L_pad,
        )
        # Project to hidden_size in the same dtype as the DiT weights.
        proj = self.noise_to_hidden(
            noise_per_token.to(self.noise_to_hidden.weight.dtype)
        ).to(x_2d.dtype)  # (B*E, L_pad, hidden_size)

        # Forward through DiT. The passthrough conditioning_embedder returns
        # `condition=proj` unchanged; FlexibleDiT sees c.ndim==3 and dispatches
        # to the per-token block forward.
        t = torch.zeros(x_2d.shape[0], device=x_2d.device, dtype=x_2d.dtype)
        y_2d = self.dit(x_2d, t, condition=proj)

        # Same post-processing as _forward_deterministic / forward_with_noise.
        if getattr(self, "detokenizer_lowpass", None) is not None:
            y_2d = self.detokenizer_lowpass(y_2d)
        if getattr(self, "conv_refinement", None) is not None:
            y_2d = y_2d + self.conv_refinement(y_2d)
        if pad_h > 0 or pad_w > 0:
            y_2d = y_2d[:, :, :H, :W]

        y = einops.rearrange(
            y_2d, "(b e) v h w -> b e (h w) v", b=B, e=E,
        ).to(dtype=input_dtype).clone()

        # state-mode persistence skip (parity with the other forwards).
        if getattr(self, "output_mode", "residual") == "state":
            x_last = x[:, -1, ...]
            y[..., self._internal_output_idx] = (
                y[..., self._internal_output_idx]
                + x_last[..., self._internal_input_idx]
            )
            for bounding in self.boundings:
                y = bounding(y)

        return y

    # ------------------------------------------------------------------
    # Forward: probabilistic (diffusion) mode
    # ------------------------------------------------------------------

    def _forward_probabilistic(
        self,
        x: Tensor,
        y_noised: Tensor,
        sigma: Tensor,
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        **kwargs,
    ) -> Tensor:
        B, T, E, G, V_in = x.shape
        H, W = self.field_shape
        input_dtype = x.dtype

        # Restrict conditioning to the configured subset (forcing/none/full).
        if self.condition_on != "full":
            x = x[..., self._cond_idx]

        # Reshape noised target to 2D
        y_2d = einops.rearrange(y_noised, "b e (h w) v -> (b e) v h w", h=H, w=W)

        # Concatenate conditioning history (if any) with the noised target.
        if self.condition_on == "none":
            combined = y_2d
        else:
            x_2d = einops.rearrange(x, "b t e (h w) v -> (b e) (t v) h w", h=H, w=W)
            combined = torch.cat([x_2d, y_2d], dim=1)
        combined, (pad_h, pad_w) = self._pad_to_patch_size(combined)

        # Use log(sigma)/4 as timestep (EDM convention)
        t = sigma.flatten()[: combined.shape[0]].log() / 4.0
        out_2d = self.dit(combined, t)  # (B*E, V_out, H_padded, W_padded)

        # Optional Gaussian LPF (anti-aliasing) applied in-line.
        if getattr(self, "detokenizer_lowpass", None) is not None:
            out_2d = self.detokenizer_lowpass(out_2d)

        # Optional conv refinement to smooth patch-boundary artifacts (residual).
        if getattr(self, "conv_refinement", None) is not None:
            out_2d = out_2d + self.conv_refinement(out_2d)

        if pad_h > 0 or pad_w > 0:
            out_2d = out_2d[:, :, :H, :W]

        y = einops.rearrange(out_2d, "(b e) v h w -> b e (h w) v", b=B, e=E).to(dtype=input_dtype).clone()
        return y

    # ------------------------------------------------------------------
    # Unified forward dispatch
    # ------------------------------------------------------------------

    def forward(self, x, *args, **kwargs):
        """Dispatch to deterministic or probabilistic forward based on args.

        Deterministic: forward(x, model_comm_group=..., grid_shard_shapes=...)
        Probabilistic: forward(x, y_noised, sigma, model_comm_group=..., grid_shard_shapes=...)
        """
        if self.mode == "probabilistic" and len(args) >= 2:
            y_noised, sigma = args[0], args[1]
            return self._forward_probabilistic(x, y_noised, sigma, **kwargs)
        return self._forward_deterministic(x, **kwargs)

    # ------------------------------------------------------------------
    # EDM preconditioning (probabilistic mode)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_preconditioning(sigma, sigma_data):
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        c_in = 1.0 / (sigma_data**2 + sigma**2) ** 0.5
        c_noise = sigma.log() / 4.0
        return c_skip, c_out, c_in, c_noise

    def fwd_with_preconditioning(self, x, y_noised, sigma, **kwargs):
        """Forward with EDM preconditioning for diffusion training."""
        c_skip, c_out, c_in, _c_noise = self._get_preconditioning(sigma, self.sigma_data)
        # Pass RAW sigma: _forward_probabilistic computes the EDM timestep
        # t = log(sigma)/4 itself. Passing c_noise (already log(sigma)/4) would
        # double-log -> log(negative)=NaN for sigma<1 (the intermittent-NaN bug).
        pred = self._forward_probabilistic(x, c_in * y_noised, sigma, **kwargs)
        return c_skip * y_noised + c_out * pred

    # ------------------------------------------------------------------
    # Predict step (deterministic \u2014 residual prediction)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_normalizer_buffers(pre_processors: nn.Module) -> tuple[Tensor, Tensor]:
        for processor in pre_processors.processors.values():
            if hasattr(processor, "_norm_mul") and hasattr(processor, "_norm_add"):
                return processor._norm_mul, processor._norm_add
        raise RuntimeError("InputNormalizer buffers not found in pre_processors.")

    def predict_step(
        self,
        batch: Tensor,
        pre_processors: nn.Module,
        post_processors: nn.Module,
        residual_normalizer: nn.Module,
        data_indices: dict,
        multi_step: int,
        model_comm_group: Optional[ProcessGroup] = None,
        gather_out: bool = True,
        **kwargs,
    ) -> Tensor:
        """Prediction step matching AnemoiResidualModelEncProcDec.predict_step.

        Predicts normalized residuals for prognostic variables and direct
        predictions for diagnostic variables, then denormalizes to physical space.
        """
        from anemoi.models.distributed.shapes import apply_shard_shapes

        with torch.no_grad():
            assert len(batch.shape) == 4, (
                f"Expected 4D batch (batch, timesteps, grid, variables), got {batch.shape}"
            )

            # Add dummy ensemble dimension
            x = batch[:, 0:multi_step, None, ...].clone()  # (B, T, 1, G, V)

            # Handle distributed sharding
            grid_shard_shapes = None
            if model_comm_group is not None:
                shard_shapes = get_shard_shapes(x, -2, model_comm_group)
                grid_shard_shapes = [shape[-2] for shape in shard_shapes]
                x = shard_tensor(x, -2, shard_shapes, model_comm_group)

            # Normalize input
            x = pre_processors(x, in_place=True)

            # Forward pass \u2014 output interpretation depends on output_mode:
            #   "residual" \u2192 model output is a normalised RESIDUAL; reconstruct
            #               physical state by adding the previous step.
            #   "state"    \u2192 model output is the predicted STATE in normalised
            #               space (with state-skip already applied inside
            #               _forward_deterministic). Just denormalise the
            #               output through the input normaliser; the residual
            #               normaliser path would double-add the input state
            #               and produce explosive rollouts.
            # Route through forward_with_noise when the model is an FGN-style
            # ensemble model (noise_encoder is configured). The noise vector is
            # read from ``self._inference_noise_vec`` (a (B, E=1, noise_dim) tensor
            # set by the caller) so the same vector is reused across every
            # autoregressive step in one forecast, but different members can be
            # produced by setting different vectors before each inference run.
            # If no vector is attached, a single fresh one is sampled and stored
            # (default member-0 trajectory; the model becomes a deterministic-
            # under-fixed-noise predictor).
            if getattr(self, "noise_injector", None) is not None:
                # AIFS-style ensemble model: per-grid-point noise is sampled
                # fresh on every call inside forward_with_spatial_noise (no
                # caller-controlled noise vector to thread through, unlike the
                # FGN path). Different members emerge because the AR loop
                # invokes this once per member with the same x.
                model_output = self.forward_with_spatial_noise(
                    x,
                    model_comm_group=model_comm_group,
                    grid_shard_shapes=grid_shard_shapes,
                    **kwargs,
                )
            elif getattr(self, "noise_encoder", None) is not None:
                B = x.shape[0]
                E = x.shape[2]
                noise_vec = getattr(self, "_inference_noise_vec", None)
                if noise_vec is None:
                    noise_vec = torch.randn(
                        B, E, self.noise_vector_dim,
                        device=x.device, dtype=x.dtype,
                    )
                    self._inference_noise_vec = noise_vec
                # Allow caller to pre-set a (1, 1, D) vector; broadcast to (B, E, D).
                if noise_vec.shape[0] != B or noise_vec.shape[1] != E:
                    noise_vec = noise_vec.expand(B, E, -1).contiguous()
                model_output = self.forward_with_noise(
                    x,
                    noise_vec.to(device=x.device, dtype=x.dtype),
                    model_comm_group=model_comm_group,
                    grid_shard_shapes=grid_shard_shapes,
                    **kwargs,
                )
            else:
                model_output = self.forward(
                    x, model_comm_group=model_comm_group, grid_shard_shapes=grid_shard_shapes, **kwargs
                )  # (B, E=1, G, V_out)
            output_mode = getattr(self, "output_mode", "residual")

            # Variable indices
            model_prog_idx = data_indices.model.output.prognostic
            model_diag_idx = data_indices.model.output.diagnostic
            # data.input.prognostic: data space (indexes the n_data-wide normalizer buffers).
            input_prog_idx = data_indices.data.input.prognostic
            # model.input.prognostic: model-input space (indexes the sliced input tensor x,
            # which is prognostic+forcing only). These DIVERGE once diagnostics exist; using
            # the data-space index on x then runs off the end. Equal when diagnostic=[].
            model_input_prog_idx = data_indices.model.input.prognostic

            # Normalizer buffers
            norm_mul, norm_add = self._get_normalizer_buffers(pre_processors)

            if output_mode == "state":
                # Output is normalised state. Denormalise prognostic outputs
                # via input normaliser (state-stats), no residual reconstruction.
                # x_phys = (x_norm \u2212 norm_add) / norm_mul   <=>   x_norm = x_phys\u00b7norm_mul + norm_add
                state_norm_prog = model_output[..., model_prog_idx].float()
                prog_mul = norm_mul[input_prog_idx].float()
                prog_add = norm_add[input_prog_idx].float()
                y_hat_prog_phys = ((state_norm_prog - prog_add) / prog_mul).to(model_output.dtype)
            else:
                # Residual mode (legacy): output is delta-from-input in residual
                # normalised space; reconstruct physical state.
                delta_norm_prog = model_output[..., model_prog_idx]  # (B, 1, G, n_prog)
                x_last_norm_prog = x[:, -1, ..., model_input_prog_idx]  # (B, 1, G, n_prog); x is model-input space
                if getattr(self, "reference_truncation", 0):
                    H_rt, W_rt = self.field_shape
                    if x_last_norm_prog.shape[-2] == H_rt * W_rt:
                        from fnmatch import fnmatch
                        from anemoi.models.models.flexible_dit import reference_truncate
                        excl = list(getattr(self, "reference_truncation_exclude",
                                            ["pressure_*", "t2m", "skintemp", "snowh"]) or [])
                        i2n = {int(v): k for k, v in data_indices.name_to_index.items()}
                        prog_names = [i2n[int(i)] for i in data_indices.data.input.prognostic]
                        chans = [c for c, nm in enumerate(prog_names)
                                 if not any(fnmatch(nm, pat) for pat in excl)]
                        xt = x_last_norm_prog.float()
                        if len(chans) == xt.shape[-1]:
                            xt = reference_truncate(xt, H_rt, W_rt, self.reference_truncation)
                        else:
                            xt = xt.clone()
                            xt[..., chans] = reference_truncate(
                                xt[..., chans], H_rt, W_rt, self.reference_truncation,
                            )
                        x_last_norm_prog = xt.to(x_last_norm_prog.dtype)
                y_hat_prog_phys = residual_normalizer.inverse_transform_physical_from_normalized(
                    x_last_norm_prog,
                    delta_norm_prog,
                    norm_mul,
                    norm_add,
                )

            # Build output tensor
            n_output = len(data_indices.model.output.full)
            batch_size = model_output.shape[0]
            ensemble_size = model_output.shape[1]
            grid_size = model_output.shape[2]

            y_hat = torch.zeros(
                batch_size, ensemble_size, grid_size, n_output,
                dtype=model_output.dtype, device=model_output.device,
            )
            y_hat[..., model_prog_idx] = y_hat_prog_phys

            # Diagnostic: direct denormalization
            if len(model_diag_idx) > 0:
                diag_output_norm = model_output[..., model_diag_idx]
                # Diagnostics are OUTPUT-only -> their normalizer coeffs live at the
                # data.output.diagnostic positions (data.input.diagnostic is empty, which
                # would silently leave diagnostics in normalized units).
                input_diag_idx = (
                    data_indices.data.output.diagnostic
                    if hasattr(data_indices.data.output, "diagnostic")
                    else []
                )
                if len(input_diag_idx) > 0:
                    diag_mul = norm_mul[input_diag_idx].float()
                    diag_add = norm_add[input_diag_idx].float()
                    y_hat_diag_phys = (diag_output_norm.float() - diag_add) / diag_mul
                    y_hat[..., model_diag_idx] = y_hat_diag_phys.to(model_output.dtype)
                else:
                    y_hat[..., model_diag_idx] = diag_output_norm

            # Squeeze ensemble and gather
            y_hat = y_hat.squeeze(1)  # (B, G, V_out)

            if gather_out and model_comm_group is not None:
                y_hat = gather_tensor(
                    y_hat, -2, apply_shard_shapes(y_hat, -2, grid_shard_shapes), model_comm_group
                )

        return y_hat
