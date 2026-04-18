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


def _swap_activation(module: nn.Module, old_cls: type, new_cls: type) -> int:
    """Walk a module tree and replace every instance of ``old_cls`` with
    ``new_cls``. Returns the number of swaps performed.

    Used to retrofit activation choices (e.g., GELU→SiLU) onto the physicsnemo
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

        # For probabilistic: input = [x_history, y_noised] concatenated on channels
        if self.mode == "probabilistic":
            in_channels = self.multi_step * self.num_input_channels + self.num_output_channels
        else:
            in_channels = self.multi_step * self.num_input_channels

        # Convert OmegaConf objects to plain dicts for physicsnemo
        tokenizer_kwargs = dict(getattr(dit_cfg, "tokenizer_kwargs", {}))
        attn_kwargs = dict(getattr(dit_cfg, "attn_kwargs", {}))
        conditioning_embedder_kwargs = dict(getattr(dit_cfg, "conditioning_embedder_kwargs", {}))

        LOGGER.info(
            f"Initializing FlexibleDiT: mode={self.mode}, in_channels={in_channels}, "
            f"out_channels={self.num_output_channels}, field_shape={self.field_shape}, "
            f"patch_size={dit_cfg.patch_size}, hidden_size={dit_cfg.hidden_size}, "
            f"depth={dit_cfg.depth}, num_heads={dit_cfg.num_heads}, "
            f"attention_backend={dit_cfg.attention_backend}"
        )

        self.dit = FlexibleDiT(
            input_size=self.field_shape,
            in_channels=in_channels,
            out_channels=self.num_output_channels,
            patch_size=int(dit_cfg.patch_size),
            hidden_size=int(dit_cfg.hidden_size),
            depth=int(dit_cfg.depth),
            num_heads=int(dit_cfg.num_heads),
            mlp_ratio=float(getattr(dit_cfg, "mlp_ratio", 4.0)),
            attention_backend=str(dit_cfg.attention_backend),
            conditioning_embedder=str(getattr(dit_cfg, "conditioning_embedder", "zero")),
            condition_dim=getattr(dit_cfg, "condition_dim", None),
            tokenizer_kwargs=tokenizer_kwargs,
            attn_kwargs=attn_kwargs,
            conditioning_embedder_kwargs=conditioning_embedder_kwargs,
            force_tokenization_fp32=bool(getattr(dit_cfg, "force_tokenization_fp32", True)),
        )

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
            # Zero-init the final layer so the refinement starts as an identity residual
            # (training-stability trick: don't perturb a DiT checkpoint if initialized cold).
            last_conv = [m for m in self.conv_refinement.modules() if isinstance(m, nn.Conv2d)][-1]
            nn.init.zeros_(last_conv.weight)
            if last_conv.bias is not None:
                nn.init.zeros_(last_conv.bias)
            LOGGER.info("DiT conv refinement enabled: %d blocks, kernel=%d, hidden=%d, act=%s",
                        n_refine, refine_kernel, refine_hidden, refine_act_cls.__name__)
        else:
            self.conv_refinement = None

        # Store data indices for predict_step
        self.data_indices = data_indices
        self._internal_input_idx = data_indices.model.input.prognostic
        self._internal_output_idx = data_indices.model.output.prognostic

        # Boundings (e.g., ReLU for precipitation)
        self.boundings = build_boundings(config, data_indices, statistics)

        # Diffusion parameters (probabilistic mode)
        if self.mode == "probabilistic":
            self.sigma_data = float(getattr(dit_cfg, "sigma_data", 1.0))
            self.sigma_max = float(getattr(dit_cfg, "sigma_max", 100.0))
            self.sigma_min = float(getattr(dit_cfg, "sigma_min", 0.02))
            self.rho = float(getattr(dit_cfg, "rho", 7.0))
            self.inference_defaults = dict(getattr(dit_cfg, "inference_defaults", {}))

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

        # Optional conv refinement to smooth patch-boundary artifacts (zero-init residual)
        # hasattr guard for backward compatibility with pre-refinement checkpoints.
        if getattr(self, "conv_refinement", None) is not None:
            y_2d = y_2d + self.conv_refinement(y_2d)

        # Crop padding
        if pad_h > 0 or pad_w > 0:
            y_2d = y_2d[:, :, :H, :W]

        # Reshape back and cast to input dtype (matches GNN's _assemble_output pattern)
        y = einops.rearrange(y_2d, "(b e) v h w -> b e (h w) v", b=B, e=E).to(dtype=input_dtype).clone()

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

        # Reshape input history and noised target to 2D
        x_2d = einops.rearrange(x, "b t e (h w) v -> (b e) (t v) h w", h=H, w=W)
        y_2d = einops.rearrange(y_noised, "b e (h w) v -> (b e) v h w", h=H, w=W)

        # Concatenate: [x_history, y_noised] along channel dim
        combined = torch.cat([x_2d, y_2d], dim=1)
        combined, (pad_h, pad_w) = self._pad_to_patch_size(combined)

        # Use log(sigma)/4 as timestep (EDM convention)
        t = sigma.flatten()[: combined.shape[0]].log() / 4.0
        out_2d = self.dit(combined, t)  # (B*E, V_out, H_padded, W_padded)

        # Optional conv refinement to smooth patch-boundary artifacts (zero-init residual)
        # hasattr guard for backward compatibility with pre-refinement checkpoints.
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
        c_skip, c_out, c_in, c_noise = self._get_preconditioning(sigma, self.sigma_data)
        pred = self._forward_probabilistic(x, c_in * y_noised, c_noise, **kwargs)
        return c_skip * y_noised + c_out * pred

    # ------------------------------------------------------------------
    # Predict step (deterministic — residual prediction)
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

            # Forward pass — predicts normalized residuals
            model_output = self.forward(
                x, model_comm_group=model_comm_group, grid_shard_shapes=grid_shard_shapes, **kwargs
            )  # (B, E=1, G, V_out)

            # Variable indices
            model_prog_idx = data_indices.model.output.prognostic
            model_diag_idx = data_indices.model.output.diagnostic
            input_prog_idx = data_indices.data.input.prognostic

            # Normalizer buffers
            norm_mul, norm_add = self._get_normalizer_buffers(pre_processors)

            # Prognostic: residual denormalization
            delta_norm_prog = model_output[..., model_prog_idx]  # (B, 1, G, n_prog)
            x_last_norm_prog = x[:, -1, ..., input_prog_idx]  # (B, 1, G, n_prog)

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
                input_diag_idx = (
                    data_indices.data.input.diagnostic
                    if hasattr(data_indices.data.input, "diagnostic")
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
