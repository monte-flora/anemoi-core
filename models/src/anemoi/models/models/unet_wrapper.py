"""Anemoi-compatible wrapper for the PhysicsNeMo SongUNet.

Supports two modes:
  - deterministic: MSE/MAE training with embedding_type="zero"
  - probabilistic: EDM diffusion training with noise-level conditioning

The wrapper handles:
  - Reshaping from Anemoi's flat grid (B, T, E, H*W, V) to 2D spatial (B, C, H, W)
  - Dynamic reflect-padding for UNet divisibility (2^(n_levels-1))
  - Optional domain-parallel sharding for CONUS-scale inference via ShardTensor
  - Residual prediction and denormalization in predict_step
  - Same forward/predict_step interface as AnemoiDiTModel
"""

import logging
import math
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
from anemoi.utils.config import DotDict

from physicsnemo.models.diffusion_unets import SongUNet

LOGGER = logging.getLogger(__name__)


class LargeKernelStem(nn.Module):
    """Depthwise-separable large-kernel 2D conv (RepLKNet-style), drop-in
    replacement for SongUNet's first-level encoder Conv2d.

    Gives the model a ~O(kernel) receptive field on the very first layer
    without the param/compute blowup of a dense large-kernel conv:
      depthwise kernel × 1 (Cin params)  +  pointwise 1×1 (Cin×Cout)
    vs dense kernel² × Cin × Cout for a vanilla large conv.

    Uses reflect padding — better for atmospheric fields at domain
    boundaries than zero or replicate.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel: int = 51):
        super().__init__()
        assert kernel % 2 == 1, "kernel must be odd"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dw = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel, padding=kernel // 2,
            padding_mode="reflect",
            groups=in_channels,
            bias=True,
        )
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.dw.weight, mode="fan_out", nonlinearity="linear")
        nn.init.kaiming_normal_(self.pw.weight, mode="fan_out", nonlinearity="linear")
        nn.init.zeros_(self.dw.bias)
        nn.init.zeros_(self.pw.bias)

    def forward(self, x, *args, **kwargs):
        # SongUNet calls block(x) on the non-UNetBlock stem; *args/**kwargs absorb anything extra.
        return self.pw(self.dw(x))


class AnemoiUNetModel(nn.Module):
    """SongUNet model wrapped for the Anemoi training/inference pipeline.

    Replaces the encoder-processor-decoder GNN with a 2D U-Net that operates
    on spatial grids. Transfer learning (train on patch, infer on CONUS) is
    enabled by using embedding_type="zero" (no positional embeddings) and
    reflect-padding for divisibility.
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
        unet_cfg = config.model.model.unet

        self.field_shape = tuple(unet_cfg.field_shape)
        self.multi_step = config.training.multistep_input
        self.mode = getattr(unet_cfg, "mode", "deterministic")

        # Variable counts from data_indices
        self.num_input_channels = len(data_indices.model.input)
        self.num_output_channels = len(data_indices.model.output)

        if self.mode == "probabilistic":
            in_channels = self.multi_step * self.num_input_channels + self.num_output_channels
        else:
            in_channels = self.multi_step * self.num_input_channels

        # Compute padding divisor: 2^(n_levels - 1)
        channel_mult = list(getattr(unet_cfg, "channel_mult", [1, 2, 3, 4]))
        n_levels = len(channel_mult)
        self._pad_divisor = 2 ** (n_levels - 1)

        # Compute padded resolution for attn_resolutions
        H_padded = self._next_multiple(self.field_shape[0], self._pad_divisor)
        W_padded = self._next_multiple(self.field_shape[1], self._pad_divisor)

        # Attention at N coarsest resolution levels
        n_attn_levels = int(getattr(unet_cfg, "n_attn_levels", 2))
        all_resolutions_h = [H_padded >> i for i in range(n_levels)]
        attn_resolutions = all_resolutions_h[-n_attn_levels:]

        LOGGER.info(
            "Initializing SongUNet: mode=%s, in_channels=%d, out_channels=%d, "
            "field_shape=%s, padded=%dx%d, model_channels=%d, channel_mult=%s, "
            "num_blocks=%d, attn_resolutions=%s, dropout=%.2f",
            self.mode, in_channels, self.num_output_channels,
            self.field_shape, H_padded, W_padded,
            unet_cfg.model_channels, channel_mult,
            unet_cfg.num_blocks, attn_resolutions,
            getattr(unet_cfg, "dropout", 0.10),
        )

        self.unet = SongUNet(
            img_resolution=[H_padded, W_padded],
            in_channels=in_channels,
            out_channels=self.num_output_channels,
            model_channels=int(unet_cfg.model_channels),
            channel_mult=channel_mult,
            num_blocks=int(unet_cfg.num_blocks),
            attn_resolutions=attn_resolutions,
            dropout=float(getattr(unet_cfg, "dropout", 0.10)),
            embedding_type="zero" if self.mode == "deterministic" else "positional",
            encoder_type=str(getattr(unet_cfg, "encoder_type", "standard")),
            decoder_type=str(getattr(unet_cfg, "decoder_type", "standard")),
            bottleneck_attention=bool(getattr(unet_cfg, "bottleneck_attention", True)),
            amp_mode=True,  # Required for Lightning's bf16-mixed precision autocast
        )

        # Optional: replace SongUNet's 3x3 encoder stem with a depthwise-separable
        # large-kernel stem (RepLKNet-style). Gives the model a ~O(stem_kernel)
        # receptive field on the very first layer, comparable to GNN encoder
        # connectivity, at ~1% the param cost of a dense large-kernel conv.
        stem_kernel = int(getattr(unet_cfg, "large_kernel_stem", 0))
        if stem_kernel > 0:
            self._patch_stem_large_kernel(stem_kernel)
            LOGGER.info("SongUNet stem replaced with depthwise-separable %dx%d kernel", stem_kernel, stem_kernel)

        # Store data indices for predict_step
        self.data_indices = data_indices
        self._internal_input_idx = data_indices.model.input.prognostic
        self._internal_output_idx = data_indices.model.output.prognostic

        # Boundings (e.g., ReLU for precipitation)
        self.boundings = build_boundings(config, data_indices, statistics)

        # Diffusion parameters (probabilistic mode)
        if self.mode == "probabilistic":
            self.sigma_data = float(getattr(unet_cfg, "sigma_data", 1.0))
            self.sigma_max = float(getattr(unet_cfg, "sigma_max", 100.0))
            self.sigma_min = float(getattr(unet_cfg, "sigma_min", 0.02))
            self.rho = float(getattr(unet_cfg, "rho", 7.0))
            self.inference_defaults = dict(getattr(unet_cfg, "inference_defaults", {}))

        # Domain parallelism (optional, for CONUS-scale inference)
        self._domain_parallel_size = int(getattr(unet_cfg, "domain_parallel_size", 1))
        self._shard_dim = int(getattr(unet_cfg, "shard_dim", 2))  # shard along H by default
        self._domain_helper = None  # lazy init on first use

    def _get_domain_helper(self):
        """Lazily initialize domain parallel helper if needed."""
        if self._domain_helper is None and self._domain_parallel_size > 1:
            try:
                from physicsnemo.domain_parallel import ShardTensor, scatter_tensor
                # Import the ParallelHelper from stormcast utils
                # or build a minimal version inline
                import torch.distributed as dist
                from torch.distributed._tensor import Shard, Replicate, DeviceMesh

                if not dist.is_initialized():
                    LOGGER.warning("Domain parallel requested but torch.distributed not initialized. Disabling.")
                    self._domain_parallel_size = 1
                    return None

                world_size = dist.get_world_size()
                if world_size < self._domain_parallel_size:
                    LOGGER.warning(
                        "domain_parallel_size=%d but only %d GPUs available. Disabling.",
                        self._domain_parallel_size, world_size,
                    )
                    self._domain_parallel_size = 1
                    return None

                self._domain_helper = {
                    "scatter_tensor": scatter_tensor,
                    "ShardTensor": ShardTensor,
                    "Shard": Shard,
                    "Replicate": Replicate,
                    "mesh": DeviceMesh(
                        "cuda",
                        torch.arange(self._domain_parallel_size),
                    ),
                    "shard_dim": self._shard_dim,
                }
                LOGGER.info(
                    "Domain parallel enabled: %d GPUs, shard_dim=%d",
                    self._domain_parallel_size, self._shard_dim,
                )
            except ImportError:
                LOGGER.warning("physicsnemo.domain_parallel not available. Disabling domain sharding.")
                self._domain_parallel_size = 1
                return None
        return self._domain_helper

    def _shard_input(self, x: Tensor) -> Tensor:
        """Shard a (B, C, H, W) tensor across the domain mesh if domain parallel is active."""
        if self._domain_parallel_size <= 1:
            return x
        helper = self._get_domain_helper()
        if helper is None:
            return x
        source_rank = 0
        placement = helper["Shard"](helper["shard_dim"])
        return helper["scatter_tensor"](
            x, source_rank, helper["mesh"],
            placements=(placement,),
            global_shape=x.shape,
            dtype=x.dtype,
        )

    def _gather_output(self, x) -> Tensor:
        """Gather a ShardTensor back to a full tensor if domain parallel is active."""
        if self._domain_parallel_size <= 1:
            return x
        # ShardTensor.full_tensor() gathers all shards
        if hasattr(x, "full_tensor"):
            return x.full_tensor()
        return x

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _next_multiple(n: int, d: int) -> int:
        """Round up n to the next multiple of d."""
        return n + (d - n % d) % d

    def _patch_stem_large_kernel(self, kernel: int) -> None:
        """Replace SongUNet's first-level encoder Conv2d with LargeKernelStem.

        The SongUNet encoder ModuleDict keys the first-level conv as
        f"{res}x{res}_conv" where res = img_resolution[0]. We locate it as the
        first '_conv' (non-aux) entry and swap in the depthwise-separable
        large-kernel module.
        """
        enc = self.unet.enc
        stem_key = None
        for key in enc:
            if "_conv" in key and "aux" not in key:
                stem_key = key
                break
        if stem_key is None:
            raise RuntimeError("Could not locate SongUNet encoder stem conv")
        original = enc[stem_key]
        new_stem = LargeKernelStem(
            in_channels=original.in_channels,
            out_channels=original.out_channels,
            kernel=kernel,
        )
        # Match device/dtype
        new_stem = new_stem.to(
            device=next(original.parameters()).device,
            dtype=next(original.parameters()).dtype,
        )
        enc[stem_key] = new_stem

    def _pad_to_divisor(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        """Reflect-pad spatial dims to be divisible by self._pad_divisor."""
        _, _, H, W = x.shape
        pad_h = (self._pad_divisor - H % self._pad_divisor) % self._pad_divisor
        pad_w = (self._pad_divisor - W % self._pad_divisor) % self._pad_divisor
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

        x_2d = einops.rearrange(x, "b t e (h w) v -> (b e) (t v) h w", h=H, w=W)
        x_2d, (pad_h, pad_w) = self._pad_to_divisor(x_2d)

        # Optional domain-parallel sharding for large domains
        x_2d = self._shard_input(x_2d)

        # SongUNet requires noise_labels even with embedding_type="zero"
        noise_labels = torch.zeros(x_2d.shape[0], device=x_2d.device)
        y_2d = self.unet(x_2d, noise_labels)

        # Gather shards back if domain parallel
        y_2d = self._gather_output(y_2d)

        if pad_h > 0 or pad_w > 0:
            y_2d = y_2d[:, :, :H, :W]

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

        x_2d = einops.rearrange(x, "b t e (h w) v -> (b e) (t v) h w", h=H, w=W)
        y_2d = einops.rearrange(y_noised, "b e (h w) v -> (b e) v h w", h=H, w=W)

        combined = torch.cat([x_2d, y_2d], dim=1)
        combined, (pad_h, pad_w) = self._pad_to_divisor(combined)

        # Optional domain-parallel sharding
        combined = self._shard_input(combined)

        # noise_labels = log(sigma)/4 for EDM convention
        noise_labels = sigma.flatten()[: combined.shape[0]].log() / 4.0
        out_2d = self.unet(combined, noise_labels)

        # Gather shards back
        out_2d = self._gather_output(out_2d)

        if pad_h > 0 or pad_w > 0:
            out_2d = out_2d[:, :, :H, :W]

        y = einops.rearrange(out_2d, "(b e) v h w -> b e (h w) v", b=B, e=E).to(dtype=input_dtype).clone()
        return y

    # ------------------------------------------------------------------
    # Unified forward dispatch
    # ------------------------------------------------------------------

    def forward(self, x, *args, **kwargs):
        """Dispatch to deterministic or probabilistic forward."""
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
        """Prediction step matching AnemoiResidualModelEncProcDec.predict_step."""
        from anemoi.models.distributed.shapes import apply_shard_shapes

        with torch.no_grad():
            assert len(batch.shape) == 4, (
                f"Expected 4D batch (batch, timesteps, grid, variables), got {batch.shape}"
            )

            x = batch[:, 0:multi_step, None, ...].clone()

            grid_shard_shapes = None
            if model_comm_group is not None:
                shard_shapes = get_shard_shapes(x, -2, model_comm_group)
                grid_shard_shapes = [shape[-2] for shape in shard_shapes]
                x = shard_tensor(x, -2, shard_shapes, model_comm_group)

            x = pre_processors(x, in_place=True)

            model_output = self.forward(
                x, model_comm_group=model_comm_group, grid_shard_shapes=grid_shard_shapes, **kwargs
            )

            model_prog_idx = data_indices.model.output.prognostic
            model_diag_idx = data_indices.model.output.diagnostic
            input_prog_idx = data_indices.data.input.prognostic

            norm_mul, norm_add = self._get_normalizer_buffers(pre_processors)

            delta_norm_prog = model_output[..., model_prog_idx]
            x_last_norm_prog = x[:, -1, ..., input_prog_idx]

            y_hat_prog_phys = residual_normalizer.inverse_transform_physical_from_normalized(
                x_last_norm_prog, delta_norm_prog, norm_mul, norm_add,
            )

            n_output = len(data_indices.model.output.full)
            batch_size = model_output.shape[0]
            ensemble_size = model_output.shape[1]
            grid_size = model_output.shape[2]

            y_hat = torch.zeros(
                batch_size, ensemble_size, grid_size, n_output,
                dtype=model_output.dtype, device=model_output.device,
            )
            y_hat[..., model_prog_idx] = y_hat_prog_phys

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

            y_hat = y_hat.squeeze(1)

            if gather_out and model_comm_group is not None:
                y_hat = gather_tensor(
                    y_hat, -2, apply_shard_shapes(y_hat, -2, grid_shard_shapes), model_comm_group
                )

        return y_hat
