"""Effective rank monitoring callback for encoder and processor latent spaces.

Registers forward hooks on encoder and processor to capture intermediate
latent tensors, computes SVD periodically, and logs effective rank metrics
to MLflow via pl_module.log.

Effective rank = number of singular values explaining 99% of total variance.
This diagnoses rank collapse where only a few dimensions carry signal.
"""

import logging

import torch
from pytorch_lightning import Callback

LOGGER = logging.getLogger(__name__)


class EffectiveRankMonitor(Callback):
    """Monitor effective rank of encoder and processor latent representations.

    Uses forward hooks to capture output tensors from encoder and processor,
    then computes SVD to track rank evolution during training.

    Parameters
    ----------
    config : DictConfig
        Full training configuration (required by Anemoi callback interface).
    log_interval : int
        Compute and log rank metrics every this many training steps. Default 50.
    variance_threshold : float
        Fraction of total variance for effective rank computation. Default 0.99.
    """

    def __init__(self, config, log_interval: int = 50, variance_threshold: float = 0.99):
        super().__init__()
        self.log_interval = log_interval
        self.variance_threshold = variance_threshold

        self._step_count = 0
        self._should_capture = False
        self._encoder_output = None
        self._processor_output = None
        self._hooks = []

    def setup(self, trainer, pl_module, stage=None):
        """Register forward hooks on encoder and processor."""
        try:
            model = pl_module.model.model
        except AttributeError:
            LOGGER.warning("EffectiveRankMonitor: Cannot access pl_module.model.model, skipping hook setup.")
            return

        encoder = getattr(model, "encoder", None)
        processor = getattr(model, "processor", None)

        if encoder is not None:
            hook = encoder.register_forward_hook(self._encoder_hook)
            self._hooks.append(hook)
            LOGGER.info("EffectiveRankMonitor: Registered forward hook on encoder.")

        if processor is not None:
            hook = processor.register_forward_hook(self._processor_hook)
            self._hooks.append(hook)
            LOGGER.info("EffectiveRankMonitor: Registered forward hook on processor.")

    def _encoder_hook(self, module, input, output):
        """Capture encoder output. Encoder returns tuple (x_src, x_dst) — take x_dst."""
        if not self._should_capture:
            return
        if isinstance(output, tuple):
            self._encoder_output = output[1].detach()
        else:
            self._encoder_output = output.detach()

    def _processor_hook(self, module, input, output):
        """Capture processor output tensor."""
        if not self._should_capture:
            return
        if isinstance(output, tuple):
            self._processor_output = output[0].detach()
        else:
            self._processor_output = output.detach()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Set capture flag on logging steps."""
        self._step_count += 1
        self._should_capture = (self._step_count % self.log_interval == 0)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Compute SVD and log rank metrics after forward pass on logging steps."""
        if not self._should_capture:
            return

        for name, tensor in [("encoder", self._encoder_output), ("processor", self._processor_output)]:
            if tensor is None:
                continue
            self._compute_and_log_rank(name, tensor, pl_module)

        # Clear captured tensors
        self._encoder_output = None
        self._processor_output = None
        self._should_capture = False

    def _compute_and_log_rank(self, component_name, tensor, pl_module):
        """Compute SVD and log effective rank metrics for a component.

        Parameters
        ----------
        component_name : str
            "encoder" or "processor"
        tensor : torch.Tensor
            Latent tensor of shape (N_nodes, hidden_dim)
        pl_module : LightningModule
            For logging
        """
        try:
            # Move to CPU and upcast to float32 for stable SVD
            t = tensor.float().cpu()

            # Handle batched tensors: reshape to 2D if needed
            if t.ndim > 2:
                t = t.reshape(-1, t.shape[-1])

            # Compute SVD (economy mode)
            s = torch.linalg.svdvals(t)

            # Effective rank: number of SVs explaining threshold% of variance
            sv_sq = s ** 2
            total_var = sv_sq.sum()
            if total_var == 0:
                effective_rank = 0.0
            else:
                cumvar = torch.cumsum(sv_sq, dim=0) / total_var
                effective_rank = float((cumvar < self.variance_threshold).sum().item() + 1)

            # Log effective rank
            pl_module.log(
                f"rank/{component_name}/effective_rank",
                effective_rank,
                on_step=True, on_epoch=False, logger=True, rank_zero_only=True,
            )

            # Spectral gap: ratio of largest to 10th singular value
            if len(s) >= 10:
                sv1_sv10_ratio = float(s[0] / (s[9] + 1e-12))
                pl_module.log(
                    f"rank/{component_name}/sv1_sv10_ratio",
                    sv1_sv10_ratio,
                    on_step=True, on_epoch=False, logger=True, rank_zero_only=True,
                )

            # Log key individual singular values
            for idx in [1, 5, 10, 50]:
                if len(s) >= idx:
                    pl_module.log(
                        f"rank/{component_name}/sv{idx}",
                        float(s[idx - 1]),
                        on_step=True, on_epoch=False, logger=True, rank_zero_only=True,
                    )

        except Exception:
            LOGGER.exception("EffectiveRankMonitor: Failed to compute rank for %s", component_name)

    def on_train_end(self, trainer, pl_module):
        """Remove forward hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        LOGGER.info("EffectiveRankMonitor: Removed forward hooks. Training ended.")
