"""Per-layer weight gradient norm monitoring callback.

Logs weight gradient L2 norms per component (encoder, each processor layer, decoder)
to MLflow every N steps. This measures the actual learning signal, not per-node
activation gradients which are scaled by 1/N from loss normalization.
"""

import logging
import re

import torch
from pytorch_lightning import Callback

LOGGER = logging.getLogger(__name__)


class GradientActivationMonitor(Callback):
    """Monitor per-layer weight gradient norms.

    After each backward pass, iterates over named parameters in encoder,
    processor, and decoder. Computes L2 norm of .grad for each parameter,
    then aggregates by component (encoder, processor.layer_N, decoder).

    Parameters
    ----------
    config : DictConfig
        Full training configuration (required by Anemoi callback interface).
    log_interval : int
        Log statistics every this many training steps. Default 250.
    """

    def __init__(self, config, log_interval: int = 250):
        super().__init__()
        self.log_interval = log_interval
        self._step_count = 0

    def _collect_grad_norms(self, module, prefix):
        """Collect per-parameter gradient norms, grouped by layer.

        Returns dict mapping group_name -> list of (param_name, grad_norm).
        For the processor, groups by layer index (e.g., "processor.layer_0").
        For encoder/decoder, uses prefix directly.
        """
        groups = {}
        for name, param in module.named_parameters():
            if param.grad is None:
                continue
            grad_norm = param.grad.float().norm(2).item()
            num_elements = param.grad.numel()

            # Extract processor layer index if present
            # Matches patterns like "proc.0.", "chunks.0.", "layers.0.", "blocks.0."
            layer_match = re.search(r'(?:proc|chunks|layers|blocks)\.(\d+)', name)
            if layer_match and prefix == "processor":
                group = f"{prefix}.layer_{layer_match.group(1)}"
            else:
                group = prefix

            if group not in groups:
                groups[group] = {'sq_sum': 0.0, 'max_norm': 0.0, 'count': 0, 'total_elements': 0}
            groups[group]['sq_sum'] += grad_norm ** 2
            groups[group]['max_norm'] = max(groups[group]['max_norm'], grad_norm)
            groups[group]['count'] += 1
            groups[group]['total_elements'] += num_elements

        return groups

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Log weight gradient norms before the optimizer step (after backward, before clipping)."""
        self._step_count += 1
        if self._step_count % self.log_interval != 0:
            return

        try:
            model = pl_module.model.model
        except AttributeError:
            return

        all_groups = {}
        for component_name in ['encoder', 'processor', 'decoder']:
            component = getattr(model, component_name, None)
            if component is not None:
                groups = self._collect_grad_norms(component, component_name)
                all_groups.update(groups)

        # Log per-group stats
        for group_name, stats in sorted(all_groups.items()):
            # Total L2 norm for this group (sqrt of sum of squared per-param norms)
            total_norm = stats['sq_sum'] ** 0.5
            pl_module.log(
                f"wgrad/{group_name}/norm",
                total_norm,
                on_step=True, on_epoch=False, logger=True, rank_zero_only=True,
            )
            pl_module.log(
                f"wgrad/{group_name}/max_param_norm",
                stats['max_norm'],
                on_step=True, on_epoch=False, logger=True, rank_zero_only=True,
            )
            pl_module.log(
                f"wgrad/{group_name}/num_params",
                float(stats['count']),
                on_step=True, on_epoch=False, logger=True, rank_zero_only=True,
            )

        # Log total model gradient norm
        total_sq = sum(s['sq_sum'] for s in all_groups.values())
        pl_module.log(
            "wgrad/total_norm",
            total_sq ** 0.5,
            on_step=True, on_epoch=False, logger=True, rank_zero_only=True,
        )

    def on_train_end(self, trainer, pl_module):
        LOGGER.info("GradientActivationMonitor: Training ended.")
