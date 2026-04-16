"""Muon optimizer wrapper for Hydra-compatible instantiation in anemoi-training.

Muon (Momentum-Orthogonalized Update) requires parameter group splitting:
  - 2D+ weight matrices → Muon (momentum-orthogonalized updates)
  - Biases, norms, embeddings → auxiliary optimizer (AdamW or StableAdamW)

This wrapper accepts a flat `params` iterable (as Hydra passes it) and
splits internally based on parameter dimensionality.

Reference: Jordan et al., "Muon: An optimizer for hidden layers in neural
networks", 2024. https://kellerjordan.github.io/posts/muon/

Usage in YAML config:
    optimizer:
      _target_: anemoi.training.optimizers.MuonOptimizer
      muon_lr: 0.02
      aux_lr: 3e-4
      momentum: 0.95
      weight_decay: 0.01
"""

import logging
from typing import Iterator

import torch
from muon import SingleDeviceMuonWithAuxAdam

LOGGER = logging.getLogger(__name__)


class MuonOptimizer(SingleDeviceMuonWithAuxAdam):
    """Hydra-compatible Muon optimizer with automatic param group splitting.

    Parameters
    ----------
    params : iterable
        Model parameters (flat iterable, as passed by Hydra instantiate).
    lr : float
        Base learning rate (set by anemoi-training's LR scheduler). Applied
        to the auxiliary group; Muon group uses muon_lr.
    muon_lr : float
        Learning rate for Muon-optimized weight matrices. Default: 0.02.
    aux_lr : float or None
        Learning rate for auxiliary params (biases, norms). If None, uses lr.
    momentum : float
        Muon momentum coefficient. Default: 0.95.
    nesterov : bool
        Use Nesterov momentum in Muon. Default: True.
    ns_steps : int
        Newton-Schulz orthogonalization steps. Default: 5.
    weight_decay : float
        Weight decay for both groups. Default: 0.01.
    aux_betas : tuple
        Adam betas for auxiliary group. Default: (0.9, 0.95).
    """

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr: float = 3e-4,
        muon_lr: float = 0.02,
        aux_lr: float = None,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.01,
        aux_betas: tuple = (0.9, 0.95),
    ):
        # Split params: 2D+ weights → Muon, rest → auxiliary (Adam)
        muon_params = []
        aux_params = []

        for p in params:
            if p.ndim >= 2:
                muon_params.append(p)
            else:
                aux_params.append(p)

        actual_aux_lr = aux_lr if aux_lr is not None else lr

        LOGGER.info(
            "MuonOptimizer: %d Muon params (lr=%.4f), %d aux params (lr=%.6f)",
            len(muon_params), muon_lr, len(aux_params), actual_aux_lr,
        )

        # MuonWithAuxAdam requires exact key sets per group:
        #   Muon: {params, lr, momentum, weight_decay, use_muon}
        #   Aux:  {params, lr, betas, eps, weight_decay, use_muon}
        param_groups = [
            dict(
                params=muon_params,
                use_muon=True,
                lr=muon_lr,
                momentum=momentum,
                weight_decay=weight_decay,
            ),
            dict(
                params=aux_params,
                use_muon=False,
                lr=actual_aux_lr,
                betas=aux_betas,
                eps=1e-10,
                weight_decay=weight_decay,
            ),
        ]

        super().__init__(param_groups)
