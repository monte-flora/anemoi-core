# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import math
from typing import Optional

from hydra.errors import InstantiationException
from hydra.utils import instantiate
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class TruncNormalLinear(nn.Linear):
    """Linear layer with Haiku-style truncated normal weight initialization.

    Matches the default initialization in JAX/Haiku's hk.Linear:
        w_init = TruncatedNormal(stddev=1/sqrt(fan_in))
        b_init = zeros

    This differs from PyTorch's default Kaiming uniform initialization.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with truncated normal, biases with zeros."""
        # Haiku default: TruncatedNormal(stddev=1/sqrt(fan_in)), truncated at ±2σ
        stddev = 1.0 / math.sqrt(self.in_features)
        nn.init.trunc_normal_(self.weight, mean=0.0, std=stddev, a=-2 * stddev, b=2 * stddev)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class XavierUniformLinear(nn.Linear):
    """Linear layer with Xavier uniform weight initialization.

    Balanced fan_in/fan_out scaling: bounds ~sqrt(6/(fan_in+fan_out)).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.weight, gain=1.0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class XavierNormalLinear(nn.Linear):
    """Linear layer with Xavier normal weight initialization.

    Gaussian variant: stddev = sqrt(2/(fan_in+fan_out)).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_normal_(self.weight, gain=1.0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class KaimingNormalLinear(nn.Linear):
    """Linear layer with Kaiming (He) normal weight initialization.

    Fan_out mode preserves backward-pass variance; leaky_relu a=0.01
    is used as a proxy for SiLU.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.weight, a=0.01, mode="fan_out", nonlinearity="leaky_relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class OrthogonalLinear(nn.Linear):
    """Linear layer with orthogonal weight initialization.

    All singular values = 1.0; maximal rank by construction.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.orthogonal_(self.weight, gain=1.0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class ScaledTruncNormalLinear(nn.Linear):
    """Linear layer with scaled truncated normal weight initialization.

    Same as TruncNormalLinear but with gain=2.0 (stddev = 2/sqrt(fan_in)).
    Doubles stddev to spread singular values wider.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self._init_weights()

    def _init_weights(self) -> None:
        stddev = 2.0 / math.sqrt(self.in_features)
        nn.init.trunc_normal_(self.weight, mean=0.0, std=stddev, a=-2 * stddev, b=2 * stddev)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class CheckpointWrapper(nn.Module):
    """Wrapper for checkpointing a module."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return checkpoint(self.module, *args, **kwargs, use_reentrant=False)


def load_layer_kernels(kernel_config: Optional[DotDict] = None, instance: bool = True) -> DotDict["str" : nn.Module]:
    """Load layer kernels from the config.

    This function tries to load the layer kernels from the config. If the layer kernel is not supplied, it will fall back to the torch.nn implementation.

    Parameters
    ----------
    kernel_config : DotDict
        Kernel configuration, e.g. {"Linear": {"_target_": "torch.nn.Linear"}}
    instance : bool
        If True, instantiate the kernels. If False, return the config.
        This is useful for testing purposes.
        Defaults to True.

    Returns
    -------
    DotDict
        Container with layer factories.
    """
    # If self.layer_kernels entry is missing from the config, use torch.nn kernels
    default_kernels = {
        "Linear": {"_target_": "torch.nn.Linear"},
        "LayerNorm": {"_target_": "torch.nn.LayerNorm"},
        "Activation": {"_target_": "torch.nn.GELU"},
        "QueryNorm": {
            "_target_": "anemoi.models.layers.normalization.AutocastLayerNorm",
            "_partial_": True,
            "bias": False,
        },
        "KeyNorm": {
            "_target_": "anemoi.models.layers.normalization.AutocastLayerNorm",
            "_partial_": True,
            "bias": False,
        },
    }

    if kernel_config is None:
        kernel_config = DotDict()

    layer_kernels = DotDict()

    # Loop through all kernels in the layer_kernels config entry and try import them
    for name, kernel_entry in {**default_kernels, **kernel_config}.items():
        if instance:
            try:
                layer_kernels[name] = instantiate(kernel_entry, _partial_=True)
            except InstantiationException:
                LOGGER.info(
                    f"{kernel_entry['_target_']} not found! Check your config.model.layer_kernel. {name} entry. Maybe your desired kernel is not installed or the import string is incorrect?"
                )
                raise InstantiationException
            else:
                LOGGER.info(f"{name} kernel: {kernel_entry['_target_']}.")
        else:
            layer_kernels[name] = kernel_entry
    return layer_kernels
