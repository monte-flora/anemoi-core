# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from .combined import CombinedLoss
from .graphcast_gaussian_nll import GraphCastGaussianNLLLoss
from .graphcast_huber import GraphCastHuberLoss
from .graphcast_logcosh import GraphCastLogCoshLoss
from .graphcast_mahalanobis import GraphCastMahalanobisLoss
from .graphcast_crps import GraphCastCRPSLoss
from .graphcast_mae import GraphCastMAELoss
from .graphcast_mse import GraphCastMSELoss
from .graphcast_mse import WeightedGraphCastMSELoss
from .graphcast_robust import GraphCastClippedMSELoss
from .graphcast_robust import GraphCastPseudoHuberLoss
from .huber import HuberLoss
from .kcrps import AlmostFairKernelCRPS
from .kcrps import KernelCRPS
from .logcosh import LogCoshLoss
from .loss import get_loss_function
from .mae import MAELoss
from .mse import MSELoss
from .multiscale import MultiscaleLossWrapper
from .normalized_rmse import RMSELossNormalized
from .rmse import RMSELoss
from .spectral import FourierCorrelationLoss
from .spectral import LogSpectralDistance
from .spectral import SpectralL2Loss
from .weighted_mse import WeightedMSELoss

__all__ = [
    "AlmostFairKernelCRPS",
    "CombinedLoss",
    "FourierCorrelationLoss",
    "GraphCastCRPSLoss",
    "GraphCastClippedMSELoss",
    "GraphCastGaussianNLLLoss",
    "GraphCastHuberLoss",
    "GraphCastLogCoshLoss",
    "GraphCastMahalanobisLoss",
    "GraphCastMAELoss",
    "GraphCastMSELoss",
    "GraphCastPseudoHuberLoss",
    "HuberLoss",
    "KernelCRPS",
    "LogCoshLoss",
    "LogSpectralDistance",
    "MAELoss",
    "MSELoss",
    "MultiscaleLossWrapper",
    "RMSELoss",
    "RMSELossNormalized",
    "SpectralL2Loss",
    "WeightedMSELoss",
    "get_loss_function",
]
