# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor as pl_LearningRateMonitor
from pytorch_lightning.callbacks import WeightAveraging as pl_WeightAveraging
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging as pl_StochasticWeightAveraging
from torch.optim.swa_utils import get_ema_avg_fn

LOGGER = logging.getLogger(__name__)


class LearningRateMonitor(pl_LearningRateMonitor):
    """Provide LearningRateMonitor from pytorch_lightning as a callback."""

    def __init__(
        self,
        config: DictConfig,
        logging_interval: str = "step",
        log_momentum: bool = False,
    ) -> None:
        super().__init__(logging_interval=logging_interval, log_momentum=log_momentum)
        self.config = config


class StochasticWeightAveraging(pl_StochasticWeightAveraging):
    """Provide StochasticWeightAveraging from pytorch_lightning as a callback."""

    def __init__(
        self,
        config: DictConfig,
        swa_lrs: int | None = None,
        swa_epoch_start: int | None = None,
        annealing_epochs: int | None = None,
        annealing_strategy: str | None = None,
        device: str | None = None,
        **kwargs,
    ) -> None:
        """Stochastic Weight Averaging Callback.

        Parameters
        ----------
        config : OmegaConf
            Full configuration object
        swa_lrs : int, optional
            Stochastic Weight Averaging Learning Rate, by default None
        swa_epoch_start : int, optional
            Epoch start, by default 0.75 * config.training.max_epochs
        annealing_epochs : int, optional
            Annealing Epoch, by default 0.25 * config.training.max_epochs
        annealing_strategy : str, optional
            Annealing Strategy, by default 'cos'
        device : str, optional
            Device to use, by default None
        """
        kwargs["swa_lrs"] = swa_lrs or config.training.swa.lr
        kwargs["swa_epoch_start"] = swa_epoch_start or min(
            int(0.75 * config.training.max_epochs),
            config.training.max_epochs - 1,
        )
        kwargs["annealing_epochs"] = annealing_epochs or max(int(0.25 * config.training.max_epochs), 1)
        kwargs["annealing_strategy"] = annealing_strategy or "cos"
        kwargs["device"] = device

        super().__init__(**kwargs)
        self.config = config


class ExponentialWeightAveraging(pl_WeightAveraging):
    """Provide Exponential Weight Averaging from pytorch_lightning as a callback."""

    def __init__(
        self,
        config: DictConfig,
        avg_fn: str = "ema",
        ema_decay: float = 0.999,
        epoch_start: int | None = None,
        **kwargs,
    ) -> None:
        """Exponential Weight Averaging Callback.

        Parameters
        ----------
        config : OmegaConf
            Full configuration object
        avg_fn : str, optional
            Averaging function to use. Default "ema" for exponential moving average.
        ema_decay : float, optional
            Decay factor for exponential moving average. Default 0.999.
            Higher values (closer to 1.0) give more weight to recent weights.
        epoch_start : int, optional
            Epoch to start averaging. Default 0.75 * config.training.max_epochs
        """
        # Lightning's WeightAveraging forwards **kwargs straight to
        # torch.optim.swa_utils.AveragedModel, which expects ``avg_fn`` as a
        # callable (not a string) and does NOT accept ``ema_decay``. Translate
        # the convenience pair ("ema", decay) into the proper callable here.
        decay = ema_decay if ema_decay is not None else config.training.ewa.ema_decay
        if avg_fn == "ema" or avg_fn is None:
            kwargs["avg_fn"] = get_ema_avg_fn(decay=decay)
        else:
            kwargs["avg_fn"] = avg_fn

        # Resolve when averaging should start. Lightning's WeightAveraging
        # does NOT accept ``epoch_start`` as a constructor kwarg (kwargs get
        # forwarded to AveragedModel which rejects it). Instead, the timing
        # is encoded by overriding ``should_update`` -- we store the
        # resolved epoch_start on the instance and gate updates below.
        if epoch_start is None:
            if config.training.max_epochs is not None:
                resolved_epoch_start = min(
                    int(0.75 * config.training.max_epochs),
                    max(0, config.training.max_epochs - 1),
                )
            elif config.training.max_steps is not None:
                # Without max_epochs we cannot infer epoch length reliably
                # (true steps/epoch depends on dataset size + batch + workers).
                # Default to averaging from epoch 0 -- safest for fine-tunes
                # where we want to average across the whole short FT anyway.
                resolved_epoch_start = 0
                LOGGER.info(
                    "EWA: max_epochs not set; defaulting epoch_start=0 "
                    "(averaging from training start over max_steps=%d).",
                    config.training.max_steps,
                )
            else:
                resolved_epoch_start = 0
        else:
            resolved_epoch_start = epoch_start

        self._ewa_epoch_start = int(resolved_epoch_start)
        # If we're starting at epoch 0, enable averaging from the very first
        # optimizer step. Otherwise we wait for the first end-of-epoch hook
        # to set the flag.
        self._ewa_started = (self._ewa_epoch_start == 0)

        super().__init__(**kwargs)
        self.config = config

    def should_update(self, step_idx=None, epoch_idx=None) -> bool:
        """Gate on the resolved epoch_start while preserving Lightning's
        per-step update semantics. Returns True once we've reached the
        configured epoch_start, then on every optimizer step (the default
        Lightning behavior).
        """
        # Per-step calls (epoch_idx is None) are gated by the trainer's
        # current_epoch which we don't have access to here without a
        # trainer reference. Defer to a stored flag set on epoch transitions.
        if step_idx is not None:
            return getattr(self, "_ewa_started", False)
        if epoch_idx is not None and epoch_idx >= self._ewa_epoch_start:
            self._ewa_started = True
            return True
        return False
