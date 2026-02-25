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
        kwargs["avg_fn"] = avg_fn
        kwargs["ema_decay"] = ema_decay or config.training.ewa.ema_decay

        # Handle epoch_start - use max_epochs if available, otherwise estimate from max_steps
        if epoch_start is None:
            if config.training.max_epochs is not None:
                # Use max_epochs directly
                kwargs["epoch_start"] = min(
                    int(0.75 * config.training.max_epochs),
                    config.training.max_epochs - 1,
                )
            elif config.training.max_steps is not None:
                # Estimate max_epochs from max_steps
                # Need to estimate steps per epoch from dataset size and batch configuration
                # Use check_val_every_n_epoch to help estimate
                check_val_every_n_epoch = getattr(config.diagnostics, 'check_val_every_n_epoch', 1)

                # Approximate steps per epoch by looking at validation frequency
                # Typical setup: validation runs every N epochs at specific step intervals
                # Conservative estimate: ~1400-1500 steps per epoch for large datasets
                # More accurate: try to infer from checkpoint frequency

                # Look for checkpoint frequency as a proxy for epoch length
                if hasattr(config.diagnostics, 'checkpoint') and hasattr(config.diagnostics.checkpoint, 'every_n_train_steps'):
                    checkpoint_freq = config.diagnostics.checkpoint.every_n_train_steps.save_frequency
                    # If checkpoints are every 20000 steps and every ~14 epochs, that's ~1428 steps/epoch
                    # Use a reasonable estimate based on this
                    estimated_steps_per_epoch = 1400  # Conservative default
                else:
                    estimated_steps_per_epoch = 1400  # Default estimate

                # Calculate estimated max_epochs
                estimated_max_epochs = config.training.max_steps / estimated_steps_per_epoch

                # Start EWA at 75% of training
                kwargs["epoch_start"] = max(1, int(0.75 * estimated_max_epochs))

                LOGGER.info(
                    "EWA: Estimated %d epochs from max_steps=%d (%.1f steps/epoch). Starting EWA at epoch %d",
                    int(estimated_max_epochs),
                    config.training.max_steps,
                    estimated_steps_per_epoch,
                    kwargs["epoch_start"]
                )
            else:
                # Neither max_steps nor max_epochs set - use a safe default
                kwargs["epoch_start"] = 1000  # Start very late
        else:
            kwargs["epoch_start"] = epoch_start

        super().__init__(**kwargs)
        self.config = config
