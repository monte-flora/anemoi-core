"""Single source of truth for resolving config-level LR into optimizer LR.

The anemoi LR config historically exposed only a "per-rank rate" knob that
was silently multiplied by ``num_nodes × num_gpus_per_node / num_gpus_per_model``
to obtain the optimizer's peak LR — while ``lr.min`` was left at its literal
config value. The asymmetry made the cosine sweep span hardware-dependent
and was a frequent source of bugs when warm-starting from a previous run
on a different rank count.

This helper centralises the resolution under an explicit ``lr.semantics``
field on the config (see :class:`anemoi.training.schemas.training.LR`).
"""

from __future__ import annotations

import logging
from typing import Tuple

LOGGER = logging.getLogger(__name__)


def _hw_multiplier(config) -> float:
    """How many ``num_gpus_per_model``-sized model groups exist in the job."""
    return (
        config.system.hardware.num_nodes
        * config.system.hardware.num_gpus_per_node
        / config.system.hardware.num_gpus_per_model
    )


def resolve_lr(config) -> Tuple[float, float, str]:
    """Return ``(peak_lr, min_lr, semantics)`` the optimizer will actually use.

    Parameters
    ----------
    config
        Full training config (pydantic-validated). Required fields:
        ``training.lr.rate``, ``training.lr.min``, optional
        ``training.lr.semantics``, plus the standard hardware fields.

    Returns
    -------
    peak_lr : float
        The peak (post-warmup) learning rate the optimizer will see.
    min_lr : float
        The cosine-decay floor the optimizer will reach.
    semantics : str
        One of ``"per_rank_legacy"``, ``"per_rank"``, ``"global"`` — passed
        through for logging.

    Notes
    -----
    Three semantics are supported:

    - ``per_rank_legacy``  *(default for back-compat)*: ``rate × mult``,
      ``min`` literal. Reproduces historical anemoi behaviour exactly —
      including the asymmetry that makes the cosine sweep span depend on
      ``num_gpus``.
    - ``per_rank``        : both ``rate × mult`` and ``min × mult``. The
      asymmetry is fixed; the cosine span is hardware-independent.
    - ``global``          : both ``rate`` and ``min`` are taken literally;
      no hardware scaling. The cleanest semantics, recommended for all
      new configs.
    """
    # The schema field has a default, but if a downstream config dict was
    # built without the schema we fall back to "per_rank_legacy".
    semantics = getattr(config.training.lr, "semantics", None) or "per_rank_legacy"
    rate = float(config.training.lr.rate)
    min_lr = float(config.training.lr.min)
    mult = _hw_multiplier(config)

    if semantics == "global":
        return rate, min_lr, semantics
    if semantics == "per_rank":
        return rate * mult, min_lr * mult, semantics
    if semantics == "per_rank_legacy":
        return rate * mult, min_lr, semantics
    msg = (
        f"Unknown lr.semantics={semantics!r}; expected one of "
        "'per_rank_legacy', 'per_rank', 'global'."
    )
    raise ValueError(msg)


def log_lr_banner(config, *, source: str) -> None:
    """Log a startup banner that makes the actual optimizer LR obvious.

    ``source`` is a short label included in the log line so the user can
    tell which task / call-site produced it (e.g. ``"BaseForecaster"``).
    """
    peak, floor, semantics = resolve_lr(config)
    mult = _hw_multiplier(config)
    LOGGER.info(
        "[lr_resolution:%s] semantics=%s  hw_mult=%g  "
        "config(rate=%.3e, min=%.3e)  →  optimizer(peak=%.3e, floor=%.3e)",
        source, semantics, mult,
        config.training.lr.rate, config.training.lr.min,
        peak, floor,
    )
