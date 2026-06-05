"""LatentResidualNormalizer — tendency-normalize latent residuals.

Mirrors v17's ``ResidualNormalizer`` (this package's :mod:`residual_normalizer`)
but at the LATENT grid scale used by v30 Atlas-style architectures. Reads
``statistics_tendencies_<freq>_latent_stdev`` from a training zarr and
exposes ``transform`` (mean-std residual → tendency-norm residual) and
``inverse_transform`` (tendency-norm → mean-std).

Why this exists (2026-05-24 diagnostic):
    v30b's predictive model emits residuals in mean-std normalized space
    (the dataloader's natural representation). At 15-minute timestep this
    gives per-channel target magnitudes from O(0.05) (pressure) to O(1)
    (w). The predictive collapses to per-variable shortcuts:
        - pressure_*: predicts ≈ 0 (persistence shortcut)
        - w_*:        predicts ~5× over-amplified residuals
    in BOTH MSE and CRPS recipes. The model's last-layer linear cannot
    naturally produce wildly-different output magnitudes per channel, and
    a loss scaler ((σ_state/σ_tend)²) does not fix this because the
    gradient at the model OUTPUT remains small for low-magnitude targets.

    v17 solved this by training against TENDENCY-NORMALIZED residual
    targets (every channel O(1)). This module ports that trick to v30's
    latent grid.

Usage (training):
    norm = LatentResidualNormalizer.from_zarr(
        latent_stats_path="/lustre/.../graf-conus-patches-train.zarr",
        prog_channel_names=["t2m", "comp_refl", "u_0", ...],  # prog channel order
    )
    r_true_tendnorm = norm.transform(r_true_meanstd)
    loss = MSE(model_output_in_tendnorm_space, r_true_tendnorm)
    # For AR feedback in mean-std state space:
    r_meanstd = norm.inverse_transform(model_output_in_tendnorm_space)
    z_next = z_t + r_meanstd

Usage (inference, composed model):
    r_lat_tendnorm = predictive(z_t, z_prev)
    r_lat_meanstd  = norm.inverse_transform(r_lat_tendnorm)
    delta_phys     = decoder(r_lat_meanstd, x_t)
    x_next         = x_t[:, :n_prog] + delta_phys

Buffers are persistent so the loaded σ travels with checkpoints.
"""

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)


def _load_latent_tendency_stdev_in_prog_order(
    latent_stats_path: str,
    prog_channel_names: Sequence[str],
    latent_stats_key: Optional[str] = None,
    min_stdev: float = 1e-7,
) -> torch.Tensor:
    """Load σ_lat_tend / σ_state (mean-std-space tendency stdev) per prog channel.

    The zarr stores σ_lat_tend in PHYSICAL units (e.g. Pa for pressure, K for
    t2m). Latent residuals at training/inference live in MEAN-STD normalized
    space (because z_lat = bilinear(x_norm)). Since bilinear is linear and
    x_norm = (x_phys - μ)/σ_state, the divisor we actually need for mean-std
    residuals is ``σ_lat_tend_phys / σ_state_phys``. Computing the ratio here.

    Parameters
    ----------
    latent_stats_path : str
        Path to the training zarr.
    prog_channel_names : sequence of str
        Names of the predictive model's prognostic channels in OUTPUT order.
    latent_stats_key : str, optional
        Override σ_lat_tend key; defaults to ``statistics_tendencies_<freq>_latent_stdev``.
    min_stdev : float
        Clip σ values below this to avoid divide-by-zero in transform.
        Defaults to ``1e-7`` matching v17's ResidualNormalizer.

    Returns
    -------
    torch.Tensor of shape (len(prog_channel_names),), dtype float32.
    """
    import zarr

    z = zarr.open(latent_stats_path, mode="r")
    freqstr = z.attrs.get("frequency")
    if not freqstr:
        error = (
            f"LatentResidualNormalizer: zarr at {latent_stats_path!r} has no "
            "'frequency' attribute; cannot resolve latent stats key."
        )
        raise RuntimeError(error)
    key = latent_stats_key or f"statistics_tendencies_{freqstr}_latent_stdev"
    if key not in z:
        error = (
            f"LatentResidualNormalizer: {key!r} missing from "
            f"{latent_stats_path!r}. Run grafai/datasets/"
            "compute_latent_tendency_stats.py first."
        )
        raise RuntimeError(error)
    if "stdev" not in z:
        error = (
            f"LatentResidualNormalizer: state stdev array 'stdev' missing from "
            f"{latent_stats_path!r}; cannot compute mean-std-space ratio."
        )
        raise RuntimeError(error)
    stdev_lat_phys = np.asarray(z[key][:])      # (n_raw_zarr_vars,) physical units
    stdev_state_phys = np.asarray(z["stdev"][:])  # (n_raw_zarr_vars,) physical state stdev

    zarr_names = list(z.attrs.get("variables", []))
    if not zarr_names or len(zarr_names) != len(stdev_lat_phys):
        error = (
            f"LatentResidualNormalizer: zarr's `variables` attr "
            f"({len(zarr_names)}) does not match the latent-stdev array length "
            f"({len(stdev_lat_phys)}). Cannot re-index."
        )
        raise RuntimeError(error)
    zarr_name_to_idx = {n: i for i, n in enumerate(zarr_names)}

    stdev_prog = np.ones(len(prog_channel_names), dtype=np.float32)
    n_filled = 0
    n_clipped = 0
    zero_or_neg: list[str] = []
    not_in_zarr: list[str] = []
    for i, name in enumerate(prog_channel_names):
        if name not in zarr_name_to_idx:
            not_in_zarr.append(name)
            continue
        zarr_idx = zarr_name_to_idx[name]
        lat_phys = float(stdev_lat_phys[zarr_idx])
        state_phys = float(stdev_state_phys[zarr_idx])
        if state_phys < min_stdev:
            zero_or_neg.append(f"{name}[σ_state={state_phys:.2e}]")
            state_phys = min_stdev
        if lat_phys <= 0.0:
            zero_or_neg.append(f"{name}[σ_lat={lat_phys:.2e}]")
            lat_phys = min_stdev
        val = lat_phys / state_phys   # mean-std-space σ_lat_tend
        if val < min_stdev:
            n_clipped += 1
            val = min_stdev
        stdev_prog[i] = val
        n_filled += 1

    LOGGER.info(
        "LatentResidualNormalizer: computed mean-std-space σ_lat_tend (= σ_lat_phys / σ_state) "
        "for %d/%d prog channels from %s. Range [%.3e, %.3e]. min_stdev clip=%.0e (%d below threshold).",
        n_filled, len(prog_channel_names), latent_stats_path,
        float(stdev_prog.min()), float(stdev_prog.max()), min_stdev, n_clipped,
    )
    if zero_or_neg:
        LOGGER.warning(
            "LatentResidualNormalizer: %d prog channels have σ_lat_tend ≤ 0 "
            "(clipped to %.0e). Likely cause: forcing-like variable misclassified "
            "as prognostic. Offenders: %s",
            len(zero_or_neg), min_stdev, zero_or_neg,
        )
    if not_in_zarr:
        LOGGER.warning(
            "LatentResidualNormalizer: %d prog channels not in zarr stats "
            "(left σ=1, i.e. no-op normalization for these channels). "
            "Offenders: %s",
            len(not_in_zarr), not_in_zarr,
        )

    return torch.from_numpy(stdev_prog)


class LatentResidualNormalizer(nn.Module):
    """Tendency-normalize latent-space residuals; v17 ResidualNormalizer at latent scale.

    Stores σ_lat_tend as a (1, n_prog, 1, 1) buffer for broadcasting against
    (B, n_prog, h, w) tensors. Computations cast through float32 for
    numerical stability (matches v17 ResidualNormalizer convention).

    Parameters
    ----------
    stdev : torch.Tensor
        1-D tensor of length n_prog, σ_lat_tend in prog-channel order.
    """

    def __init__(self, stdev: torch.Tensor) -> None:
        super().__init__()
        if stdev.dim() != 1:
            error = f"LatentResidualNormalizer: stdev must be 1-D, got shape {tuple(stdev.shape)}"
            raise ValueError(error)
        # (1, n_prog, 1, 1) for broadcast against (B, n_prog, h, w)
        self.register_buffer("_std_tend", stdev.view(1, -1, 1, 1).float(), persistent=True)

    @classmethod
    def from_zarr(
        cls,
        latent_stats_path: str,
        prog_channel_names: Sequence[str],
        latent_stats_key: Optional[str] = None,
        min_stdev: float = 1e-7,
    ) -> "LatentResidualNormalizer":
        stdev = _load_latent_tendency_stdev_in_prog_order(
            latent_stats_path=latent_stats_path,
            prog_channel_names=prog_channel_names,
            latent_stats_key=latent_stats_key,
            min_stdev=min_stdev,
        )
        return cls(stdev)

    def transform(self, r_meanstd: torch.Tensor) -> torch.Tensor:
        """Mean-std-normalized latent residual → tendency-normalized residual.

        Parameters
        ----------
        r_meanstd : torch.Tensor, shape (B, n_prog, h, w)
            Latent residual in mean-std normalized space (the dataloader's
            natural output after ``bilinear_downsample(x_norm[t+1] - x_norm[t])``).

        Returns
        -------
        torch.Tensor of the same shape, in tendency-normalized space (every
        channel ~O(1) variance).
        """
        original_dtype = r_meanstd.dtype
        r_f32 = r_meanstd.float()
        out = r_f32 / self._std_tend.to(r_f32.device)
        return out.to(original_dtype)

    def inverse_transform(self, r_tendnorm: torch.Tensor) -> torch.Tensor:
        """Tendency-normalized residual → mean-std-normalized residual.

        Parameters
        ----------
        r_tendnorm : torch.Tensor, shape (B, n_prog, h, w)
            The predictive model's natural output after training under
            ``transform``.

        Returns
        -------
        torch.Tensor of the same shape, in mean-std normalized space (suitable
        for adding directly to ``z_t`` for AR feedback OR for passing to a
        decoder that was trained on mean-std residuals).
        """
        original_dtype = r_tendnorm.dtype
        r_f32 = r_tendnorm.float()
        out = r_f32 * self._std_tend.to(r_f32.device)
        return out.to(original_dtype)
