"""Spatial-resample utilities for the Atlas-style latent-rollout architecture.

Two distinct uses:

  1. **Data downsample / upsample** — :func:`bilinear_downsample` and
     :func:`bilinear_upsample` wrap ``F.interpolate`` with ``mode='bilinear'``.
     Used as the (parameter-free) encoder / inverse-decoder in the Atlas
     pipeline (paper §2.2 literal).
  2. **Positional-embedding resize** — :func:`resize_pos_embed` uses
     BICUBIC interpolation on a pos-embed buffer when the input field_shape
     (and hence token grid) differs between training and inference, the
     ViT / DEiT transfer-learning pattern. Sine-cosine pos embeds are
     analytic so the bicubic resize is essentially lossless for them; the
     same code path also handles learnable pos embeds (future configs).

Bilinear-downsample original notes (Atlas §2.2):

Pure PyTorch, no learned parameters, runs on whatever device the input is on.
The "encoder" of the Atlas pattern (Kossaifi et al., NVIDIA, Jan 2026) is just
a bilinear interpolation that downsamples a high-resolution atmospheric field
to a low-resolution "latent" grid. The probabilistic dynamics model then
operates on the latent, and a separate decoder upsamples back to full-res.

Why bilinear instead of a learned VAE encoder (Atlas §2.2):

  * Learned VAEs mix channels and produce latent spectra with a long, flat
    high-frequency tail (a known spectral-bias artifact when aggressively
    compressing multi-channel fields).
  * Learned VAEs can map temporally adjacent fields far apart in latent
    space, hurting autoregressive rollout stability.
  * Bilinear preserves physical structure exactly at scales above the
    Nyquist of the latent grid; it just removes the small-scale physics
    that are deterministically unpredictable at the model's timestep
    anyway (per the predictability cutoff analysis).

For GRAF AI's 15-min timestep we use a 4x downsample (4 km native -> 16 km
latent), which sits just above the predictability frontier of vertical
velocity w (~50 km cutoff).

The companion decoder is responsible for recovering small-scale detail at
inference time, conditioned on the full-resolution initial state. See
``anemoi.models.models.decoder_dit_wrapper``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def bilinear_downsample(
    x: torch.Tensor,
    scale_factor: float | int | None = None,
    *,
    target_shape: tuple[int, int] | None = None,
    align_corners: bool = False,
) -> torch.Tensor:
    """Bilinear downsample a 2-D spatial field on its device.

    Wraps ``torch.nn.functional.interpolate`` with ``mode='bilinear'``. No
    learned parameters; differentiable; runs on whatever device ``x`` lives
    on (typically GPU during training and inference).

    Provide exactly one of ``scale_factor`` or ``target_shape``. Prefer
    ``target_shape`` when the output grid size is fixed by downstream
    architecture (e.g. a DiT token grid). ``scale_factor=0.25`` on a
    250-wide input floors to 62, not 63; use ``target_shape=(63, 63)``
    to get exact ceil-style 4x compression matching our tokenized grid.

    Parameters
    ----------
    x : torch.Tensor
        Spatial field of shape ``(..., C, H, W)``. Leading dimensions are
        preserved. Float dtype (fp32, fp16, bf16 all supported).
    scale_factor : float or int, optional
        Multiplicative factor for output size. ``0.25`` for an exact 4x
        downsample (floored).
    target_shape : tuple[int, int], optional
        Explicit output ``(H', W')``. Use this when the latent grid size
        is constrained by architecture.
    align_corners : bool, optional
        Passed through to ``F.interpolate``. Default False, matching the
        Atlas paper's convention and PyTorch's recommended default for
        bilinear interpolation of natural fields.

    Returns
    -------
    torch.Tensor
        Downsampled field of shape ``(..., C, H', W')``.
    """
    if (scale_factor is None) == (target_shape is None):
        raise ValueError(
            "bilinear_downsample: provide exactly one of scale_factor, target_shape"
        )
    if x.ndim < 3:
        raise ValueError(
            f"bilinear_downsample expects at least 3 dims (C, H, W); got {x.shape!r}"
        )
    # F.interpolate operates on (N, C, H, W). Fold any extra leading dims
    # into the batch dim so we can also downsample (B, T, C, H, W) etc.
    orig_shape = x.shape
    if x.ndim > 4:
        leading = orig_shape[:-3]
        x = x.reshape(-1, *orig_shape[-3:])
    elif x.ndim == 3:
        x = x.unsqueeze(0)
        leading = ()
    else:
        leading = orig_shape[:-3]
    if target_shape is not None:
        y = F.interpolate(
            x, size=target_shape, mode="bilinear", align_corners=align_corners,
        )
    else:
        y = F.interpolate(
            x, scale_factor=scale_factor, mode="bilinear", align_corners=align_corners,
        )
    if leading:
        y = y.reshape(*leading, *y.shape[-3:])
    elif x.shape[0] == 1 and len(orig_shape) == 3:
        y = y.squeeze(0)
    return y


def bilinear_upsample(
    x: torch.Tensor,
    target_shape: tuple[int, int] | None = None,
    scale_factor: float | int | None = None,
    *,
    align_corners: bool = False,
) -> torch.Tensor:
    """Bilinear upsample to a target spatial shape (or by a scale factor).

    Companion of :func:`bilinear_downsample`. Used by the identity-decoder
    sanity-check path and by the latent residual upsampling step inside
    the real decoder model.

    Provide exactly one of ``target_shape`` or ``scale_factor``.
    """
    if (target_shape is None) == (scale_factor is None):
        raise ValueError(
            "bilinear_upsample: provide exactly one of target_shape, scale_factor"
        )
    orig_shape = x.shape
    if x.ndim > 4:
        leading = orig_shape[:-3]
        x = x.reshape(-1, *orig_shape[-3:])
    elif x.ndim == 3:
        x = x.unsqueeze(0)
        leading = ()
    else:
        leading = orig_shape[:-3]

    if target_shape is not None:
        y = F.interpolate(
            x, size=target_shape, mode="bilinear", align_corners=align_corners,
        )
    else:
        y = F.interpolate(
            x, scale_factor=scale_factor, mode="bilinear", align_corners=align_corners,
        )

    if leading:
        y = y.reshape(*leading, *y.shape[-3:])
    elif x.shape[0] == 1 and len(orig_shape) == 3:
        y = y.squeeze(0)
    return y


def resize_pos_embed(
    pos_embed: torch.Tensor,
    old_shape: tuple[int, int],
    new_shape: tuple[int, int],
    *,
    align_corners: bool = False,
) -> torch.Tensor:
    """Bicubic-resize a 2-D positional-embedding buffer to a new token grid.

    ViT / DEiT / Atlas convention: at inference time when the input
    field_shape differs from training, the token grid changes, but the
    pre-trained pos_embed buffer is sized for the training grid. The
    canonical fix is bicubic interpolation of the pos_embed to the new
    token grid.

    For sine-cosine pos embeds (which we use today) the result is
    essentially lossless — the formula is analytic so bicubic just
    re-evaluates it on a finer/coarser grid with negligible error. For
    learnable pos embeds (future configs) the same path applies and is
    the standard transfer-learning recipe.

    Parameters
    ----------
    pos_embed : torch.Tensor
        Either a ``(1, h_old*w_old, D)`` 3-D tensor (the registered
        buffer convention in our DiT wrappers) or a ``(1, D, h_old, w_old)``
        4-D spatial layout. The 3-D form is unflattened internally before
        interpolation.
    old_shape : tuple[int, int]
        ``(h_old, w_old)`` — the spatial layout that ``pos_embed`` was
        sized for.
    new_shape : tuple[int, int]
        ``(h_new, w_new)`` — the target token grid.
    align_corners : bool, optional
        Passed through to ``F.interpolate``. Default False (matches the
        rest of this module).

    Returns
    -------
    torch.Tensor
        Same layout as the input but with ``h_new * w_new`` tokens
        (3-D form) or ``(D, h_new, w_new)`` (4-D form). dtype + device
        are preserved.
    """
    h_old, w_old = int(old_shape[0]), int(old_shape[1])
    h_new, w_new = int(new_shape[0]), int(new_shape[1])
    if (h_old, w_old) == (h_new, w_new):
        return pos_embed

    if pos_embed.dim() == 3:
        B1, N, D = pos_embed.shape
        if N != h_old * w_old:
            error = (
                f"resize_pos_embed: pos_embed token count {N} != h*w "
                f"= {h_old * w_old}"
            )
            raise ValueError(error)
        x = pos_embed.reshape(B1, h_old, w_old, D).permute(0, 3, 1, 2).contiguous()
        out_dtype = x.dtype
        # F.interpolate's bicubic kernel doesn't support bf16/fp16; promote.
        x_f = x.float() if x.dtype not in (torch.float32, torch.float64) else x
        y = F.interpolate(
            x_f, size=(h_new, w_new), mode="bicubic", align_corners=align_corners,
        ).to(out_dtype)
        return y.permute(0, 2, 3, 1).reshape(B1, h_new * w_new, D).contiguous()

    if pos_embed.dim() == 4:
        out_dtype = pos_embed.dtype
        x_f = pos_embed.float() if pos_embed.dtype not in (torch.float32, torch.float64) else pos_embed
        y = F.interpolate(
            x_f, size=(h_new, w_new), mode="bicubic", align_corners=align_corners,
        ).to(out_dtype)
        return y

    error = f"resize_pos_embed: expected pos_embed with ndim 3 or 4, got {pos_embed.dim()}"
    raise ValueError(error)
