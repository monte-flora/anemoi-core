"""Atlas-composed model: encoder + predictive + decoder, end-to-end.

Wraps the three pieces of an Atlas-style latent-rollout architecture into
a single ``nn.Module`` whose forward is one full prediction step in
PHYSICAL space:

    x_t, x_{t-1}, xi  ->  x_{t+1}

via the pipeline

    z_t   = B(x_t)              # bilinear downsample (no params)
    z_{t-1} = B(x_{t-1})        # bilinear downsample (no params)
    r_t   = predictive(z_t, z_{t-1}, xi)        # latent residual
    delta = decoder(r_t, x_t)                    # full-res residual
    x_{t+1} = x_t[:prog_idx] + delta            # state update

The two trainable submodels (``predictive`` and ``decoder``) are
independent (Atlas §3.1: trained separately). At inference, this module
runs them sequentially per AR step. At training time, EACH submodel is
trained against its OWN loss (decoder: L1 in full-res; predictive: CRPS
in latent), so the composed forward is invoked only at evaluation /
deployment time.

The encoder is a pure function (``bilinear_downsample``) — no learned
parameters live in this wrapper for it.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from anemoi.models.layers.bilinear_encoder import bilinear_downsample
from anemoi.models.models.latent_dit_wrapper import AnemoiLatentDiTModel
from anemoi.models.models.decoder_dit_wrapper import (
    AnemoiDecoderDiTModel,
    IdentityBilinearDecoder,
)

LOGGER = logging.getLogger(__name__)


class AnemoiAtlasModel(nn.Module):
    """End-to-end Atlas-style composed forecaster.

    Parameters
    ----------
    predictive : nn.Module
        Latent predictive model. Forward signature
        ``(z_curr, z_prev, noise) -> r_t``.
    decoder : nn.Module
        Decoder model. Forward signature ``(r_t, x_t) -> delta_t`` (full-
        resolution residual).
    full_res_shape : tuple[int, int]
        Spatial shape of the full-resolution grid.
    latent_shape : tuple[int, int]
        Spatial shape of the latent grid.
    prognostic_channels : int
        Number of prognostic-only channels in the input ``x_t``. The
        first ``prognostic_channels`` channels of ``x_t`` are the
        prognostic state used to compute the latent ``z_t`` and updated
        by the decoder's output residual. Forcings (HGT, land/sea, hour
        of day, etc.) live in channels at index ``prognostic_channels``
        and beyond.
    forcings_channels : int
        Number of forcing channels in ``x_t`` beyond the prognostic
        slice. When >0, forcings are bilinear-encoded to the latent grid
        in parallel with prognostics and passed to the predictive model
        as conditioning (geographic + temporal features that v17-era
        prognostic latents would otherwise lose).
    """

    def __init__(
        self,
        predictive: nn.Module | None = None,
        decoder: nn.Module | None = None,
        *,
        model_config=None,
        data_indices=None,
        statistics=None,
        graph_data=None,
        full_res_shape: tuple[int, int] | None = None,
        latent_shape: tuple[int, int] | None = None,
        prognostic_channels: int | None = None,
        forcings_channels: int = 0,
        **_extra_kwargs,
    ) -> None:
        super().__init__()

        # Anemoi-standard path: pull config from model_config.model.atlas.
        # The composed model is for inference; predictive + decoder are
        # loaded externally by the Predictor (see grafai/runners/predictor.py).
        # The constructor here only records shape metadata; submodels are
        # attached by the predictor's _compose_atlas_model() step.
        if model_config is not None:
            from anemoi.utils.config import DotDict
            cfg = DotDict(model_config).model.model.atlas
            full_res_shape = tuple(cfg.full_res_shape)
            latent_shape = tuple(cfg.latent_shape)
            prognostic_channels = int(cfg.prognostic_channels)
            forcings_channels = int(getattr(cfg, "forcings_channels", forcings_channels))

        if any(v is None for v in (full_res_shape, latent_shape, prognostic_channels)):
            error = (
                "AnemoiAtlasModel: must provide either model_config or all "
                "of full_res_shape / latent_shape / prognostic_channels."
            )
            raise ValueError(error)

        self.predictive = predictive
        self.decoder = decoder
        self.full_res_shape = tuple(full_res_shape)
        self.latent_shape = tuple(latent_shape)
        self.prognostic_channels = prognostic_channels
        self.forcings_channels = forcings_channels

        # If the predictive was trained with a LatentResidualNormalizer
        # (v30c+: tendency-normalized latent residual targets), its output is
        # in tendency-norm space. The decoder is trained on mean-std r_lat,
        # so we denormalize before passing to the decoder.
        # The normalizer is attached to predictive as an attribute by the
        # training task; we surface it here for use in `forward`.
        self.latent_residual_normalizer = getattr(
            predictive, "latent_residual_normalizer", None,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Bilinear-downsample a full-resolution prognostic field to latent.

        Only the first ``prognostic_channels`` are encoded; forcings are
        handled separately by :meth:`encode_forcings`.
        """
        # If x has forcings appended, slice to prognostic only first.
        if x.shape[1] > self.prognostic_channels:
            x = x[:, : self.prognostic_channels]
        return bilinear_downsample(x, target_shape=self.latent_shape)

    def encode_forcings(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Bilinear-downsample the forcing slice of ``x`` to the latent grid.

        Returns None when ``forcings_channels == 0``. When >0, returns the
        ``(B, forcings_channels, h_lat, w_lat)`` tensor expected by the
        predictive model.
        """
        if self.forcings_channels <= 0:
            return None
        if x.shape[1] < self.prognostic_channels + self.forcings_channels:
            error = (
                f"x has {x.shape[1]} channels, expected "
                f"{self.prognostic_channels} prognostic + {self.forcings_channels} forcings."
            )
            raise ValueError(error)
        f = x[:, self.prognostic_channels : self.prognostic_channels + self.forcings_channels]
        return bilinear_downsample(f, target_shape=self.latent_shape)

    def forward(
        self,
        x_curr: torch.Tensor,
        x_prev: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        *,
        return_intermediates: bool = False,
    ) -> torch.Tensor | dict:
        """Single AR step in physical space.

        Parameters
        ----------
        x_curr : torch.Tensor
            Current full-resolution state + forcings, shape
            ``(B, C_total, H, W)`` where the first ``prognostic_channels``
            are prognostic.
        x_prev : torch.Tensor, optional
            Previous full-resolution state (history). If None, uses
            ``x_curr`` for both (cold-start; tendency = 0).
        noise : torch.Tensor, optional
            FGN noise vector ``(B, [E,] noise_dim)``. None → predictive
            samples internally.
        return_intermediates : bool
            If True, return a dict containing ``z_curr, z_prev, r_t,
            delta_t, x_next`` for diagnostics.

        Returns
        -------
        torch.Tensor
            Next full-resolution prognostic state ``x_{t+1}``, shape
            ``(B, prognostic_channels, H, W)`` (or
            ``(B, E, prognostic_channels, H, W)`` for ensemble noise).
        """
        # 1) Bilinear-encode prognostic channels to latent.
        z_curr = self.encode(x_curr)
        z_prev = self.encode(x_prev) if x_prev is not None else z_curr

        # 1b) Bilinear-encode forcings (if any) for predictive conditioning.
        forcings_curr_lat = self.encode_forcings(x_curr)
        forcings_prev_lat = (
            self.encode_forcings(x_prev) if x_prev is not None else forcings_curr_lat
        )

        # 2) Predictive model in latent space.
        r_t = self.predictive(
            z_curr, z_prev, noise,
            forcings_curr=forcings_curr_lat,
            forcings_prev=forcings_prev_lat,
        )

        # 2b) If the predictive was trained with a LatentResidualNormalizer
        #     (v30c+), its output is in tendency-normalized space. Denormalize
        #     back to mean-std space — the decoder was trained on mean-std
        #     r_lat targets, so it needs the same input distribution at inference.
        if self.latent_residual_normalizer is not None:
            if r_t.dim() == 5:
                B, E = r_t.shape[:2]
                r_t_flat = r_t.reshape(B * E, *r_t.shape[2:])
                r_t_flat = self.latent_residual_normalizer.inverse_transform(r_t_flat)
                r_t = r_t_flat.reshape(B, E, *r_t_flat.shape[1:])
            else:
                r_t = self.latent_residual_normalizer.inverse_transform(r_t)

        # 3) Decoder produces full-res residual conditioned on x_curr
        #    (including forcings).
        prog_slice = x_curr[:, : self.prognostic_channels]
        if r_t.dim() == 5:
            # Ensemble (B, E, C, h, w) — decode each member separately.
            B, E = r_t.shape[:2]
            r_flat = r_t.reshape(B * E, *r_t.shape[2:])
            x_curr_flat = x_curr.unsqueeze(1).expand(B, E, *x_curr.shape[1:]).reshape(B * E, *x_curr.shape[1:])
            prog_flat = prog_slice.unsqueeze(1).expand(B, E, *prog_slice.shape[1:]).reshape(B * E, *prog_slice.shape[1:])
            delta_flat = self.decoder(r_flat, x_curr_flat)
            x_next_flat = prog_flat + delta_flat
            x_next = x_next_flat.reshape(B, E, *x_next_flat.shape[1:])
            delta_t = delta_flat.reshape(B, E, *delta_flat.shape[1:])
        else:
            delta_t = self.decoder(r_t, x_curr)
            x_next = prog_slice + delta_t

        if return_intermediates:
            return {
                "z_curr": z_curr, "z_prev": z_prev,
                "r_t": r_t, "delta_t": delta_t,
                "x_next": x_next,
            }
        return x_next

    def predict_step(
        self,
        batch: torch.Tensor,
        pre_processors: nn.Module,
        post_processors: nn.Module,
        data_indices,
        multi_step: int,
        model_comm_group=None,
        gather_out: bool = True,
        residual_normalizer=None,
        **kwargs,
    ) -> torch.Tensor:
        """Anemoi-inference-compatible predict_step for the composed Atlas model.

        Mirrors the contract of ``AnemoiDiTModel.predict_step`` (v17) so the
        existing ``AnemoiModelInterface.predict_step`` can call us
        transparently — no runner changes required other than substituting
        this ``AnemoiAtlasModel`` for the standalone latent-predictive at
        ``predictor.py:_compose_atlas_model``.

        Variant B normalization (locked v30 design):
          1. ``pre_processors`` normalize the full batch (mean-std) so x_curr,
             x_prev arrive in σ-normalized space.
          2. The composed forward returns ``x_next_prog`` in σ-normalized
             space (delta is added to σ-normalized x_curr_prog).
          3. We denormalize via ``norm_mul / norm_add`` from the input
             normalizer (same path v17's predict_step uses) so output is in
             physical units, matching what the runner expects.

        Parameters
        ----------
        batch : torch.Tensor
            ``(B, T, G, V)`` with ``T == multi_step`` history frames.
        pre_processors, post_processors : nn.Module
            Anemoi normalizers (from AnemoiModelInterface).
        data_indices : IndexCollection
            Provides prognostic/forcing/full index lists in the input space.
        multi_step : int
            Number of history frames in ``batch``. Must be ``>= 2`` —
            the composed model needs both x_curr and x_prev.

        Returns
        -------
        torch.Tensor
            ``(B, 1, G, V_out_full)`` with prognostic channels filled by
            the composed forecast and diagnostic channels zeroed
            (the composed model doesn't predict diagnostics — matches v17
            residual-mode behavior).
        """
        # Unused parameters accepted for runner compat:
        del model_comm_group, gather_out, residual_normalizer

        with torch.no_grad():
            assert batch.dim() == 4, (
                f"AnemoiAtlasModel.predict_step expects 4D batch (B, T, G, V); got {batch.shape}"
            )
            assert multi_step >= 2, (
                "AnemoiAtlasModel.predict_step requires multi_step >= 2 "
                f"(needs x_curr + x_prev); got multi_step={multi_step}"
            )

            B, T, G, V = batch.shape

            # Add ensemble dim and normalize the history window (B, T, 1, G, V).
            x = batch[:, 0:multi_step, None, ...].clone()
            x = pre_processors(x, in_place=True)

            # Pull the input-space index lists (full = prog + forcings).
            input_full_idx = data_indices.data.input.full
            input_prog_idx = data_indices.data.input.prognostic
            input_forc_idx = data_indices.data.input.forcing

            # Slice last + second-to-last frames as full-state tensors
            # (B, V_full, H, W). Squeeze ensemble dim — the composed forward
            # accepts (B, C, H, W); ensemble re-fans-out via the noise vector.
            H, W = self.full_res_shape
            def _to_image(t, idx_list):
                """(B, 1, G, V) → (B, len(idx_list), H, W)."""
                sl = t[..., idx_list]                    # (B, 1, G, n)
                # collapse ensemble singleton then channels-last → channels-first
                sl = sl.squeeze(1)                       # (B, G, n)
                sl = sl.permute(0, 2, 1).contiguous()    # (B, n, G)
                return sl.reshape(B, -1, H, W)

            x_curr_full = _to_image(x[:, -1], input_full_idx)         # (B, V_full, H, W)
            x_prev_full = _to_image(x[:, -2], input_full_idx) if multi_step >= 2 else x_curr_full

            # Composed forecast returns the normalized next-step prog state.
            # The composed wrapper auto-samples noise if predictive.noise_vector_dim > 0.
            x_next_prog_norm = self.forward(
                x_curr_full, x_prev=x_prev_full, noise=None,
                return_intermediates=False,
            )                                            # (B, prog_channels, H, W)

            # Reshape back to anemoi (B, 1, G, V_prog).
            n_prog = self.prognostic_channels
            x_next_prog_norm = (
                x_next_prog_norm.reshape(B, n_prog, G).permute(0, 2, 1).contiguous()
            )                                            # (B, G, n_prog)
            x_next_prog_norm = x_next_prog_norm.unsqueeze(1)   # (B, 1, G, n_prog)

            # Denormalize prognostic-only via input normalizer
            # (state lives in σ-normalized space; mean-std denorm gives physical).
            norm_mul, norm_add = self._get_normalizer_buffers(pre_processors)
            prog_mul = norm_mul[input_prog_idx].float()
            prog_add = norm_add[input_prog_idx].float()
            x_next_prog_phys = (
                (x_next_prog_norm.float() - prog_add) / prog_mul
            ).to(x_next_prog_norm.dtype)

            # Build full output tensor (B, 1, G, V_out_full). Decoder outputs
            # are prognostic-only; diagnostic channels (if any) stay zero —
            # the v30 stack does not model diagnostics.
            model_prog_idx = data_indices.model.output.prognostic
            n_output = len(data_indices.model.output.full)
            y_hat = torch.zeros(
                B, 1, G, n_output,
                dtype=x_next_prog_phys.dtype, device=x_next_prog_phys.device,
            )
            y_hat[..., model_prog_idx] = x_next_prog_phys
        return y_hat

    @staticmethod
    def _get_normalizer_buffers(pre_processors: nn.Module):
        """Pull (norm_mul, norm_add) from the InputNormalizer in pre_processors."""
        for processor in pre_processors.processors.values():
            if hasattr(processor, "_norm_mul") and hasattr(processor, "_norm_add"):
                return processor._norm_mul, processor._norm_add
        raise RuntimeError(
            "AnemoiAtlasModel.predict_step: InputNormalizer buffers not "
            "found in pre_processors. The composed model needs mean-std stats "
            "to denormalize the predicted prognostic state."
        )


_RECIPE_DEFAULTS = {
    "fgn": {
        # Alet et al. (June 2025) — low-d noise, no spectral CRPS,
        # almost-fair CRPS estimator handles small-N degeneracy.
        "noise_vector_dim": 32,
        "spectral_crps_weight": 0.0,
        "crps_alpha": 0.95,  # almost-fair to mitigate biased-fair gap
    },
    "atlas": {
        # Kossaifi et al. (Jan 2026) — high-d noise, spectral CRPS in
        # latent stabilizes high-d noise, standard fair-CRPS estimator.
        # noise_vector_dim choice between 256 and ~latent_channels x H x W
        # is an open ablation; start mid-range.
        "noise_vector_dim": 256,
        "spectral_crps_weight": 0.1,
        "crps_alpha": 1.0,  # fully-fair
    },
}


def make_default_atlas_model(
    *,
    full_res_shape: tuple[int, int] = (250, 250),
    latent_shape: tuple[int, int] = (63, 63),
    prognostic_channels: int = 105,
    forcings_channels: int = 11,
    predictive_hidden_size: int = 512,
    predictive_depth: int = 16,
    decoder_hidden_size: int = 512,
    decoder_depth: int = 8,
    noise_vector_dim: int | None = None,
    recipe: str | None = None,
) -> AnemoiAtlasModel:
    """Construct a default GRAF-AI / Atlas-style composed model.

    Convenience constructor for the locked v30 design choices. Override
    individual fields for ablations.

    Parameters
    ----------
    recipe : {"fgn", "atlas", None}
        Optional shortcut for paper-defined defaults. ``"fgn"`` uses
        Alet-et-al low-d noise (32) defaults; ``"atlas"`` uses Kossaifi-
        et-al high-d noise (256) defaults. The recipe ONLY affects the
        ``noise_vector_dim`` here — the loss-side knobs
        (``spectral_crps_weight``, CRPS ``alpha``) are read by the
        training task, not this constructor. See ``_RECIPE_DEFAULTS``
        for the full per-recipe loss config.
    noise_vector_dim : int, optional
        Overrides the recipe default when provided.

    Returns
    -------
    AnemoiAtlasModel
        Composed model with the locked v30 design choices baked in.
    """
    # Resolve noise dim from recipe if not given explicitly.
    if noise_vector_dim is None:
        if recipe is None:
            noise_vector_dim = 32
        else:
            if recipe not in _RECIPE_DEFAULTS:
                raise ValueError(
                    f"recipe must be one of {list(_RECIPE_DEFAULTS)}, got {recipe!r}"
                )
            noise_vector_dim = _RECIPE_DEFAULTS[recipe]["noise_vector_dim"]
    pred = AnemoiLatentDiTModel(
        latent_shape=latent_shape,
        in_channels=prognostic_channels,
        out_channels=prognostic_channels,
        forcings_channels=forcings_channels,
        hidden_size=predictive_hidden_size,
        depth=predictive_depth,
        num_heads=8,
        history_len=2,
        noise_vector_dim=noise_vector_dim,
    )
    dec = AnemoiDecoderDiTModel(
        full_res_shape=full_res_shape,
        latent_shape=latent_shape,
        in_channels_xt=prognostic_channels + forcings_channels,
        in_channels_r=prognostic_channels,
        out_channels=prognostic_channels,
        hidden_size=decoder_hidden_size,
        depth=decoder_depth,
        num_heads=8,
        attn_kernel=9,
    )
    return AnemoiAtlasModel(
        predictive=pred,
        decoder=dec,
        full_res_shape=full_res_shape,
        latent_shape=latent_shape,
        prognostic_channels=prognostic_channels,
        forcings_channels=forcings_channels,
    )
