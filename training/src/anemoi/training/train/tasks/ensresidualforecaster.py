# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Ensemble residual forecaster — combines ``GraphResidualForecaster``'s
residual reconstruction with ``GraphEnsForecaster``'s ensemble-aware loss
path. Injects a per-member noise vector at each forward pass, à la FGN.

Use-case: fine-tune a deterministic residual model (e.g. v17, trained
with ``GraphResidualForecaster`` + ``GraphCastMSELoss``) into an
ensemble model trained with fair-CRPS (``GraphCastCRPSLoss``) by:

1. Cat-replicating the input along the ensemble dim
   (``nens_per_device``).
2. Sampling a fresh ``(B, nens_per_device, noise_vector_dim)`` noise
   tensor each rollout step.
3. Calling ``self.model.forward_with_noise(x, noise_vec)`` instead of
   the standard ``self(x)``.
4. Reconstructing the next-state prediction per member via the residual
   normalizer (same math as ``GraphResidualForecaster``).
5. Computing the CRPS loss on the per-member residuals
   (``Δx̂_norm`` ensemble) against the single-realisation truth
   residual (``Δx_true_norm`` broadcast across ensemble).

At rollout=1 the truth-side ``x_last`` is identical across all members
(input is cat-replicated), so ``Δx_true_norm`` is the same for every
member and the residual-space CRPS is equivalent to state-space CRPS up
to the residual normalization. At rollout > 1 members diverge — handled
in the same way but with the caveat that the truth-residual is computed
per-member (each sees its own diverged ``x_last``); this is consistent
with how ``GraphResidualForecaster`` handles rollout > 1.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import einops
import torch
from torch.utils.checkpoint import checkpoint

from anemoi.training.train.tasks.ensforecaster import GraphEnsForecaster
from anemoi.training.train.tasks.residualforecaster import GraphResidualForecaster

if TYPE_CHECKING:
    from collections.abc import Generator

LOGGER = logging.getLogger(__name__)


class GraphEnsResidualForecaster(GraphEnsForecaster):
    """Ensemble residual forecaster with per-member noise injection."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # FGN-style noise vector knob. Read from config.training; fall back to 32.
        cfg = kwargs.get("config") or args[0]  # config is the first positional in BaseGraphModule
        train_cfg = cfg.training
        self.noise_vector_dim = int(getattr(train_cfg, "noise_vector_dim", 32))
        LOGGER.info(
            "GraphEnsResidualForecaster: noise_vector_dim=%d, nens_per_device=%d",
            self.noise_vector_dim, self.nens_per_device,
        )

        # Per-channel state bounding (WoFSCast-colleague recipe). When
        # ``training.state_bounding.enabled: True``, the reconstructed
        # state in normalized space is clipped to ``[-n_sigma, +n_sigma]``
        # at every rollout step. For mean-std-normalised prognostics this
        # is equivalent to clipping the physical state to ``μ ± n_sigma·σ``
        # (per-channel μ, σ from the training distribution) — no per-
        # channel lookup needed because the normaliser already maps to
        # μ=0, σ=1. The clip teaches the model to stay in-distribution
        # under multi-step rollout AND prevents exploding amplitudes at
        # inference. Applied symmetrically — does not affect ensemble
        # spread, only outlier suppression.
        sb_cfg = getattr(train_cfg, "state_bounding", None)
        self.state_bounding_enabled = bool(getattr(sb_cfg, "enabled", False)) if sb_cfg is not None else False
        self.state_bounding_n_sigma = float(getattr(sb_cfg, "n_sigma", 4.0)) if sb_cfg is not None else 4.0
        if self.state_bounding_enabled:
            LOGGER.info(
                "GraphEnsResidualForecaster: state bounding ENABLED  (clip "
                "normalised state to ±%.2f σ at every rollout step)",
                self.state_bounding_n_sigma,
            )

        # In-loop PHYSICAL bounding — same recipe v48a's deterministic forecaster
        # uses: positive bounding (qv>=0 via ``variables``+``min_val``), per-variable
        # ``var_ranges``, and low-precip ``zero_below``, applied to the reconstructed
        # physical state BEFORE renormalization/feedback at every rollout step. The
        # ensemble thus trains under the same clamp the operational post-hoc stack
        # applies, and never learns to propagate states (negative humidity, drizzle)
        # inference would clip away. Distinct from the ±n_sigma normalized clip above
        # (both can be active): physical bounding enforces variable ranges, n_sigma
        # suppresses outliers. Read from the same training.state_bounding block.
        self._sb_phys_enabled = False
        self._sb_min, self._sb_ranges, self._sb_zero = [], [], []
        if sb_cfg is not None and getattr(sb_cfg, "enabled", False):
            from fnmatch import fnmatch as _fnmatch
            _i2n = {int(v): k for k, v in self.data_indices.name_to_index.items()}
            _prog = [_i2n[int(i)] for i in self.data_indices.data.input.prognostic]
            _n2p = {nm: p for p, nm in enumerate(_prog)}
            _minv = float(getattr(sb_cfg, "min_val", 0.0))
            for pat in (getattr(sb_cfg, "variables", []) or []):
                self._sb_min += [(p, _minv) for nm, p in _n2p.items() if _fnmatch(nm, pat)]
            for nm, rng in (getattr(sb_cfg, "var_ranges", {}) or {}).items():
                if nm in _n2p:
                    lo = None if rng[0] is None else float(rng[0])
                    hi = None if (len(rng) < 2 or rng[1] is None) else float(rng[1])
                    self._sb_ranges.append((_n2p[nm], lo, hi))
            for nm, thr in (getattr(sb_cfg, "zero_below", {}) or {}).items():
                if nm in _n2p:
                    self._sb_zero.append((_n2p[nm], float(thr)))
            self._sb_phys_enabled = bool(self._sb_min or self._sb_ranges or self._sb_zero)
            if self._sb_phys_enabled:
                LOGGER.info(
                    "GraphEnsResidualForecaster: PHYSICAL state bounding ENABLED "
                    "(fed-back state): %d min-clamp, %d ranges, %d zero_below",
                    len(self._sb_min), len(self._sb_ranges), len(self._sb_zero),
                )

        # Sanity: the underlying AnemoiDiTModel must support forward_with_noise.
        # We can't check this at __init__ because self.model is built lazily by
        # the base class; we'll error at the first forward call if it's missing.

    def _apply_state_bounding(self, x: torch.Tensor) -> torch.Tensor:
        """Physical bounding (qv>=0 / var_ranges / low-precip zero_below) on the
        reconstructed physical state before feedback — FUNCTIONAL (broadcast clamp
        + where, autograd-safe). ``x`` is (..., grid, n_prog) in
        data.input.prognostic order; per-channel vectors built once and cached.
        No-op unless physical state bounding is configured."""
        if not getattr(self, "_sb_phys_enabled", False):
            return x
        n = x.shape[-1]
        if getattr(self, "_sb_lo", None) is None or self._sb_lo.numel() != n or self._sb_lo.device != x.device:
            lo = torch.full((n,), float("-inf"), device=x.device, dtype=torch.float32)
            hi = torch.full((n,), float("inf"), device=x.device, dtype=torch.float32)
            zt = torch.full((n,), float("-inf"), device=x.device, dtype=torch.float32)
            for p, mn in self._sb_min:
                lo[p] = max(float(lo[p]), mn)
            for p, l, h in self._sb_ranges:
                if l is not None:
                    lo[p] = max(float(lo[p]), l)
                if h is not None:
                    hi[p] = min(float(hi[p]), h)
            for p, thr in self._sb_zero:
                zt[p] = thr
            self._sb_lo, self._sb_hi, self._sb_zt = lo, hi, zt
        x = torch.clamp(x, min=self._sb_lo.to(x.dtype), max=self._sb_hi.to(x.dtype))
        x = torch.where(x < self._sb_zt.to(x.dtype), torch.zeros((), device=x.device, dtype=x.dtype), x)
        return x

    @staticmethod
    def _get_normalizer_buffers(processors) -> tuple[torch.Tensor, torch.Tensor]:
        """Reuse the buffer accessor from GraphResidualForecaster."""
        return GraphResidualForecaster._get_normalizer_buffers(processors)

    def _sample_noise(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Per-step, per-member noise vector ``z ~ N(0, I)^{noise_vector_dim}``.

        Returns shape ``(B, nens_per_device, noise_vector_dim)``. A fresh
        sample is drawn at every call so members within a step are
        independent. Different ranks see different draws because their
        default RNG state diverges; this is fine for CRPS where we only
        need the ensemble to be a valid sample, not perfectly
        synchronized across ranks.
        """
        return torch.randn(
            batch_size,
            self.nens_per_device,
            self.noise_vector_dim,
            device=device,
            dtype=dtype,
        )

    def _rollout_step(
        self,
        batch: torch.Tensor,
        rollout: int | None = None,
        validation_mode: bool = False,
    ) -> Generator[tuple[torch.Tensor | None, dict, list]]:
        """Ensemble residual rollout with per-member noise injection.

        Structurally mirrors ``GraphResidualForecaster._rollout_step``
        but threads a noise vector through the model on every step and
        keeps the ensemble dim active in all per-member computations.
        """
        # Initial input window (ensemble dim = 1 from the dataloader).
        x = batch[
            :,
            0 : self.multi_step,
            ...,
            self.data_indices.data.input.full,
        ]  # (B, multi_step, 1, latlon, nvar)

        # Cat-replicate along the ensemble dim (same as GraphEnsForecaster line 207).
        x = torch.cat([x] * self.nens_per_device, dim=2)  # (B, ms, nens, latlon, nvar)
        assert x.shape[2] == self.nens_per_device

        msg = (
            "Batch length not sufficient for requested multi_step length!"
            f", {batch.shape[1]} !>= {rollout + self.multi_step}"
        )
        assert batch.shape[1] >= rollout + self.multi_step, msg

        model_prog_idx = self.data_indices.model.output.prognostic
        model_diag_idx = self.data_indices.model.output.diagnostic
        # data-space prognostic indices: for the full data tensor `batch` (n_data-wide)
        # and the n_data-wide normalizer buffers (norm_mul/norm_add).
        input_prog_idx = self.data_indices.data.input.prognostic
        # model-input-space prognostic indices: for the SLICED input tensor `x`
        # (= batch[..., data.input.full], i.e. input.full-wide). When diagnostic-only
        # outputs are present (e.g. v39 surface 2D fields), data-space != model-input
        # space, so indexing `x` with the data-space idx overruns -> device-side assert.
        # Without diagnostics this equals input_prog_idx (no-op). Matches residualforecaster.
        model_input_prog_idx = self.data_indices.model.input.prognostic

        norm_mul, norm_add = self._get_normalizer_buffers(self.model.pre_processors)

        # Cached reference for inverse normalization (small tensors, persist for the rollout).
        nm_mul_prog = norm_mul[input_prog_idx].float()
        nm_add_prog = norm_add[input_prog_idx].float()

        # Decide noise path at the start of the trajectory:
        #
        #   AIFS-style (per-grid-point): the model has a ``noise_injector``;
        #       ``forward_with_spatial_noise`` samples noise INTERNALLY on every
        #       call (fresh per step, AIFS convention). No noise vector to thread
        #       through; member diversity is produced inside the model.
        #
        #   FGN-style (global per-member): the model has a ``noise_encoder``;
        #       we sample ONE noise vector per (batch, member, trajectory) and
        #       reuse it on every rollout step. This matches the FGN inference
        #       contract where ``forward_with_noise`` is called repeatedly with
        #       the same z, and at rollout=1 is identical to per-step sampling.
        use_aifs_noise = getattr(self.model, "noise_injector", None) is not None
        use_fgn_noise = (not use_aifs_noise) and hasattr(self.model, "forward_with_noise")

        B = x.shape[0]
        if use_aifs_noise:
            noise_vec_traj = None
        elif use_fgn_noise:
            noise_vec_traj = self._sample_noise(B, x.device, x.dtype)
        else:
            msg = (
                "GraphEnsResidualForecaster requires the model to expose either "
                "an AIFS-style ``noise_injector`` (per-grid-point noise) or an "
                "FGN-style ``forward_with_noise(x, noise_vec)`` method. Neither "
                "was found. Configure dit.noise_injector OR dit.noise_vector_dim."
            )
            raise AttributeError(msg)

        for rollout_step in range(rollout or self.rollout):
            if use_aifs_noise:
                # AIFS: fresh per-grid-point noise sampled inside the model.
                model_output = self.model.forward_with_spatial_noise(x)
            else:
                # FGN: same trajectory-level noise across every rollout step.
                model_output = self.model.forward_with_noise(x, noise_vec_traj)
            # model_output: (B, nens_per_device, latlon, n_output)

            # ---- residual reconstruction (same math as GraphResidualForecaster) ----
            # Note: x_last per member may differ at rollout > 1; the residual normalizer
            # is applied per-member so each one self-consistently reconstructs its
            # next state.
            Δx̂_norm_prog = model_output[..., model_prog_idx]  # (B, nens, G, n_prog)

            x_last_norm = x[:, -1, ..., model_input_prog_idx]      # (B, nens, G, n_prog)
            y_true_norm = batch[
                :,
                self.multi_step + rollout_step,
                ...,
                input_prog_idx,
            ]                                                       # (B, 1, G, n_prog)

            # Unnormalize to physical space (fp32 for precision).
            x_last_phys = (x_last_norm.float() - nm_add_prog) / nm_mul_prog
            y_true_phys = (y_true_norm.float() - nm_add_prog) / nm_mul_prog

            # Target residual in normalized residual space.
            # Broadcasting: x_last_phys (B, nens, G, n_prog), y_true_phys (B, 1, G, n_prog).
            # The residual_normalizer.transform will produce (B, nens, G, n_prog) — per
            # member it's (y_true - x_last^m) / σ_Δx, which is the natural per-member
            # truth residual.
            Δx_true_norm = self.model.residual_normalizer.transform(
                x_last_phys,
                y_true_phys,
                in_place=False,
            )  # (B, nens, G, n_prog)

            # Reconstruct next-state predictions per member (state space).
            y_pred_phys_prog = self.model.residual_normalizer.inverse_transform(
                x_last_phys,
                Δx̂_norm_prog,
                in_place=False,
            )

            # In-loop PHYSICAL bounding (qv>=0 / var_ranges / low-precip zero_below),
            # applied to the fed-back state — same as v48a. Broadcasts over the
            # ensemble dim (per-member). Affects feedback only; the CRPS loss is on
            # the residual Δx̂. Distinct from the ±n_sigma clip applied below.
            y_pred_phys_prog = self._apply_state_bounding(y_pred_phys_prog)

            # Renormalize for next-step input.
            y_pred_prog = (
                y_pred_phys_prog * nm_mul_prog + nm_add_prog
            ).to(model_output.dtype)

            # State bounding: clip normalised state to ±n_sigma per channel.
            # Mean-std normaliser maps physical (μ, σ) → (0, 1), so this is
            # equivalent to physical-space clip at μ ± n_sigma·σ. Applied
            # symmetrically to every prognostic channel; outlier suppression
            # only, does not reduce ensemble spread within ±n_sigma. The
            # bound is what the WoFSCast colleague found load-bearing for
            # CRPS rollout stability at storm scale.
            if self.state_bounding_enabled:
                y_pred_prog = torch.clamp(
                    y_pred_prog,
                    -self.state_bounding_n_sigma,
                    self.state_bounding_n_sigma,
                )

            # Build full prediction tensor (prognostic + diagnostic) for _advance_input.
            n_output = len(self.data_indices.model.output.full)
            y_pred = torch.zeros(
                *model_output.shape[:-1], n_output,
                dtype=model_output.dtype, device=model_output.device,
            )
            y_pred[..., model_prog_idx] = y_pred_prog
            if len(model_diag_idx) > 0:
                y_pred[..., model_diag_idx] = model_output[..., model_diag_idx]

            # ---- loss (CRPS in normalized residual space) ------------------
            # _compute_loss in the base class expects an ensemble dim on pred.
            # Δx̂_norm_prog: (B, nens, G, n_prog).
            # Δx_true_norm: (B, nens, G, n_prog) — same nens dim (broadcast happened
            # in transform). We collapse to a single target per (B, G, n_prog) by
            # taking the truth-residual at member 0 — at rollout=1, all members
            # share the same x_last, so this is exact; at rollout > 1 members
            # diverge but the truth-residual is intrinsically per-member, so we
            # keep the per-member targets and let CRPS broadcast.

            grid_shard_slice = self.grid_shard_slice

            # Target shape: pass 4-D ``(B, 1, G, V)`` with singleton ensemble
            # dim. GraphCastCRPSLoss.calculate_difference accepts both 3-D
            # ``(B, G, V)`` and 4-D ``(B, 1, G, V)`` and squeezes internally.
            # GraphCast*MSE*/MAE/Huber etc. need the 4-D form so that
            # ``pred (B, E, G, V) - target (B, 1, G, V)`` broadcasts cleanly
            # along the ensemble axis; the 3-D form caused a right-aligned
            # broadcast that compared pred's ensemble dim against target's
            # batch dim (3 vs 4) and crashed v31c CombinedLoss[CRPS, MSE].
            #
            # Member-0 slicing (vs all-members average) is exact at
            # rollout=1 (all members share x_last); at rollout > 1 the
            # truth-residual is intrinsically per-member, but the standard
            # FGN formulation uses a single reference truth so we pick m0.
            Δx_true_for_loss = Δx_true_norm[:, 0:1]  # (B, 1, G, n_prog)

            loss = checkpoint(
                self.compute_loss_metrics_residual,
                Δx̂_norm_prog,
                Δx_true_for_loss,                    # (B, 1, G, n_prog)
                rollout_step,
                validation_mode,
                use_reentrant=False,
            )[0]

            # Per-group loss logging (matches GraphResidualForecaster pattern).
            if hasattr(self.loss, "_last_per_group_losses") and self.loss._last_per_group_losses is not None:
                for group_name, group_loss in self.loss._last_per_group_losses.items():
                    self.log(
                        f"train_loss/{group_name}",
                        group_loss,
                        on_step=True,
                        on_epoch=False,
                        logger=True,
                        rank_zero_only=True,
                    )

            # ---- validation metrics (in state space) -----------------------
            metrics_next = {}
            if validation_mode:
                # Take the ensemble-mean prediction for state-space metrics
                # comparison. y_true is the single-realisation truth.
                y_true_for_metrics = batch[
                    :,
                    self.multi_step + rollout_step,
                    ...,
                    self.data_indices.data.output.full,
                ]  # (B, 1, G, n_output)
                # Use ensemble mean for the pointwise metrics; spread metrics
                # would need a separate path, deferred.
                metrics_next = self.calculate_val_metrics(
                    y_pred.mean(dim=1, keepdim=True),
                    y_true_for_metrics,
                    step=rollout_step,
                    grid_shard_slice=grid_shard_slice,
                )

            # Feed prediction back into the input window (one slot per member).
            x = self._advance_input(x, y_pred, batch, rollout_step)

            yield loss, metrics_next, y_pred

    def compute_loss_metrics_residual(
        self,
        pred_residual: torch.Tensor,
        target_residual: torch.Tensor,
        step: int,
        validation_mode: bool,
    ) -> tuple[torch.Tensor, dict, torch.Tensor]:
        """Compute CRPS loss in normalized residual space.

        Parameters
        ----------
        pred_residual : torch.Tensor
            Per-member predicted residuals in normalized residual space,
            shape ``(B, nens_per_device, G, n_prog)``.
        target_residual : torch.Tensor
            Single-realisation truth residual, shape ``(B, 1, G, n_prog)``
            (4-D with singleton ensemble dim — broadcasts cleanly for MSE
            children of CombinedLoss; CRPS internally squeezes the singleton).
        step : int
            Rollout step index (kept for API compatibility; unused here).
        validation_mode : bool
            Kept for API compatibility; metrics are computed in the caller
            in state space.

        Returns
        -------
        loss : torch.Tensor
            Scalar loss (after GraphCast reduction).
        metrics : dict
            Empty here — state-space metrics are handled in the rollout
            loop using the un-normalized predictions.
        y_pred_residual_ens : torch.Tensor
            The (gathered) per-member residuals, kept for downstream
            diagnostics.
        """
        # Gather across the ensemble communication subgroup (no-op when
        # nens_per_device == nens_per_group / world).
        # Note: residual is in (B, E, G, V) layout — matches GraphEnsForecaster's
        # gather dim=1.
        from anemoi.models.distributed.graph import gather_tensor

        pred_ens = gather_tensor(
            pred_residual.clone(),
            dim=1,
            shapes=[pred_residual.shape] * self.ens_comm_subgroup_size,
            mgroup=self.ens_comm_subgroup,
        )

        loss = self._compute_loss(
            pred_ens,
            target_residual,
            grid_dim=self.grid_dim,
            grid_shard_shape=self.grid_shard_shapes,
        )

        return loss, {}, pred_ens
