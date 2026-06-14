# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.utils.checkpoint import checkpoint

from anemoi.training.train.tasks.rollout import BaseRolloutGraphModule

if TYPE_CHECKING:
    from collections.abc import Generator

LOGGER = logging.getLogger(__name__)


class GraphResidualForecaster(BaseRolloutGraphModule):
    """Graph neural-network forecaster that predicts *normalized residuals*.

    Δx_norm = (x_{t+1} - x_t) / std_Δx  (mean difference assumed zero).
    Matches the GraphCast framework.

    Diagnostic variables (``data_indices.model.output.diagnostic``) are
    predicted as DIRECT normalized STATE (not residuals). Their loss term
    (``L_diag``) is computed in normalized-state space against the
    InputNormalizer-normalized batch truth — so σ_state is already baked in
    and MSE gives a ~1.0 no-skill baseline per diagnostic, commensurate with
    the prognostic residual term. It is combined with the prognostic residual
    loss (``L_prog``) count-weighted per-variable-equal:

        total = (N_prog · L_prog + N_diag · L_diag) / (N_prog + N_diag)

    When ``N_diag == 0`` (v17 and all prior configs) the entire diagnostic
    path is skipped and ``total == L_prog`` exactly (byte-identical behaviour).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Diagnostic state-space loss wiring. Built ONCE here, not per-step.
        # When there are no diagnostics this is an exact no-op (see _rollout_step).
        self._diag_out_idx = self.data_indices.model.output.diagnostic
        self._diag_data_out_idx = self.data_indices.data.output.diagnostic
        self._n_diag = len(self._diag_out_idx)
        self._n_prog = len(self.data_indices.model.output.prognostic)

        if self._n_diag > 0:
            # ----------------------------------------------------------------
            # (1) Cache diagnostic-subset variable weights + grid weights from
            #     the FULL (n_output-wide) loss scaler, BEFORE subsetting it to
            #     prognostic. var_w is indexed in model-output space, so the
            #     diagnostic positions self._diag_out_idx select the right cols.
            #     Stored as registered buffers (correct device, moved with module).
            # ----------------------------------------------------------------
            var_w, grid_w = self._extract_loss_scalers(self.loss)
            self.register_buffer(
                "_diag_var_weights",
                var_w[self._diag_out_idx].clone() if var_w is not None else None,
                persistent=False,
            )
            self.register_buffer(
                "_diag_grid_weights",
                grid_w.clone() if grid_w is not None else None,
                persistent=False,
            )
            self.register_buffer(
                "_diag_data_out_idx_buf",
                torch.as_tensor(self._diag_data_out_idx, dtype=torch.long),
                persistent=False,
            )

            # ----------------------------------------------------------------
            # (2) Fix the prognostic-loss variable-scaler width mismatch.
            #     `general_variable` is built over data.output.full (n_output-
            #     wide, e.g. 115), but the prognostic loss tensor is n_prog-wide
            #     (e.g. 112). Subset the loss's variable scaler to the prognostic
            #     OUTPUT positions so scale_iteratively's expand_as aligns and the
            #     prognostic-space group slices in _reduce_per_variable match.
            #     Done AFTER caching the diagnostic weights above. For
            #     N_diag == 0 this code never runs (prognostic == full).
            # ----------------------------------------------------------------
            prog_out_idx = torch.as_tensor(
                self.data_indices.model.output.prognostic, dtype=torch.long
            )
            self._subset_loss_variable_scaler(self.loss, prog_out_idx)

            LOGGER.info(
                "GraphResidualForecaster: diagnostic state-space loss ENABLED "
                "(N_prog=%d, N_diag=%d, var_weights=%s)",
                self._n_prog, self._n_diag,
                None if var_w is None else self._diag_var_weights.tolist(),
            )
        else:
            LOGGER.info(
                "GraphResidualForecaster: no diagnostics (N_diag=0); "
                "diagnostic loss path disabled (L_prog only).",
            )

        # ------------------------------------------------------------------
        # Colored input-noise robustness training (v42-P1). Injects band-
        # limited, spectrally-shaped noise into the NORMALIZED input state's
        # prognostic channels with per-variable amplitudes measured from the
        # model's own 1-step error spectrum; target unchanged -> the single-
        # step map is trained to DAMP its own grid-scale noise (the measured
        # closed-loop gain is 1.04-1.37 per step in the 9-25 km band, which
        # compounds to the 18 h rollout failure). Disabled by default.
        # ------------------------------------------------------------------
        self._inoise = None
        in_cfg = getattr(getattr(self.config, "training", None), "input_noise", None)
        if in_cfg is not None and getattr(in_cfg, "enabled", False):
            self._init_input_noise(in_cfg)

        # AIFS-CRPS reference-field truncation (their eq. 1):
        #   x_{t+1} = U(D(x_t)) + f(x_t)
        # The target residual AND the reconstruction both use the truncated
        # reference, so the loss trains the tendency to regenerate the
        # removed scales while the identity path can no longer carry
        # sub-(factor*dx) junk (leftover-advection artifacts, lattice noise)
        # between steps. Read from model config so train/inference agree.
        self._ref_trunc = 0
        self._ref_trunc_hw = None
        dit_cfg = getattr(getattr(self.config.model, "model", None), "dit", None)
        if dit_cfg is not None:
            self._ref_trunc = int(getattr(dit_cfg, "reference_truncation", 0) or 0)
            if self._ref_trunc and getattr(dit_cfg, "field_shape", None) is not None:
                self._ref_trunc_hw = tuple(int(v) for v in dit_cfg.field_shape)
                # Channel gating: positions (in input.prognostic order) to truncate;
                # surface/terrain-anchored fields are excluded (their small scales
                # are stationary truth, and truncation inflates their targets).
                from fnmatch import fnmatch
                excl = list(getattr(dit_cfg, "reference_truncation_exclude",
                                    ["pressure_*", "t2m", "skintemp", "snowh"]) or [])
                idx_to_name = {int(v): k for k, v in self.data_indices.name_to_index.items()}
                prog_names = [idx_to_name[int(i)] for i in self.data_indices.data.input.prognostic]
                self._ref_trunc_chans = [
                    c for c, nm in enumerate(prog_names)
                    if not any(fnmatch(nm, pat) for pat in excl)
                ]
                LOGGER.info(
                    "reference_truncation ENABLED: factor %d, grid %s, %d/%d prognostic "
                    "channels (excluded: %s)",
                    self._ref_trunc, self._ref_trunc_hw,
                    len(self._ref_trunc_chans), len(prog_names),
                    [nm for nm in prog_names if any(fnmatch(nm, p_) for p_ in excl)][:8],
                )

    def _truncate_reference(self, x_phys: torch.Tensor) -> torch.Tensor:
        """U(D(x)) on (..., grid, n_prog); no-op when disabled or sharded."""
        if not self._ref_trunc or self._ref_trunc_hw is None:
            return x_phys
        H, W = self._ref_trunc_hw
        if x_phys.shape[-2] != H * W:
            return x_phys
        from anemoi.models.models.flexible_dit import reference_truncate
        chans = getattr(self, "_ref_trunc_chans", None)
        if chans is None or len(chans) == x_phys.shape[-1]:
            return reference_truncate(x_phys, H, W, self._ref_trunc)
        out = x_phys.clone()
        out[..., chans] = reference_truncate(x_phys[..., chans], H, W, self._ref_trunc)
        return out

    def _init_input_noise(self, cfg) -> None:
        import numpy as np

        # Per-channel sigma in MODEL-INPUT space (prognostic channels only;
        # forcings are clean external inputs and must not be perturbed).
        name_to_index = self.data_indices.name_to_index
        input_full = [int(i) for i in self.data_indices.data.input.full]
        idx_to_name = {int(v): k for k, v in name_to_index.items()}
        input_names = [idx_to_name[i] for i in input_full]   # model-input channel order
        prog_pos = [int(p) for p in self.data_indices.model.input.prognostic]

        spec = {}
        if cfg.spec_path:
            d = np.load(cfg.spec_path, allow_pickle=True)
            spec = {str(v): float(s) for v, s in zip(d["vars"], d["sigma_band_rms"])}
        sigma = torch.zeros(len(input_names))
        missing = []
        for p in prog_pos:
            n = input_names[p]
            if n in spec:
                sigma[p] = spec[n]
            else:
                sigma[p] = float(cfg.default_sigma)
                missing.append(n)
        if missing:
            LOGGER.warning("input_noise: %d prognostic vars missing from spec (default_sigma=%g): %s",
                           len(missing), cfg.default_sigma, missing[:5])
        self.register_buffer("_inoise_sigma", sigma, persistent=False)

        # Field shape (DiT regular grid) for the spectral mask.
        dit = getattr(getattr(self.config.model, "model", None), "dit", None)
        if dit is None or getattr(dit, "field_shape", None) is None:
            LOGGER.warning("input_noise: no model.model.dit.field_shape; noise DISABLED")
            return
        H, W = [int(v) for v in dit.field_shape]
        self._inoise = {
            "prob": float(cfg.prob), "scale_max": float(cfg.scale_max),
            "H": H, "W": W, "lmin": float(cfg.lambda_min_km),
            "lmax": float(cfg.lambda_max_km), "cell": float(cfg.cell_km),
            "p": float(cfg.shape_powerlaw_p), "mask": None,
        }
        LOGGER.info(
            "input_noise ENABLED: band %.1f-%.1f km, shape p=%.1f, prob=%.2f, "
            "scale U(0,%.1f), %d/%d channels (max sigma=%.4f)",
            cfg.lambda_min_km, cfg.lambda_max_km, cfg.shape_powerlaw_p, cfg.prob,
            cfg.scale_max, int((sigma > 0).sum()), len(input_names), float(sigma.max()),
        )

    def _inject_input_noise(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., grid, vars) NORMALIZED model-input tensor. Returns x (+ noise).

        Dimension-agnostic over leading dims (handles both (bs, ms, grid, v)
        and (bs, ms, ens, grid, v) layouts).
        """
        st = self._inoise
        if st is None or not self.training or torch.rand(()) > st["prob"]:
            return x
        H, W = st["H"], st["W"]
        *lead, grid, nv = x.shape
        if grid != H * W:
            if not st.get("warned_shape"):
                st["warned_shape"] = True
                LOGGER.warning(
                    "input_noise: grid size %d != field_shape %dx%d — noise is "
                    "NEVER injected this run (check model.model.dit.field_shape)",
                    grid, H, W,
                )
            return x
        if st["mask"] is None:
            ky = torch.fft.fftfreq(H, d=st["cell"], device=x.device)
            kx = torch.fft.rfftfreq(W, d=st["cell"], device=x.device)
            kk = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
            wl = torch.where(kk > 0, 1.0 / kk.clamp_min(1e-12), torch.full_like(kk, 1e9))
            band = (wl >= st["lmin"]) & (wl <= st["lmax"])
            amp = torch.zeros_like(kk)
            amp[band] = (wl[band] / st["lmin"]) ** (st["p"] / 2.0)  # power ~ lambda^p
            # calibrate so a generated field has unit spatial std
            probe = torch.fft.irfft2(torch.fft.rfft2(torch.randn(8, H, W, device=x.device)) * amp, s=(H, W))
            st["mask"] = amp / probe.std().clamp_min(1e-12)
        n_lead = 1
        for d in lead:
            n_lead *= d
        white = torch.randn(n_lead * nv, H, W, device=x.device)
        eps = torch.fft.irfft2(torch.fft.rfft2(white) * st["mask"], s=(H, W))
        eps = eps.reshape(*lead, nv, grid).movedim(-2, -1)  # (..., grid, vars)
        scale = self._inoise_sigma.to(device=x.device, dtype=x.dtype) * (
            torch.rand((), device=x.device) * st["scale_max"]
        )
        return x + eps.to(x.dtype) * scale

    @staticmethod
    def _extract_loss_scalers(loss) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Pull the (VARIABLE, GRID) scaler tensors from a loss / its MSE leaf.

        Returns (variable_scaler, grid_scaler), either may be None if absent.
        Multiple scalers on the same dim are multiplied together (matches how
        scale_iteratively composes them).
        """
        from anemoi.training.utils.enums import TensorDim

        scaler = getattr(loss, "mse", loss).scaler
        var_w = None
        grid_w = None
        for dims, tensor in scaler.tensors.values():
            dims = (dims,) if isinstance(dims, int) else tuple(dims)
            if TensorDim.VARIABLE in dims:
                var_w = tensor if var_w is None else var_w * tensor
            elif TensorDim.GRID in dims:
                grid_w = tensor if grid_w is None else grid_w * tensor
        return var_w, grid_w

    @staticmethod
    def _subset_loss_variable_scaler(loss, keep_idx: torch.Tensor) -> None:
        """Subset every VARIABLE-dim scaler on the loss to `keep_idx` (override)."""
        from anemoi.training.utils.enums import TensorDim

        leaf = getattr(loss, "mse", loss)
        for name, (dims, tensor) in list(leaf.scaler.tensors.items()):
            dims = (dims,) if isinstance(dims, int) else tuple(dims)
            if TensorDim.VARIABLE not in dims:
                continue
            if tensor.shape[0] == len(keep_idx):
                continue  # already the right width
            loss.update_scaler(name, tensor[keep_idx.to(tensor.device)].clone(), override=True)

    @staticmethod
    def _get_normalizer_buffers(processors) -> tuple[torch.Tensor, torch.Tensor]:
        for processor in processors.processors.values():
            if hasattr(processor, "_norm_mul") and hasattr(processor, "_norm_add"):
                return processor._norm_mul, processor._norm_add
        raise RuntimeError("InputNormalizer buffers not found in pre_processors.")

    def _compute_diag_loss(
        self,
        yhat_diag_state: torch.Tensor,
        ytrue_diag_state: torch.Tensor,
        grid_shard_slice: slice | None,
    ) -> torch.Tensor:
        """Diagnostic state-space MSE, reduced identically to L_prog.

        Replicates BaseLoss.reduce: scale by general_variable (diag subset) on
        VARIABLE + limited_area_mask on GRID, then per-variable MEAN over
        VARIABLE, node-weighted SUM over GRID, MEAN over BATCH/ENSEMBLE.
        With unit-sum grid weights the grid-SUM is a weighted mean over the
        LAM interior — matching the prognostic term's spatial reduction.
        """
        from anemoi.training.utils.enums import TensorDim

        diff = yhat_diag_state.float() - ytrue_diag_state.float()
        out = torch.square(diff)  # (B, E, G, n_diag)

        if self._diag_var_weights is not None:
            out = out * self._diag_var_weights.to(out.dtype).view(1, 1, 1, -1)
        if self._diag_grid_weights is not None:
            gw = self._diag_grid_weights.to(out.dtype)
            if grid_shard_slice is not None and gw.shape[0] >= getattr(grid_shard_slice, "stop", 0):
                gw = gw[grid_shard_slice]
            out = out * gw.view(1, 1, -1, 1)

        # per-variable MEAN, then node-weighted SUM over grid, then mean B/E
        out = out.mean(dim=TensorDim.VARIABLE)
        out = out.sum(dim=TensorDim.GRID)
        out = out.mean(dim=(TensorDim.BATCH_SIZE, TensorDim.ENSEMBLE_DIM))
        if grid_shard_slice is not None and self.model_comm_group is not None:
            from anemoi.models.distributed.graph import reduce_tensor

            out = reduce_tensor(out, self.model_comm_group)
        return out

    def _rollout_step(
        self,
        batch: torch.Tensor,
        rollout: int | None = None,
        validation_mode: bool = False,
    ) -> Generator[tuple[torch.Tensor | None, dict, list]]:
        """Rollout step for the residual forecaster (GraphCast-style).

        The model predicts *normalized residuals*:
            Δx_norm = (y_true - x_last) / σ_Δx
        where y_true and x_last are in physical space.

        The loss is computed in this normalized residual space.

        At each step, the next-state prediction is reconstructed as:
            x̂_{t+1} = x_last + Δx̂_norm * σ_Δx
        """
        # Start rollout of preprocessed batch
        x = batch[
            :,
            0 : self.multi_step,
            ...,
            self.data_indices.data.input.full,
        ]  # (bs, multi_step, latlon, nvar)

        # Colored input-noise robustness training (v42-P1): perturb the
        # NORMALIZED input's prognostic channels; target unchanged. No-op
        # unless training.input_noise.enabled (and self.training).
        x = self._inject_input_noise(x)

        msg = (
            "Batch length not sufficient for requested multi_step length!"
            f", {batch.shape[1]} !>= {rollout + self.multi_step}"
        )
        assert batch.shape[1] >= rollout + self.multi_step, msg

        # Get indices for prognostic and diagnostic variables
        # model.output indices are for the model output tensor
        model_prog_idx = self.data_indices.model.output.prognostic
        model_diag_idx = self.data_indices.model.output.diagnostic
        # data-space prognostic indices: for the full data tensor `batch` (n_data-wide)
        # and the n_data-wide normalizer buffers (norm_mul/norm_add).
        input_prog_idx = self.data_indices.data.input.prognostic
        # model-input-space prognostic indices: for the SLICED input tensor `x`
        # (= batch[..., data.input.full], i.e. input.full-wide). When diagnostics are
        # present, data-space != model-input space, so `x` must be indexed here (matches
        # _advance_input). Without diagnostics this equals input_prog_idx (no-op).
        model_input_prog_idx = self.data_indices.model.input.prognostic

        # Get normalizer buffers for unnormalization
        # IMPORTANT: norm_mul and norm_add are indexed by INPUT variable positions,
        # so we must use data.input.prognostic (not data.output.prognostic) to access them.
        norm_mul, norm_add = self._get_normalizer_buffers(self.model.pre_processors)

        for rollout_step in range(rollout or self.rollout):
            # Forward prediction (normalized residual)
            # model_output shape: (batch, ensemble, grid, n_output)
            model_output = self(x)

            # ============================================================
            # Extract PROGNOSTIC variables for residual prediction
            # ============================================================
            # Slice model output to only prognostic variables
            Δx̂_norm_prog = model_output[..., model_prog_idx]  # (batch, ensemble, grid, n_prog)

            # Get normalized values from batch (only prognostic). `x` is the sliced
            # model-input tensor -> index with model_input_prog_idx; `batch` is the full
            # data tensor -> index with input_prog_idx (data space). Same prognostic
            # ordering in both, so downstream physical-space math stays aligned.
            x_last_norm = x[:, -1, ..., model_input_prog_idx]  # (batch, ensemble, grid, n_prog)
            y_true_norm = batch[
                :,
                self.multi_step + rollout_step,
                ...,
                input_prog_idx,
            ]  # (batch, ensemble, grid, n_prog)

            # ============================================================
            # GraphCast-style: Compute residuals in PHYSICAL space
            # ============================================================

            # Unnormalize to physical space (in float32 for precision)
            # x_norm = x_phys * norm_mul + norm_add  =>  x_phys = (x_norm - norm_add) / norm_mul
            x_last_phys = (x_last_norm.float() - norm_add[input_prog_idx].float()) / norm_mul[input_prog_idx].float()
            y_true_phys = (y_true_norm.float() - norm_add[input_prog_idx].float()) / norm_mul[input_prog_idx].float()

            # Reference-field truncation (AIFS eq. 1): both the target residual
            # and the reconstruction below are taken w.r.t. U(D(x_last)).
            x_ref_phys = self._truncate_reference(x_last_phys)

            # Compute target residual in physical space, then normalize by diff_std only
            Δx_true_norm = self.model.residual_normalizer.transform(
                x_ref_phys,
                y_true_phys,
                in_place=False,
            )

            # ============================================================
            # GraphCast-style: Reconstruct in PHYSICAL space, then renormalize
            # ============================================================

            # Reconstruct next state in physical space (only prognostic)
            y_pred_phys_prog = self.model.residual_normalizer.inverse_transform(
                x_ref_phys,
                Δx̂_norm_prog,
                in_place=False,
            )

            # Renormalize prognostic predictions for next rollout step (normalized state space)
            y_pred_prog = (y_pred_phys_prog * norm_mul[input_prog_idx].float() + norm_add[input_prog_idx].float()).to(model_output.dtype)

            # ============================================================
            # Build full prediction tensor with prognostic + diagnostic
            # ============================================================
            # y_pred needs to have shape compatible with model output for _advance_input
            # which expects y_pred[..., model.output.prognostic]
            n_output = len(self.data_indices.model.output.full)
            y_pred = torch.zeros(
                *model_output.shape[:-1], n_output,
                dtype=model_output.dtype, device=model_output.device
            )
            y_pred[..., model_prog_idx] = y_pred_prog

            # Handle diagnostic variables if present (direct prediction, no residual)
            if len(model_diag_idx) > 0:
                # Diagnostics are predicted directly - keep as normalized for metrics
                y_pred[..., model_diag_idx] = model_output[..., model_diag_idx]

            # ============================================================
            # Loss in normalized RESIDUAL space (for backprop)
            # Only compute loss on prognostic variables (residual prediction)
            # ============================================================
            Δx̂_norm_full, Δx_true_norm_full, grid_shard_slice = self._prepare_tensors_for_loss(
                Δx̂_norm_prog,
                Δx_true_norm,
                validation_mode,
            )

            loss = checkpoint(
                self._compute_loss,
                Δx̂_norm_full,
                Δx_true_norm_full,
                grid_shard_slice,
                use_reentrant=False,
            )

            # ============================================================
            # Diagnostic STATE-space loss term (only when diagnostics exist).
            # Diagnostics are DIRECT normalized-state predictions; truth is the
            # InputNormalizer-normalized batch (σ_state baked in -> ~1.0 no-skill
            # baseline). Combined count-weighted per-variable-equal with L_prog:
            #   total = (N_prog·L_prog + N_diag·L_diag) / (N_prog + N_diag)
            # For N_diag == 0 this branch is skipped -> total == L_prog exactly.
            # ============================================================
            if self._n_diag > 0:
                yhat_diag = model_output[..., self._diag_out_idx]
                ytrue_diag = batch[
                    :,
                    self.multi_step + rollout_step,
                    ...,
                    self._diag_data_out_idx_buf,
                ]
                yhat_diag_full, ytrue_diag_full, diag_grid_shard_slice = self._prepare_tensors_for_loss(
                    yhat_diag,
                    ytrue_diag,
                    validation_mode,
                )
                loss_diag = checkpoint(
                    self._compute_diag_loss,
                    yhat_diag_full,
                    ytrue_diag_full,
                    diag_grid_shard_slice,
                    use_reentrant=False,
                )
                if not validation_mode:
                    self.log("train_loss/prognostic_residual", loss.detach().mean(),
                             on_step=True, on_epoch=False, logger=True, rank_zero_only=True)
                    self.log("train_loss/diagnostic_state", loss_diag.detach().mean(),
                             on_step=True, on_epoch=False, logger=True, rank_zero_only=True)
                loss = (self._n_prog * loss + self._n_diag * loss_diag) / (self._n_prog + self._n_diag)

            # Log per-variable-group losses to MLflow (every 250 steps, matching total loss interval)
            if hasattr(self.loss, '_last_per_group_losses') and self.loss._last_per_group_losses is not None:
                for group_name, group_loss in self.loss._last_per_group_losses.items():
                    self.log(
                        f"train_loss/{group_name}",
                        group_loss,
                        on_step=True,
                        on_epoch=False,
                        logger=True,
                        rank_zero_only=True,
                    )

            # ============================================================
            # Validation metrics in STATE space (not residual space!)
            # The post_processors expect tensors with shape matching data.output.full
            # so they can correctly index the normalization buffers.
            # ============================================================
            metrics_next = {}
            if validation_mode:
                # Get ground truth with full output shape (to match y_pred)
                y_true_for_metrics = batch[
                    :,
                    self.multi_step + rollout_step,
                    ...,
                    self.data_indices.data.output.full,
                ]

                # Pass full tensors (n_output shape) so post_processors can
                # correctly apply denormalization using _output_idx
                metrics_next = self.calculate_val_metrics(
                    y_pred,
                    y_true_for_metrics,
                    step=rollout_step,
                    grid_shard_slice=self.grid_shard_slice,
                )

            # Feed next-state prediction back into input window
            x = self._advance_input(x, y_pred, batch, rollout_step)

            yield loss, metrics_next, y_pred
