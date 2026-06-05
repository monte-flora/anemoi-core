# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Ensemble + residual encoder–processor–decoder.

Minimal extension of ``AnemoiEnsModelEncProcDec`` for **residual-prediction
ensemble training** (e.g. graphcast_gnn family + CRPS + anchored MSE).

The parent ``AnemoiEnsModelEncProcDec``:
  - folds the ensemble dim into the batch dim,
  - instantiates a ``noise_injector`` between encoder and processor,
  - passes the resulting per-mesh-node noise to the processor as
    ``cond`` for conditional layer-norm conditioning,
  - then in ``_assemble_output`` adds the input prognostic-skip
    ``x_skip[..., input_prog_idx]`` to the output prognostic slots
    (state-mode skip).

For RESIDUAL mode the state-mode skip is wrong — the residual reconstruction
happens at the task level (``GraphEnsResidualForecaster._rollout_step`` uses
``residual_normalizer.inverse_transform`` on the model output, treating it
as a normalised tendency). So we override ``_assemble_output`` to drop the
skip add and leave the model output untouched (bounding still runs).

We also surface a ``forward_with_spatial_noise`` alias because
``GraphEnsResidualForecaster._rollout_step`` keys its noise-dispatch on
``hasattr(model, "forward_with_spatial_noise")`` (originally added for the
AIFS-style DiT path). The alias just routes to ``forward()`` with
``fcstep=0``, which matches GNN-residual semantics (no forecast-step
embedding, since the task drives multi-step rollout externally).
"""

from __future__ import annotations

import logging
from typing import Optional

import einops
import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.models.ens_encoder_processor_decoder import AnemoiEnsModelEncProcDec

LOGGER = logging.getLogger(__name__)


class AnemoiEnsResidualModelEncProcDec(AnemoiEnsModelEncProcDec):
    """Ensemble encoder-processor-decoder configured for residual prediction.

    Drop-in replacement for ``AnemoiEnsModelEncProcDec`` when used with
    ``GraphEnsResidualForecaster``. Required differences from the parent:

    1. No state-mode ``x_skip`` add in ``_assemble_output`` — the task
       computes the physical-state reconstruction itself using the
       ``ResidualNormalizer`` (so the model output stays as a normalised
       residual, as the task expects).
    2. Exposes ``forward_with_spatial_noise(x)`` so the task's existing
       noise-path dispatch (added for DiT-AIFS) routes here without a
       second branch.
    """

    def _assemble_output(
        self,
        x_out: torch.Tensor,
        x_skip: torch.Tensor,
        batch_size: int,
        batch_ens_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Reshape (B*E, n, f) → (B, E, n, f); apply boundings.

        Critically, do NOT add ``x_skip`` to the prognostic slots. That add
        is the state-mode skip that the parent class performs; for residual
        mode the model output is interpreted as a residual by the task and
        adding the skip here would double-count.
        """
        x_out = einops.rearrange(x_out, "(bse n) f -> bse n f", bse=batch_ens_size)
        x_out = (
            einops.rearrange(x_out, "(bs e) n f -> bs e n f", bs=batch_size)
            .to(dtype=dtype)
            .clone()
        )

        # Residual mode: skip the state-add. The task
        # (GraphEnsResidualForecaster._rollout_step) reconstructs physical
        # state via residual_normalizer.inverse_transform on this output.
        #
        # NOTE: ``x_skip`` is still received from the parent's _assemble_input
        # because ``forward`` passes it through, but we deliberately do not
        # use it here. Keeping the signature unchanged keeps us as a true
        # drop-in for the parent.

        for bounding in self.boundings:
            x_out = bounding(x_out)
        return x_out

    def forward_with_spatial_noise(
        self,
        x: torch.Tensor,
        *,
        fcstep: int = 0,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Alias used by ``GraphEnsResidualForecaster`` to detect ensemble
        noise-conditioned models. Routes to the standard ``forward()``.

        The task code (anemoi-core/training/.../tasks/ensresidualforecaster.py
        line ~155 in the dispatch I added during the v31 AIFS-DiT work) checks
        for this method's existence to decide whether to thread an explicit
        noise vector through. Graph-ensemble models inject noise INSIDE
        ``forward()`` via ``self.noise_injector`` (see parent class) so no
        explicit noise argument is needed — this alias makes the dispatch a
        no-op for us.
        """
        return self.forward(
            x,
            fcstep=fcstep,
            model_comm_group=model_comm_group,
            grid_shard_shapes=grid_shard_shapes,
            **kwargs,
        )

    @staticmethod
    def _get_normalizer_buffers(pre_processors):
        """Same accessor as AnemoiResidualModelEncProcDec — finds the
        InputNormalizer in pre_processors and returns (norm_mul, norm_add).
        """
        for processor in pre_processors.processors.values():
            if hasattr(processor, "_norm_mul") and hasattr(processor, "_norm_add"):
                return processor._norm_mul, processor._norm_add
        raise RuntimeError("InputNormalizer buffers not found in pre_processors.")

    def predict_step(
        self,
        batch: torch.Tensor,
        pre_processors,
        post_processors,
        residual_normalizer,
        data_indices: dict,
        multi_step: int,
        model_comm_group: Optional[ProcessGroup] = None,
        gather_out: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Residual-prediction inference, ensemble-aware.

        Mirrors ``AnemoiResidualModelEncProcDec.predict_step`` but
        - calls ``forward_with_spatial_noise`` (so noise is injected per call
          inside ``self.noise_injector`` — fresh per AR step, AIFS convention),
        - keeps the ensemble dim alive throughout: input is ``(B, T, E=1, G, V)``
          by default (single-member inference, like the deterministic path),
          but an ensemble dim with E>1 also works if the caller pre-replicates.

        Output shape: ``(B, G, V_out)`` after squeezing the singleton ensemble
        dim, matching the deterministic-residual predict_step contract.
        """
        from anemoi.models.distributed.graph import gather_tensor
        from anemoi.models.distributed.graph import shard_tensor
        from anemoi.models.distributed.shapes import apply_shard_shapes
        from anemoi.models.distributed.shapes import get_shard_shapes

        with torch.no_grad():
            assert len(batch.shape) == 4, (
                f"Expected 4-D batch (B, T, G, V); got {batch.shape}"
            )

            # Add singleton ensemble dim as 3rd index. Clone so pre_processors
            # in-place normalisation doesn't corrupt the caller's tensor.
            x = batch[:, 0:multi_step, None, ...].clone()  # (B, T, 1, G, V)

            grid_shard_shapes = None
            if model_comm_group is not None:
                shard_shapes = get_shard_shapes(x, -2, model_comm_group)
                grid_shard_shapes = [shape[-2] for shape in shard_shapes]
                x = shard_tensor(x, -2, shard_shapes, model_comm_group)

            # Normalise input
            x = pre_processors(x, in_place=True)

            # Forward → (B, E, G, V_out). The noise_injector samples fresh
            # noise INSIDE forward_with_spatial_noise so this is one AIFS
            # ensemble member per call.
            model_output = self.forward_with_spatial_noise(
                x,
                fcstep=0,
                model_comm_group=model_comm_group,
                grid_shard_shapes=grid_shard_shapes,
                **kwargs,
            )

            # Variable indices
            model_prog_idx = data_indices.model.output.prognostic
            model_diag_idx = data_indices.model.output.diagnostic
            input_prog_idx = data_indices.data.input.prognostic

            # Reconstruct physical state via residual_normalizer
            norm_mul, norm_add = self._get_normalizer_buffers(pre_processors)

            # Prognostic: residual reconstruction
            delta_norm_prog = model_output[..., model_prog_idx]      # (B, E, G, n_prog)
            x_last_norm_prog = x[:, -1, ..., input_prog_idx]         # (B, E, G, n_prog)
            y_hat_prog_phys = residual_normalizer.inverse_transform_physical_from_normalized(
                x_last_norm_prog, delta_norm_prog, norm_mul, norm_add,
            )

            # Assemble full output (prognostic + diagnostic)
            n_output = len(data_indices.model.output.full)
            batch_size = model_output.shape[0]
            ensemble_size = model_output.shape[1]
            grid_size = model_output.shape[2]
            y_hat = torch.zeros(
                batch_size, ensemble_size, grid_size, n_output,
                dtype=model_output.dtype, device=model_output.device,
            )
            y_hat[..., model_prog_idx] = y_hat_prog_phys

            # Diagnostic vars (predicted directly, not residual)
            if len(model_diag_idx) > 0:
                diag_out_norm = model_output[..., model_diag_idx]
                input_diag_idx = getattr(data_indices.data.input, "diagnostic", [])
                if len(input_diag_idx) > 0:
                    diag_mul = norm_mul[input_diag_idx].float()
                    diag_add = norm_add[input_diag_idx].float()
                    diag_phys = (diag_out_norm.float() - diag_add) / diag_mul
                    y_hat[..., model_diag_idx] = diag_phys.to(model_output.dtype)
                else:
                    y_hat[..., model_diag_idx] = diag_out_norm

            # Squeeze E=1 to match the deterministic predict_step contract
            # ((B, G, V_out)). If E>1 we keep the ensemble dim — caller can
            # set squeeze=False via kwargs if needed (future extension).
            if y_hat.shape[1] == 1:
                y_hat = y_hat.squeeze(1)

            if gather_out and model_comm_group is not None:
                y_hat = gather_tensor(
                    y_hat, -2,
                    apply_shard_shapes(y_hat, -2, grid_shard_shapes),
                    model_comm_group,
                )

        return y_hat
