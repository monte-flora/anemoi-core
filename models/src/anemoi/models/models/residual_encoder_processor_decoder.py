# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Optional

import einops
import numpy as np
import torch
from hydra.utils import instantiate
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import gather_channels
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_channels
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import get_or_apply_shard_shapes
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.models.layers.mapper import GraphTransformerBaseMapper
from anemoi.utils.config import DotDict

from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.preprocessing.residual_normalizer import ResidualNormalizer 

LOGGER = logging.getLogger(__name__)


class AnemoiResidualModelEncProcDec(AnemoiModelEncProcDec):
    """Message passing graph neural network with a residual prediction."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        statistics: dict,
        graph_data: HeteroData,
        **kwargs,
    ) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        model_config : DotDict
            Model configuration
        data_indices : dict
            Data indices
        graph_data : HeteroData
            Graph definition
        **kwargs
            Additional arguments (ignored for compatibility)
        """
        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
        )
        
    def _assemble_input(self, x, batch_size, grid_shard_shapes=None, model_comm_group=None):
        node_attributes_data = self.node_attributes(self._graph_name_data, batch_size=batch_size)
        if grid_shard_shapes is not None:
            shard_shapes_nodes = get_or_apply_shard_shapes(
                node_attributes_data, 0, shard_shapes_dim=grid_shard_shapes, model_comm_group=model_comm_group
            )
            node_attributes_data = shard_tensor(node_attributes_data, 0, shard_shapes_nodes, model_comm_group)

        # normalize and add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                node_attributes_data,
            ),
            dim=-1,  # feature dimension
        )
        shard_shapes_data = get_or_apply_shard_shapes(
            x_data_latent, 0, shard_shapes_dim=grid_shard_shapes, model_comm_group=model_comm_group
        )

        return x_data_latent, shard_shapes_data    
    
    def _assemble_output(self, x_out, batch_size, ensemble_size, dtype):
        x_out = (
            einops.rearrange(
                x_out,
                "(batch ensemble grid) vars -> batch ensemble grid vars",
                batch=batch_size,
                ensemble=ensemble_size,
            )
            .to(dtype=dtype)
            .clone()
        )

        # Note: 9 Nov 2025, commented out the residual connection
        # residual connection (just for the prognostic variables)
        #x_out[..., self._internal_output_idx] += x_skip[..., self._internal_input_idx]

        for bounding in self.boundings:
            # bounding performed in the order specified in the config file
            x_out = bounding(x_out)
            
        return x_out

    @staticmethod
    def _get_normalizer_buffers(pre_processors: nn.Module) -> tuple[Tensor, Tensor]:
        for processor in pre_processors.processors.values():
            if hasattr(processor, "_norm_mul") and hasattr(processor, "_norm_add"):
                return processor._norm_mul, processor._norm_add
        raise RuntimeError("InputNormalizer buffers not found in pre_processors.")
    
    def forward(
        self,
        x: Tensor,
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass of the model.

        No skip connections. 

        Parameters
        ----------
        x : Tensor
            Input data
        model_comm_group : Optional[ProcessGroup], optional
            Model communication group, by default None
        grid_shard_shapes : list, optional
            Shard shapes of the grid, by default None

        Returns
        -------
        Tensor
            Output of the model, with the same shape as the input (sharded if input is sharded)
        """
        batch_size = x.shape[0]
        ensemble_size = x.shape[2]
        in_out_sharded = grid_shard_shapes is not None
        self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded, model_comm_group)

        x_data_latent, shard_shapes_data = self._assemble_input(
            x, batch_size, grid_shard_shapes, model_comm_group
        )

        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)
        shard_shapes_hidden = get_shard_shapes(x_hidden_latent, 0, model_comm_group)

        # Encoder
        x_data_latent, x_latent = self.encoder(
            (x_data_latent, x_hidden_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_data, shard_shapes_hidden),
            model_comm_group=model_comm_group,
            x_src_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
            x_dst_is_sharded=False,  # x_latent does not come sharded
            keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
        )

        # Processor
        x_latent_proc = self.processor(
            x_latent,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        # 9 Nov 2025: removing the skip connection. 
        # Skip
        # x_latent_proc = x_latent_proc + x_latent

        # Decoder
        x_out = self.decoder(
            (x_latent_proc, x_data_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_hidden, shard_shapes_data),
            model_comm_group=model_comm_group,
            x_src_is_sharded=True,  # x_latent always comes sharded
            x_dst_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
            keep_x_dst_sharded=in_out_sharded,  # keep x_out sharded iff in_out_sharded
        )

        x_out = self._assemble_output(x_out, batch_size, ensemble_size, x.dtype)

        return x_out
    
    def predict_step(
        self,
        batch: torch.Tensor,
        pre_processors: nn.Module,
        post_processors: nn.Module,
        residual_normalizer: nn.Module,
        data_indices: dict,
        multi_step: int,
        model_comm_group: Optional[ProcessGroup] = None,
        gather_out: bool = True,
        **kwargs,
    ) -> Tensor:
        """Prediction step for the residual model.

        The model predicts normalized residuals for prognostic variables.
        Diagnostic variables (if any) are predicted directly.

        Parameters
        ----------
        batch : torch.Tensor
            Input batched data (before pre-processing), shape (batch, timesteps, grid, variables)
        pre_processors : nn.Module
            Pre-processing module (normalizer)
        post_processors : nn.Module
            Post-processing module (denormalizer)
        residual_normalizer : nn.Module
            Residual normalizer for converting residuals to physical space
        data_indices : dict
            Data indices for variable mapping
        multi_step : int
            Number of input timesteps
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training
        gather_out : bool
            Whether to gather output tensors across distributed processes
        **kwargs
            Additional arguments

        Returns
        -------
        Tensor
            Model output in physical space, shape (batch, grid, n_output)
        """
        from anemoi.models.distributed.shapes import apply_shard_shapes

        with torch.no_grad():

            assert (
                len(batch.shape) == 4
            ), f"The input tensor has an incorrect shape: expected a 4-dimensional tensor, got {batch.shape}!"

            # Dimensions are: batch, timesteps, grid, variables
            # Add dummy ensemble dimension as 3rd index
            # Clone to avoid corrupting the caller's tensor when pre_processors
            # normalizes in-place (the slice+None creates a view that shares storage).
            x = batch[:, 0:multi_step, None, ...].clone()  # shape: (batch, time, 1, grid, n_input)

            # Handle distributed processing
            grid_shard_shapes = None
            if model_comm_group is not None:
                shard_shapes = get_shard_shapes(x, -2, model_comm_group)
                grid_shard_shapes = [shape[-2] for shape in shard_shapes]
                x = shard_tensor(x, -2, shard_shapes, model_comm_group)

            # ============================================================
            # Step 1: Normalize the input
            # ============================================================
            x = pre_processors(x, in_place=True)

            # ============================================================
            # Step 2: Forward pass - model predicts normalized residuals
            # ============================================================
            # Output shape: (batch, ensemble=1, grid, n_output)
            model_output = self.forward(x, model_comm_group=model_comm_group, grid_shard_shapes=grid_shard_shapes, **kwargs)

            # Get indices for prognostic and diagnostic variables
            # model.output indices are for the model output tensor
            model_prog_idx = data_indices.model.output.prognostic
            model_diag_idx = data_indices.model.output.diagnostic
            # data.input indices are for the input/batch tensor (used to index normalizer buffers)
            input_prog_idx = data_indices.data.input.prognostic

            # Get normalizer buffers
            norm_mul, norm_add = self._get_normalizer_buffers(pre_processors)

            # ============================================================
            # Step 3: Handle PROGNOSTIC variables (residual prediction)
            # ============================================================
            # Extract prognostic residuals from model output
            # model_output shape: (batch, ensemble=1, grid, n_output)
            Δx̂_norm_prog = model_output[..., model_prog_idx]  # (batch, 1, grid, n_prog)

            # Get last normalized input state for prognostic variables
            # x shape: (batch, time, ensemble=1, grid, n_input)
            x_last_norm_prog = x[:, -1, ..., input_prog_idx]  # (batch, 1, grid, n_prog)

            # Reconstruct prognostic variables in physical space
            # inverse_transform_physical_from_normalized expects inputs with same shape
            y_hat_prog_phys = residual_normalizer.inverse_transform_physical_from_normalized(
                x_last_norm_prog,
                Δx̂_norm_prog,
                norm_mul,
                norm_add,
            )  # (batch, 1, grid, n_prog)

            # ============================================================
            # Step 4: Handle DIAGNOSTIC variables (direct prediction)
            # ============================================================
            n_output = len(data_indices.model.output.full)
            batch_size = model_output.shape[0]
            ensemble_size = model_output.shape[1]
            grid_size = model_output.shape[2]

            # Initialize output tensor in physical space
            y_hat = torch.zeros(
                batch_size, ensemble_size, grid_size, n_output,
                dtype=model_output.dtype, device=model_output.device
            )

            # Place prognostic predictions
            y_hat[..., model_prog_idx] = y_hat_prog_phys

            # Handle diagnostic variables if present
            if len(model_diag_idx) > 0:
                # Diagnostic variables are predicted directly (not as residuals)
                # They need to be denormalized using the standard post-processor
                diag_output_norm = model_output[..., model_diag_idx]
                # For diagnostics, apply standard denormalization
                # Create a temporary tensor with just diagnostics for post-processing
                # Note: post_processors expect full output shape, so we apply denorm manually
                input_diag_idx = data_indices.data.input.diagnostic if hasattr(data_indices.data.input, 'diagnostic') else []
                if len(input_diag_idx) > 0:
                    diag_mul = norm_mul[input_diag_idx].float()
                    diag_add = norm_add[input_diag_idx].float()
                    y_hat_diag_phys = (diag_output_norm.float() - diag_add) / diag_mul
                    y_hat[..., model_diag_idx] = y_hat_diag_phys.to(model_output.dtype)
                else:
                    # If no input diagnostic indices, just pass through
                    y_hat[..., model_diag_idx] = diag_output_norm

            # ============================================================
            # Step 5: Remove ensemble dimension and gather if needed
            # ============================================================
            # Squeeze ensemble dimension: (batch, 1, grid, n_output) -> (batch, grid, n_output)
            y_hat = y_hat.squeeze(1)

            # Gather output if needed for distributed processing
            if gather_out and model_comm_group is not None:
                y_hat = gather_tensor(y_hat, -2, apply_shard_shapes(y_hat, -2, grid_shard_shapes), model_comm_group)

        return y_hat
