# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import importlib
import io
import logging
import pickle
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from pytorch_lightning import Callback
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer

from anemoi.models.migrations import Migrator
from anemoi.training.train.tasks.base import BaseGraphModule
from anemoi.utils.checkpoints import save_metadata

chunking_fix_migration = importlib.import_module("anemoi.models.migrations.scripts.1762857428_chunking_fix").migrate

LOGGER = logging.getLogger(__name__)


def load_and_prepare_model(lightning_checkpoint_path: str) -> tuple[torch.nn.Module, dict]:
    """Load the lightning checkpoint and extract the pytorch model and its metadata.

    Parameters
    ----------
    lightning_checkpoint_path : str
        path to lightning checkpoint

    Returns
    -------
    tuple[torch.nn.Module, dict]
        pytorch model, metadata

    """
    module = BaseGraphModule.load_from_checkpoint(lightning_checkpoint_path, weights_only=False)
    model = module.model

    metadata = dict(**model.metadata)
    model.metadata = None
    model.config = None

    return model, metadata


def save_inference_checkpoint(model: torch.nn.Module, metadata: dict, save_path: Path | str) -> Path:
    """Save a pytorch checkpoint for inference with the model metadata.

    Parameters
    ----------
    model : torch.nn.Module
        Pytorch model
    metadata : dict
        Anemoi Metadata to inject into checkpoint
    save_path : Path | str
        Directory to save anemoi checkpoint

    Returns
    -------
    Path
        Path to saved checkpoint
    """
    save_path = Path(save_path)
    inference_filepath = save_path.parent / f"inference-{save_path.name}"

    torch.save(model, inference_filepath)
    save_metadata(inference_filepath, metadata)
    return inference_filepath


# Normalization / index buffers are computed from the NEW dataset's statistics and
# variable layout; they must always come from the new model, never be transferred or
# name-mapped from the source (doing so corrupts normalization).
_TRANSFER_KEEP_NEW = ("pre_processors", "post_processors", "residual_normalizer")


def _di_ordered_names(data_indices, which: str) -> list[str]:
    """Variable names in channel order for `which` in {'input','output'}."""
    n2i = data_indices.name_to_index
    i2n = {int(v): k for k, v in n2i.items()}
    return [i2n[int(x)] for x in list(getattr(data_indices.data, which).full)]


def _name_map_channels(src_w: torch.Tensor, tgt_w: torch.Tensor, dim: int, new_names: list[str], old_idx: dict) -> torch.Tensor:
    """Copy source channel-slices into the target by variable NAME along `dim`.

    Shared variables get the source weights; genuinely-new channels stay zero (so a
    new input contributes nothing to the trunk and a new output predicts 0 / the
    climatological mean initially).
    """
    out = torch.zeros_like(tgt_w)
    new_pos = [i for i, nm in enumerate(new_names) if nm in old_idx]
    src_pos = [old_idx[nm] for nm in new_names if nm in old_idx]
    if new_pos:
        out.index_copy_(
            dim,
            torch.tensor(new_pos, dtype=torch.long, device=out.device),
            src_w.index_select(dim, torch.tensor(src_pos, dtype=torch.long, device=src_w.device)),
        )
    return out


def remap_state_dict_for_transfer(
    state_dict: dict,
    model_state_dict: dict,
    old_data_indices: Any = None,
    new_data_indices: Any = None,
) -> tuple[dict, int]:
    """Filter/remap a source state_dict onto a target model's shapes (pure function).

    For each tensor present in the target model:
      * shape-identical -> passed through unchanged (direct load);
      * shape differs in exactly ONE dim whose (old,new) sizes equal the (old,new)
        INPUT or OUTPUT variable counts -> NAME-MAPPED along that dim: source weights
        for variables present in both models are copied to their (possibly reordered)
        new positions, genuinely-new channels are zero-inited;
      * anything else (multi-dim mismatch, dim size not matching a variable count, a
        normalization/index buffer, or data_indices unavailable) -> DROPPED, so the
        target model keeps its own freshly-initialised tensor (the historical behaviour).

    Returns the (possibly reduced) state_dict to load with ``strict=False`` and the
    number of tensors name-mapped. Pure and data-driven (no model/ckpt/IO), so it is
    unit-testable and generalises to adding/removing inputs or outputs in any project.
    """
    sd = dict(state_dict)
    can_map = old_data_indices is not None and new_data_indices is not None
    if can_map:
        old_in, new_in = _di_ordered_names(old_data_indices, "input"), _di_ordered_names(new_data_indices, "input")
        old_out, new_out = _di_ordered_names(old_data_indices, "output"), _di_ordered_names(new_data_indices, "output")
        old_in_idx = {n: i for i, n in enumerate(old_in)}
        old_out_idx = {n: i for i, n in enumerate(old_out)}

    n_mapped = 0
    for key in list(sd):
        if key not in model_state_dict or sd[key].shape == model_state_dict[key].shape:
            continue  # absent (loaded loosely) or shape-identical (direct load)

        sw, tw = sd[key], model_state_dict[key]
        mapped = None
        if (
            can_map
            and not any(s in key for s in _TRANSFER_KEEP_NEW)
            and tw.is_floating_point()
            and sw.dim() == tw.dim()
        ):
            diffs = [d for d in range(tw.dim()) if sw.shape[d] != tw.shape[d]]
            if len(diffs) == 1:
                d = diffs[0]
                if sw.shape[d] == len(old_in) and tw.shape[d] == len(new_in):
                    mapped = _name_map_channels(sw, tw, d, new_in, old_in_idx)
                elif sw.shape[d] == len(old_out) and tw.shape[d] == len(new_out):
                    mapped = _name_map_channels(sw, tw, d, new_out, old_out_idx)

        if mapped is not None:
            sd[key] = mapped
            n_mapped += 1
            LOGGER.info("Name-mapped channel transfer: %s  %s -> %s", key, tuple(sw.shape), tuple(tw.shape))
        else:
            LOGGER.info("Skipping loading parameter (re-init): %s  ckpt %s vs model %s", key, tuple(sw.shape), tuple(tw.shape))
            del sd[key]
    return sd, n_mapped


def transfer_learning_loading(model: torch.nn.Module, ckpt_path: Path | str) -> nn.Module:
    # Load the checkpoint
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=model.device)

    # apply chunking migration (fails silently otherwise leading to hard to debug issues)
    # this is due to loading with strict=False, planning to make this more robust in the future
    checkpoint = chunking_fix_migration(checkpoint)

    # Name-map shape-mismatched channel tensors (added/removed inputs or outputs) instead
    # of dropping them, so a warm-start across a variable-set change preserves the source
    # model's encoding/decoding for shared variables. See remap_state_dict_for_transfer.
    state_dict, n_mapped = remap_state_dict_for_transfer(
        checkpoint["state_dict"],
        model.state_dict(),
        checkpoint.get("hyper_parameters", {}).get("data_indices"),
        getattr(model, "data_indices", None),
    )
    if n_mapped:
        LOGGER.info("Transfer learning: %d tensors name-mapped by variable across a channel change.", n_mapped)

    # Load the filtered state_dict into the model
    model.load_state_dict(state_dict, strict=False)
    # Needed for data indices check
    model._ckpt_model_name_to_index = checkpoint["hyper_parameters"]["data_indices"].name_to_index
    return model


def freeze_submodule_by_name(module: nn.Module, target_name: str) -> None:
    """Recursively freezes the parameters of a submodule with the specified name.

    Parameters
    ----------
    module : torch.nn.Module
        Pytorch model
    target_name : str
        The name of the submodule to freeze.
    """
    for name, child in module.named_children():
        # If this is the target submodule, freeze its parameters
        if name == target_name:
            for param in child.parameters():
                param.requires_grad = False
        else:
            # Recursively search within children
            freeze_submodule_by_name(child, target_name)


class LoggingUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> str:
        if "anemoi.training" in module:
            msg = (
                f"anemoi-training Pydantic schemas found in model's metadata: "
                f"({module}, {name}) Please review Pydantic schemas to avoid this."
            )
            raise ValueError(msg)
        return super().find_class(module, name)


def check_classes(model: torch.nn.Module) -> None:
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)
    _ = LoggingUnpickler(buffer).load()


class RegisterMigrations(Callback):
    """Callback that register all existing migrations to a checkpoint before storing it."""

    def __init__(self):
        self.migrator = Migrator()

    def on_save_checkpoint(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        checkpoint: dict[str, Any],
    ) -> None:
        self.migrator.register_migrations(checkpoint)
