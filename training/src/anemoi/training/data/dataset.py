# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
import random
from collections.abc import Callable
from functools import cached_property

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import IterableDataset

from anemoi.training.data.grid_indices import BaseGridIndices
from anemoi.training.utils.seeding import get_base_seed
from anemoi.training.utils.usable_indices import get_usable_indices

LOGGER = logging.getLogger(__name__)


class NativeGridDataset(IterableDataset):
    """Iterable dataset for AnemoI data on the arbitrary grids."""

    def __init__(
        self,
        data_reader: Callable,
        grid_indices: type[BaseGridIndices],
        relative_date_indices: list,
        timestep: str = "6h",
        shuffle: bool = True,
        label: str = "generic",
        num_gpus_per_ens: int = 1,
        num_gpus_per_model: int = 1,
        trajectory_diverse_batching: bool = False,
        trajectory_filter: int | list[int] | None = None,
    ) -> None:
        """Initialize (part of) the dataset state.

        Parameters
        ----------
        data_reader : Callable
            user function that opens and returns the anemoi-datasets array data
        grid_indices : Type[BaseGridIndices]
            indices of the grid to keep. Defaults to None, which keeps all spatial indices.
        relative_date_indices: list
            list of time indices to load from the data relative to the current sample i in __iter__
        timestep : int, optional
            the time frequency of the samples, by default '6h'
        shuffle : bool, optional
            Shuffle batches, by default True
        label : str, optional
            label for the dataset, by default "generic"
        num_gpus_per_ens : int, optional
            Number of GPUs per ensemble, by default 1
        num_gpus_per_model : int, optional
            Number of GPUs per model, by default 1
        trajectory_diverse_batching : bool, optional
            Enable trajectory-diverse batching for better batch diversity, by default False
        """
        self.data = data_reader
        self.timestep = timestep
        self.grid_indices = grid_indices
        self.label = label
        self.trajectory_diverse_batching = trajectory_diverse_batching
        self.trajectory_filter = trajectory_filter
        self.relative_date_indices = relative_date_indices  # relative index of dates to extract

        self.num_gpus_per_ens = num_gpus_per_ens
        self.num_gpus_per_model = num_gpus_per_model

        # lazy init model and reader group info, will be set by the DDPGroupStrategy:
        self.model_comm_group_rank = 0
        self.model_comm_num_groups = 1
        self.model_comm_group_id = 0
        self.global_rank = 0

        self.reader_group_rank = 0
        self.reader_group_size = 1

        self.sample_comm_num_groups = 1  # groups that work on the same sample / batch
        self.sample_comm_group_id = 0

        self.ens_comm_group_rank = 0
        self.ens_comm_num_groups = 1
        self.ens_comm_group_id = 0

        # additional state vars (lazy init)
        self.n_samples_per_worker = 0
        self.chunk_index_range: np.ndarray | None = None
        self.shuffle = shuffle

    @cached_property
    def statistics(self) -> dict:
        """Return dataset statistics."""
        return self.data.statistics

    @cached_property
    def statistics_tendencies(self) -> dict:
        """Return dataset tendency statistics."""
        try:
            return self.data.statistics_tendencies(self.timestep)
        except (KeyError, AttributeError):
            return None

    @cached_property
    def metadata(self) -> dict:
        """Return dataset metadata."""
        return self.data.metadata()

    @cached_property
    def supporting_arrays(self) -> dict:
        """Return dataset supporting_arrays."""
        return self.data.supporting_arrays()

    @cached_property
    def name_to_index(self) -> dict:
        """Return dataset name_to_index mapping."""
        return self.data.name_to_index

    @cached_property
    def resolution(self) -> dict:
        """Return dataset resolution."""
        return self.data.resolution

    @cached_property
    def valid_date_indices(self) -> np.ndarray:
        """Return valid date indices.

        A date t is valid if we can sample the elements t + i
        for every relative_date_index i.
        """
        indices = get_usable_indices(
            self.data.missing,
            len(self.data),
            np.array(self.relative_date_indices, dtype=np.int64),
            self.data.trajectory_ids,
        )
        if self.trajectory_filter is not None and self.data.trajectory_ids is not None:
            traj_ids = self.data.trajectory_ids
            if isinstance(self.trajectory_filter, int):
                mask = traj_ids[indices] == self.trajectory_filter
            else:
                mask = np.isin(traj_ids[indices], self.trajectory_filter)
            LOGGER.info("Trajectory filter: keeping %d / %d samples (trajectory_filter=%s)",
                        mask.sum(), len(indices), self.trajectory_filter)
            indices = indices[mask]
        return indices

    def set_comm_group_info(
        self,
        global_rank: int,
        model_comm_group_id: int,
        model_comm_group_rank: int,
        model_comm_num_groups: int,
        reader_group_rank: int,
        reader_group_size: int,
    ) -> None:
        """Set model and reader communication group information (called by DDPGroupStrategy).

        Parameters
        ----------
        global_rank : int
            Global rank
        model_comm_group_id : int
            Model communication group ID
        model_comm_group_rank : int
            Model communication group rank
        model_comm_num_groups : int
            Number of model communication groups
        reader_group_rank : int
            Reader group rank
        reader_group_size : int
            Reader group size
        """
        self.global_rank = global_rank
        self.model_comm_group_id = model_comm_group_id
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

        self.sample_comm_group_id = model_comm_group_id
        self.sample_comm_num_groups = model_comm_num_groups

        assert self.reader_group_size >= 1, f"reader_group_size(={self.reader_group_size}) must be positive"

        LOGGER.info(
            "NativeGridDataset.set_group_info(): global_rank %d, model_comm_group_id %d, "
            "model_comm_group_rank %d, model_comm_num_groups %d, reader_group_rank %d",
            global_rank,
            model_comm_group_id,
            model_comm_group_rank,
            model_comm_num_groups,
            reader_group_rank,
        )

    def set_ens_comm_group_info(
        self,
        ens_comm_group_id: int,
        ens_comm_group_rank: int,
        ens_comm_num_groups: int,
    ) -> None:
        """Set ensemble communication group information (called by DDPGroupStrategy).

        Parameters
        ----------
        ens_comm_group_id : int
            Ensemble communication group ID
        ens_comm_group_rank : int
            Ensemble communication group rank
        ens_comm_num_groups : int
            Number of ensemble communication groups
        """
        self.ens_comm_group_id = ens_comm_group_id
        self.ens_comm_group_rank = ens_comm_group_rank
        self.ens_comm_num_groups = ens_comm_num_groups

        LOGGER.info(
            "NativeGridDataset.set_group_info(): global_rank %d, ens_comm_group_id %d, "
            "ens_comm_group_rank %d, ens_comm_num_groups %d, reader_group_rank %d",
            self.global_rank,
            ens_comm_group_id,
            ens_comm_group_rank,
            ens_comm_num_groups,
            self.reader_group_rank,
        )

    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        """Called by worker_init_func on each copy of dataset.

        This initialises after the worker process has been spawned.

        Parameters
        ----------
        n_workers : int
            Number of workers
        worker_id : int
            Worker ID

        """
        self.worker_id = worker_id

        # Divide this equally across shards (one shard per group!)
        shard_size = len(self.valid_date_indices) // self.sample_comm_num_groups
        shard_start = self.sample_comm_group_id * shard_size
        shard_end = (self.sample_comm_group_id + 1) * shard_size

        shard_len = shard_end - shard_start
        self.n_samples_per_worker = shard_len // n_workers

        low = shard_start + worker_id * self.n_samples_per_worker
        high = min(shard_start + (worker_id + 1) * self.n_samples_per_worker, shard_end)
        self.chunk_index_range = np.arange(low, high, dtype=np.uint32)

        LOGGER.info(
            "Worker %d (pid %d, global_rank %d, model comm group %d)  has low/high range %d / %d",
            worker_id,
            os.getpid(),
            self.global_rank,
            self.model_comm_group_id,
            low,
            high,
        )

        base_seed = get_base_seed()

        torch.manual_seed(base_seed)
        random.seed(base_seed)
        self.rng = np.random.default_rng(seed=base_seed)
        sanity_rnd = self.rng.random(1)

        LOGGER.info(
            (
                "Worker %d (%s, pid %d, glob. rank %d, model comm group %d, "
                "group_rank %d, seed group id %d, base_seed %d, sanity rnd %f)"
            ),
            worker_id,
            self.label,
            os.getpid(),
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
            self.sample_comm_group_id,
            base_seed,
            sanity_rnd,
        )

    def _get_trajectory_diverse_indices(self, chunk_indices: np.ndarray) -> np.ndarray:
        """Reorder indices to ensure consecutive samples come from different trajectories.

        This ensures that when the DataLoader batches consecutive samples, each batch
        contains samples from diverse forecast trajectories (different weather regimes).

        The algorithm:
        1. Group indices by trajectory ID
        2. Shuffle within each trajectory group
        3. Yield indices in round-robin fashion across trajectories

        Parameters
        ----------
        chunk_indices : np.ndarray
            The indices to reorder (already assigned to this worker's chunk)

        Returns
        -------
        np.ndarray
            Reordered indices with trajectory diversity
        """
        traj_ids = getattr(self.data, 'trajectory_ids', None)
        if traj_ids is None:
            LOGGER.debug("No trajectory_ids available, falling back to standard shuffle")
            return chunk_indices

        # Group chunk indices by their trajectory ID
        traj_to_indices = {}
        for idx in chunk_indices:
            traj_id = traj_ids[idx]
            if traj_id not in traj_to_indices:
                traj_to_indices[traj_id] = []
            traj_to_indices[traj_id].append(idx)

        # Shuffle within each trajectory group
        for traj_id in traj_to_indices:
            self.rng.shuffle(traj_to_indices[traj_id])

        # Get list of trajectories and shuffle their order
        trajectory_list = list(traj_to_indices.keys())
        self.rng.shuffle(trajectory_list)

        # Round-robin across trajectories to create diverse ordering
        diverse_indices = []
        traj_iterators = {t: iter(indices) for t, indices in traj_to_indices.items()}
        active_trajs = list(trajectory_list)

        while active_trajs:
            # Take one sample from each active trajectory
            next_round_trajs = []
            for traj_id in active_trajs:
                try:
                    idx = next(traj_iterators[traj_id])
                    diverse_indices.append(idx)
                    next_round_trajs.append(traj_id)
                except StopIteration:
                    # This trajectory is exhausted
                    pass
            active_trajs = next_round_trajs

        diverse_indices = np.array(diverse_indices, dtype=chunk_indices.dtype)

        LOGGER.debug(
            "Trajectory-diverse batching (worker %d): %d indices from %d trajectories",
            getattr(self, 'worker_id', 0),
            len(diverse_indices),
            len(traj_to_indices),
        )

        return diverse_indices

    def __iter__(self) -> torch.Tensor:
        """Return an iterator over the dataset.

        The datasets are retrieved by anemoi.datasets from anemoi datasets. This iterator yields
        chunked batches for DDP and sharded training.

        Currently it receives data with an ensemble dimension, which is discarded for
        now. (Until the code is "ensemble native".)
        """
        if self.shuffle:
            shuffled_chunk_indices = self.rng.choice(
                self.valid_date_indices,
                size=len(self.valid_date_indices),
                replace=False,
            )[self.chunk_index_range]

            # Apply trajectory-diverse reordering if enabled
            if self.trajectory_diverse_batching:
                shuffled_chunk_indices = self._get_trajectory_diverse_indices(shuffled_chunk_indices)
        else:
            shuffled_chunk_indices = self.valid_date_indices[self.chunk_index_range]

        LOGGER.debug(
            (
                "Worker pid %d, label %s, worker id %d, global_rank %d, "
                "model comm group %d, group_rank %d, seed comm group id %d, using indices[0:10]: %s"
            ),
            os.getpid(),
            self.label,
            self.worker_id,
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
            self.sample_comm_group_id,
            shuffled_chunk_indices[:10],
        )

        for sample_count, i in enumerate(shuffled_chunk_indices):
            start = i + self.relative_date_indices[0]
            end = i + self.relative_date_indices[-1] + 1
            timeincrement = self.relative_date_indices[1] - self.relative_date_indices[0]
            # NOTE: this is temporary until anemoi datasets allows indexing with arrays or lists
            # data[start...] will be replaced with data[self.relative_date_indices + i]

            grid_shard_indices = self.grid_indices.get_shard_indices(self.reader_group_rank)
            if isinstance(grid_shard_indices, slice):
                # Load only shards into CPU memory
                x = self.data[start:end:timeincrement, :, :, grid_shard_indices]

            else:
                # Load full grid in CPU memory, select grid_shard after
                # Note that anemoi-datasets currently doesn't support slicing + indexing
                # in the same operation.
                x = self.data[start:end:timeincrement, :, :, :]
                x = x[..., grid_shard_indices]  # select the grid shard
            x = rearrange(x, "dates variables ensemble gridpoints -> dates ensemble gridpoints variables")
            self.ensemble_dim = 1

            # Convert to torch tensor
            x_tensor = torch.from_numpy(x)

            # Diagnostic plot: save first sample's first channel as 2D image
            if sample_count == 0 and self.label == "train" and self.global_rank == 0:
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    # x_tensor shape: [dates, ensemble, gridpoints, variables]
                    input_field = x_tensor[0, 0, :, 0].numpy()  # first timestep, first channel
                    target_field = x_tensor[-1, 0, :, 0].numpy()  # last timestep, first channel

                    grid_shape = (445, 595)

                    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                    inp_2d = input_field.reshape(grid_shape)
                    tgt_2d = target_field.reshape(grid_shape)
                    diff_2d = tgt_2d - inp_2d
                    im0 = axes[0].imshow(inp_2d, origin="lower", aspect="auto")
                    plt.colorbar(im0, ax=axes[0])
                    im1 = axes[1].imshow(tgt_2d, origin="lower", aspect="auto")
                    plt.colorbar(im1, ax=axes[1])
                    im2 = axes[2].imshow(diff_2d, origin="lower", aspect="auto", cmap="RdBu_r")
                    plt.colorbar(im2, ax=axes[2])
                    axes[0].set_title(f"Input (t=0), ch0, normalized")
                    axes[1].set_title(f"Target (t+1), ch0, normalized")
                    axes[2].set_title(f"Target - Input, ch0")
                    plt.suptitle(f"label={self.label}, idx={i}, npts={len(input_field)}")
                    plt.tight_layout()
                    plt.savefig("/home/mflora/dataloader_diagnostic.png", dpi=150)
                    plt.close()
                    LOGGER.info("Saved dataloader diagnostic plot to /home/mflora/dataloader_diagnostic.png")
                except Exception as e:
                    LOGGER.warning("Failed to save diagnostic plot: %s", e)

            yield x_tensor

    def __repr__(self) -> str:
        return f"""
            {super().__repr__()}
            Dataset: {self.data}
            Relative dates: {self.relative_date_indices}
        """
