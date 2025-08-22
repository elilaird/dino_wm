import abc
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Sequence, List
from torch.utils.data import Dataset, Subset
from torch import default_generator, randperm
from einops import rearrange


# https://github.com/JaidedAI/EasyOCR/issues/1243
def _accumulate(iterable, fn=lambda x, y: x + y):
    "Return running totals"
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total


class TrajDataset(Dataset, abc.ABC):
    @abc.abstractmethod
    def get_seq_length(self, idx):
        """
        Returns the length of the idx-th trajectory.
        """
        raise NotImplementedError


class TrajSubset(TrajDataset, Subset):
    """
    Subset of a trajectory dataset at specified indices.

    Args:
        dataset (TrajectoryDataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset: TrajDataset, indices: Sequence[int]):
        Subset.__init__(self, dataset, indices)
        self.dataset = dataset
        self.proprio_dim = dataset.proprio_dim
        self.action_dim = dataset.action_dim
        self.state_dim = dataset.state_dim
        self.state_mean = dataset.state_mean
        self.state_std = dataset.state_std
        self.action_mean = dataset.action_mean
        self.action_std = dataset.action_std
        self.proprio_mean = dataset.proprio_mean
        self.proprio_std = dataset.proprio_std
        self.transform = dataset.transform

    def get_seq_length(self, idx):
        return self.dataset.get_seq_length(self.indices[idx])

    # def __getattr__(self, name):
    #     # Use getattr with a default to avoid infinite recursion
    #     try:
    #         return getattr(self.dataset, name)
    #     except AttributeError:
    #         raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class TrajSlicerDataset(TrajDataset):
    def __init__(
        self,
        dataset: TrajDataset,
        num_frames: int,
        frameskip: int = 1,
        process_actions: str = "concat",
    ):
        self.dataset = dataset
        self.num_frames = num_frames
        self.frameskip = frameskip
        self.slices = []
        for i in range(len(self.dataset)):
            T = self.dataset.get_seq_length(i)
            if T - num_frames < 0:
                print(
                    f"Ignored short sequence #{i}: len={T}, num_frames={num_frames}"
                )
            else:
                self.slices += [
                    (i, start, start + num_frames * self.frameskip)
                    for start in range(T - num_frames * frameskip + 1)
                ]  # slice indices follow convention [start, end)
        # randomly permute the slices
        self.slices = np.random.permutation(self.slices)

        self.proprio_dim = self.dataset.proprio_dim
        if process_actions == "concat":
            self.action_dim = self.dataset.action_dim * self.frameskip
        else:
            self.action_dim = self.dataset.action_dim

        self.state_dim = self.dataset.state_dim
        self.action_mean = self.dataset.action_mean
        self.action_std = self.dataset.action_std
        self.proprio_mean = self.dataset.proprio_mean
        self.proprio_std = self.dataset.proprio_std
        self.state_mean = self.dataset.state_mean
        self.state_std = self.dataset.state_std
        self.transform = self.dataset.transform

    def get_seq_length(self, idx: int) -> int:
        return self.num_frames

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]
        obs, act, state, _ = self.dataset[i]
        for k, v in obs.items():
            obs[k] = v[start : end : self.frameskip]
        state = state[start : end : self.frameskip]
        act = act[start:end]
        act = rearrange(
            act, "(n f) d -> n (f d)", n=self.num_frames
        )  # concat actions
        return tuple([obs, act, state])

    # def __getattr__(self, name):
    #     # Use getattr with a default to avoid infinite recursion
    #     try:
    #         return getattr(self.dataset, name)
    #     except AttributeError:
    #         raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class TrajFullSequenceDataset(TrajDataset):
    def __init__(
        self,
        dataset: TrajDataset,
        frameskip: int = 1,
        process_actions: str = "concat",
        min_seq_length: int = 10,  # minimum sequence length to include
    ):
        self.dataset = dataset
        self.frameskip = frameskip
        self.min_seq_length = min_seq_length

        # Filter sequences that are long enough after frameskip
        self.valid_indices = []
        for i in range(len(self.dataset)):
            T = self.dataset.get_seq_length(i)
            # Calculate effective length after frameskip
            effective_length = (T - 1) // frameskip + 1
            if effective_length >= min_seq_length:
                self.valid_indices.append(i)

        self.num_frames = (
            self.dataset.get_seq_length(self.valid_indices[0]) - 1
        ) // frameskip + 1

        print(
            f"Using {len(self.valid_indices)} sequences out of {len(self.dataset)}"
        )
        print(
            f"Frameskip: {frameskip}, Min effective length: {min_seq_length}"
        )

        # Copy attributes from original dataset
        self.proprio_dim = self.dataset.proprio_dim
        if process_actions == "concat":
            self.action_dim = self.dataset.action_dim * self.frameskip
        else:
            self.action_dim = self.dataset.action_dim

        self.state_dim = self.dataset.state_dim
        self.action_mean = self.dataset.action_mean
        self.action_std = self.dataset.action_std
        self.proprio_mean = self.dataset.proprio_mean
        self.proprio_std = self.dataset.proprio_std
        self.state_mean = self.dataset.state_mean
        self.state_std = self.dataset.state_std
        self.transform = self.dataset.transform

    def get_seq_length(self, idx: int) -> int:
        """Returns the effective length after frameskip"""
        original_idx = self.valid_indices[idx]
        T = self.dataset.get_seq_length(original_idx)
        return (T - 1) // self.frameskip + 1

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        original_idx = self.valid_indices[idx]
        obs, act, state, _ = self.dataset[original_idx]

        # Apply frameskip to observations and state
        for k, v in obs.items():
            obs[k] = v[:: self.frameskip]  # Take every frameskip-th frame
        state = state[:: self.frameskip]

        # Handle actions - take frameskip consecutive actions and concatenate
        act = rearrange(
            act, "(n f) d -> n (f d)", n=self.num_frames
        )  # concat actions

        return tuple([obs, act, state])


def random_split_traj(
    dataset: TrajDataset,
    lengths: Sequence[int],
    generator: Optional[torch.Generator] = default_generator,
) -> List[TrajSubset]:
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()
    # print(
    #     [
    #         indices[offset - length : offset]
    #         for offset, length in zip(_accumulate(lengths), lengths)
    #     ]
    # )
    return [
        TrajSubset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


def split_traj_datasets(dataset, train_fraction=0.95, random_seed=42):
    dataset_length = len(dataset)
    lengths = [
        int(train_fraction * dataset_length) + 1,
        dataset_length - int(train_fraction * dataset_length) - 1,
    ]
    train_set, val_set = random_split_traj(
        dataset, lengths, generator=torch.Generator().manual_seed(random_seed)
    )
    return train_set, val_set


def get_train_val_sliced(
    traj_dataset: TrajDataset,
    train_fraction: float = 0.8,
    random_seed: int = 42,
    num_frames: int = 10,
    frameskip: int = 1,
):
    train, val = split_traj_datasets(
        traj_dataset,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )

    train_slices = TrajSlicerDataset(train, num_frames, frameskip)
    val_slices = TrajSlicerDataset(val, num_frames, frameskip)

    return train, val, train_slices, val_slices


def get_train_val_full_sequence(
    traj_dataset: TrajDataset,
    train_fraction: float = 0.8,
    random_seed: int = 42,
    frameskip: int = 1,
    min_seq_length: int = 10,
):
    train, val = split_traj_datasets(
        traj_dataset,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )

    train_full = TrajFullSequenceDataset(
        train, frameskip, min_seq_length=min_seq_length
    )
    val_full = TrajFullSequenceDataset(
        val, frameskip, min_seq_length=min_seq_length
    )

    return train, val, train_full, val_full
