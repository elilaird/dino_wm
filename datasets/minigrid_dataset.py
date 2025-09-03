import torch
import numpy as np
import json
from pathlib import Path
from typing import Optional, Callable, Any
from torch.utils.data import Dataset
from .traj_dset import get_train_val_sliced, get_train_val_full_sequence
from einops import rearrange


class MiniGridDataset(Dataset):
    """
    Dataset class for MiniGrid trajectories that follows the same pattern as TrajDataset.
    Loads data from chunked NPZ files for efficient memory usage.
    """

    def __init__(
        self,
        data_path: str,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale: float = 1.0,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action
        self.action_scale = action_scale

        # Load index
        with open(self.data_path / 'index.json', 'r') as f:
            self.index = json.load(f)

        self.episodes_per_chunk = self.index['episodes_per_chunk']
        self.total_episodes = self.index['total_episodes']
        self.n_chunks = self.index['n_chunks']

        # Limit to n_rollout if specified
        if n_rollout:
            self.n_rollout = min(n_rollout, self.total_episodes)
        else:
            self.n_rollout = self.total_episodes

        # Load sequence lengths and metadata from first chunk to get dimensions
        self._load_chunk_metadata()

        # Initialize normalization
        self._init_default_normalization()

        print(f"Loaded {self.n_rollout} rollouts from {self.n_chunks} chunks")

    def _load_chunk_metadata(self):
        """Load metadata from chunks to determine dimensions."""
        # Load first chunk to get dimensions
        first_chunk_path = self.data_path / f"chunk_0000.npz"
        with np.load(first_chunk_path, mmap_mode='r') as data:
            sample_obs = data['observations'][0]
            sample_act = data['actions'][0].reshape(-1, 1)

        # Determine dimensions
        if len(sample_obs.shape) == 3:  # (H, W, C)
            self.obs_shape = sample_obs.shape
            self.obs_dim = sample_obs.shape[-1] if len(sample_obs.shape) > 2 else 1
        else:
            self.obs_shape = sample_obs.shape
            self.obs_dim = sample_obs.shape[-1] if len(sample_obs.shape) > 1 else 1

        self.action_dim = sample_act.shape[-1] if len(sample_act.shape) > 1 else 1
        self.proprio_dim = self.action_dim
        self.state_dim = self.action_dim

        # Load sequence lengths from all chunks
        self.seq_lengths = []
        for chunk_idx in range(self.n_chunks):
            chunk_path = self.data_path / f"chunk_{chunk_idx:04d}.npz"
            with np.load(chunk_path, mmap_mode='r') as data:
                chunk_size = len(data['observations'])
                for i in range(chunk_size):
                    if len(self.seq_lengths) >= self.n_rollout:
                        break
                    self.seq_lengths.append(len(data['actions'][i]))

        self.seq_lengths = torch.tensor(self.seq_lengths[:self.n_rollout])

    def _init_default_normalization(self):
        """Initialize default normalization (no scaling)."""
        self.obs_mean = torch.zeros(self.obs_dim)
        self.obs_std = torch.ones(self.obs_dim)
        self.action_mean = torch.zeros(self.action_dim)
        self.action_std = torch.ones(self.action_dim)
        self.proprio_mean = torch.zeros(self.action_dim)
        self.proprio_std = torch.ones(self.action_dim)
        self.state_mean = torch.zeros(self.action_dim)
        self.state_std = torch.ones(self.action_dim)

    def get_seq_length(self, idx):
        """Returns the length of the idx-th trajectory."""
        return self.seq_lengths[idx]

    def _get_chunk_and_local_idx(self, global_idx):
        """Convert global trajectory index to chunk index and local index within chunk."""
        chunk_idx = global_idx // self.episodes_per_chunk
        local_idx = global_idx % self.episodes_per_chunk
        return chunk_idx, local_idx

    def get_frames(self, idx, frame_indices):
        """Get specific frames from trajectory at index idx."""
        if idx >= self.n_rollout:
            raise IndexError(f"Index {idx} out of bounds for {self.n_rollout} rollouts")

        chunk_idx, local_idx = self._get_chunk_and_local_idx(idx)
        chunk_path = self.data_path / f"chunk_{chunk_idx:04d}.npz"

        with np.load(chunk_path, mmap_mode='r') as data:
            # Only load the frames we actually need
            obs = data['observations'][local_idx][frame_indices]
            actions = data['actions'][local_idx][frame_indices]

        # Convert to torch tensors
        obs = torch.from_numpy(obs).float()
        obs = obs / 255.0
        obs = rearrange(obs, "T H W C -> T C H W")
        if self.transform:
            obs = self.transform(obs)

        actions = torch.from_numpy(actions).float().reshape(-1, 1)

        # dummy proprio and state
        proprio = torch.zeros_like(actions)
        state = torch.zeros_like(actions)

        # Create observation dict
        obs_dict = {
            "visual": obs,
            "proprio": proprio,
        }

        return obs_dict, actions, state, {}

    def __getitem__(self, idx):
        """Get trajectory at index idx."""
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return self.n_rollout

    def get_all_actions(self):
        """Get all actions from all trajectories (useful for training)."""
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            obs = self[i]
            result.append(obs['actions'][:T])
        return torch.cat(result, dim=0)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0


# Factory functions following the same pattern as point_maze_dset.py
def make_minigrid_dataset(
    data_path: str,
    n_rollout: Optional[int] = None,
    transform: Optional[Callable] = None,
    normalize_action: bool = False,
    action_scale: float = 1.0,
):
    """Factory function to create MiniGridDataset."""
    return MiniGridDataset(
        data_path=data_path,
        n_rollout=n_rollout,
        transform=transform,
        normalize_action=normalize_action,
        action_scale=action_scale,
    )


def load_minigrid_slice_train_val(
    transform,
    n_rollout=50,
    data_path='data/minigrid_env',
    normalize_action=False,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    num_frames=None,
    full_sequence=False,
):
    """Load and split MiniGrid dataset following the same pattern as point_maze_dset.py."""
    
    dset = MiniGridDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path,
        normalize_action=normalize_action,
    )
    
    if full_sequence:
        dset_train, dset_val, train_slices, val_slices = get_train_val_full_sequence(
            traj_dataset=dset, 
            train_fraction=split_ratio, 
            frameskip=frameskip,
            min_seq_length=num_frames if num_frames else num_hist + num_pred,
        )
    else:
        dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
            traj_dataset=dset, 
            train_fraction=split_ratio, 
            num_frames=num_frames if num_frames else num_hist + num_pred, 
            frameskip=frameskip
        )

    datasets = {}
    datasets['train'] = train_slices
    datasets['valid'] = val_slices
    traj_dset = {}
    traj_dset['train'] = dset_train
    traj_dset['valid'] = dset_val
    return datasets, traj_dset
