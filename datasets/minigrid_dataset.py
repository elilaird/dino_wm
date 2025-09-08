import torch
import numpy as np
import json
from pathlib import Path
from typing import Optional, Callable, Any, Dict, List, Tuple
from torch.utils.data import Dataset
from .traj_dset import get_train_val_sliced, get_train_val_full_sequence
from einops import rearrange

class MiniGridInMemoryDataset(Dataset):

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

        # Load all data into memory
        self._load_all_data_into_tensors()

        # Initialize normalization
        self._init_default_normalization()

        print(f"Loaded {self.n_rollout} rollouts from {self.n_chunks} chunks into memory")

    def _load_all_data_into_tensors(self):
        """Load all chunk data into memory as concatenated tensors."""
        print("Loading all chunks into memory as tensors...")
        
        all_observations = []
        all_actions = []
        
        # Load all data from chunks
        for chunk_idx in range(self.n_chunks):
            chunk_path = self.data_path / f"chunk_{chunk_idx:04d}.npz"
            with np.load(chunk_path) as data:
                chunk_obs = data['observations']
                chunk_actions = data['actions']
                
                # Add all trajectories from this chunk
                all_observations.extend(chunk_obs)
                all_actions.extend(chunk_actions)
                
                # Stop if we have enough rollouts
                if len(all_observations) >= self.n_rollout:
                    break
        
        # Limit to n_rollout
        all_observations = all_observations[:self.n_rollout]
        all_actions = all_actions[:self.n_rollout]
        
        # Get dimensions from first sample
        sample_obs = all_observations[0]
        sample_act = all_actions[0].reshape(-1, 1)
        
        if len(sample_obs.shape) == 3:  # (H, W, C)
            self.obs_shape = sample_obs.shape
            self.obs_dim = sample_obs.shape[-1] if len(sample_obs.shape) > 2 else 1
        else:
            self.obs_shape = sample_obs.shape
            self.obs_dim = sample_obs.shape[-1] if len(sample_obs.shape) > 1 else 1

        self.action_dim = sample_act.shape[-1] if len(sample_act.shape) > 1 else 1
        self.proprio_dim = self.action_dim
        self.state_dim = self.action_dim
        
        # Since all sequences have the same length, we can use concatenation
        seq_len = len(all_actions[0])
        self.seq_lengths = torch.full((self.n_rollout,), seq_len, dtype=torch.long)
        
        # Convert to tensors using concatenation - much more efficient
        self.observations = torch.from_numpy(np.stack(all_observations)).to(torch.uint8)
        self.actions = torch.from_numpy(np.stack(all_actions)).to(torch.float32)
        
        # Reshape actions to have proper action_dim
        if self.actions.dim() == 2:  # (n_rollout, seq_len)
            self.actions = self.actions.unsqueeze(-1)  # (n_rollout, seq_len, 1)
        
        print(f"Loaded {len(all_observations)} trajectories into memory tensors")
        print(f"Tensor shapes - obs: {self.observations.shape}, actions: {self.actions.shape}")
        print(f"All sequences have length: {seq_len}")

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

    def get_frames(self, idx, frame_indices):
        """Get specific frames from trajectory at index idx."""
        if idx >= self.n_rollout:
            raise IndexError(f"Index {idx} out of bounds for {self.n_rollout} rollouts")

        # Direct tensor access - very fast
        obs = self.observations[idx, frame_indices].float()
        obs = obs / 255.0
        obs = rearrange(obs, "T H W C -> T C H W")
        if self.transform:
            obs = self.transform(obs)

        actions = self.actions[idx, frame_indices]
        
        # Convert one-hot actions to discrete integers if action_dim > 1
        if actions.ndim >= 2 and actions.shape[-1] > 1:
            # Convert one-hot to discrete: [T, action_dim] -> [T, 1]
            actions = torch.argmax(actions, dim=-1, keepdim=True).float()

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
            actions = self.actions[i, :T]
            
            # Convert one-hot actions to discrete integers if action_dim > 1
            if actions.shape[-1] > 1:
                # Convert one-hot to discrete: [T, action_dim] -> [T, 1]
                actions = torch.argmax(actions, dim=-1, keepdim=True).float()
            
            result.append(actions)
        return torch.cat(result, dim=0)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0



class MiniGridMemmapDataset(Dataset):
    """
    Memory-mapped MiniGrid dataset using NPY files for efficient loading.

    - Uses .npy chunk files with NumPy memmap (np.load(..., mmap_mode='r')).
    - Supports fixed-length episodes only (no offset logic).
    - Keeps per-worker memmap caches (safe for DDP + num_workers > 0).
    - Public interface: constructor args, __len__, __getitem__, get_frames(), etc.

    Expected files per chunk:
      - observations_{idx:04d}.npy  (shape: [episodes, timesteps, height, width, channels])
      - actions_{idx:04d}.npy       (shape: [episodes, timesteps])

    index.json:
      {
        "episodes_per_chunk": <int>,
        "total_episodes": <int>,
        "n_chunks": <int>
      }
    """

    def __init__(
        self,
        data_path: str,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale: float = 1.0,
        total_episodes: Optional[int] = None,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action
        self.action_scale = float(action_scale)

        # Load index.json
        with open(self.data_path / "index.json", "r") as f:
            self.index = json.load(f)

        self.episodes_per_chunk: int = int(self.index.get("episodes_per_chunk", 0))
        self.total_episodes: int = int(self.index["total_episodes"]) if total_episodes is None else total_episodes
        self.n_chunks: int = int(self.index["n_chunks"]) if total_episodes is None else total_episodes // self.episodes_per_chunk
        self.seq_length = int(self.index["max_steps"])

        print(f"Total episodes: {self.total_episodes}")
        print(f"N chunks: {self.n_chunks}")
        print(f"Episodes per chunk: {self.episodes_per_chunk}")

        # Limit to n_rollout if specified
        self.n_rollout = min(n_rollout, self.total_episodes) if n_rollout else self.total_episodes

        # Lazily opened per-worker memmap handles
        self._mmaps: Dict[str, Dict[int, np.ndarray]] = None  # set in worker
        # Episode index: global episode id -> (chunk_id, local_ep_idx)
        self._episode_map: List[Tuple] = []

        # Build episode map for fixed-length episodes
        self._build_episode_map()

        # Infer shapes & dims cheaply from the first episode
        self._init_shapes_and_norms()

        print(f"[MiniGridMemmapDataset] Using memory-mapped .npy chunks "
              f"with {len(self)} rollouts across {self.n_chunks} chunks.")

    # ---------- storage probing & maps ----------

    def _chunk_path(self, chunk_idx: int, kind: str) -> Path:
        # kind: "observations", "actions"
        return self.data_path / f"{kind}_{chunk_idx:04d}.npy"

    def _ensure_worker_caches(self):
        # Initialize per-worker cache dicts lazily (so each worker has its own)
        if self._mmaps is None:
            self._mmaps = {"observations": {}, "actions": {}}

    def _get_mmap(self, kind: str, chunk_idx: int) -> np.ndarray:
        self._ensure_worker_caches()
        cache = self._mmaps[kind]
        if chunk_idx in cache:
            return cache[chunk_idx]
        path = self._chunk_path(chunk_idx, kind)
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        arr = np.load(path, mmap_mode="r")
        cache[chunk_idx] = arr
        return arr

    def _build_episode_map(self):
        """
        Build a list that maps global episode indices [0, n_rollout) to chunk slices.
        Only supports fixed-length episodes: self._episode_map[ep] = (chunk_id, local_ep_idx)
        """
        episode_count = 0

        for cid in range(self.n_chunks):
            # Count episodes by episodes_per_chunk (or infer from observations file shape)
            if self.episodes_per_chunk <= 0:
                observations = np.load(self._chunk_path(cid, "observations"), mmap_mode="r")
                if observations.ndim < 2:
                    raise ValueError(f"Unexpected observations shape in chunk {cid}: {observations.shape}")
                e_per_chunk = observations.shape[0]
            else:
                e_per_chunk = self.episodes_per_chunk

            for local_idx in range(e_per_chunk):
                if episode_count >= self.n_rollout:
                    break
                self._episode_map.append((cid, local_idx))
                episode_count += 1

            if episode_count >= self.n_rollout:
                break

        # Clip in case index.json total_episodes > actually mapped
        self.n_rollout = min(self.n_rollout, len(self._episode_map))

    def _peek_episode(self, global_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load *metadata* to infer shapes without copying full arrays.
        Returns a tiny view (not fully copied until Torch conversion).
        """
        cid, local_idx = self._episode_map[global_idx]
        observations = self._get_mmap("observations", cid)
        actions = self._get_mmap("actions", cid)
        obs = observations[local_idx]        # shape [T, H, W, C]
        act = actions[local_idx]             # shape [T, A] or [T]
        return obs, act

    def _init_shapes_and_norms(self):
        # Peek the first episode to infer dims
        img_np, act_np = self._peek_episode(0)

        # image dims
        if img_np.ndim == 3:
            H, W, C = img_np.shape
            T = 1
        elif img_np.ndim == 4:
            T, H, W, C = img_np.shape
        else:
            raise ValueError(f"Unexpected image shape: {img_np.shape}")

        # action dims
        if act_np.ndim == 1:
            # get unique action values
            unique_actions = np.unique(act_np)
            action_dim = len(unique_actions)
        elif act_np.ndim == 2:
            action_dim = act_np.shape[-1]
        else:
            raise ValueError(f"Unexpected action shape: {act_np.shape}")

        self.obs_shape = (H, W, C)
        self.obs_dim = C
        self.action_dim = action_dim
        self.proprio_dim = action_dim
        self.state_dim = action_dim

        # Default "no normalization"
        self._init_default_normalization()

    # ---------- public API ----------

    def _init_default_normalization(self):
        self.obs_mean = torch.zeros(self.obs_dim)
        self.obs_std = torch.ones(self.obs_dim)
        self.action_mean = torch.zeros(self.action_dim)
        self.action_std = torch.ones(self.action_dim)
        self.proprio_mean = torch.zeros(self.action_dim)
        self.proprio_std = torch.ones(self.action_dim)
        self.state_mean = torch.zeros(self.action_dim)
        self.state_std = torch.ones(self.action_dim)

    def __len__(self):
        return self.n_rollout

    # Helpers to fetch episode length for a given global idx
    def get_seq_length(self, idx: int) -> int:
        cid, local_idx = self._episode_map[idx]
        observations = self._get_mmap("observations", cid)
        # observations shape: [E, T, H, W, C]
        return int(observations.shape[1])

    def get_frames(self, idx: int, frame_indices) -> Tuple[dict, torch.Tensor, torch.Tensor, dict]:
        """
        Get specific frames from trajectory at global index `idx`.
        Returns (obs_dict, actions, state, extras) to match previous behavior.
        """
        if idx >= self.n_rollout:
            raise IndexError(f"Index {idx} out of bounds for {self.n_rollout} rollouts")

        cid, local_idx = self._episode_map[idx]

        # Resolve numpy views from memmaps
        obs_np = self._get_mmap("observations", cid)[local_idx]     # [T, H, W, C] (uint8 recommended)
        acts_np = self._get_mmap("actions", cid)[local_idx]         # [T, A] or [T]

        # Fancy indexing with a list/array is fine on memmap (pages on demand)
        if not isinstance(frame_indices, slice):
            frame_indices = list(frame_indices)

        # Observations → float in [0,1], then channel-first
        obs = torch.from_numpy(obs_np[frame_indices])
        if obs.dtype == torch.uint8:
            obs = obs.float().div_(255.0)
        else:
            obs = obs.float()
        obs = rearrange(obs, "T H W C -> T C H W")
        if self.transform:
            obs = self.transform(obs)

        # Actions → float, (T, A)
        acts = torch.from_numpy(acts_np[frame_indices]).float()
        
        # Convert one-hot actions to discrete integers if action_dim > 1
        if acts.ndim >= 2 and acts.shape[-1] > 1:
            # Convert one-hot to discrete: [T, action_dim] -> [T, 1]
            acts = torch.argmax(acts, dim=1, keepdim=True).float() 

        if acts.ndim == 1:
            acts = acts.unsqueeze(-1)

        # Dummy proprio & state to preserve interface
        proprio = torch.zeros_like(acts)
        state = torch.zeros_like(acts)

        obs_dict = {"visual": obs, "proprio": proprio}
        return obs_dict, acts, state, {}

    def __getitem__(self, idx: int):
        # Return the full trajectory (all frames), matching your previous behavior
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    # For compatibility with your training utilities
    def get_all_actions(self) -> torch.Tensor:
        """
        Concatenate actions from all episodes. Still zero-copy from memmaps
        until cast to torch.float (which is necessary anyway).
        """
        out = []
        for i in range(len(self)):
            T = self.get_seq_length(i)
            # Efficient slice without reading observations
            entry = self._episode_map[i]
            cid, local_idx = entry
            acts_np = self._get_mmap("actions", cid)[local_idx, :T]
            t = torch.from_numpy(acts_np).float()
            if t.ndim == 1:
                t = t.unsqueeze(-1)
            
            # Convert one-hot actions to discrete integers if action_dim > 1
            if self.action_dim > 1:
                # Convert one-hot to discrete: [T, action_dim] -> [T, 1]
                t = torch.argmax(t, dim=-1, keepdim=True).float()
            
            out.append(t)
        return torch.cat(out, dim=0)

    # Ensure per-worker caches are not accidentally serialized to children
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mmaps"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._mmaps = None

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
    in_memory=True,
    total_episodes=None,
):
    """Load and split MiniGrid dataset following the same pattern as point_maze_dset.py."""
    
    if in_memory:
        dset = MiniGridInMemoryDataset(
            n_rollout=n_rollout,
            transform=transform,
            data_path=data_path,
            normalize_action=normalize_action,
        )
    else:
        dset = MiniGridMemmapDataset(
            n_rollout=n_rollout,
            transform=transform,
            data_path=data_path,
            normalize_action=normalize_action,
            total_episodes=total_episodes,
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
