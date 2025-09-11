# Adapted from https://github.com/facebookresearch/ijepa/blob/main/src/models/vision_transformer.py 
import numpy as np
import torch.nn as nn
import torch


def get_1d_sincos_pos_embed(emb_dim, grid_size, cls_token=False):
    """
    emb_dim: output dimension for each position
    grid_size: int of the grid length
    returns:
        pos_embed: [grid_size, emb_dim] (w/o cls_token)
                or [1+grid_size, emb_dim] (w/ cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(emb_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, emb_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(emb_dim, pos):
    """
    emb_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    returns: (M, D)
    """
    assert emb_dim % 2 == 0
    omega = np.arange(emb_dim // 2, dtype=float)
    omega /= emb_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class ProprioceptiveEmbedding(nn.Module):
    def __init__(
        self,
        num_frames=16, # horizon
        tubelet_size=1,
        in_chans=8, # action_dim
        emb_dim=384, # output_dim
        use_3d_pos=False # always False for now
    ):
        super().__init__()
        print(f'using 3d prop position {use_3d_pos=}')

        # Map input to predictor dimension
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        self.emb_dim = emb_dim

        self.patch_embed = nn.Conv1d(
            in_chans,
            emb_dim,
            kernel_size=tubelet_size,
            stride=tubelet_size)

    def forward(self, x):
        # x: proprioceptive vectors of shape [B T D]
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        return x

class MiniGridProprioceptiveEmbedding(nn.Module):
    def __init__(self, in_chans=None, world_size=17, emb_dim=16):
        super().__init__()
        self.world_size = world_size
        self.emb_dim = emb_dim
        self.in_chans = in_chans # dummy to match interface
        
        # Grid embedding for coordinates
        self.grid_embedding = nn.Embedding(world_size * world_size, emb_dim // 2)
        
        # Direction embedding (discrete: 0, 1, 2, 3)
        self.dir_embedding = nn.Embedding(4, emb_dim // 2)
        
    def forward(self, x):
        # x: [B, T, 3] where [x, y, dir]
        coords = x[..., :2].long()  # [B, T, 2]
        dirs = x[..., -1].long()    # [B, T, 1]
        
        # Grid coordinates
        grid_indices = coords[..., 0] * self.world_size + coords[..., 1]
        coord_emb = self.grid_embedding(grid_indices)  # [B, T, emb_dim//2]
        
        # Directions (treat each direction separately)
        dir_emb = self.dir_embedding(dirs)  # [B, T, emb_dim//2]
        
        return torch.cat([coord_emb, dir_emb], dim=-1)  # [B, T, emb_dim]

class MemoryMazeProprioceptiveEmbedding(nn.Module):
    def __init__(self, in_chans, emb_dim=16):
        super().__init__()
        self.emb_dim = emb_dim
        self.in_chans = in_chans
        
        self.pos_linear = nn.Linear(2, emb_dim // 2)
        self.dir_linear = nn.Linear(in_chans - 2, emb_dim // 2)

    def forward(self, x):
        # x: [B, T, 4] where [x, y, dir_x, dir_y]
        coords = x[..., :2]  # [B, T, 2]
        dirs = x[..., 2:]    # [B, T, in_chans - 2]
        
        # Grid coordinates
        coord_emb = self.pos_linear(coords)  # [B, T, emb_dim//2]
        
        # Directions (treat each direction separately)
        dir_emb = self.dir_linear(dirs)  # [B, T, emb_dim//2]
        
        return torch.cat([coord_emb, dir_emb], dim=-1)  # [B, T, emb_dim]
