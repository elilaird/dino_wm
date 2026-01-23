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
        use_3d_pos=False, # always False for now
        frameskip=1,
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

class ActionEmbeddingMLP(nn.Module):
    def __init__(self, in_chans, emb_dim, frameskip):
        super().__init__()
        self.in_chans = in_chans # Action_dim * frameskip
        self.emb_dim = emb_dim
        self.frameskip = frameskip
        self.action_dim = in_chans // frameskip

        self.action_embed = nn.Sequential(
            nn.Linear(self.action_dim, emb_dim * 2),
            nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim)
        )

    def forward(self, x):
        B, T, D = x.shape # [B, T, D * F] where D=original_action_dim, F=frameskip
        test_frameskip = D // self.action_dim

        # reshape to [B, T, F, D]
        x = x.view(B, T, test_frameskip, self.action_dim)
        x = self.action_embed(x)
        
        return x

class ActionMLPLSTM(nn.Module):
    def __init__(self, in_chans, emb_dim, frameskip):
        super().__init__()
        self.in_chans = in_chans # Action_dim * frameskip
        self.emb_dim = emb_dim
        self.frameskip = frameskip
        self.action_dim = in_chans // frameskip

        self.action_embed = nn.Sequential(
            nn.Linear(self.action_dim, emb_dim * 2),
            nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim * 2)
        )
        self.action_lstm = nn.LSTM(
            input_size=emb_dim * 2,
            hidden_size=emb_dim,  # Match your embedding dimension
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

    def forward(self, x):
        B, T, D = x.shape # [B, T, D * F] where D=original_action_dim, F=frameskip
        test_frameskip = D // self.action_dim

        # reshape to [B, T, F, D]
        x = x.view(B, T, test_frameskip, self.action_dim)
        x = self.action_embed(x)

        x_flat = x.reshape(B * T, test_frameskip, x.shape[-1])
        _, (h, _) = self.action_lstm(x_flat)
        action_emb = h[-1].view(B, T, -1)

        return action_emb

class VariableProprioceptiveEmbedding(nn.Module):
    def __init__(
        self,
        num_frames=16,
        tubelet_size=1,
        in_chans=8,  # original action_dim (not multiplied by frameskip)
        emb_dim=384,
        use_3d_pos=False,
        frameskip=1,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans // frameskip
        self.emb_dim = emb_dim
        self.frameskip = frameskip 
        

        # Embed individual actions
        self.action_embed = nn.Conv1d(
            in_chans,
            emb_dim,
            kernel_size=1,  # Each action gets embedded independently
            stride=1,
        )

    def forward(self, x):
        # x: [B, T, D * F] where D=original_action_dim, F=frameskip
        B, T, action_dim = x.shape
        F = action_dim // self.in_chans  # Infer frameskip from input

        # Reshape to process individual actions: [B, T*F, D]
        x = x.view(B, T * F, self.in_chans)

        # Embed each individual action
        x = x.permute(0, 2, 1)  # [B, D, T*F]
        x = self.action_embed(x)  # [B, emb_dim, T*F]

        # Reshape back to temporal sequence: [B, T, F, emb_dim]
        x = x.permute(0, 2, 1).view(B, T, F, self.emb_dim)

        x = x.mean(dim=2)  # [B, T, emb_dim] - simple averaging
        
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
        
        # Clamp coordinates to valid range
        coords = torch.clamp(coords, 0, self.world_size - 1)
        
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
