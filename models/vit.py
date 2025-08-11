# adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
import torch
from torch import nn
from einops import rearrange, repeat

# helpers
NUM_FRAMES = 1
NUM_PATCHES = 1

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def generate_mask_matrix(npatch, nwindow):
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)
    rows = []
    for i in range(nwindow):
        row = torch.cat([ones] * (i+1) + [zeros] * (nwindow - i-1), dim=1)
        rows.append(row)
    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        bias = generate_mask_matrix(NUM_PATCHES, NUM_FRAMES)
        self.register_buffer("bias", bias)

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # apply causal mask
        dots = dots.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViTPredictor(nn.Module):
    def __init__(self, *, num_patches, num_frames, dim, depth, heads, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        # update params for adding causal attention masks
        global NUM_FRAMES, NUM_PATCHES
        NUM_FRAMES = num_frames
        NUM_PATCHES = num_patches

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * (num_patches), dim)) # dim for the pos encodings
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool

    def forward(self, x): # x: (b, window_size * H/patch_size * W/patch_size, 384)
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x) 
        x = self.transformer(x) 
        return x


class AdditiveControlViTPredictor(nn.Module):
    """
    ViT predictor with additive control injection for actions.
    Separates action processing from visual/proprio processing to avoid gradient entanglement.
    """
    def __init__(self, *, num_patches, num_frames, dim, depth, heads, mlp_dim, 
                 action_dim=0, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        # update params for adding causal attention masks
        global NUM_FRAMES, NUM_PATCHES
        NUM_FRAMES = num_frames
        NUM_PATCHES = num_patches

        self.num_patches = num_patches
        self.num_frames = num_frames
        self.dim = dim
        self.action_dim = action_dim
        self.pool = pool

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        # Additive control components
        if action_dim > 0:
            self.action_projection = nn.Linear(action_dim, dim)
            self.action_norm = nn.LayerNorm(dim)
            
            # Control injection layers for each transformer layer
            self.control_injection = nn.ModuleList([
                nn.Linear(dim, dim) for _ in range(depth)
            ])

    def forward(self, x, actions=None):
        """
        x: (b, num_frames * num_patches, dim) - visual/proprio embeddings only
        actions: (b, num_frames, action_dim) - separate action input (optional)
        """
        b, n, _ = x.shape
        
        # Add position embeddings
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        
        # Process actions if provided
        if actions is not None and self.action_dim > 0:
            # Reshape actions to match token sequence
            actions = rearrange(actions, 'b t d -> b (t p) d', p=1)
            actions = actions.repeat(1, self.num_patches, 1)
            
            # Project actions to embedding dimension
            action_control = self.action_projection(actions)
            action_control = self.action_norm(action_control)
            
            # Process through transformer with additive control injection
            for i, (attn, ff) in enumerate(self.transformer.layers):
                # Standard transformer processing
                x = attn(x) + x
                x = ff(x) + x
                
                # Additive control injection at each layer
                if i < len(self.control_injection):
                    control_injection = self.control_injection[i](action_control)
                    x = x + control_injection
        else:
            # Standard transformer processing without actions
            x = self.transformer(x)
            
        return x
