import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional
from einops import rearrange, repeat

class TitansMemory(nn.Module):
    """
    Per-sequence memory: y = W q, with inner-loop update on each step:
      W_t = (1 - alpha)*W_{t-1} + momentum - lr * grad_{W} ||W k_t - v_t||^2
    Vectorized across the batch. Reset between sequences.
    """
    def __init__(self, d_k: int, d_v: int, alpha: float = 0.02, lr: float = 0.1, beta: float = 0.9):
        super().__init__()
        self.d_k, self.d_v = d_k, d_v
        self.alpha = alpha  # forget
        self.lr = lr        # inner-loop step size
        self.beta = beta    # momentum

    def reset(self, B: int, device=None, dtype=None):
        self.W = torch.zeros(B, self.d_v, self.d_k, device=device, dtype=dtype)
        self.m = torch.zeros_like(self.W)  # momentum buffer

    def read(self, q: Tensor) -> Tensor:
        # q: [B, L, d_k]
        return torch.einsum('bvk,blk->blv', self.W, q)  # [B, L, d_v]

    def update_step(self, k: Tensor, v: Tensor):
        """
        One online step given the *current* (k, v).
        k: [B, d_k], v: [B, d_v]
        Loss: ||W k - v||^2 (per batch item)
        grad_W = 2 * (Wk - v) k^T
        """
        Wk = torch.einsum('bvk,bk->bv', self.W, k)              # [B, d_v]
        resid = (Wk - v)                                        # [B, d_v]
        grad = 2.0 * torch.einsum('bv,bk->bvk', resid, k)       # [B, d_v, d_k]

        # Momentum + forget (all ops are differentiable; no .data/.detach here)
        self.m = self.beta * self.m + (1 - self.beta) * grad
        self.W = (1.0 - self.alpha) * self.W - self.lr * self.m

    def update_window(self, K: Tensor, V: Tensor, n_inner: int = 1):
        """
        K: [B, C, d_k], V: [B, C, d_v] for a window/chunk of length C.
        Loss: ||W K^T - V^T||_F^2 (batched)
        grad_W = 2 * (W K^T - V^T) K  (properly batched)
        We can take a few inner steps (n_inner) if desired.
        """
        for _ in range(n_inner):
            # [B, d_v, C] = [B, d_v, d_k] @ [B, d_k, C]
            K_t = K.transpose(1, 2)                 # [B, d_k, C]
            VK = torch.einsum('bvk,bkc->bvc', self.W, K_t)  # [B, d_v, C]
            V_t = V.transpose(1, 2)                 # [B, d_v, C]
            E = (VK - V_t)                          # [B, d_v, C]

            # grad = 2 * E @ K^T  -> [B, d_v, d_k]
            grad = 2.0 * torch.einsum('bvc,bck->bvk', E, K)  # [B, d_v, d_k]

            # Simple gradient descent (omit momentum/forget for window updates)
            self.W = self.W - self.lr * grad


class KVQProj(nn.Module):
    """Key/Value/Query projections for memory"""
    def __init__(self, d_in: int, d_k: int, d_v: int):
        super().__init__()
        self.k = nn.Linear(d_in, d_k)
        self.v = nn.Linear(d_in, d_v)
        self.q = nn.Linear(d_in, d_k)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return self.k(x), self.v(x), self.q(x)  # [B, L, d_k/d_v]


class FeedForward(nn.Module):
    """Feed-forward network for transformer blocks"""
    def __init__(self, dim, hidden_dim, dropout=0.):
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
    """Standard multi-head attention"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        B, T, C = x.size()
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        if mask is not None:
            dots = dots.masked_fill(mask == 0, float("-inf"))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# ============================================================================
# MAC - Memory-as-Context
# ============================================================================

class MACBlock(nn.Module):
    """
    Memory-as-Context: concatenate memory readout as extra tokens in each segment,
    then apply full causal attention within the segment.
    """
    def __init__(self, dim, heads, dim_head, mlp_dim, d_k, d_v, n_persist=16, 
                 alpha=0.02, lr=0.1, beta=0.9, dropout=0.):
        super().__init__()
        self.mem = TitansMemory(d_k, d_v, alpha, lr, beta)
        self.proj = KVQProj(dim, d_k, d_v)
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout)
        self.P = nn.Parameter(torch.randn(1, n_persist, dim))  # persistent tokens
        self.out = nn.Linear(d_v, dim)  # memory→token projection
        self.n_persist = n_persist

    def forward(self, x):
        """
        x: [B, T, dim] - one segment/chunk
        """
        B, T, _ = x.shape
        if not hasattr(self.mem, "W"):
            self.mem.reset(B, x.device, x.dtype)

        K, V, Q = self.proj(x)  # [B, T, d_k/d_v]
        
        # Read memory and create summary token
        h = self.mem.read(Q).mean(dim=1, keepdim=True)  # [B, 1, d_v]
        memory_token = self.out(h)  # [B, 1, dim]
        
        # Concatenate: persistent + memory + current segment
        tokens = torch.cat([
            self.P.repeat(B, 1, 1),      # persistent tokens
            memory_token,                 # memory token
            x                             # current segment
        ], dim=1)  # [B, n_persist + 1 + T, dim]
        
        # Apply attention
        y = self.attn(tokens)
        y = self.ff(y)
        
        # Return only the positions corresponding to the original segment
        out = y[:, self.n_persist + 1:, :]  # drop persistent + memory tokens
        
        # Update memory
        self.mem.update_window(K, V)
        
        return out


# ============================================================================
# MAG - Memory-as-Gating
# ============================================================================

class MAGBlock(nn.Module):
    """
    Memory-as-Gating: parallel branches with gated fusion.
    Short-term SWA ⊗ Long-term memory
    """
    def __init__(self, dim, heads, dim_head, mlp_dim, d_k, d_v, window=512,
                 alpha=0.02, lr=0.1, beta=0.9, dropout=0.):
        super().__init__()
        self.mem = TitansMemory(d_k, d_v, alpha, lr, beta)
        self.proj = KVQProj(dim, d_k, d_v)
        self.swa = Attention(dim, heads, dim_head, dropout)  # short-term window
        self.g_lin = nn.Linear(dim, dim)  # gate network
        self.U = nn.Linear(d_v, dim)  # project memory read to model dim
        self.ff = FeedForward(dim, mlp_dim, dropout)
        self.window = window

    def forward(self, x):
        """
        x: [B, L, dim] - full sequence chunk
        """
        B, L, _ = x.shape
        if not hasattr(self.mem, "W"):
            self.mem.reset(B, x.device, x.dtype)

        K, V, Q = self.proj(x)
        y_mem = self.mem.read(Q)  # [B, L, d_v]
        y_swa = self.swa(x)  # short-term attention

        # Gated fusion
        gate = torch.sigmoid(self.g_lin(x))  # learned gate per position
        fused = gate * y_swa + (1 - gate) * self.U(y_mem)
        out = self.ff(fused)

        # Update memory
        self.mem.update_window(K, V)
        
        return out


# ============================================================================
# MAL - Memory-as-Layer
# ============================================================================

class MALBlock(nn.Module):
    """
    Memory-as-Layer: memory layer → attention layer
    The memory replaces one Transformer block; attention operates on its output.
    """
    def __init__(self, dim, heads, dim_head, mlp_dim, d_k, d_v,
                 alpha=0.02, lr=0.1, beta=0.9, dropout=0.):
        super().__init__()
        self.mem = TitansMemory(d_k, d_v, alpha, lr, beta)
        self.proj = KVQProj(dim, d_k, d_v)
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout)
        self.U = nn.Linear(d_v, dim)  # project memory to model dim

    def forward(self, x):
        """
        x: [B, L, dim] - full sequence
        """
        B, L, _ = x.shape
        if not hasattr(self.mem, "W"):
            self.mem.reset(B, x.device, x.dtype)

        K, V, Q = self.proj(x)
        
        # Memory layer output
        x_mem = self.U(self.mem.read(Q))  # [B, L, dim]
        
        # Attention layer on memory output
        x_att = self.attn(x_mem)
        out = self.ff(x_att)

        # Update memory
        self.mem.update_window(K, V)
        
        return out


# ============================================================================
# Drop-in replacement for existing Transformer
# ============================================================================

class TitansTransformer(nn.Module):
    """
    Drop-in replacement for the existing Transformer that supports all three variants.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, 
                 memory_variant="MAC", d_k=64, d_v=64, n_persist=16, window=512,
                 alpha=0.02, lr=0.1, beta=0.9, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        
        # Choose memory variant
        if memory_variant == "MAC":
            block_cls = MACBlock
            block_kwargs = {"n_persist": n_persist}
        elif memory_variant == "MAG":
            block_cls = MAGBlock
            block_kwargs = {"window": window}
        elif memory_variant == "MAL":
            block_cls = MALBlock
            block_kwargs = {}
        else:
            raise ValueError(f"Unknown memory variant: {memory_variant}")
        
        # Create memory blocks
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(block_cls(
                dim, heads, dim_head, mlp_dim, d_k, d_v,
                alpha=alpha, lr=lr, beta=beta, dropout=dropout,
                **block_kwargs
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x  # residual connection
        return self.norm(x)


# ============================================================================
# Drop-in replacement for ViTPredictor
# ============================================================================

class TitansViTPredictor(nn.Module):
    """
    Drop-in replacement for ViTPredictor that uses Titans memory.
    """
    def __init__(self, *, num_patches, num_frames, dim, depth, heads, mlp_dim, 
                 memory_variant="MAC", d_k=64, d_v=64, n_persist=16, window=512,
                 pool='cls', dim_head=64, dropout=0., emb_dropout=0.,
                 alpha=0.02, lr=0.1, beta=0.9):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = TitansTransformer(
            dim, depth, heads, dim_head, mlp_dim,
            memory_variant=memory_variant, d_k=d_k, d_v=d_v,
            n_persist=n_persist, window=window,
            alpha=alpha, lr=lr, beta=beta, dropout=dropout
        )
        self.pool = pool

    def forward(self, x):
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x)
        return x
