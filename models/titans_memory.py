import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional
from einops import rearrange, repeat
from .vit import Attention, FeedForward


class DeepMemory(nn.Module):
    def __init__(self, dim, num_layers, hidden_dim, alpha, theta, eta, from_base=True):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.theta = theta
        self.eta = eta
        self.act = nn.SiLU()
        self.from_base = from_base
        
        self.base_layers = nn.ModuleList([])
        for i in range(num_layers):
            if i == 0:
                self.base_layers.append(nn.Linear(dim, hidden_dim))
            elif i == num_layers - 1:
                self.base_layers.append(nn.Linear(hidden_dim, dim))
            else:
                self.base_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # register buffer for fast update weights based on base layer weights
        for i, layer in enumerate(self.base_layers):
            self.register_buffer(
                f"Memory_W_{i}_fast", layer.weight.data.detach().clone()
            )
            self.register_buffer(
                f"Memory_b_{i}_fast", layer.bias.data.detach().clone()
            )
            self.register_buffer(
                f"Momentum_W_{i}_fast", layer.weight.data.detach().clone()
            )
            self.register_buffer(
                f"Momentum_b_{i}_fast", layer.bias.data.detach().clone()
            )

    @torch.no_grad()
    def reset_weights(self):
        for i, layer in enumerate(self.base_layers):
            W = getattr(self, f"Memory_W_{i}_fast")
            b = getattr(self, f"Memory_b_{i}_fast")
            M = getattr(self, f"Momentum_W_{i}_fast")
            m = getattr(self, f"Momentum_b_{i}_fast")
            if self.from_base:
                W.copy_(layer.weight.data)
                b.copy_(layer.bias.data)
            else:
                W.zero_()
                b.zero_()
            M.zero_()
            m.zero_()

    def fast_params(self):
        params = []
        for i, _ in enumerate(self.base_layers):
            params.append(
                (
                    getattr(self, f"Memory_W_{i}_fast"),
                    getattr(self, f"Memory_b_{i}_fast"),
                )
            )
        return params

    def fast_params_flattened(self):
        params = []
        for i in range(self.num_layers):
            params.extend(
                (
                    getattr(self, f"Memory_W_{i}_fast"),
                    getattr(self, f"Memory_b_{i}_fast"),
                )
            )
        return params

    def moments_flattened(self):
        params = []
        for i in range(self.num_layers):
            params.extend(
                (
                    getattr(self, f"Momentum_W_{i}_fast"),
                    getattr(self, f"Momentum_b_{i}_fast"),
                )
            )
        return params

    def base_params(self):
        params = []
        for _, layer in enumerate(self.base_layers):
            params.append((layer.weight, layer.bias))
        return params

    def forward(self, x):
        for i, (W, b) in enumerate(self.fast_params()):
            x = x @ W.t() + b
            if i < self.num_layers - 1:
                x = self.act(x)
        return x

    def read(self, x):
        return self.forward(x)

    def _loss(self, k, v):
        return torch.mean((self.forward(k) - v) ** 2)

    def write(self, k, v):
        k = k.detach()
        v = v.detach()
        weights = []

        for w in self.fast_params_flattened():
            w.requires_grad_(True)
            weights.append(w)

        loss = self._loss(k, v)
        grads = torch.autograd.grad(
            loss, weights, retain_graph=False, create_graph=False
        )

        # turn off gradients
        for w in weights:
            w.requires_grad_(False)
        for s in self.moments_flattened():
            s.requires_grad_(False)

        for w, grad, s in zip(weights, grads, self.moments_flattened()):
            # surprise momentum update
            s.mul_(self.eta).add_(grad, alpha=-self.theta)

            # forget + surprise step
            w.mul_(1 - self.alpha).add_(s)

        return loss


# ============================================================================
# MAC - Memory-as-Context
# ============================================================================

class MACBlock(nn.Module):
    """
    Memory-as-Context: concatenate memory readout as extra tokens in each segment,
    then apply full causal attention within the segment.
    """
    def __init__(self, dim, heads, mem_layers, mem_hidden_dim, ff_mlp_dim, chunk_size,n_persist=16, 
                 alpha=0.02, theta=0.1, eta=0.9, dropout=0., from_base=True):
        super().__init__()

        self.chunk_size = chunk_size

        # Memory-as-Context
        self.mem = DeepMemory(dim, mem_layers, mem_hidden_dim, alpha, theta, eta, from_base)
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.theta = nn.Parameter(torch.tensor(theta)) # surprise gradient step size
        self.eta = nn.Parameter(torch.tensor(eta)) # surprise decay
        self.alpha = nn.Parameter(torch.tensor(alpha)) # forget weight

        # Core
        self.attn = Attention(dim, heads, dropout=dropout)
        self.ff = FeedForward(dim, ff_mlp_dim, dropout)

        # Persistent tokens
        self.P = nn.Parameter(torch.randn(1, n_persist, dim))  # persistent tokens
        self.n_persist = n_persist    

    def reset_memory(self):
        self.mem.reset_weights()

    def forward(self, x):
        """
        x: [B, T, dim] - one segment/chunk
        """
        B, T, _ = x.shape
        assert T % self.chunk_size == 0, "Chunk size must divide sequence length"
        
        # reset memory for batch
        self.reset_memory()

        # inner loop (train memory on sequence)
        memory_tokens = []
        for chunk_idx in range(0, T, self.chunk_size):
            chunk = x[:, chunk_idx:chunk_idx + self.chunk_size, :]
            Q_c, K_c, V_c = self.to_qkv(chunk).chunk(3, dim = -1)
            
            token = self.mem.read(Q_c)
            


            # memory gate
            chunk = chunk * self.mem.read(new_Q)
        Q, K, V = self.to_qkv(x).chunk(3, dim = -1)

        # Read memory and create summary token
        memory_token = self.mem.read(Q)

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
        new_qkv = self.to_qkv(y)
        new_K, new_V, new_Q = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), new_qkv)
        self.mem.update_step(new_K, new_V)

        # memory gate
        out = y * self.mem.read(new_Q)

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
