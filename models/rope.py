import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
    
    Args:
        dim: Dimension of the embedding
        end: End position for the sequence
        theta: Scaling factor for frequency computation
    
    Returns:
        Complex tensor of shape (end, dim // 2)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.
    
    Args:
        xq: Query tensor of shape (batch_size, seq_len, dim)
        xk: Key tensor of shape (batch_size, seq_len, dim)
        freqs_cis: Precomputed frequency tensor of shape (seq_len, dim // 2)
    
    Returns:
        Tuple of (rotated_query, rotated_key) tensors
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RoPEPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding (RoPE) for transformers.
    
    RoPE encodes relative positional information by rotating the query and key vectors
    in the complex plane. This allows the model to understand relative positions
    without being limited by absolute position embeddings.
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.scale = scale
        
        # Precompute frequency tensor
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(dim, max_seq_len, theta),
            persistent=False
        )
    
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Apply RoPE to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            start_pos: Starting position in the sequence (for sliding window)
        
        Returns:
            Tensor with RoPE applied
        """
        seq_len = x.shape[1]
        freqs_cis = self.freqs_cis[start_pos:start_pos + seq_len]
        
        # Split into query and key (for attention)
        xq, xk = x, x  # In RoPE, we apply the same rotation to both
        
        # Apply rotary embeddings
        xq_rotated, xk_rotated = apply_rotary_emb(xq, xk, freqs_cis)
        
        return xq_rotated, xk_rotated
    
    def get_freqs_cis(self, start_pos: int = 0, seq_len: int = None) -> torch.Tensor:
        """
        Get frequency tensor for a specific sequence range.
        
        Args:
            start_pos: Starting position
            seq_len: Length of sequence (if None, uses max_seq_len - start_pos)
        
        Returns:
            Frequency tensor of shape (seq_len, dim // 2)
        """
        if seq_len is None:
            seq_len = self.max_seq_len - start_pos
        return self.freqs_cis[start_pos:start_pos + seq_len]


class GlobalRoPEPositionalEncoding(nn.Module):
    """
    Global RoPE that can handle variable sequence lengths and sliding windows.
    
    This version is designed for world modeling where we need to handle:
    1. Variable sequence lengths
    2. Sliding window processing
    3. Global position awareness
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 10000,  # Large enough for long sequences
        theta: float = 10000.0,
        scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.scale = scale
        
        # We'll compute freqs_cis on-the-fly to handle variable lengths
        self._freqs_cis_cache = {}
    
    def _get_freqs_cis(self, seq_len: int) -> torch.Tensor:
        """Get or compute frequency tensor for given sequence length."""
        if seq_len not in self._freqs_cis_cache:
            self._freqs_cis_cache[seq_len] = precompute_freqs_cis(
                self.dim, seq_len, self.theta
            )
        return self._freqs_cis_cache[seq_len]
    
    def forward(
        self,
        x: torch.Tensor,
        global_start_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to input tensor with global position awareness.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            global_start_pos: Global starting position in the full sequence
        
        Returns:
            Tuple of (rotated_query, rotated_key) tensors
        """
        seq_len = x.shape[1]
        
        # Get frequency tensor for the global sequence range
        freqs_cis = self._get_freqs_cis(global_start_pos + seq_len)
        freqs_cis = freqs_cis[global_start_pos:global_start_pos + seq_len]
        
        # Apply rotary embeddings
        xq, xk = x, x
        xq_rotated, xk_rotated = apply_rotary_emb(xq, xk, freqs_cis)
        
        return xq_rotated, xk_rotated
    
    def clear_cache(self):
        """Clear the frequency tensor cache to free memory."""
        self._freqs_cis_cache.clear()


class RoPEAttention(nn.Module):
    """
    Attention module with RoPE positional encoding.
    
    This replaces the standard attention mechanism with one that uses RoPE
    for positional encoding instead of absolute position embeddings.
    """
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        max_seq_len: int = 10000,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        )
        
        # RoPE positional encoding
        self.rope = GlobalRoPEPositionalEncoding(
            dim=dim_head,
            max_seq_len=max_seq_len,
            theta=theta
        )
    
    def forward(
        self,
        x: torch.Tensor,
        global_start_pos: int = 0,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with RoPE attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            global_start_pos: Global starting position in the full sequence
            mask: Optional attention mask
        
        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        b, n, _ = x.shape
        
        x = self.norm(x)
        
        # Get Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.reshape(b, n, self.heads, self.dim_head).transpose(1, 2),
            qkv
        )
        
        # Apply RoPE to Q and K
        q_rotated, k_rotated = self.rope(
            q.transpose(1, 2).reshape(b, n, self.dim_head),
            global_start_pos
        )
        q_rotated = q_rotated.reshape(b, n, self.heads, self.dim_head).transpose(1, 2)
        k_rotated = k_rotated.reshape(b, n, self.heads, self.dim_head).transpose(1, 2)
        
        # Compute attention
        dots = torch.matmul(q_rotated, k_rotated.transpose(-1, -2)) * self.scale
        
        if mask is not None:
            dots = dots.masked_fill(mask == 0, float("-inf"))
        
        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, self.inner_dim)
        
        return self.to_out(out)
