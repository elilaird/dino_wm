import torch
import torch.nn as nn

class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN) module.

    This module extends standard Layer Normalization by dynamically generating
    the scaling (γ) and shifting (β) parameters based on a conditioning input.
    This allows the model to adapt its normalization process to varying inputs.

    Args:
        normalized_shape (int or tuple): The shape of the input tensor to normalize.
            If int, it's treated as the last dimension.
        cond_dim (int): The dimension of the conditioning input.
        eps (float): A small value added to the denominator for numerical stability.
        elementwise_affine (bool): If True, applies learnable affine transformation.
            If False, only uses adaptive parameters from conditioning input.
        bias (bool): If True, includes bias in the adaptive parameters.

    Shape:
        - Input: (N, ..., L) where L is normalized_shape
        - Conditioning: (N, cond_dim) or (N, ..., cond_dim)
        - Output: (N, ..., L) same shape as input

    Example:
        >>> ada_ln = AdaptiveLayerNorm(64, 128)
        >>> x = torch.randn(32, 10, 64)  # (batch, seq_len, hidden_dim)
        >>> cond = torch.randn(32, 128)  # (batch, cond_dim)
        >>> output = ada_ln(x, cond)
    """

    def __init__(
        self,
        normalized_shape,
        cond_dim,
        eps=1e-5,
        elementwise_affine=True,
        bias=True,
        zero_init=False,
    ):
        super(AdaptiveLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.cond_dim = cond_dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape

        # Adaptive parameter generators
        self.ada_gamma = nn.Linear(cond_dim, normalized_shape[0], bias=bias)
        self.ada_beta = nn.Linear(cond_dim, normalized_shape[0], bias=bias)

        if zero_init:
            nn.init.zeros_(self.ada_gamma.weight)
            nn.init.zeros_(self.ada_beta.weight)
            if bias:
                nn.init.zeros_(self.ada_gamma.bias)
                nn.init.zeros_(self.ada_beta.bias)

        # Optional learnable affine parameters (like standard LayerNorm)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias_param = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias_param", None)

    def forward(self, x, cond_input):
        """
        Forward pass of adaptive layer normalization.

        Args:
            x (torch.Tensor): Input tensor to normalize
            cond_input (torch.Tensor): Conditioning input for adaptive parameters

        Returns:
            torch.Tensor: Normalized and adaptively scaled/shifted tensor
        """
        # Compute mean and variance over the last len(normalized_shape) dimensions
        mean = x.mean(
            dim=tuple(range(-len(self.normalized_shape), 0)), keepdim=True
        )
        var = x.var(
            dim=tuple(range(-len(self.normalized_shape), 0)),
            keepdim=True,
            unbiased=False,
        )

        # Normalize input
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Generate adaptive parameters from conditioning input
        # Handle different conditioning input shapes
        if cond_input.dim() == 2:
            # (batch, cond_dim) -> (batch, 1, ..., 1, normalized_shape[0])
            gamma = self.ada_gamma(cond_input)
            beta = self.ada_beta(cond_input)
            # Expand to match input dimensions
            for _ in range(x.dim() - 2):
                gamma = gamma.unsqueeze(-2)
                beta = beta.unsqueeze(-2)
        else:
            # (batch, ..., cond_dim) -> (batch, ..., normalized_shape[0])
            gamma = self.ada_gamma(cond_input)
            beta = self.ada_beta(cond_input)

        # Apply adaptive scaling and shifting
        output = gamma * x_norm + beta

        # Apply optional learnable affine transformation
        if self.elementwise_affine:
            output = output * self.weight + self.bias_param

        return output

# LoRA pre-attn adapter
class MemoryLoRAProj(nn.Module):
    """
    Memory-conditioned low-rank adapter that augments a linear map:
        y = x @ W  +  (x @ B) @ A(m)

    in_dim:  input feature size
    out_dim: output feature size
    rank:    low-rank r
    alpha:   scale (typically alpha / rank is used)
    zero_init: if True, starts as a no-op
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int = 16,
        alpha: float = 8.0,
        zero_init: bool = True,
        mem_hidden_mul: int = 2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.scale = alpha / max(1, rank)

        # shared low-rank basis B: [in_dim, rank]
        self.B = nn.Parameter(
            torch.zeros(in_dim, rank)
            if zero_init
            else torch.randn(in_dim, rank) * 0.02
        )

        # A(m) generator: memory_summary (dim) -> [rank, out_dim]
        self.mem_ln = nn.LayerNorm(
            in_dim
        )  # assumes memory tokens have same dim as model dim
        self.A_gen = nn.Sequential(
            nn.Linear(in_dim, mem_hidden_mul * in_dim),
            nn.GELU(),
            nn.Linear(mem_hidden_mul * in_dim, rank * out_dim),
        )
        # start near zero for stability
        nn.init.zeros_(self.A_gen[-1].weight)
        nn.init.zeros_(self.A_gen[-1].bias)

        # the base (frozen or trainable) linear we augment lives outside this class

    def forward(
        self, x: torch.Tensor, memory_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [B, T, in_dim]
        memory_tokens: [B, S, in_dim] (SSM memory concatenated over frames/patches)
        returns Δy: [B, T, out_dim]
        """
        B, T, Din = x.shape
        assert Din == self.in_dim, f"in_dim mismatch: {Din} vs {self.in_dim}"

        # mean-pool memory (swap to attention pooling if desired)
        m = self.mem_ln(memory_tokens.mean(dim=1))  # [B, in_dim]

        # A(m): [B, rank, out_dim]
        A = self.A_gen(m).view(B, self.rank, self.out_dim)

        # x @ B -> [B, T, rank]
        xB = x @ self.B

        # (x @ B) @ A(m) -> [B, T, out_dim]
        delta = torch.einsum("btr,bro->bto", xB, A)
        return delta * self.scale

# loRA post-attn adapter
class MemoryLoRAAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        rank: int = 64,  # low-rank r
        lora_alpha: float = 1.0,  # scale multiplier (often alpha / r)
        dropout: float = 0.0,
        zero_init: bool = True,  # if True, initialize B to zeros so adapter starts as no-op
        hidden_mul: int = 2,  # width of memory MLP that generates A(m)
    ):

        super().__init__()
        self.dim = dim
        self.rank = rank
        self.scale = lora_alpha / max(1, rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Shared low-rank basis B (d x r)
        self.B = nn.Parameter(
            torch.zeros(dim, rank)
            if zero_init
            else torch.randn(dim, rank) * (0.02)
        )

        # Generator for A(m): takes a memory summary (dim) -> (r x d)
        self.A_gen = nn.Sequential(
            nn.Linear(dim, hidden_mul * dim),
            nn.GELU(),
            nn.Linear(hidden_mul * dim, rank * dim),
        )

        # Optional: layernorm on memory summary for stability
        self.mem_ln = nn.LayerNorm(dim)

        # Initialize the last layer close to zero so Δ starts tiny
        nn.init.zeros_(self.A_gen[-1].weight)
        nn.init.zeros_(self.A_gen[-1].bias)

    def forward(
        self, x: torch.Tensor, memory_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [B, T, D]
        memory_tokens: [B, S, D] (concatenated per-frame memory from SSM)
        returns Δy: [B, T, D]
        """
        Bsz, T, D = x.shape
        assert (
            D == self.dim
        ), f"LoRA dim mismatch: got {D}, expected {self.dim}"

        # Summarize memory tokens -> [B, D]
        # (Swap to attention pooling if you prefer selective recall)
        m_summary = self.mem_ln(memory_tokens.mean(dim=1))  # [B, D]

        # Generate A(m): [B, r, d]
        A = self.A_gen(m_summary).view(Bsz, self.rank, self.dim)

        # Project x into low-rank space: [B, T, r]
        xB = x @ self.B  # (B,T,D) @ (D,r) -> (B,T,r)

        # Contract back to D with A(m): [B, T, D]
        # Δy[b, t, d] = sum_r xB[b, t, r] * A[b, r, d]
        delta = torch.einsum("btr,brd->btd", xB, A)

        return self.dropout(delta * self.scale)
