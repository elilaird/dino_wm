from typing import Optional, Tuple
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
        m_summary = self.mem_ln(memory_tokens.mean(dim=1))  # [B, D]

        # Generate A(m): [B, r, d]
        A = self.A_gen(m_summary).view(Bsz, self.rank, self.dim)

        # Project x into low-rank space: [B, T, r]
        xB = x @ self.B  # (B,T,D) @ (D,r) -> (B,T,r)

        # Contract back to D with A(m): [B, T, D]
        # Δy[b, t, d] = sum_r xB[b, t, r] * A[b, r, d]
        delta = torch.einsum("btr,brd->btd", xB, A)

        return self.dropout(delta * self.scale)


class LoRAGenerator(nn.Module):
    """
    Generates token-wise A,B from per-token memory m:
      m: [B, T, d_m]
      A: [B, T, r, d_in]
      B: [B, T, d_out, r]
    """

    def __init__(
        self,
        d_m: int,
        d_in: int,
        d_out: int,
        r: int,
        hidden: Optional[int] = None,
    ):
        super().__init__()
        gen_size = r * (d_in + d_out)
        self.fc = nn.Sequential(
            nn.Linear(d_m, gen_size * 2),
            nn.GELU(),
            nn.Linear(gen_size * 2, gen_size, bias=False),
        )
        self.d_in, self.d_out, self.r = d_in, d_out, r

        # Stability gates (start near identity: no LoRA effect)
        self.gate_A = nn.Parameter(torch.tensor(1e-3), requires_grad=True)
        self.gate_B = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.reset_parameters()

    def forward(self, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # m: [B,d_m]
        Bsz, _ = m.shape
        vec = self.fc(m)  # [B,T, r*(d_in + d_out)]

        a_sz = self.r * self.d_in
        Avec, Bvec = vec[..., :a_sz], vec[..., a_sz:]
        A = Avec.view(Bsz, self.r, self.d_in)  # [B,r,d_in]
        B = Bvec.view(Bsz, self.r, self.d_out)  # [B,r,d_out]

        # Apply small gates
        A = self.gate_A * A
        B = self.gate_B * B
        return A, B

    def reset_parameters(self):
        nn.init.constant_(self.gate_A, 1e-3)
        nn.init.constant_(self.gate_B, 0.0)

        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5**0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class LowRankGenerator(nn.Module):
    "Generates only one low-rank matrix such as A or B"
    def __init__(self, in_features: int, out_features: int, r: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        gen_size = r * out_features
        self.fc = nn.Sequential(
            nn.Linear(in_features, gen_size * 2),
            nn.GELU(),
            nn.Linear(gen_size * 2, gen_size, bias=False),
        )
        self.gate = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        return self.gate * self.fc(x)

    def reset_parameters(self):
        nn.init.constant_(self.gate, 0.0)
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5**0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# DEPRECATED
class DynamicLoRALinearDEPRECATED(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        alpha: float = 2.0,
        gen_hidden: Optional[int] = None,
        use_bias: bool = True,
        dropout: float = 0.0,
        gen_type: str = "mm", # "mm": B(m) @ A(m), "xm": B(x) @ A(m), "mx": B(m) @ A(x)
        n_heads: int = 1,
        heads_is_input: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gen_type = gen_type
        assert gen_type in {"mm", "xm", "mx"}, "Invalid type"
        self.n_heads = n_heads
        self.heads_is_input = heads_is_input

        self.W0 = nn.Parameter(torch.empty(out_features, in_features), requires_grad=True)
        if use_bias:
            self.b0 = nn.Parameter(torch.empty(out_features), requires_grad=True)
        else:
            self.b0 = None

        if not heads_is_input:
            gen_out = out_features // n_heads # share BA across heads
            gen_in = in_features
            mem_in = in_features
        else:
            gen_out = out_features
            gen_in = in_features // n_heads
            mem_in = out_features

        self.gen = LoRAGenerator(
            d_m=mem_in, d_in=gen_in, d_out=gen_out, r=r, hidden=gen_hidden
        )
        self.scale = alpha / float(r)
        self.r = r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def forward(self, x: torch.Tensor, m_tok: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,in_features] where in_features = D*H (heads flattened)
        m_tok: [B,T, in_features] (per-token memory)
        """
        # generate A and B
        A, B = self.gen(
            self.dropout(m_tok)
        )  # A: [B,T,r,gen_in], B: [B,T,gen_out,r]

        # y = W0 @ x + (alpha/r) * B @ (A @ x)
        base = torch.einsum("od,btd->bto", self.W0, x)  # [B,T,out_features]

        if not self.heads_is_input:
            # QKV case: heads are in output dimension
            # A: [B,T,r,in_features], B: [B,T,out_features//n_heads,r]

            # Compute LoRA contribution for one head
            Ax = torch.einsum("btrd,btd->btr", A, x)  # [B,T,r]
            lora_contrib_single = torch.einsum(
                "btor,btr->bto", B, Ax
            )  # [B,T,out_features//n_heads]

            # Stack the full BA for each head
            lora_contrib = lora_contrib_single.repeat(1, 1, self.n_heads)  # [B,T,out_features]
        else:
            # Output projection case: heads are in input dimension
            # A: [B,T,r,in_features//n_heads], B: [B,T,out_features,r]

            # Compute LoRA contribution: B @ (A @ x)
            Ax = torch.einsum("btrd,btd->btr", A, x)  # [B,T,r] #TODO: fix shape mismatch on heads
            lora_contrib = torch.einsum(
                "btor,btr->bto", B, Ax
            )  # [B,T,out_features]

        y = base + self.scale * lora_contrib  # [B,T,out_features]

        if self.b0 is not None:
            y = y + self.b0.view(1, 1, -1)

        return y

    def reset_parameters(self):
        # Base weight/bias init
        nn.init.kaiming_uniform_(self.W0, a=5**0.5)
        if self.b0 is not None:
            nn.init.zeros_(self.b0)

        self.gen.reset_parameters()


class DynamicLoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        alpha: float = 2.0,
        use_bias: bool = True,
        gen_type: str = "A", 
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gen_type = gen_type
        assert gen_type in {"A", "B", "AB"}, "Invalid type"
   
        # base linear parameters
        self.W0 = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=True
        )
        self.b0 = nn.Parameter(
            torch.empty(out_features), requires_grad=True
        ) if use_bias else None
        
        if gen_type == "A":
            gen_out = in_features # A: r x in_features
            self.B = nn.Parameter(torch.empty(r, out_features), requires_grad=True)
        elif gen_type == "B":
            gen_out = out_features # B: r x out_features (will transpose in forward)
            self.A = nn.Parameter(torch.empty(r, in_features), requires_grad=True)
        elif gen_type == "AB":
            # gen both AB
            gen_out = out_features # AB: r x in_features
            
        self.gen = LowRankGenerator(in_features, gen_out, r) if not gen_type == "AB" else LoRAGenerator(in_features, in_features, gen_out, r)
        self.scale = alpha / float(r)
        self.r = r
        self.reset_parameters()
    
    def generate_weights(self, m_tok: torch.Tensor) -> torch.Tensor:
        if self.gen_type == "A":
            return self.gen(m_tok).view(-1, self.r, self.in_features), self.B
        elif self.gen_type == "B":
            return self.A, self.gen(m_tok).view(-1, self.r, self.out_features)
        elif self.gen_type == "AB":
            return self.gen(m_tok)

    def forward(self, x: torch.Tensor, m_tok: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,in_features] where in_features = D*H (heads flattened)
        m_tok: [B,T, in_features] (per-token memory)
        """

        # y = W0 @ x
        base = torch.einsum("od,btd->bto", self.W0, x)  # [B,T,out_features]

        # pool memory tokens over time
        m_tok_pooled = m_tok.mean(dim=1)

        # generate A and B
        A, B = self.generate_weights(m_tok_pooled)

        Am = torch.einsum("...rd,btd->...tr", A, m_tok)  # [B,T,r]

        BAm = torch.einsum("...ro,btr->...to", B, Am)  # [B,T,out_features]

        y = base + self.scale * BAm  # [B,T,out_features]

        if self.b0 is not None:
            y = y + self.b0.view(1, 1, -1)

        return y

    def reset_parameters(self):
        # Base weight/bias init
        nn.init.kaiming_uniform_(self.W0, a=5**0.5)
        if self.b0 is not None:
            nn.init.zeros_(self.b0)
        
        if self.gen_type == "A":
            nn.init.zeros_(self.B) # B init with zeros to start
        elif self.gen_type == "B":
            nn.init.constant_(self.A, 1e-3) # A init with small near identity weights

        self.gen.reset_parameters()


class FFNMemGenerator(nn.Module):
    """
    F(m): [B,T,in_features] → flattened params for memory FFN:
      W1m: (2r, out_features),  W2m: (r, 2r),  W3m: (out_features, r)
    We output a single big vector per token and split/reshape.
    """

    def __init__(
        self, in_features: int, out_features: int, r: int, hidden_mul=2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        hidden = hidden_mul * in_features

        # total params per token
        self.sz_W1 = r * out_features
        self.sz_W2 = r * out_features
        self.sz_W3 = out_features * r
        total = self.sz_W1 + self.sz_W2 + self.sz_W3
        # hidden = total * hidden_mul

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, total, bias=False),
        )

        # Stability gates: start the memory path near zero
        # (paper: W3_mem often zero-init; here we softly gate all)
        self.gate_all = nn.Parameter(
            torch.tensor(0.0)
        )  # 0 → no effect initially
        self.scale_W1 = nn.Parameter(torch.tensor(1e-3))  # tiny slope at start
        self.scale_W2 = nn.Parameter(torch.tensor(1e-3))
        self.scale_W3 = nn.Parameter(torch.tensor(1e-3))

        self.reset_parameters()

    def forward(
        self, m_tok: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        m_tok: [B,T,in_features]
        returns:
          W1m: [B,T,r,out_features]
          W2m: [B,T,r,out_features]
          W3m: [B,T,out_features,r]
        """
        B, T, _ = m_tok.shape
        vec = self.net(m_tok)  # [B,T,total]

        # split
        i1 = self.sz_W1
        i2 = i1 + self.sz_W2
        v1 = vec[..., :i1]
        v2 = vec[..., i1:i2]
        v3 = vec[..., i2:]

        W1m = v1.view(B, T, self.r, self.out_features)
        W2m = v2.view(B, T, self.r, self.out_features)
        W3m = v3.view(B, T, self.out_features, self.r)

        # gated/scaled output (start ~0; train to increase)
        g = torch.sigmoid(self.gate_all)  # [scalar in (0,1)]
        W1m = g * self.scale_W1 * W1m
        W2m = g * self.scale_W2 * W2m
        W3m = g * self.scale_W3 * W3m

        return W1m, W2m, W3m

    def reset_parameters(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5**0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.constant_(self.gate_all, 0.0)
        nn.init.constant_(self.scale_W1, 1e-3)
        nn.init.constant_(self.scale_W2, 1e-3)
        nn.init.constant_(self.scale_W3, 1e-3)
