import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import FeedForward


# ---------------------------------------------------------------------------
# Pure-PyTorch Mamba block (no mamba_ssm dependency)
# ---------------------------------------------------------------------------


def selective_scan(u, delta, A, B, C, D):
    """Non-optimised selective scan (S6) in pure PyTorch.

    Args:
        u:     (B, L, d_inner)  — input after conv + SiLU
        delta: (B, L, d_inner)  — per-step time-scale (after softplus)
        A:     (d_inner, d_state) — state transition (log-space on entry)
        B:     (B, L, d_state)  — input-dependent selector
        C:     (B, L, d_state)  — output-dependent selector
        D:     (d_inner,)       — skip/residual gain

    Returns:
        y:     (B, L, d_inner)
    """
    B_batch, L, d_inner = u.shape
    d_state = A.shape[1]

    # Discretise: Ā_t = exp(Δ_t · A),  B̄_t = Δ_t · B_t
    # delta: (B,L,d_inner) -> (B,L,d_inner,1); A: (d_inner,d_state) -> (1,1,d_inner,d_state)
    deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))   # (B,L,d_inner,d_state)
    deltaB_u = (
        delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
    )  # (B,L,d_inner,d_state)

    # Sequential scan (cannot be parallelised without associative-scan kernels)
    h = torch.zeros(B_batch, d_inner, d_state, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(L):
        h = deltaA[:, t] * h + deltaB_u[:, t]          # (B, d_inner, d_state)
        y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)    # (B, d_inner)
        ys.append(y_t)

    y = torch.stack(ys, dim=1)  # (B, L, d_inner)
    y = y + u * D.unsqueeze(0).unsqueeze(0)
    return y


class MambaBlock(nn.Module):
    """Single Mamba block: in-projection → 1-D causal conv → SSM → out-projection.

    Follows the architecture from *Mamba: Linear-Time Sequence Modeling with
    Selective State Spaces* (Gu & Dao, 2023).  Uses a naive sequential scan
    instead of the custom CUDA selective-scan kernel.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        self.dt_rank = max(d_model // 16, 1)

        # --- projections ---
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # --- 1-D causal depthwise convolution ---
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )

        # --- SSM parameters ---
        # Δ projection: x → dt_rank → d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        # x → (Δ_rank + B + C) combined projection for efficiency
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)

        # A is kept in log-space for numerical stability
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D (skip connection scalar per channel)
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, L, d_model)
        """
        B, L, _ = x.shape

        # Split into two branches: one for SSM, one for gating
        xz = self.in_proj(x)                              # (B, L, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)                    # each (B, L, d_inner)

        # Causal 1-D conv  (channels-first for Conv1d)
        x_ssm = x_ssm.transpose(1, 2)                     # (B, d_inner, L)
        x_ssm = self.conv1d(x_ssm)[:, :, :L]              # trim to causal length
        x_ssm = x_ssm.transpose(1, 2)                     # (B, L, d_inner)
        x_ssm = F.silu(x_ssm)

        # Compute Δ, B, C from x_ssm
        x_dbc = self.x_proj(x_ssm)                        # (B, L, dt_rank + 2*d_state)
        dt, B_sel, C_sel = x_dbc.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt))               # (B, L, d_inner)

        A = -torch.exp(self.A_log)                         # (d_inner, d_state)
        y = selective_scan(x_ssm, delta, A, B_sel, C_sel, self.D)

        # Gate and project out
        y = y * F.silu(z)
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# Full predictor
# ---------------------------------------------------------------------------


class MambaPredictor(nn.Module):
    """
    Mamba-based predictor that replaces the ViT transformer with selective
    state space model blocks. Processes the flattened sequence of
    (num_frames * num_patches) DINOv2 tokens with O(L) complexity.

    Uses a pure-PyTorch selective scan (no mamba_ssm dependency).

    Causality is implicit: Mamba's left-to-right recurrence ensures that
    patches in frame t only depend on frames 0..t, with no explicit
    attention masks needed.
    """

    def __init__(
        self,
        *,
        num_patches: int,
        num_frames: int,
        dim: int,
        depth: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        emb_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.dim = dim
        self.depth = depth

        # Learned positional embedding (same pattern as ViTPredictor)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames * num_patches, dim)
        )
        self.emb_dropout = nn.Dropout(emb_dropout)

        # Stack of Mamba + FeedForward blocks with pre-norm residual connections
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(dim),
                        MambaBlock(
                            d_model=dim,
                            d_state=d_state,
                            d_conv=d_conv,
                            expand=expand,
                        ),
                        nn.Dropout(dropout),
                        # FeedForward from vit.py includes its own LayerNorm
                        FeedForward(dim, dim * expand, dropout),
                    ]
                )
            )

        self.ln_out = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_frames * num_patches, dim) flattened visual embeddings
        Returns:
            (B, num_frames * num_patches, dim) predicted embeddings
        """
        b, n, _ = x.shape

        x = x + self.pos_embedding[:, :n]
        x = self.emb_dropout(x)

        for ln, mamba, drop, ff in self.layers:
            x = x + drop(mamba(ln(x)))
            x = ff(x) + x

        return self.ln_out(x), None

    def reset_memory(self):
        """No-op. Mamba uses parallel scan with no persistent state between
        forward calls (same stateless behavior as ViTPredictor). Included
        for interface compatibility with visual_world_model.py rollout()."""
        pass
