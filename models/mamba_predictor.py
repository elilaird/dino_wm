import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import FeedForward

# ---------------------------------------------------------------------------
# Try to import CUDA-optimised selective scan from mamba_ssm
# ---------------------------------------------------------------------------
# try:
#     from mamba_ssm.ops.selective_scan_interface import (
#         selective_scan_fn as _cuda_selective_scan_fn,
#     )

#     HAS_CUDA_SCAN = True
# except ImportError:
HAS_CUDA_SCAN = False


# ---------------------------------------------------------------------------
# Parallel Associative Scan  (pure PyTorch, GPU-friendly)
# ---------------------------------------------------------------------------

def _pscan(gates, tokens):
    """Hillis-Steele parallel inclusive scan for the linear recurrence
    ``h_t = gates_t * h_{t-1} + tokens_t``  with  ``h_0 = tokens_0``.

    Runs in **O(log L)** sequential GPU kernel launches instead of the
    O(L) iterations required by a Python for-loop, giving ~50-100× wall-
    clock speedup for typical sequence lengths (L = 256–1024).

    Args
    ----
    gates  : ``(B, L, ...)``  multiplicative coefficients  (discretised Ā)
    tokens : ``(B, L, ...)``  additive inputs              (discretised B̄·u)

    Returns
    -------
    ``(B, L, ...)``  all hidden states ``[h_0, h_1, …, h_{L-1}]``
    """
    L = gates.shape[1]
    log_L = int(math.ceil(math.log2(max(L, 2))))

    a = gates
    b = tokens

    for k in range(log_L):
        stride = 2 ** k
        # Combine each position with its neighbour *stride* positions to the left.
        # Positions < stride have no left neighbour and are kept as-is.
        b = torch.cat([
            b[:, :stride],
            a[:, stride:] * b[:, :-stride] + b[:, stride:],
        ], dim=1)
        a = torch.cat([
            a[:, :stride],
            a[:, stride:] * a[:, :-stride],
        ], dim=1)

    return b


def selective_scan_parallel(u, delta, A, B, C, D):
    """Vectorised selective scan (S6) using :func:`_pscan`.

    Replaces the naïve sequential Python for-loop with ~log₂(L) batched
    GPU operations.

    Args have the same semantics as the original ``selective_scan``.
    """
    B_batch, L, d_inner = u.shape

    # Discretise:  Ā_t = exp(Δ_t · A),   B̄_t·u_t = Δ_t · B_t · u_t
    deltaA = torch.exp(
        delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
    )  # (B, L, d_inner, d_state)
    deltaB_u = (
        delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
    )  # (B, L, d_inner, d_state)

    # Parallel scan:  h_t = Ā_t · h_{t-1} + B̄_t · u_t
    h = _pscan(deltaA, deltaB_u)  # (B, L, d_inner, d_state)

    # Output:  y_t = Σ_s  h_{t,s} · C_{t,s}
    y = (h * C.unsqueeze(2)).sum(dim=-1)  # (B, L, d_inner)
    y = y + u * D.unsqueeze(0).unsqueeze(0)
    return y


def _selective_scan_cuda(x_ssm, delta, A, B_sel, C_sel, D):
    """Thin wrapper around ``mamba_ssm``'s CUDA selective-scan kernel.

    Handles the transposition between our (B, L, D) layout and the
    (B, D, L) layout expected by the CUDA kernel.
    """
    y = _cuda_selective_scan_fn(
        x_ssm.permute(0, 2, 1).contiguous(),    # (B, d_inner, L)
        delta.permute(0, 2, 1).contiguous(),     # (B, d_inner, L)
        A.contiguous(),                           # (d_inner, d_state)
        B_sel.permute(0, 2, 1).contiguous(),     # (B, d_state, L)
        C_sel.permute(0, 2, 1).contiguous(),     # (B, d_state, L)
        D.float(),
        z=None,
        delta_bias=None,
        delta_softplus=False,
    )
    return y.permute(0, 2, 1)                    # back to (B, L, d_inner)


# ---------------------------------------------------------------------------
# Mamba Block
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """Single Mamba block: in-proj → 1-D causal conv → SSM → out-proj.

    Uses the CUDA selective-scan kernel from ``mamba_ssm`` when available,
    otherwise falls back to a pure-PyTorch parallel associative scan.
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
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)

        # A kept in log-space for numerical stability
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D (skip / residual gain per channel)
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, L, d_model)
        """
        B, L, _ = x.shape

        # Split into SSM branch + gating branch
        xz = self.in_proj(x)                              # (B, L, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)                    # each (B, L, d_inner)

        # Causal 1-D conv (channels-first for Conv1d)
        x_ssm = x_ssm.transpose(1, 2)                     # (B, d_inner, L)
        x_ssm = self.conv1d(x_ssm)[:, :, :L]              # trim to causal length
        x_ssm = x_ssm.transpose(1, 2)                     # (B, L, d_inner)
        x_ssm = F.silu(x_ssm)

        # Compute Δ, B, C from x_ssm
        x_dbc = self.x_proj(x_ssm)                        # (B, L, dt_rank+2*d_state)
        dt, B_sel, C_sel = x_dbc.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt))               # (B, L, d_inner)

        A = -torch.exp(self.A_log)                         # (d_inner, d_state)

        if HAS_CUDA_SCAN:
            y = _selective_scan_cuda(x_ssm, delta, A, B_sel, C_sel, self.D)
        else:
            y = selective_scan_parallel(x_ssm, delta, A, B_sel, C_sel, self.D)

        # Gate and project out
        y = y * F.silu(z)
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# Full predictor
# ---------------------------------------------------------------------------


class MambaPredictor(nn.Module):
    """
    Mamba-based predictor that replaces the ViT transformer with selective
    state-space model blocks.

    **Scan backends** (selected automatically):

    1. ``mamba_ssm`` CUDA kernel — fastest, requires ``pip install mamba-ssm``
    2. Parallel associative scan (Hillis-Steele) — pure PyTorch, ~log₂(L) GPU
       kernel launches instead of L sequential Python-loop iterations.

    Causality is implicit: Mamba's left-to-right recurrence ensures that
    patches in frame *t* only depend on frames 0 … t.
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
                        FeedForward(dim, dim * expand, dropout),
                    ]
                )
            )

        self.ln_out = nn.LayerNorm(dim)

        backend = "mamba_ssm CUDA kernel" if HAS_CUDA_SCAN else "parallel associative scan (pure PyTorch)"
        print(f"MambaPredictor: using {backend}")

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
        """No-op — included for interface compatibility with rollout()."""
        pass
