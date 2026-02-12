import torch
import torch.nn as nn
from mamba_ssm import Mamba

from .vit import FeedForward


class MambaPredictor(nn.Module):
    """
    Mamba-based predictor that replaces the ViT transformer with selective
    state space model blocks. Processes the flattened sequence of
    (num_frames * num_patches) DINOv2 tokens with O(L) complexity.

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
                        Mamba(
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

        return self.ln_out(x)

    def reset_memory(self):
        """No-op. Mamba uses parallel scan with no persistent state between
        forward calls (same stateless behavior as ViTPredictor). Included
        for interface compatibility with visual_world_model.py rollout()."""
        pass
