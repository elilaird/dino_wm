import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralMemory(nn.Module):
    """
    Simple deep memory MLP:
      retrieve: y = M*(q)   (forward without update)
      update:   M <- M - step * grad(||M(k)-v||^2), with momentum & decay
    """

    def __init__(self, d_model: int, hidden_scale: int = 2, depth: int = 2, 
                 eta: float = 0.9, theta: float = 1e-3, alpha: float = 1e-5):
        super().__init__()
        self.eta = torch.tensor(eta)  # surprise momentum decay (η_t)
        self.theta = torch.tensor(theta)  # learning rate (θ_t)
        self.alpha = torch.tensor(alpha)  # weight decay (α_t) ~ forgetting
        d_hidden = d_model * hidden_scale

        layers = []
        dims = [d_model] + [d_hidden] * (depth - 1) + [d_model]
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1])]
            if i < len(dims) - 2:
                layers += [nn.SiLU()]
        self.net = nn.Sequential(*layers)

        # momentum buffer for "surprise" (Equation: S_t)
        self.register_buffer(
            "S_buf", torch.zeros(1)
        )  # placeholder; per-param momentum lives in optimizer-like buffers
        # per-parameter momentum:
        self.momentum_buffers = [
            torch.zeros_like(p) for p in self.net.parameters()
        ]

    def _memory_update_online(
        self,
        k: torch.Tensor,  # [B, T, d]
        v: torch.Tensor,  # [B, T, d]
    ):
        """
        One online update over a (mini)batch of (k, v).
        Implements: S_t = eta * S_{t-1} - theta * grad; W <- (1-alpha) * W + S_t
        Here we maintain per-parameter momentum buffers.
        """

        k = k.detach()
        v = v.detach()

        # compute associative loss over the batch
        def assoc_loss():
            y = self.net(k)  # predict v
            return F.mse_loss(y, v)

        # compute grads
        for p in self.net.parameters():
            p.requires_grad_(True)

        loss = assoc_loss()
        grads = torch.autograd.grad(
            loss,
            list(self.net.parameters()),
            create_graph=False,
            retain_graph=False,
        )

        # manual momentum + decay update
        with torch.no_grad():
            for p, g, m in zip(
                self.net.parameters(), grads, self.momentum_buffers
            ):
                # momentum (S_t)
                m.mul_(self.eta).add_(
                    g, alpha=-self.theta
                )  # m = eta*m - theta*g  
                # forgetting (weight decay) and step:
                p.mul_(1.0 - self.alpha).add_(m)  # W = (1-alpha)W + m

        # clear grads to keep graph light
        for p in self.net.parameters():
            p.grad = None
            p.requires_grad_(False)

    def retrieve(self, q: torch.Tensor) -> torch.Tensor:
        """M*(q): forward pass without weight update."""
        return self.net(q)

    def update_from_batch(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """public method to update memory weights online"""
        self._memory_update_online(k, v)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.retrieve(q)

    def reset_weights(self):
        self.net.reset_parameters()
        self.S_buf.zero_()
        for m in self.momentum_buffers:
            m.zero_()





# ============================================================================
# Drop-in replacement for existing Transformer
# ============================================================================

# class TitansTransformer(nn.Module):
#     """
#     Drop-in replacement for the existing Transformer that supports all three variants.
#     """
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, 
#                  memory_variant="MAC", d_k=64, d_v=64, n_persist=16, window=512,
#                  alpha=0.02, lr=0.1, beta=0.9, dropout=0.):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
        
#         # Choose memory variant
#         if memory_variant == "MAC":
#             block_cls = MACBlock
#             block_kwargs = {"n_persist": n_persist}
#         elif memory_variant == "MAG":
#             block_cls = MAGBlock
#             block_kwargs = {"window": window}
#         elif memory_variant == "MAL":
#             block_cls = MALBlock
#             block_kwargs = {}
#         else:
#             raise ValueError(f"Unknown memory variant: {memory_variant}")
        
#         # Create memory blocks
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(block_cls(
#                 dim, heads, dim_head, mlp_dim, d_k, d_v,
#                 alpha=alpha, lr=lr, beta=beta, dropout=dropout,
#                 **block_kwargs
#             ))

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x) + x  # residual connection
#         return self.norm(x)


# # ============================================================================
# # Drop-in replacement for ViTPredictor
# # ============================================================================

# class TitansViTPredictor(nn.Module):
#     """
#     Drop-in replacement for ViTPredictor that uses Titans memory.
#     """
#     def __init__(self, *, num_patches, num_frames, dim, depth, heads, mlp_dim, 
#                  memory_variant="MAC", d_k=64, d_v=64, n_persist=16, window=512,
#                  pool='cls', dim_head=64, dropout=0., emb_dropout=0.,
#                  alpha=0.02, lr=0.1, beta=0.9):
#         super().__init__()
#         assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
#         self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * num_patches, dim))
#         self.dropout = nn.Dropout(emb_dropout)
#         self.transformer = TitansTransformer(
#             dim, depth, heads, dim_head, mlp_dim,
#             memory_variant=memory_variant, d_k=d_k, d_v=d_v,
#             n_persist=n_persist, window=window,
#             alpha=alpha, lr=lr, beta=beta, dropout=dropout
#         )
#         self.pool = pool

#     def forward(self, x):
#         b, n, _ = x.shape
#         x = x + self.pos_embedding[:, :n]
#         x = self.dropout(x)
#         x = self.transformer(x)
#         return x
