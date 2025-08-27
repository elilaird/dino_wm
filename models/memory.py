import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralMemory(nn.Module):
    """
    Simple deep memory MLP:
      retrieve: y = M*(q)   (forward without update)
      update:   M <- M - step * grad(||M(k)-v||^2), with momentum & decay
    """

    def __init__(
        self,
        d_model: int,
        hidden_scale: int = 2,
        depth: int = 2,
        eta: float = 0.9,
        theta: float = 1e-3,
        alpha: float = 1e-5,
        max_grad_norm: float = 1,
        momentum_clip: float = 1.0,
        weight_clip: float = 5.0,
        update_steps: int = 1,
    ):
        super().__init__()
        self.register_buffer(
            "eta", torch.tensor(eta)
        )  # surprise momentum decay (η_t)
        self.register_buffer(
            "theta", torch.tensor(theta)
        )  # learning rate (θ_t)
        self.register_buffer(
            "alpha", torch.tensor(alpha)
        )  # weight decay (α_t) ~ forgetting
        self.max_grad_norm = max_grad_norm
        self.momentum_clip = momentum_clip
        self.weight_clip = weight_clip
        self.update_steps = update_steps
        d_hidden = d_model * hidden_scale

        layers = []
        dims = [d_model] + [d_hidden] * (depth - 1) + [d_model]
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1])]
            if i < len(dims) - 2:
                layers += [nn.SiLU()]
        self.net = nn.Sequential(*layers)

        # update weights manually using surprise
        for param in self.net.parameters():
            param.requires_grad = False

        # momentum buffer for "surprise" (Equation: S_t)
        self.register_buffer(
            "S_buf", torch.zeros(1)
        )  # placeholder; per-param momentum lives in optimizer-like buffers
        # per-parameter momentum buffers - will be initialized after first forward pass
        self.momentum_buffers = []

    @torch.enable_grad()
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

        k = k.detach().to(torch.float32)
        v = v.detach().to(torch.float32)

        # Initialize momentum buffers on the correct device if not already done
        if len(self.momentum_buffers) == 0:
            self.momentum_buffers = [
                torch.zeros_like(p, dtype=torch.float32, device=p.device) for p in self.net.parameters()
            ]

        # compute grads & ensure float 32
        for p in self.net.parameters():
            if p.dtype != torch.float32:
                p = p.to(torch.float32)
            p.requires_grad_(True)

        # compute associative loss over the batch
        def assoc_loss():
            y = self.net(k)  # predict v
            return F.mse_loss(y, v)

        for step in range(self.update_steps):

            loss = assoc_loss()
            grads = torch.autograd.grad(
                loss,
                list(self.net.parameters()),
                create_graph=False,
                retain_graph=False,
            )

            # finite check + cast to fp32
            safe_grads = []
            for g in grads:
                g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).float()
                safe_grads.append(g)

            # global L2 norm (fp32)
            gnorm = torch.sqrt(sum((g.norm(p=2) ** 2 for g in safe_grads)))
            if not torch.isfinite(gnorm):
                # skip this step entirely
                continue

            clip_coef = min(1.0, self.max_grad_norm / (gnorm + 1e-12))
            with torch.no_grad():
                for p, g, m in zip(
                    self.net.parameters(), safe_grads, self.momentum_buffers
                ):
                    g = g * clip_coef

                    # momentum update: m = eta*m - theta*g
                    m.mul_(float(self.eta)).add_(g, alpha=-float(self.theta))

                    # clamp momentum and apply decoupled weight decay + momentum
                    m.clamp_(-self.momentum_clip, self.momentum_clip)
                    p.mul_(1.0 - float(self.alpha)).add_(m)

                # optional: clip **by tensor norm** instead of elementwise
                for p in self.net.parameters():
                    pn = p.norm()
                    if pn > self.weight_clip:
                        p.mul_(self.weight_clip / (pn + 1e-12))

            for p in self.net.parameters():
                p.grad = None

        for p in self.net.parameters():
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
        self.momentum_buffers = []  # Reset momentum buffers


class LookupMemory(nn.Module):
    def __init__(self, d_model: int, bank_size: int):
        super().__init__()
        self.d_model = d_model
        self.bank_size = bank_size
        self.register_buffer(
            "memory_bank", torch.empty(bank_size, 0, d_model)
        )

    def retrieve(self):
        return self.memory_bank.clone()
    
    def update(self, batch):
        self.memory_bank = torch.cat([
            self.memory_bank,
            batch.detach().clone()
        ], dim=1)

    def forward(self):
        return self.retrieve()

    def reset_weights(self):
        self.memory_bank = torch.empty(self.bank_size, 0, self.d_model)
