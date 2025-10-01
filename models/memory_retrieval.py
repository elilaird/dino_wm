import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class NeuralMemory(nn.Module):
    """
    Simple deep memory MLP:
      retrieve: y = M*(q)   (forward without update)
      update:   M <- M - theta * grad(||M(k)-v||^2), with momentum & decay
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
                torch.zeros_like(p, dtype=torch.float32, device=p.device, requires_grad=False) for p in self.net.parameters()
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
                continue

            clip_coef = min(1.0, self.max_grad_norm / (gnorm + 1e-12))
            with torch.no_grad():
                for p, g, m in zip(
                    self.net.parameters(), safe_grads, self.momentum_buffers
                ):
                    g = g * clip_coef

                    # momentum update: m = eta*m - theta*g
                    m = m * float(self.eta) - g * float(self.theta)
                    m = torch.clamp(m, -self.momentum_clip, self.momentum_clip)
                    p = p * (1.0 - float(self.alpha)) + m

                    pn = p.norm()
                    if pn > self.weight_clip:
                        p.copy_(p * (self.weight_clip / (pn + 1e-12)))

        for p, m in zip(self.net.parameters(), self.momentum_buffers):
            p.requires_grad_(False)
            m.requires_grad_(False)

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
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

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


## --- SSM-based Memory Generators --- ##
class SSMCell(nn.Module):
    def __init__(self, d_model: int, n_state: int):
        super().__init__()
        self.U = nn.Linear(d_model, n_state, bias=False)  # write encoder
        self.C = nn.Linear(n_state, d_model, bias=False)  # read head
        self.log_tau = nn.Parameter(torch.zeros(n_state))  # time constants

    def discretize(self, dt: float = 1.0):
        tau = F.softplus(self.log_tau) + 1e-4
        Abar = torch.exp(-dt / tau)  # [S]
        Bbar = 1.0 - Abar  # [S]
        return Abar, Bbar

    @torch.no_grad()
    def init_state(self, B: int, P: int, n_state: int, device=None):
        return torch.zeros(B, P, n_state, device=device)

    def forward(self, X_t, H_t, dt: float = 1.0):
        """
        X_t: [B, P, D], H_t: [B, P, S]
        returns: H_{t+1}, M_{t+1}=[B,P,D]
        """
        Abar, Bbar = self.discretize(dt)  # [S]
        Abar = Abar.view(1, 1, -1)  # [1,1,S]
        Bbar = Bbar.view(1, 1, -1)
        Ux = self.U(X_t)  # [B,P,S]
        H_tp1 = Abar * H_t + Bbar * Ux
        M_tp1 = self.C(H_tp1)  # [B,P,D]
        return H_tp1, M_tp1


class MambaSSMCell(nn.Module):
    """
    Full Mamba-style selective SSM cell (diagonal state matrix).

    Shapes:
      X_t:  [B, P, D]
      H_t:  [B, P, S]
      step(...) -> (H_{t+1}, Y_{t+1}=[B,P,D])
      scan(X_win, H0) with X_win: [B, T, P, D] -> (H_seq: [B,T,P,S], Y_seq: [B,T,P,D], H_T: [B,P,S])
    """

    def __init__(self, d_model: int, n_state: int, dt_rank: int = 4):
        super().__init__()
        self.D = d_model
        self.S = n_state

        # ---- Fixed A (diagonal), stable: logA = -exp(a_hat) <= 0
        self.a_hat = nn.Parameter(torch.randn(self.S))

        # ---- Input-dependent carriers and selectors
        # u_t = U(x_t) (carrier that enters the state via B_t)
        self.U = nn.Linear(self.D, self.S, bias=False)

        # B_t = s_B(x_t), C_t = s_C(x_t)  (per-token modulators)
        self.sB = nn.Linear(self.D, self.S, bias=True)
        self.sC = nn.Linear(self.D, self.S, bias=True)

        # Δ_t = softplus( W2( SiLU(W1 x_t) ) )  -> scalar per (B,P,1) then broadcast to S
        self.sDelta1 = nn.Linear(self.D, dt_rank, bias=True)
        self.sDelta2 = nn.Linear(dt_rank, 1, bias=True)

        # ---- Readout: (C_t ⊙ h_t) -> D
        self.readout = nn.Linear(self.S, self.D, bias=False)

    def _logA(self):
        # logA ≤ 0 (vector of size S)
        return -torch.exp(self.a_hat)

    def _selective_params(self, X):
        """
        X: [B, P, D]
        Returns:
          A_t: [B, P, S], b_t: [B, P, S], C_t: [B, P, S]
        where recurrence is h' = A_t ⊙ h + b_t and output uses C_t.
        """
        B, P, D = X.shape
        logA = self._logA().view(1, 1, self.S)  # [1,1,S]

        # u_t and selectors
        u_t = self.U(X)  # [B,P,S]
        B_t = self.sB(X)  # [B,P,S]
        C_t = self.sC(X)  # [B,P,S]

        # Δ_t scalar per (B,P,1) -> broadcast to S
        dt_feat = F.silu(self.sDelta1(X))  # [B,P,R]
        dt = F.softplus(self.sDelta2(dt_feat))  # [B,P,1] >= 0
        dt = dt.expand(-1, -1, self.S)  # [B,P,S]

        # Discretized per-step decay: A_t = exp( dt * logA ), in (0,1]
        A_t = torch.exp(dt * logA)  # [B,P,S]

        # Input injection already discretized: b_t = (1 - A_t) * (B_t ⊙ u_t)
        b_t = (1.0 - A_t) * (B_t * u_t)  # [B,P,S]

        return A_t, b_t, C_t

    @torch.no_grad()
    def init_state(
        self, B: int, P: int, n_state: int = None, device=None, dtype=None
    ):
        S = self.S if n_state is None else n_state
        return torch.zeros(B, P, S, device=device, dtype=dtype)

    # ---------- One-step recurrent update (streaming/inference) ----------
    def step(self, X_t, H_t):
        """
        One selective step:
          X_t: [B,P,D], H_t: [B,P,S]
          returns: (H_{t+1}, Y_{t+1}=[B,P,D])
        """
        A_t, b_t, C_t = self._selective_params(X_t)  # [B,P,S]
        H_tp1 = A_t * H_t + b_t  # [B,P,S]
        Y_tp1 = self.readout(C_t * H_tp1)  # [B,P,D]
        return H_tp1, Y_tp1

    # ---------- Fast parallel scan over a window (training) ----------
    @staticmethod
    def _scan_window(A, b, H0):
        """
        Parallel prefix for diagonal recurrence with initial state:
          h_t = A_t ⊙ h_{t-1} + b_t
        Closed form (inclusive):
          P_t = Π_{j<=t} A_j
          h_t = P_t ⊙ (H0 + Σ_{k<=t} b_k / P_k)
        A,b:  [B, T, P, S]
        H0:   [B, P, S]
        returns: H_seq [B,T,P,S]
        """
        eps = 1e-12
        P = torch.cumprod(A.clamp_min(eps), dim=1)  # [B,T,P,S]
        invP = 1.0 / P.clamp_min(eps)  # [B,T,P,S]
        c = torch.cumsum(b * invP, dim=1)  # [B,T,P,S]
        H0e = H0.unsqueeze(1)  # [B,1,P,S]
        H_seq = P * (H0e + c)  # [B,T,P,S]
        return H_seq

    def scan(self, X_win, H0):
        """
        Parallel selective scan over a window:
          X_win: [B, T, P, D]
          H0:    [B, P, S]  (carried state from previous window)
        returns:
          H_seq: [B, T, P, S]
          Y_seq: [B, T, P, D]
          H_T:   [B, P, S]
        """
        B, T, P, D = X_win.shape
        # Flatten time into batch for parameter eval, then reshape back
        Xf = rearrange(X_win, "b t p d -> (b t) p d")
        A_t, b_t, C_t = self._selective_params(Xf)  # [BT,P,S] each
        A_t = rearrange(A_t, "(b t) p s -> b t p s", b=B, t=T)
        b_t = rearrange(b_t, "(b t) p s -> b t p s", b=B, t=T)
        C_t = rearrange(C_t, "(b t) p s -> b t p s", b=B, t=T)

        # Parallel recurrence with initial state H0
        H_seq = self._scan_window(A_t, b_t, H0)  # [B,T,P,S]

        # Readout (token-dependent C_t gates the state)
        Y_seq = self.readout((C_t * H_seq).reshape(B * T, P, self.S)).view(
            B, T, P, self.D
        )
        Y_seq = rearrange(Y_seq, "b t p d -> b (t p) d")

        H_T = H_seq[:, -1, :, :]  # [B,P,S]
        return H_seq, Y_seq, H_T

    # ---------- Backward-compatible forward ----------
    def forward(self, X_t, H_t=None, dt: float = 1.0, mode: str = "step"):
        """
        mode == "step":   X_t: [B,P,D], H_t:[B,P,S] -> (H_{t+1}, Y_{t+1})
        mode == "scan":   X_t: [B,T,P,D], H_t:[B,P,S] -> (H_seq, Y_seq, H_T)
        """
        if mode == "step":
            if H_t is None:
                H_t = self.init_state(
                    X_t.size(0),
                    X_t.size(1),
                    device=X_t.device,
                    dtype=X_t.dtype,
                )
            return self.step(X_t, H_t)
        elif mode == "scan":
            assert (
                H_t is not None
            ), "H_t (initial state) required for scan mode"
            return self.scan(X_t, H_t)
        else:
            raise ValueError("mode must be 'step' or 'scan'")