# adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F

from .memory import NeuralMemory, LookupMemory

# helpers
NUM_FRAMES = 1
NUM_PATCHES = 1


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def generate_mask_matrix(npatch, nwindow):
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)
    rows = []
    for i in range(nwindow):
        row = torch.cat([ones] * (i + 1) + [zeros] * (nwindow - i - 1), dim=1)
        rows.append(row)
    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask


def generate_sliding_window_mask(seq_len, window_size):
    """Generate mask for sliding window attention"""
    mask = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, start : i + 1] = 1
    return mask.unsqueeze(0).unsqueeze(0)


def generate_mac_mask_matrix(npatch, nwindow, n_persistent, n_retrieved):
    """
    Generate frame-level mask for MAC transformer using the same pattern as generate_mask_matrix
    but accounting for persistent tokens and memory frames.
    """
    total_frames = n_persistent + n_retrieved + nwindow

    # Create blocks for each frame type
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)

    rows = []

    # Persistent token rows (can attend to everything)
    for i in range(n_persistent):
        row = torch.cat([ones] * total_frames, dim=1)
        rows.append(row)

    # Memory frame rows (can attend to everything)
    for i in range(n_retrieved):
        row = torch.cat([ones] * total_frames, dim=1)
        rows.append(row)

    # Main sequence rows (frame-level causality + access to persistent/memory)
    for i in range(nwindow):
        # Allow attention to persistent tokens (all frames can attend to persistent tokens)
        persistent_blocks = [ones] * n_persistent

        # Allow attention to memory frames (all frames can attend to memory frames)
        memory_blocks = [ones] * n_retrieved

        # Allow attention to current and previous frames in main sequence (frame-level causality)
        main_blocks = [ones] * (i + 1) + [zeros] * (nwindow - i - 1)

        row = torch.cat(persistent_blocks + memory_blocks + main_blocks, dim=1)
        rows.append(row)

    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

        bias = generate_mask_matrix(NUM_PATCHES, NUM_FRAMES)
        self.register_buffer("bias", bias)

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # apply causal mask
        dots = dots.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViTPredictor(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        # update params for adding causal attention masks
        global NUM_FRAMES, NUM_PATCHES
        NUM_FRAMES = num_frames
        NUM_PATCHES = num_patches

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames * (num_patches), dim)
        )  # dim for the pos encodings
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )
        self.pool = pool

    def forward(self, x):
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x)
        return x


class ViTPredictorWithPersistentTokens(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        n_persist=0,
    ):
        super().__init__()
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        # update params for adding causal attention masks
        global NUM_FRAMES, NUM_PATCHES
        NUM_FRAMES = num_frames + n_persist
        NUM_PATCHES = num_patches

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames * (num_patches), dim)
        )  # dim for the pos encodings
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )
        self.pool = pool
        self.n_persist = n_persist
        if n_persist > 0:
            self.persistent_tokens = nn.Parameter(
                torch.randn(1, n_persist, dim)
            )
        else:
            self.persistent_tokens = None

    def forward(
        self, x
    ):  # x: (b, window_size * H/patch_size * W/patch_size, 384)
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)

        # add persistent tokens
        if self.persistent_tokens is not None:
            x = torch.cat([self.persistent_tokens.repeat(b, 1, 1), x], dim=1)

        x = self.transformer(x)

        # drop persistent tokens
        if self.persistent_tokens is not None:
            x = x[:, self.n_persist :, :]
        return x


class AdditiveControlTransformer(Transformer):

    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        action_emb_dim,
        alpha_init=0.1,
        dropout=0.0,
    ):
        super().__init__(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.action_emb_dim = action_emb_dim
        self.injection_layers = nn.ModuleList(
            [nn.Linear(action_emb_dim, dim) for _ in range(depth)]
        )
        # Per-layer alpha parameters
        self.alphas = nn.ParameterList(
            [nn.Parameter(torch.tensor(alpha_init)) for _ in range(depth)]
        )

    def forward(self, x, actions):
        for i, (attn, ff) in enumerate(self.layers):
            x = attn(x) + x
            x = ff(x) + x
            if i < len(self.injection_layers):
                # apply injection to visual patches only
                injection = self.injection_layers[i](actions) * self.alphas[i]
                injection = torch.cat(
                    [
                        injection.repeat(1, x.shape[1] - 1, 1),
                        torch.zeros_like(x[:, :1]),
                    ],
                    dim=1,
                )
                x = x + injection
        return self.norm(x)


class AdditiveControlViTPredictor(nn.Module):
    """
    ViT predictor with additive control injection for actions.
    Separates action processing from visual/proprio processing to avoid gradient entanglement.
    """

    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        alpha_init=0.1,
    ):
        super().__init__()
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        # update params for adding causal attention masks
        global NUM_FRAMES, NUM_PATCHES
        NUM_FRAMES = num_frames
        NUM_PATCHES = num_patches

        self.num_patches = num_patches
        self.num_frames = num_frames
        self.dim = dim
        self.pool = pool

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames * num_patches, dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = AdditiveControlTransformer(
            dim, depth, heads, dim_head, mlp_dim, dim, alpha_init, dropout
        )

    def forward(self, x):
        """
        x: (b, num_frames * num_patches, dim) - visual/proprio embeddings only
        actions: (b, num_frames, action_dim) - separate action input (optional)
        """
        b, n, _ = x.shape

        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)

        action_emb = x[:, -1:, :].clone()  # (b, 1, dim)
        x = x[:, :-1, :]  # remove the action token
        x = self.transformer(x, action_emb)
        x = torch.cat([x, action_emb], dim=1)
        return x


class MAGTransformerBlock(nn.Module):
    def __init__(
        self,
        mem: NeuralMemory,
        dim,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        gate_type: str = "sigmoid",
    ):
        super().__init__()
        self.mem = mem
        self.gate_type = gate_type
        self.norm = nn.LayerNorm(dim)
        self.attention = Attention(dim, heads, dim_head, dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout)
        self.W_y = nn.Linear(dim, dim)
        self.W_m = nn.Linear(dim, dim)
        self.W_Q = nn.Linear(dim, dim)
        

    def forward(self, x):
        x_in = x
        x = x + self.attention(x)
        x = torch.sigmoid(self.W_y(x)) * self.W_m(self.mem.retrieve(self.W_Q(x)))
        x = x + self.ff(x)

        # update memory
        self.mem.update_from_batch(F.normalize(x_in, p=2, dim=-1).detach(), F.normalize(x, p=2, dim=-1).detach())

        return self.norm(x)
    
    def reset_memory(self):
        self.mem.reset_weights()


class MAGTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        gate_type: str = "sigmoid",
        hidden_scale: float = 1.0,
        eta: float = 0.9,
        theta: float = 1e-3,
        alpha: float = 1e-5,
        mem_depth: int = 1,
    ):
        super().__init__()
        self.gate_type = gate_type
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                MAGTransformerBlock(
                    mem=NeuralMemory(
                        d_model=dim,
                        hidden_scale=hidden_scale,
                        depth=mem_depth,
                        eta=eta,
                        theta=theta,
                        alpha=alpha,
                    ),
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    gate_type=gate_type,
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def reset_memory(self):
        for layer in self.layers:
            layer.reset_memory()


class MAGViTPredictor(nn.Module):
    def __init__(
        self,
        num_patches: int,
        num_frames: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        emb_dropout=0.0,
        hidden_scale: float = 1.0,
        mem_eta: float = 0.9,
        mem_theta: float = 1e-3,
        mem_alpha: float = 1e-5,
        mem_depth: int = 1,
        dim_head: int = 64,
        gate_type: str = "sigmoid",
        pool: str = "mean",
    ):
        super().__init__()

        # update params for adding causal attention masks
        global NUM_FRAMES, NUM_PATCHES
        NUM_FRAMES = num_frames
        NUM_PATCHES = num_patches

        self.num_patches = num_patches
        self.num_frames = num_frames
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames * num_patches, dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = MAGTransformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            gate_type=gate_type,
            hidden_scale=hidden_scale,
            eta=mem_eta,
            theta=mem_theta,
            alpha=mem_alpha,
            mem_depth=mem_depth,
        )

    def forward(self, x):
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        return self.transformer(x)

    def reset_memory(self):
        self.transformer.reset_memory()


class MACTransformerBlock(nn.Module):
    """
    A standard Transformer block where the input *segment* is augmented with:
      [ Persistent P | Retrieved long-term h_t | Segment tokens ]
    and then passed through attention and FFN.
    """

    def __init__(
        self,
        mem: NeuralMemory,
        num_patches: int,  # number of patches per frame
        num_frames: int,  # number of frames in segment
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_persistent: int = 4,  # number of persistent tokens P
        n_retrieved: int = 4,  # number of memory "slots" to prepend (learned projection of M*(Q))
        dropout: float = 0.0,
        dim_head: int = 64,
        use_slots: bool = True,
        update_type: str = "selfattention",  # "selfattention" or "crossattention"
        proj_k_eq_q: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.mem = mem
        self.use_slots = use_slots
        self.n_persistent = n_persistent
        self.n_retrieved = (
            n_retrieved if self.use_slots else num_frames * num_patches
        )
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.update_type = update_type
        self.proj_k_eq_q = proj_k_eq_q

        if self.update_type == "crossattention":
            assert (
                not self.use_slots
            ), "use_slots must be False for crossattention"

        # self.h_norm = nn.LayerNorm(d_model, eps=1e-5)
        # self.h_norm = F.normalize(x, p=2, dim=-1)
        # self.q_norm = nn.LayerNorm(d_model, eps=1e-5)
        # self.q_norm = lambda x: torch.nn.functional.normalize(x, p=2, dim=-1)

        if self.n_persistent > 0:
            self.P = nn.Parameter(torch.randn(n_persistent, d_model))
            # self.p_norm = nn.LayerNorm(d_model, eps=1e-5)
            # self.p_norm = lambda x: torch.nn.functional.normalize(x, p=2, dim=-1)

        self.mem_W_Q = nn.Linear(d_model, d_model)

        if self.use_slots:
            self.mem_slots = nn.Linear(d_model, n_retrieved * d_model)

        bias = generate_mac_mask_matrix(
            num_patches, num_frames, n_persistent, n_retrieved
        )
        self.attention = Attention(d_model, n_heads, dim_head, dropout)
        self.attention.register_buffer("bias", bias)

        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def reset_memory(self):
        self.mem.reset_weights()

    def forward(
        self,
        x_seg: torch.Tensor,  # [B, T, d] current segment
        update_memory: bool = True,
    ) -> torch.Tensor:
        B, T, D = x_seg.shape
        device = x_seg.device

        # retrieve long-term memory for current segment
        q_t = self.mem_W_Q(F.normalize(x_seg, p=2, dim=-1))  # [B, T, d_model]
        h = self.mem.retrieve(q_t)  # [B, T, d_model]

        if self.use_slots:
            # compress retrieved "per-token" memory into a fixed number of slots
            h = self.mem_slots(h)  # [B, T, n_retrieved*d_model]
            h = h.mean(dim=1).view(  # TODO: try different pooling
                B, self.n_retrieved, self.d_model
            )  # [B, n_retrieved, d_model] (pool over time in segment)
        else:
            self.n_retrieved = T

        h = F.normalize(h, p=2, dim=-1)

        # prepend persistent tokens and retrieved memory slots
        if self.n_persistent > 0:
            P = (
                F.normalize(self.P, p=2, dim=-1).unsqueeze(0).expand(B, -1, -1)
            )  # [B, n_persistent, d_model]
            x_aug = torch.cat(
                [P, h, x_seg], dim=1
            )  # [B, n_persistent + n_retrieved + T, d_model]
        else:
            x_aug = torch.cat(
                [h, x_seg], dim=1
            )  # [B, n_retrieved + T, d_model]

        x_aug = self.norm1(x_aug)
        x_aug = x_aug + self.attention(x_aug)

        y2 = self.norm2(x_aug)
        x_aug = x_aug + self.ff(y2)

        out = x_aug[:, self.n_persistent + self.n_retrieved :, :]  # [B, T, d]

        #  update memory online
        if update_memory:
            memory_tokens = x_aug[
                :, self.n_persistent : self.n_persistent + self.n_retrieved, :
            ]
            if self.update_type == "selfattention":
                k = (
                    self.mem_W_Q(F.normalize(memory_tokens, p=2, dim=-1))
                    if self.proj_k_eq_q
                    else F.normalize(memory_tokens, p=2, dim=-1)
                )
                self.mem.update_from_batch(k.detach(), F.normalize(memory_tokens, p=2, dim=-1).detach())
            elif self.update_type == "crossattention":
                assert (
                    out.shape[1] == memory_tokens.shape[1]
                ), f"out.shape[1] ({out.shape[1]}) != memory_tokens.shape[1] ({memory_tokens.shape[1]})"
                k = self.mem_W_Q(F.normalize(out, p=2, dim=-1)) if self.proj_k_eq_q else F.normalize(out, p=2, dim=-1)
                self.mem.update_from_batch(k.detach(), F.normalize(memory_tokens, p=2, dim=-1).detach())
            else:
                raise ValueError(f"Invalid update_type: {self.update_type}")

        return out


class MACTransformer(nn.Module):
    def __init__(
        self,
        memory_module: NeuralMemory,
        num_patches,
        num_frames,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout=0.0,
        n_persistent=4,
        n_retrieved=4,
        dim_head=64,
        use_slots=True,
        update_type="selfattention",
        proj_k_eq_q=False,
    ):
        super().__init__()
        self.mem = memory_module
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                MACTransformerBlock(
                    mem=memory_module,
                    num_patches=num_patches,
                    num_frames=num_frames,
                    d_model=dim,
                    n_heads=heads,
                    d_ff=mlp_dim,
                    n_persistent=n_persistent,
                    n_retrieved=n_retrieved,
                    dropout=dropout,
                    dim_head=dim_head,
                    use_slots=use_slots,
                    update_type=update_type,
                    proj_k_eq_q=proj_k_eq_q,
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class MACViTPredictor(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        dropout=0.0,
        emb_dropout=0.0,
        n_persistent=4,
        n_retrieved=4,
        dim_head=64,
        hidden_scale=2,
        mem_depth=2,
        mem_eta=0.9,
        mem_theta=1e-3,
        mem_alpha=1e-5,
        max_grad_norm=1.0,
        momentum_clip=1.0,
        weight_clip=5.0,
        update_steps=1,
        use_slots=True,
        update_type="selfattention",
        proj_k_eq_q=False,
    ):
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        super().__init__()
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.dim = dim
        self.pool = pool

        self.mem = NeuralMemory(
            d_model=dim,
            hidden_scale=hidden_scale,
            depth=mem_depth,
            eta=mem_eta,
            theta=mem_theta,
            alpha=mem_alpha,
            max_grad_norm=max_grad_norm,
            momentum_clip=momentum_clip,
            weight_clip=weight_clip,
            update_steps=update_steps,
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames * num_patches, dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = MACTransformer(
            self.mem,
            num_patches,
            num_frames,
            dim,
            depth,
            heads,
            mlp_dim,
            dropout,
            n_persistent,
            n_retrieved,
            dim_head,
            use_slots,
            update_type,
            proj_k_eq_q,
        )

    def forward(self, x):
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        return self.transformer(x)

    def reset_memory(self):
        self.mem.reset_weights()


class LayerMACTransformer(nn.Module):
    def __init__(
        self,
        num_patches,
        num_frames,
        dim,
        depth,
        heads,
        mlp_dim,
        hidden_scale,
        mem_depth,
        mem_eta,
        mem_theta,
        mem_alpha,
        max_grad_norm,
        momentum_clip,
        weight_clip,
        update_steps,
        dropout=0.0,
        n_persistent=4,
        n_retrieved=4,
        dim_head=64,
        use_slots=True,
        update_type="selfattention",
        proj_k_eq_q=False,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                MACTransformerBlock(
                    mem=NeuralMemory(
                        d_model=dim,
                        hidden_scale=hidden_scale,
                        depth=mem_depth,
                        eta=mem_eta,
                        theta=mem_theta,
                        alpha=mem_alpha,
                        max_grad_norm=max_grad_norm,
                        momentum_clip=momentum_clip,
                        weight_clip=weight_clip,
                        update_steps=update_steps,
                    ),
                    num_patches=num_patches,
                    num_frames=num_frames,
                    d_model=dim,
                    n_heads=heads,
                    d_ff=mlp_dim,
                    n_persistent=n_persistent,
                    n_retrieved=n_retrieved,
                    dropout=dropout,
                    dim_head=dim_head,
                    use_slots=use_slots,
                    update_type=update_type,
                    proj_k_eq_q=proj_k_eq_q,
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def reset_memory(self):
        for layer in self.layers:
            layer.reset_memory()


class LayerMACViTPredictor(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout=0.0,
        emb_dropout=0.0,
        hidden_scale=2,
        mem_depth=2,
        mem_eta=0.9,
        mem_theta=1e-3,
        mem_alpha=1e-5,
        max_grad_norm=1.0,
        momentum_clip=1.0,
        weight_clip=5.0,
        update_steps=1,
        n_persistent=4,
        n_retrieved=4,
        dim_head=64,
        use_slots=True,
        update_type="selfattention",
        proj_k_eq_q=False,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.dim = dim

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames * num_patches, dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = LayerMACTransformer(
            num_patches,
            num_frames,
            dim,
            depth,
            heads,
            mlp_dim,
            hidden_scale,
            mem_depth,
            mem_eta,
            mem_theta,
            mem_alpha,
            max_grad_norm,
            momentum_clip,
            weight_clip,
            update_steps,
            dropout,
            n_persistent,
            n_retrieved,
            dim_head,
            use_slots,
            update_type,
            proj_k_eq_q,
        )

    def forward(self, x):
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        return self.transformer(x)

    def reset_memory(self):
        self.transformer.reset_memory()


class LookupTransformerBlock(nn.Module):
    def __init__(
        self,
        mem: LookupMemory,
        num_patches: int,
        num_frames: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        dim_head: int = 64,
    ):
        super().__init__()
        self.mem = mem
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.dim_head = dim_head

        self.attention = Attention(d_model, n_heads, dim_head, dropout)
        bias = generate_mac_mask_matrix(num_patches, num_frames, 0, num_frames)
        self.attention.register_buffer("bias", bias)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, D = x.shape

        # lookup memory
        memory = self.mem.retrieve()
        if memory.size(1) > T:
            memory = memory[:, :T, :]

            x_aug = torch.cat([memory, x], dim=1)

            x_aug = self.norm1(x_aug)
            x_aug = x_aug + self.attention(x_aug)

            # FFN
            y2 = self.norm2(x_aug)
            x_aug = x_aug + self.ff(y2)

            # strip off the prepended tokens; only return the positions corresponding to the segment
            out = x_aug[:, memory.size(1) :, :]  # [B, T, d]
        else:
            x = self.norm1(x)
            x = x + self.attention(x)
            y2 = self.norm2(x)
            x = x + self.ff(y2)
            out = x

        return out


class LookupTransformer(nn.Module):
    def __init__(
        self,
        mem: LookupMemory,
        num_patches: int,
        num_frames: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        dim_head: int = 64,
    ):
        super().__init__()
        self.mem = mem
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                LookupTransformerBlock(
                    mem=mem,
                    num_patches=num_patches,
                    num_frames=num_frames,
                    d_model=dim,
                    n_heads=heads,
                    d_ff=mlp_dim,
                    dropout=dropout,
                    dim_head=dim_head,
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class LookupViTPredictor(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        depth,
        heads,
        mlp_dim,
        batch_size,
        pool="cls",
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=64,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.dim = dim
        self.pool = pool

        self.mem = LookupMemory(dim, batch_size)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames * num_patches, dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = LookupTransformer(
            self.mem,
            num_patches,
            num_frames,
            dim,
            depth,
            heads,
            mlp_dim,
            dropout,
            dim_head,
        )

    def forward(self, x):
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        return self.transformer(x)

    def reset_memory(self):
        self.mem.reset_weights()
