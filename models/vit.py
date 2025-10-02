# adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F

from .model_utils import *
from .memory_retrieval import NeuralMemory, LookupMemory, SSMCell, MambaSSMCell
from .memory_injection import AdaptiveLayerNorm, MemoryLoRAAdapter, MemoryLoRAProj

# helpers
NUM_FRAMES = 1
NUM_PATCHES = 1

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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, bias=None):
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

        if bias is None:
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


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, context):
        B, T, C = x.size()
        B_c, T_c, C_c = context.size()

        x = self.norm(x)

        q = self.to_q(x)
        kv = self.to_kv(context)
        k, v = kv.chunk(2, dim=-1)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class AttentionWithLoRA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, bias=None,
                 lora_rank=16, lora_alpha=8.0, zero_init=True, lora_on_out=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # base packed qkv
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # optional out proj
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out else nn.Identity()
        )

        if bias is None:
            bias = generate_mask_matrix(NUM_PATCHES, NUM_FRAMES)
        self.register_buffer("bias", bias)

        # ---- LoRA: memory-conditioned adapters on Q, K, V (and optionally out) ----
        # We’ll treat the packed qkv as three separate augmented linears.
        self.q_lora = MemoryLoRAProj(in_dim=dim, out_dim=inner_dim, rank=lora_rank, alpha=lora_alpha, zero_init=zero_init)
        self.k_lora = MemoryLoRAProj(in_dim=dim, out_dim=inner_dim, rank=lora_rank, alpha=lora_alpha, zero_init=zero_init)
        self.v_lora = MemoryLoRAProj(in_dim=dim, out_dim=inner_dim, rank=lora_rank, alpha=lora_alpha, zero_init=zero_init)

        self.lora_on_out = lora_on_out
        if lora_on_out and project_out:
            self.out_lora = MemoryLoRAProj(in_dim=inner_dim, out_dim=dim, rank=lora_rank, alpha=lora_alpha, zero_init=zero_init)
        else:
            self.out_lora = None

    def forward(self, x, memory_tokens=None):
        B, T, C = x.size()

        x = self.norm(x)

        # base qkv
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # each [B, T, inner_dim]
        q_base, k_base, v_base = qkv

        if memory_tokens is not None:
            # add memory-conditioned low-rank updates to Q, K, V
            q = q_base + self.q_lora(x, memory_tokens)
            k = k_base + self.k_lora(x, memory_tokens)
            v = v_base + self.v_lora(x, memory_tokens)
        else:
            q, k, v = q_base, k_base, v_base

        # reshape into heads
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # causal mask
        dots = dots.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        out = self.to_out(out)
        if self.out_lora is not None and memory_tokens is not None:
            out = out + self.out_lora(rearrange(out, "b n c -> b n c"), memory_tokens)

        return out

class FeedForwardWithLoRA(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0,
                 lora_rank=16, lora_alpha=8.0, zero_init=True):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

        self.lin1 = nn.Linear(dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, dim)

        self.act = nn.GELU()
        self.do1 = nn.Dropout(dropout)
        self.do2 = nn.Dropout(dropout)

        # LoRA adapters for both projections
        self.lora1 = MemoryLoRAProj(in_dim=dim,       out_dim=hidden_dim, rank=lora_rank, alpha=lora_alpha, zero_init=zero_init)
        self.lora2 = MemoryLoRAProj(in_dim=hidden_dim,out_dim=dim,        rank=lora_rank, alpha=lora_alpha, zero_init=zero_init)

    def forward(self, x, memory_tokens=None):
        x = self.ln(x)

        h = self.lin1(x)
        if memory_tokens is not None:
            h = h + self.lora1(x, memory_tokens)
        h = self.act(h)
        h = self.do1(h)

        y = self.lin2(h)
        if memory_tokens is not None:
            y = y + self.lora2(h, memory_tokens)
        y = self.do2(y)
        return y


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
        use_residual: bool = False,
    ):
        super().__init__()
        assert gate_type in {"sigmoid", "sigmoid_convex"}, "Invalid gate type"
        self.mem = mem
        self.gate_type = gate_type
        self.norm = nn.LayerNorm(dim)
        self.attention = Attention(dim, heads, dim_head, dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout)
        self.W_y = nn.Linear(dim, dim)
        self.W_m = nn.Linear(dim, dim)
        self.W_Q = nn.Linear(dim, dim)
        self.use_residual = use_residual
        self.pre_norm = nn.LayerNorm(dim)

        if self.gate_type == "sigmoid_convex":
            self.V_y = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.pre_norm(x)
        x_in = x
        x = x + self.attention(x)
        x = x + self.ff(x)

        if self.gate_type == "sigmoid":
            gate = torch.sigmoid(self.W_y(x)) * self.W_m(
                self.mem.retrieve(self.W_Q(x))
            ) 
        elif self.gate_type == "sigmoid_convex":
            g = torch.sigmoid(self.W_y(x))
            y = self.V_y(x)
            m = self.W_m(self.mem.retrieve(self.W_Q(x)))
            gate = (1.0 - g) * y + g * m
        else:
            raise ValueError(f"Invalid gate type: {self.gate_type}")

        if self.use_residual:
            x = x + gate
        else:
            x = gate

        # update memory
        # self.mem.update_from_batch(
        #     F.normalize(x_in, p=2, dim=-1).detach(),
        #     F.normalize(x_in, p=2, dim=-1).detach(),
        # )
        self.mem.update_from_batch(
            x_in.detach(),
            x_in.detach(),
        )

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
        use_residual: bool = False,
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
                    use_residual=use_residual,
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
        use_residual: bool = False,
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
            use_residual=use_residual,
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
        q_t = self.mem_W_Q(x_seg)  # [B, T, d_model]
        h = self.mem.retrieve(q_t)  # [B, T, d_model]

        if self.use_slots:
            # compress retrieved "per-token" memory into a fixed number of slots
            h = self.mem_slots(h)  # [B, T, n_retrieved*d_model]
            h = h.mean(dim=1).view(  # TODO: try different pooling
                B, self.n_retrieved, self.d_model
            )  # [B, n_retrieved, d_model] (pool over time in segment)
        else:
            self.n_retrieved = T

        # prepend persistent tokens and retrieved memory slots
        if self.n_persistent > 0:
            P = (
                self.P.unsqueeze(0).expand(B, -1, -1)
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
                    self.mem_W_Q(memory_tokens)
                    if self.proj_k_eq_q
                    else memory_tokens
                )
                self.mem.update_from_batch(
                    k.detach(),
                    memory_tokens.detach(),
                )
            elif self.update_type == "crossattention":
                assert (
                    out.shape[1] == memory_tokens.shape[1]
                ), f"out.shape[1] ({out.shape[1]}) != memory_tokens.shape[1] ({memory_tokens.shape[1]})"
                k = (
                    self.mem_W_Q(out)
                    if self.proj_k_eq_q
                    else out
                )
                self.mem.update_from_batch(
                    k.detach(),
                    memory_tokens.detach(),
                )
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


class StateSpaceTransformer(nn.Module):
    def __init__(
        self,
        ssm_type,
        dim,
        state_size,
        num_patches,
        depth,
        heads,
        mlp_dim,
        dropout=0.0,
        dim_head=64,
        dt: float = 1.0,  # could change to frameskip
        use_gate: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.dim_head = dim_head
        self.state_size = state_size
        self.use_gate = use_gate
        self.dt = dt
        self.num_patches = num_patches
        self.ssm_type = ssm_type
        assert ssm_type in {"lti", "mamba"}, "Invalid SSM type"

        if use_gate:
            self.gate = nn.Linear(dim, dim)
        else:
            self.gate = None

        if ssm_type == "lti":
            self.ssm_cell = SSMCell(dim, state_size)
        elif ssm_type == "mamba":
            self.ssm_cell = MambaSSMCell(dim, state_size, dt_rank=8)
        else:
            raise ValueError(f"Invalid SSM cell type: {ssm_type}")

        self.ln_in = nn.LayerNorm(dim)
        self.ln_out = nn.LayerNorm(dim)

        self.H_buffer = None # keep track of hidden memory state w/o passing around

        self._build_transformer(depth, dim, heads, dim_head, dropout, mlp_dim, **kwargs)

    def _build_transformer(self, depth, dim, heads, dim_head, dropout, mlp_dim, **kwargs):
        self.ln_fuse = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads,
                            dim_head,
                            dropout,
                            bias=generate_mask_with_memory(
                                NUM_PATCHES, NUM_FRAMES
                            ) if not kwargs.get("use_gate", False) else None,
                        ),
                        FeedForward(dim, mlp_dim, dropout),
                    ]
                )
            )

    def _ssm_forward(self, x, mode: str = "step"):
        B, T, D = x.shape
        n_frames = T // self.num_patches

        if self.H_buffer is None:
                self.H_buffer = self.init_state(B, x.device)

        if mode == "step":  
            # iterate through frames to aggregate memory
            M_new = []
            h_new = self.H_buffer
            for i in range(n_frames):
                x_i = x[:, i * self.num_patches : (i + 1) * self.num_patches, :]
                h_new, m_i = self.ssm_cell(x_i, h_new, self.dt)
                M_new.append(m_i)

            M_new = torch.cat(M_new, dim=1)
            self.H_buffer = h_new.detach()

        elif mode == "scan":
            x = rearrange(x, "b (t p) d -> b t p d", t=T//self.num_patches)
            _, M_new, H_T = self.ssm_cell(x, self.H_buffer, mode="scan")
            self.H_buffer = H_T.detach()

        return M_new

    def forward(self, x):
        B, T, D = x.shape
        x = self.ln_in(x)

        M_new = self._ssm_forward(x)

        if self.use_gate:
            G = torch.sigmoid(self.gate(x))
            ctx = self.ln_fuse(
                x + G * M_new
            )  # inject ctx with memory (post-write)
        else:
            ctx = self.ln_fuse(torch.cat([M_new, x], dim=1))  # B, T * 2, D

        for attn, ff in self.layers:
            ctx = ctx + attn(ctx)
            ctx = ctx + ff(ctx)

        if not self.use_gate:
            # remove the prepended tokens
            ctx = ctx[:, T:, :]

        return self.ln_out(ctx)

    @torch.no_grad()
    def init_state(self, B: int, device=None):
        return self.ssm_cell.init_state(B, self.num_patches, self.state_size, device=device)

    def reset_memory(self):
        self.H_buffer = None

class MemoryInjectionSSMTransformer(StateSpaceTransformer):
    def __init__(
        self,
        ssm_type,
        dim,
        state_size,
        num_patches,
        depth,
        heads,
        mlp_dim,
        dropout=0.0,
        dim_head=64,
        dt: float = 1.0, # could change to frameskip
        alpha_init: float = 0.1,
        **kwargs,
    ):
        super().__init__(ssm_type, dim, state_size, num_patches, depth, heads, mlp_dim, dropout, dim_head, dt, alpha_init=alpha_init, **kwargs)

    def _build_transformer(self, depth, dim, heads, dim_head, dropout, mlp_dim, **kwargs):
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.ones(1) * kwargs.get("alpha_init", 0.1)) for _ in range(depth)
        ])

        self.layers = nn.ModuleList([])
        self.injection_layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads,
                            dim_head,
                            dropout,
                        ),
                        FeedForward(dim, mlp_dim, dropout),
                    ]
                )
            )
            self.injection_layers.append(
                nn.Linear(dim, dim)
            )
    
    def forward(self, x):
        B, T, D = x.shape
        x = self.ln_in(x)

        M_new = self._ssm_forward(x)

        for i, (attn, ff) in enumerate(self.layers):
            x = x + attn(x)
            x = x + ff(x)
            x = x + self.injection_layers[i](M_new) * self.alphas[i]
            
        return self.ln_out(x)

class MemoryInjectionPreSSMTransformer(MemoryInjectionSSMTransformer):
    def __init__(
        self,
        ssm_type,
        dim,
        state_size,
        num_patches,
        depth,
        heads,
        mlp_dim,
        dropout=0.0,
        dim_head=64,
        dt: float = 1.0, # could change to frameskip
        alpha_init: float = 0.1,
        **kwargs,
    ):
        super().__init__(ssm_type, dim, state_size, num_patches, depth, heads, mlp_dim, dropout, dim_head, dt, alpha_init=alpha_init, **kwargs)

    def forward(self, x):
        x = self.ln_in(x)
        M_new = self._ssm_forward(x)
        for i, (attn, ff) in enumerate(self.layers):
            x = x + attn(x)
            x = x + self.injection_layers[i](M_new) * self.alphas[i]
            x = x + ff(x)
        return self.ln_out(x)

class AdaMemSSMTransformer(StateSpaceTransformer):
    def __init__(
        self,
        ssm_type,
        dim,
        state_size,
        num_patches,
        depth,
        heads,
        mlp_dim,
        dropout=0.0,
        dim_head=64,
        dt: float = 1.0, # could change to frameskip
        zero_init: bool = False,
        **kwargs,
    ):
        super().__init__(ssm_type, dim, state_size, num_patches, depth, heads, mlp_dim, dropout, dim_head, dt, zero_init=zero_init, **kwargs)
        
    
    def _build_transformer(self, depth, dim, heads, dim_head, dropout, mlp_dim, **kwargs):
        self.layers = nn.ModuleList([])
        self.injection_layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads,
                            dim_head,
                            dropout,
                        ),
                        FeedForward(dim, mlp_dim, dropout),
                    ]
                )
            )
            self.injection_layers.append(
                AdaptiveLayerNorm(dim, dim, zero_init=kwargs.get("zero_init", False))
            )

    def forward(self, x):
        x = self.ln_in(x)

        M_new = self._ssm_forward(x)

        for i, (attn, ff) in enumerate(self.layers):
            x = x + attn(x)
            x = x + ff(x)
            x = self.injection_layers[i](x, M_new)

        return self.ln_out(x)

class LoRAMemSSMTransformer(StateSpaceTransformer):
    def __init__(
        self,
        ssm_type,
        dim,
        state_size,
        num_patches,
        depth,
        heads,
        mlp_dim,
        dropout=0.0,
        dim_head=64,
        dt: float = 1.0,
        zero_init: bool = True,    # zero-init LoRA branch so it starts as a no-op
        lora_rank: int = 128,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(ssm_type, dim, state_size, num_patches, depth, heads, mlp_dim, dropout, dim_head, dt, zero_init=zero_init, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, **kwargs)

    def _build_transformer(self, depth, dim, heads, dim_head, dropout, mlp_dim, **kwargs):
        self.layers = nn.ModuleList([])
        self.lora_post_attn = nn.ModuleList([])
        self.lora_post_ff = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads, dim_head, dropout),
                        FeedForward(dim, mlp_dim, dropout),
                    ]
                )
            )
            self.lora_post_attn.append(
                MemoryLoRAAdapter(
                    dim=dim,
                    rank=kwargs.get("lora_rank"),
                    lora_alpha=kwargs.get("lora_alpha"),
                    dropout=kwargs.get("lora_dropout"),
                    zero_init=kwargs.get("zero_init"),
                )
            )
            self.lora_post_ff.append(
                MemoryLoRAAdapter(
                    dim=dim,
                    rank=kwargs.get("lora_rank"),
                    lora_alpha=kwargs.get("lora_alpha"),
                    dropout=kwargs.get("lora_dropout"),
                    zero_init=kwargs.get("zero_init"),
                )
            )

    def forward(self, x):
        x = self.ln_in(x)

        M_new = self._ssm_forward(x)

        # --- transformer blocks with LoRA memory adapters ---
        for (attn, ff), lora_attn, lora_ff in zip(self.layers, self.lora_post_attn, self.lora_post_ff):
            x = x + attn(x)
            # LoRA after attention
            x = x + lora_attn(x, M_new)

            x = x + ff(x)
            # LoRA after feedforward
            x = x + lora_ff(x, M_new)

        return self.ln_out(x)


class MemCrossAttentionSSMTransformer(StateSpaceTransformer):
    def __init__(
        self,
        ssm_type,
        dim,
        state_size,
        num_patches,
        depth,
        heads,
        mlp_dim,
        dropout=0.0,
        dim_head=64,
        dt: float = 1.0,
        **kwargs,
    ):
        super().__init__(ssm_type, dim, state_size, num_patches, depth, heads, mlp_dim, dropout, dim_head, dt, **kwargs)
    
    def _build_transformer(self, depth, dim, heads, dim_head, dropout, mlp_dim, **kwargs):
        self.layers = nn.ModuleList([])
        self.injection_layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads, dim_head, dropout),
                        FeedForward(dim, mlp_dim, dropout),
                    ]
                )
            )
            self.injection_layers.append(
                CrossAttention(dim, heads, dim_head, dropout)
            )

    def forward(self, x):
        x = self.ln_in(x)

        M_new = self._ssm_forward(x)

        for i, (attn, ff) in enumerate(self.layers):
            x = x + attn(x)
            x = x + ff(x)
            x = x + self.injection_layers[i](x, M_new)
            
        return self.ln_out(x)


class StateSpaceViTPredictor(nn.Module):
    def __init__(self, *, num_patches, num_frames, dim, state_size, depth, heads, mlp_dim, ssm_type, injection_type, alpha_init: float = 0.1, dropout=0.0, emb_dropout=0.0, dim_head=64, use_gate: bool = False, dt: float = 1.0, zero_init: bool = False, lora_rank: int = 64, lora_alpha: float = 1.0, lora_dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.state_size = state_size
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.dim_head = dim_head
        self.use_gate = use_gate
        self.num_patches = num_patches
        self.num_frames = num_frames

        # update params for adding causal attention masks
        global NUM_FRAMES, NUM_PATCHES
        NUM_FRAMES = num_frames
        NUM_PATCHES = num_patches

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames * num_patches, dim)
        )
        self.dropout = nn.Dropout(emb_dropout)

        if injection_type == "sst":
            self.transformer = StateSpaceTransformer(
                ssm_type, dim, state_size, num_patches, depth, heads, mlp_dim, dropout, dim_head, dt, use_gate=use_gate
            )
        elif injection_type == "misst":
            self.transformer = MemoryInjectionSSMTransformer(
                ssm_type, dim, state_size, num_patches, depth, heads, mlp_dim, dropout, dim_head, dt, alpha_init=alpha_init
            )
        elif injection_type == "misst_pre":
            self.transformer = MemoryInjectionPreSSMTransformer(
                ssm_type, dim, state_size, num_patches, depth, heads, mlp_dim, dropout, dim_head, dt, alpha_init=alpha_init
            )
        elif injection_type == "adamem":
            self.transformer = AdaMemSSMTransformer(
                ssm_type, dim, state_size, num_patches, depth, heads, mlp_dim, dropout, dim_head, dt, alpha_init=alpha_init, zero_init=zero_init
            )
        elif injection_type == "loramem_post":
            self.transformer = LoRAMemSSMTransformer(
                ssm_type, dim, state_size, num_patches, depth, heads, mlp_dim, dropout, dim_head, dt, alpha_init=alpha_init, zero_init=zero_init, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout
            )
        elif injection_type == "ssm_ca":
            self.transformer = MemCrossAttentionSSMTransformer(
                ssm_type, dim, state_size, num_patches, depth, heads, mlp_dim, dropout, dim_head, dt
            )
        else:
            raise ValueError(f"Invalid injection type: {injection_type}")

    def forward(self, x):
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        return self.transformer(x)

    def reset_memory(self):
        self.transformer.reset_memory()


 ###### HYBRID ARCHITECTURES ######
class DualAttentionSSMKeys(nn.Module):
    """
    Dual attention combining:
      - alpha (memory-driven):  Q vs K_mem, where K_mem = Proj(C_t ⊙ h_t)
      - beta  (content-driven): Q vs K_cnt, where K_cnt = W_K x

    Fusion modes:
      - 'sum' : out = out_alpha + out_beta
      - 'diff': out = out_alpha - out_beta
      - 'mul' : attn = softmax( (alpha_probs * beta_probs) )
      - 'gate': out = g * out_alpha + (1 - g) * out_beta  (token-wise learned gate)

    Shapes:
      x:      [B, T, D] where T = n_frames * n_patches  (consistent with your bias mask)
      bias:   [1, 1, T, T] causal/structured mask
      returns:
        y:    [B, T, D]
        H_T:  [B, P, S] (final SSM state to carry, with P = n_patches)
    """

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        n_patches=1,
        n_frames=1,
        fusion="sum",                 # 'sum' | 'diff' | 'mul' | 'gate'
        bias=None
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner = heads * dim_head
        self.scale = dim_head ** -0.5
        self.n_patches = n_patches
        self.n_frames = n_frames
        self.fusion = fusion
        self.ssm = MambaSSMCell(d_model=dim, n_state= dim // 2) # replace dt_rank eventually

        # projections
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, self.inner, bias=False)
        self.to_k_cnt = nn.Linear(dim, self.inner, bias=False)  # content keys
        self.to_v = nn.Linear(dim, self.inner, bias=False)

        # project SSM (C_t ⊙ h_t) -> key space (per token)
        # we will construct K_mem via: Proj( (C_t ⊙ H_seq) )
        self.proj_k_mem = nn.Linear(dim, self.inner, bias=False)

        # output projection
        self.to_out = nn.Sequential(nn.Linear(self.inner, dim), nn.Dropout(dropout))

        # attention bits
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # optional learned gate for 'gate' fusion (token-wise, head-shared)
        if self.fusion == "gate":
            self.gate = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, 1),
                nn.Sigmoid()
            )

        # mask (expects T = n_frames * n_patches)
        if bias is None:
            bias = generate_mask_matrix(self.n_patches, self.n_frames)
        self.register_buffer("bias", bias)  # [1,1,T,T]

    def _reshape_for_ssm(self, x):
        """
        x: [B, T, D] with T = n_frames * n_patches
        returns X_win: [B, T_frames, P, D]
        """
        B, T, D = x.shape
        P = self.n_patches
        F = self.n_frames
        assert T == P * F, f"T ({T}) must equal n_patches*n_frames ({P*F})"
        X_win = rearrange(x, "b (t p) d -> b t p d", t=F, p=P)  # [B, F, P, D]
        return X_win

    def _heads(self, t):
        return rearrange(t, "b n (h d) -> b h n d", h=self.heads)

    def _unheads(self, t):
        return rearrange(t, "b h n d -> b n (h d)")

    def forward(self, x, H0=None):
        """
        x:  [B, T, D]
        H0: [B, P, S] or None (zeros)
        returns: y [B,T,D], H_T [B,P,S]
        """
        B, T, D = x.shape
        P, F = self.n_patches, self.n_frames

        x = self.norm(x)

        # ---- Q, K_content, V from raw features
        Q = self._heads(self.to_q(x))        # [B,H,T,dh]
        K_cnt = self._heads(self.to_k_cnt(x))# [B,H,T,dh]
        V = self._heads(self.to_v(x))        # [B,H,T,dh]

        # ---- SSM trajectory to build K_mem = Proj(C_t ⊙ h_t)
        X_win = self._reshape_for_ssm(x)     # [B,F,P,D]
        if H0 is None:
            H0 = self.ssm.init_state(B, P, device=x.device, dtype=x.dtype)  # [B,P,S]
        _, Y_seq, H_T = self.ssm(X_win, H0, mode="scan")  # Y_seq: (B, F*P, D), H_T: (B, P, S)

        # # Token-wise (C_t ⊙ h_t) already used inside self.ssm.readout for Y_seq;
        # # we need (C_t ⊙ h_t) again to produce keys. Recompute C_t for keys:
        # Xf = rearrange(X_win, "b t p d -> (b t) p d")
        # _, _, C_t = self.ssm._selective_params(Xf)                 # [BF,P,S]
        # C_t = rearrange(C_t, "(b t) p s -> b t p s", b=B, t=F)     # [B,F,P,S]
        # Ch = C_t * H_seq                                           # [B,F,P,S]

        # Project to key space per token, then flatten time*patch -> tokens
        K_mem_tokens = self.proj_k_mem(Y_seq)   # [B,T,Inner]
        K_mem = self._heads(K_mem_tokens)       # [B,H,T,dh]

        # ---- Two attention maps (beta: content, alpha: memory)
        dots_beta = torch.matmul(Q, K_cnt.transpose(-1, -2)) * self.scale  # [B,H,T,T]
        dots_alpha = torch.matmul(Q, K_mem.transpose(-1, -2)) * self.scale

        # apply same causal/structured mask
        mask = self.bias[:, :, :T, :T] == 0  # [1,1,T,T] -> bool
        dots_beta = dots_beta.masked_fill(mask, float("-inf"))
        dots_alpha = dots_alpha.masked_fill(mask, float("-inf"))

        # softmax maps
        attn_beta = self.attend(dots_beta)    # [B,H,T,T]
        attn_alpha = self.attend(dots_alpha)  # [B,H,T,T]

        # optional dropout on maps
        attn_beta = self.dropout(attn_beta)
        attn_alpha = self.dropout(attn_alpha)

        # ---- Fuse attentions / outputs
        if self.fusion == "sum":
            # (alphaV + betaV)
            out = torch.matmul(attn_alpha, V) + torch.matmul(attn_beta, V)

        elif self.fusion == "diff":
            # (alphaV - betaV)  (differential attention)
            out = torch.matmul(attn_alpha, V) - torch.matmul(attn_beta, V)

        elif self.fusion == "mul":
            # multiplicative agreement: softmax(alpha ⊙ beta)
            attn_agree = attn_alpha * attn_beta
            # renormalize per head
            attn_agree = attn_agree / (attn_agree.sum(dim=-1, keepdim=True) + 1e-9)
            out = torch.matmul(attn_agree, V)

        elif self.fusion == "gate":
            # token-wise gate g \in [0,1], shared across heads
            # use the input x to predict gate; broadcast to heads
            g = self.gate(x).clamp(0.0, 1.0)            # [B,T,1]
            g = g.transpose(1, 2)                       # [B,1,T]
            g = g.unsqueeze(-1)                         # [B,1,T,1]
            out_alpha = torch.matmul(attn_alpha, V)
            out_beta  = torch.matmul(attn_beta, V)
            out = g * out_alpha + (1.0 - g) * out_beta  # broadcast over heads

        else:
            raise ValueError(f"Unknown fusion '{self.fusion}'")

        # ---- Merge heads and project out
        out = self._unheads(out)            # [B,T,Inner]
        y = self.to_out(out)                # [B,T,D]
        return y, H_T


class HybridTransformerLayer(nn.Module):
    """
    A single Transformer layer where the attention sublayer is replaced
    by DualAttentionSSMKeys. Residual + FFN preserved.
    """
    def __init__(
        self,
        dim,
        heads,
        dim_head,
        mlp_dim,
        n_patches,
        n_frames,
        dropout=0.0,
        fusion="sum",
        bias=None
    ):
        super().__init__()
        self.attn = DualAttentionSSMKeys(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            n_patches=n_patches,
            n_frames=n_frames,
            fusion=fusion,
            bias=bias
        )
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x, H0=None):
        # attention + residual
        attn_out, H_T = self.attn(x, H0=H0)
        x = x + attn_out
        # ff + residual
        x = x + self.ff(x)
        return x, H_T


class HybridTransformer(nn.Module):
    """
    Stack of HybridTransformerLayers. We carry the SSM state across layers
    OR re-init per layer. Here we carry within each layer (common pattern:
    one SSM per layer). You can also share one SSM across layers if desired.
    """
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        n_patches,
        n_frames,
        dropout=0.0,
        fusion="sum",
        bias=None
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            layer = HybridTransformerLayer(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                n_patches=n_patches,
                n_frames=n_frames,
                dropout=dropout,
                fusion=fusion,
                bias=bias
            )
            self.layers.append(layer)

    def forward(self, x, H0=None):
        """
        x:  [B, T, D]  with T = n_frames * n_patches
        H0: Optional initial SSM state for the FIRST layer.
            If provided, it will be used for the first layer; subsequent
            layers will init zeros unless you choose to thread H across.
        """
        # Carry SSM state per-layer independently; thread H only within the layer.
        for i, layer in enumerate(self.layers):
            H0_i = H0 if (i == 0) else None
            x, _ = layer(x, H0=H0_i)
        return self.norm(x)


class HybridViTPredictor(nn.Module):
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
        fusion="sum",
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
        self.transformer = HybridTransformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            n_patches=num_patches,
            n_frames=num_frames,
            dropout=dropout,
            fusion=fusion,
        )
        self.pool = pool

    def forward(self, x):
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x)
        return x