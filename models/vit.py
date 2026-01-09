import math
import torch.utils.checkpoint as checkpoint
import torch
from torch import nn
from einops import rearrange, repeat
from torch.nn import functional as F
from torchdiffeq import odeint, odeint_adjoint
import torchdiffeq

from .model_utils import *
from .memory_retrieval import (
    BasicHiddenMambaLayer,
    BasicMambaLayer,
    NeuralMemory,
    LookupMemory,
    MambaLayer,
    MambaSSMCell,
)
from .memory_injection import (
    AdaptiveLayerNorm,
    DynamicLoRALinear,
    FFNMemGenerator,
    MemoryLoRAAdapter,
    MemoryLoRAProj,
)

# helpers
NUM_FRAMES = 1
NUM_PATCHES = 1


class TwoInputIdentity(nn.Module):
    def forward(self, x1, x2):
        return x1

class RetentionPredictor(nn.Module):
    def __init__(self, dim, cond_dim, depth=2, heads=8, dim_head=64, mlp_dim=1024, cond_repeat=7, dropout=0.0):
        super().__init__()

        self.depth = depth
        self.dim = dim
        self.cond_dim = cond_dim
        self.cond_repeat = cond_repeat
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        cond_size = cond_dim * cond_repeat

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim + cond_size, heads, dim_head, dropout, bias=generate_mask_matrix(NUM_PATCHES, 1)), # only one time frame
                FeedForward(dim + cond_size, mlp_dim, dropout)
            ]))
        self.to_out = nn.Linear(dim + cond_size, dim)

    def add_cond(self, x, cond):
        # tile cond 
        cond_repeated = cond.repeat(1, 1, 1, self.cond_repeat)
        out = torch.cat([x, cond_repeated], dim=3) # (b, t, num_patches, dim + cond_dim * cond_repeat)
        return rearrange(out, "b t p d -> b (t p) d")

    def forward(self, x, cond):
        B, T, P, D = x.shape
        x = self.add_cond(x, cond) # (b, (t p), dim + cond_dim * cond_repeat)
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)
        out = self.to_out(x)
        return rearrange(out, "b (t p) d -> b t p d", t=T)

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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, bias=None):
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

        if bias is None:
            bias = generate_diagonal_frame_mask(NUM_PATCHES, NUM_FRAMES)
        self.register_buffer("bias", bias)

    def forward(self, x, context):
        B, T, C = x.size()
        B_c, T_c, C_c = context.size()

        x = self.norm(x)
        context = self.norm(context)

        q = self.to_q(x)
        kv = self.to_kv(context)
        k, v = kv.chunk(2, dim=-1)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # apply causal mask - prevent attending to future memory tokens
        dots = dots.masked_fill(self.bias[:, :, :T, :T_c] == 0, float("-inf"))

        attn = self.attend(dots)
        if torch.isnan(attn).any():
            print("T_c: ", T_c)
            print("attn shape: ", attn.shape)

            print(f"Attn is nan, dots: {dots.shape}, q: {q.shape}, k: {k.shape}, v: {v.shape}")
            print(f"isnan: {torch.isnan(attn).sum()}")
            print(f"dots nan: {dots[torch.isnan(attn)].shape}")

        assert not torch.isnan(attn).any(), f"Attn is nan, dots: {dots.shape}, q: {q.shape}, k: {k.shape}, v: {v.shape}"
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

class CrossAttentionInjection(nn.Module):
    def __init__(self, q_dim, kv_dim, heads=8, dim_head=64, dropout=0.0, bias=None):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == q_dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(q_dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(q_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(kv_dim, inner_dim * 2, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, q_dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

        if bias is None:
            bias = generate_diagonal_frame_mask(NUM_PATCHES, NUM_FRAMES)
        self.register_buffer("bias", bias)

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
        
        # apply causal mask - prevent attending to future memory tokens
        dots = dots.masked_fill(self.bias[:, :, :T, :T_c] == 0, float("-inf"))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

class AttentionWithLoRA(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        bias=None,
        lora_rank=16,
        lora_alpha=8.0,
        zero_init=True,
        lora_on_out=False,
    ):
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
            if project_out
            else nn.Identity()
        )

        if bias is None:
            bias = generate_mask_matrix(NUM_PATCHES, NUM_FRAMES)
        self.register_buffer("bias", bias)


        self.q_lora = MemoryLoRAProj(
            in_dim=dim,
            out_dim=inner_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            zero_init=zero_init,
        )
        self.k_lora = MemoryLoRAProj(
            in_dim=dim,
            out_dim=inner_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            zero_init=zero_init,
        )
        self.v_lora = MemoryLoRAProj(
            in_dim=dim,
            out_dim=inner_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            zero_init=zero_init,
        )

        self.lora_on_out = lora_on_out
        if lora_on_out and project_out:
            self.out_lora = MemoryLoRAProj(
                in_dim=inner_dim,
                out_dim=dim,
                rank=lora_rank,
                alpha=lora_alpha,
                zero_init=zero_init,
            )
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
            out = out + self.out_lora(
                rearrange(out, "b n c -> b n c"), memory_tokens
            )

        return out


class FeedForwardWithLoRA(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        dropout=0.0,
        lora_rank=16,
        lora_alpha=8.0,
        zero_init=True,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

        self.lin1 = nn.Linear(dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, dim)

        self.act = nn.GELU()
        self.do1 = nn.Dropout(dropout)
        self.do2 = nn.Dropout(dropout)

        # LoRA adapters for both projections
        self.lora1 = MemoryLoRAProj(
            in_dim=dim,
            out_dim=hidden_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            zero_init=zero_init,
        )
        self.lora2 = MemoryLoRAProj(
            in_dim=hidden_dim,
            out_dim=dim,
            rank=lora_rank,
            alpha=lora_alpha,
            zero_init=zero_init,
        )

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


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, bias=None):
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, dropout, bias=bias)
        self.ff = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x

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

    def forward(self, x, H=None):
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x)
        return x, None
    
    def set_dt(self, new_dt):
        return
    
    def get_dt(self):
        return 1.0


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
        self, x, H=None
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
        return x, None


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
        return self.norm(x), None


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

    def forward(self, x, H=None):
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
        return x, None


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

    def forward(self, x, H=None):
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
        return self.norm(x), None

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

    def forward(self, x, H=None):
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
        dropout: float = 0.0,
        dim_head: int = 64,
        update_type: str = "selfattention",  # "selfattention" or "crossattention"
        proj_k_eq_q: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.mem = mem
        self.n_persistent = n_persistent
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.update_type = update_type
        self.proj_k_eq_q = proj_k_eq_q

        if self.n_persistent > 0:
            self.P = nn.Parameter(torch.randn(n_persistent, d_model))

        self.mem_W_Q = nn.Linear(d_model, d_model)

        bias = generate_mac_mask_matrix(
            num_patches, num_frames, n_persistent, num_frames
        )
        self.attention = Attention(d_model, n_heads, dim_head, dropout, bias=bias)

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

        x_seg = self.norm1(x_seg)

        # retrieve long-term memory for current segment
        q_t = self.mem_W_Q(x_seg)  # [B, T, d_model]
        h = self.mem.retrieve(q_t)  # [B, T, d_model]

        # prepend persistent tokens and retrieved memory slots
        if self.n_persistent > 0:
            P = self.P.unsqueeze(0).expand(
                B, -1, -1
            )  # [B, n_persistent, d_model]
            x_aug = torch.cat(
                [P, h, x_seg], dim=1
            )  # [B, n_persistent + T, d_model]
        else:
            x_aug = torch.cat(
                [h, x_seg], dim=1
            )  # [B, T, d_model]

        x_aug = x_aug + self.attention(x_aug)
        x_aug = x_aug + self.ff(x_aug)

        out = x_aug[:, self.n_persistent + T :, :]  # [B, T, d]

        out = self.norm2(out)

        #  update memory online
        if update_memory:
            memory_tokens = x_aug[
                :, self.n_persistent : self.n_persistent + T, :
            ]
            self.mem.update_from_batch(
                q_t.detach(),
                memory_tokens.detach(),
            )

        return out


class MACResidualInjectionBlock(MACTransformerBlock):
    def __init__(
        self,
        mem: NeuralMemory,
        num_patches: int,  # number of patches per frame
        num_frames: int,  # number of frames in segment
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_persistent: int = 0,  # number of persistent tokens P
        dropout: float = 0.0,
        dim_head: int = 64,
        update_type: str = "selfattention",  # "selfattention" or "crossattention"
        proj_k_eq_q: bool = False,
    ):
        super().__init__(
            mem=mem,
            num_patches=num_patches,
            num_frames=num_frames,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_persistent=n_persistent,
            dropout=dropout,
            dim_head=dim_head,
            update_type=update_type,
        )
        self.attention = Attention(d_model, n_heads, dim_head, dropout)

        self.injection_layer = nn.Linear(d_model, d_model)
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor, update_memory: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        x = self.norm1(x)

        # retrieve long-term memory for current segment
        q_t = self.mem_W_Q(x)  # [B, T, d_model]
        M = self.mem.retrieve(q_t)  # [B, T, d_model]

        x = x + self.attention(x)
        x = x + self.injection_layer(M) * self.alpha
        x = x + self.ff(x)
        out = self.norm2(x)

        #  update memory online
        if update_memory:
            self.mem.update_from_batch(
                q_t.detach(),
                M.detach(),
            )
          
        return out

class MACCrossAttentionBlock(MACTransformerBlock):
    def __init__(
        self,
        mem: NeuralMemory,
        num_patches: int,  # number of patches per frame
        num_frames: int,  # number of frames in segment
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_persistent: int = 0,  # number of persistent tokens P
        dropout: float = 0.0,
        dim_head: int = 64,
        update_type: str = "selfattention",  # "selfattention" or "crossattention"
        proj_k_eq_q: bool = False,
    ):
        super().__init__(
            mem=mem,
            num_patches=num_patches,
            num_frames=num_frames,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_persistent=n_persistent,
            dropout=dropout,
            dim_head=dim_head,
            update_type=update_type,
        )
        self.attention = Attention(d_model, n_heads, dim_head, dropout)
        self.injection_layer = CrossAttention(d_model, n_heads, dim_head, dropout, bias=generate_diagonal_frame_mask(NUM_PATCHES, NUM_FRAMES))

    def forward(self, x: torch.Tensor, update_memory: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        x = self.norm1(x)

        # retrieve long-term memory for current segment
        q_t = self.mem_W_Q(x)  # [B, T, d_model]
        M = self.mem.retrieve(q_t)  # [B, T, d_model]

        x = x + self.attention(x)
        x = x + self.injection_layer(x, M)
        x = x + self.ff(x)
        out = self.norm2(x)

        #  update memory online
        if update_memory:
            self.mem.update_from_batch(
                q_t.detach(),
                M.detach(),
            )
          
        return out


class MACAdaMemTransformerBlock(MACTransformerBlock):
    def __init__(
        self,
        mem: NeuralMemory,
        num_patches: int,  # number of patches per frame
        num_frames: int,  # number of frames in segment
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_persistent: int = 0,  # number of persistent tokens P
        dropout: float = 0.0,
        dim_head: int = 64,
        update_type: str = "selfattention",  # "selfattention" or "crossattention"
        proj_k_eq_q: bool = False,

    ):
        super().__init__(
            mem=mem,
            num_patches=num_patches,
            num_frames=num_frames,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_persistent=n_persistent,
            dropout=dropout,
            dim_head=dim_head,
            update_type=update_type,
        )
        self.attention = Attention(d_model, n_heads, dim_head, dropout)
        self.injection_layer1 = AdaptiveLayerNorm(d_model, d_model)
        self.injection_layer2 = AdaptiveLayerNorm(d_model, d_model)

    def forward(self, x: torch.Tensor, update_memory: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        x = self.norm1(x)

        # retrieve long-term memory for current segment
        q_t = self.mem_W_Q(x)  # [B, T, d_model]
        M = self.mem.retrieve(q_t)  # [B, T, d_model]


        x = x + self.attention(self.injection_layer1(x, M))
        x = x + self.ff(self.injection_layer2(x, M))
        out = self.norm2(x)

        #  update memory online
        if update_memory:
            self.mem.update_from_batch(
                q_t.detach(),
                M.detach(),
            )
          
        return out

class MACLoRATransformerBlock(MACTransformerBlock):
    def __init__(
        self,
        mem: NeuralMemory,
        num_patches: int,  # number of patches per frame
        num_frames: int,  # number of frames in segment
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_persistent: int = 0,  # number of persistent tokens P
        dropout: float = 0.0,
        dim_head: int = 64,
        update_type: str = "selfattention",  # "selfattention" or "crossattention"
        proj_k_eq_q: bool = False,

    ):
        super().__init__(
            mem=mem,
            num_patches=num_patches,
            num_frames=num_frames,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_persistent=n_persistent,
            dropout=dropout,
            dim_head=dim_head,
            update_type=update_type,
        )
        self.attention = DynamicLoRAAttention(d_model, n_heads, dim_head, r=16, alpha=0.5, dropout=dropout)

    def forward(self, x: torch.Tensor, update_memory: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        x = self.norm1(x)

        # retrieve long-term memory for current segment
        q_t = self.mem_W_Q(x)  # [B, T, d_model]
        M = self.mem.retrieve(q_t)  # [B, T, d_model]

        x = x + self.attention(x, M)
        x = x + self.ff(x)
        out = self.norm2(x)

        #  update memory online
        if update_memory:
            self.mem.update_from_batch(
                q_t.detach(),
                M.detach(),
            )
          
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
        dim_head=64,
        update_type="selfattention",
        proj_k_eq_q=False,
        mem_layer_type: str = "all",
        injection_type = "prepend",
    ):
        super().__init__()
        self.mem = memory_module
        self.norm = nn.LayerNorm(dim)
        self.mem_layer_type = mem_layer_type
        self.injection_type = injection_type

        if self.mem_layer_type == 'all':
            self.mem_layer_idx = list(range(depth))
        elif self.mem_layer_type == 'first':
            self.mem_layer_idx = [0]
        elif self.mem_layer_type == 'middle':
            self.mem_layer_idx = list(range(1, depth - 1))
        elif self.mem_layer_type == 'last':
            self.mem_layer_idx = [depth - 1]
        elif self.mem_layer_type == 'alternate':
            self.mem_layer_idx = list(range(0, depth, 2))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            if i in self.mem_layer_idx:
                if self.injection_type == "prepend":
                    injection_layer = MACTransformerBlock(
                        mem=memory_module,
                        num_patches=num_patches,
                        num_frames=num_frames,
                        d_model=dim,
                        n_heads=heads,
                        d_ff=mlp_dim,
                        n_persistent=n_persistent,
                        dropout=dropout,
                        dim_head=dim_head,
                        update_type=update_type,
                        proj_k_eq_q=proj_k_eq_q,
                    )
                elif self.injection_type == "residual":
                    injection_layer = MACResidualInjectionBlock(
                        mem=memory_module,
                        num_patches=num_patches,
                        num_frames=num_frames,
                        d_model=dim,
                        n_heads=heads,
                        d_ff=mlp_dim,
                        dropout=dropout,
                        dim_head=dim_head,
                        update_type=update_type,
                        proj_k_eq_q=proj_k_eq_q,
                    )
                elif self.injection_type == "crossattention":
                    injection_layer = MACCrossAttentionBlock(
                        mem=memory_module,
                        num_patches=num_patches,
                        num_frames=num_frames,
                        d_model=dim,
                        n_heads=heads,
                        d_ff=mlp_dim,
                        dropout=dropout,
                        dim_head=dim_head,
                        update_type=update_type,
                        proj_k_eq_q=proj_k_eq_q,
                    )
                elif self.injection_type == "admem":
                    injection_layer = MACAdaMemTransformerBlock(
                        mem=memory_module,
                        num_patches=num_patches,
                        num_frames=num_frames,
                        d_model=dim,
                        n_heads=heads,
                        d_ff=mlp_dim,
                        dropout=dropout,
                        dim_head=dim_head,
                        update_type=update_type,
                        proj_k_eq_q=proj_k_eq_q,
                    )
                elif self.injection_type == "lora":
                    injection_layer = MACLoRATransformerBlock(
                        mem=memory_module,
                        num_patches=num_patches,
                        num_frames=num_frames,
                        d_model=dim,
                        n_heads=heads,
                        d_ff=mlp_dim,
                        dropout=dropout,
                        dim_head=dim_head,
                        update_type=update_type,
                        proj_k_eq_q=proj_k_eq_q,
                    )
                else:
                    raise ValueError(f"Invalid injection type: {self.injection_type}")

                self.layers.append(injection_layer)
                
            else:
                self.layers.append(
                    TransformerBlock(dim, heads, dim_head, mlp_dim, dropout, bias=generate_mask_matrix(NUM_PATCHES, NUM_FRAMES))
                )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x), None


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
        update_type="selfattention",
        proj_k_eq_q=False,
        mem_layer_type: str = "all",
        injection_type = "prepend",
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
            dim_head,
            update_type,
            proj_k_eq_q,
            mem_layer_type,
            injection_type=injection_type,
        )

    def forward(self, x, H=None):
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
        return self.norm(x), None

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
        return self.norm(x), None


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

    def forward(self, x, H=None):
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        return self.transformer(x)

    def reset_memory(self):
        self.mem.reset_weights()


class StateSpaceTransformer(nn.Module):
    def __init__(
        self,
        dim,
        num_patches,
        depth,
        heads,
        mlp_dim,
        state_dim,
        step_size,
        n_mem_blocks,
        dropout=0.0,
        dim_head=64,
        dt_rank: int = 16,
        use_gate: bool = False,
        use_cls_token: bool = False,
        shift_memory: bool = False,
        mem_layer_type: str = "all", # "all", "first", "middle", "last", "alternate" (injection position)
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.dim_head = dim_head
        self.use_gate = use_gate
        self.dt_rank = dt_rank
        self.num_patches = num_patches
        self.state_dim = state_dim
        self.step_size = step_size
        self.n_mem_blocks = n_mem_blocks
        self.use_cls_token = use_cls_token
        self.shift_memory = shift_memory
        self.mem_layer_type = mem_layer_type

        if self.mem_layer_type == 'all':
            self.mem_layer_idx = list(range(depth))
        elif self.mem_layer_type == 'first':
            self.mem_layer_idx = [0]
        elif self.mem_layer_type == 'middle':
            self.mem_layer_idx = list(range(1, depth - 1))
        elif self.mem_layer_type == 'last':
            self.mem_layer_idx = [depth - 1]
        elif self.mem_layer_type == 'alternate':
            self.mem_layer_idx = list(range(0, depth, 2))

        if use_gate:
            self.gate = nn.Linear(dim, dim)
        else:
            self.gate = None

        self.ln_in = nn.LayerNorm(dim)
        self.ln_out = nn.LayerNorm(dim)

        self.H_buffer = (
            None  # keep track of hidden memory state w/o passing around
        )

        self._build_transformer(
            depth, dim, heads, dim_head, dropout, mlp_dim, **kwargs
        )
        self._build_mem_blocks(
            n_mem_blocks, dim, state_dim, dropout, mlp_dim, dt_rank, **kwargs
        )

    def _build_mem_blocks(
        self, n_mem_blocks, dim, state_dim, dropout, mlp_dim, dt_rank, **kwargs
    ):
        self.ln_mem_out = nn.Identity() # nn.LayerNorm(dim) #TODO: add back in
        self.mem_blocks = nn.ModuleList([])

        for _ in range(n_mem_blocks):
            self.mem_blocks.append(
                BasicMambaLayer(
                    d_model=dim,
                    n_state=state_dim,
                    step_size=self.step_size,
                    num_patches=self.num_patches if not self.use_cls_token else 1,
                    dt_rank=dt_rank,
                    dropout=dropout,
                ),
            )

    def _build_transformer(
        self, depth, dim, heads, dim_head, dropout, mlp_dim, **kwargs
    ):
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
                            bias=(
                                generate_mask_with_memory(
                                    NUM_PATCHES, NUM_FRAMES
                                )
                                if not kwargs.get("use_gate", False)
                                else None
                            ),
                        ),
                        FeedForward(dim, mlp_dim, dropout),
                    ]
                )
            )

    def _mem_blocks_forward(self, x):
        B, T, D = x.shape
        x = x.clone()
        x = rearrange(x, "b (t p) d -> b t p d", t=T // self.num_patches)
  
        for mem_block in self.mem_blocks:
            x, H_T = mem_block(x)

        return rearrange(x, "b t p d -> b (t p) d")

    def forward(self, x, H=None):
        B, T, D = x.shape

        M_new = self._mem_blocks_forward(x)
        x = self.ln_in(x)

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

        return self.ln_out(ctx), self.ln_mem_out(M_new)

    def reset_memory(self):
        for mem_block in self.mem_blocks:
            mem_block.reset_memory()

    def set_step_size(self, step_size):
        for mem_block in self.mem_blocks:
            mem_block.set_step_size(step_size)


class MemoryInjectionSSMTransformer(StateSpaceTransformer):
    def __init__(
        self,
        dim,
        num_patches,
        depth,
        heads,
        mlp_dim,
        state_dim,
        step_size,
        n_mem_blocks,
        dropout=0.0,
        dim_head=64,
        dt_rank: int = 16,
        alpha_init: float = 0.1,
        use_cls_token: bool = False,
        shift_memory: bool = False,
        mem_layer_type: str = "all", # "all", "first", "middle", "last", "alternate" (injection position)
        **kwargs,
    ):
        super().__init__(
            dim,
            num_patches,
            depth,
            heads,
            mlp_dim,
            state_dim,
            step_size,
            n_mem_blocks,
            dropout,
            dim_head,
            dt_rank,
            alpha_init=alpha_init,
            use_cls_token=use_cls_token,
            shift_memory=shift_memory,
            mem_layer_type=mem_layer_type,
            **kwargs,
        )

    def _build_transformer(
        self, depth, dim, heads, dim_head, dropout, mlp_dim, **kwargs
    ):
        self.alphas = nn.ParameterList(
            [
                nn.Parameter(torch.ones(1) * kwargs.get("alpha_init", 0.1))
                for _ in self.mem_layer_idx
            ]
        )

        self.layers = nn.ModuleList([])
        self.injection_layers = nn.ModuleList([])
        for i in range(depth):
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
            if i in self.mem_layer_idx:
                self.injection_layers.append(nn.Linear(dim, dim))

    def forward(self, x):
        B, T, D = x.shape
        M_new = self._mem_blocks_forward(x)
        x = self.ln_in(x)

        for i, (attn, ff) in enumerate(self.layers):
            x = x + attn(x)
            if i in self.mem_layer_idx:
                idx = self.mem_layer_idx.index(i)            
                x = x + self.injection_layers[idx](M_new) * self.alphas[idx]
            x = x + ff(x)

        return self.ln_out(x), self.ln_mem_out(M_new)

class AdaMemSSMTransformer(StateSpaceTransformer):
    def __init__(
        self,
        dim,
        num_patches,
        depth,
        heads,
        mlp_dim,
        state_dim,
        step_size,
        n_mem_blocks,
        dropout=0.0,
        dim_head=64,
        dt_rank: int = 16,
        zero_init: bool = False,
        use_cls_token: bool = False,
        shift_memory: bool = False,
        both_injections: bool = False,
        mem_layer_type: str = "all",
        **kwargs,
    ):
        self.both_injections = both_injections
        super().__init__(
            dim,
            num_patches,
            depth,
            heads,
            mlp_dim,
            state_dim,
            step_size,
            n_mem_blocks,
            dropout,
            dim_head,
            dt_rank,
            zero_init=zero_init,
            use_cls_token=use_cls_token,
            shift_memory=shift_memory,
            both_injections=both_injections,
            mem_layer_type=mem_layer_type,
            **kwargs,
        )
        
    def _build_transformer(
        self, depth, dim, heads, dim_head, dropout, mlp_dim, **kwargs
    ):
        self.layers = nn.ModuleList([])
        self.injection_layers = nn.ModuleList([])
        for i in range(depth):
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
            if i in self.mem_layer_idx:
                self.injection_layers.append(
                    nn.ModuleList(
                        [
                            AdaptiveLayerNorm(
                                dim, dim, zero_init=kwargs.get("zero_init", False)
                            ) if self.both_injections else TwoInputIdentity(),
                            AdaptiveLayerNorm(
                                dim, dim, zero_init=kwargs.get("zero_init", False)
                            )
                        ]
                    )                
                )
            else:
                self.injection_layers.append(nn.ModuleList([TwoInputIdentity(), TwoInputIdentity()]))
            

    def forward(self, x):
        M_new = self._mem_blocks_forward(x)
        x = self.ln_in(x)

        for i, (attn, ff) in enumerate(self.layers):            
            injection_1, injection_2 = self.injection_layers[i]
            x = x + attn(injection_1(x, M_new))
            x = x + ff(injection_2(x, M_new))

        return self.ln_out(x), self.ln_mem_out(M_new)


class MemCrossAttentionSSMTransformer(StateSpaceTransformer):
    def __init__(
        self,
        dim,
        num_patches,
        depth,
        heads,
        mlp_dim,
        state_dim,
        step_size,
        n_mem_blocks,
        dropout=0.0,
        dim_head=64,
        dt_rank: int = 16,
        use_cls_token: bool = False,
        shift_memory: bool = False,
        mem_layer_type: str = "all",
        **kwargs,
    ):
        super().__init__(
            dim,
            num_patches,
            depth,
            heads,
            mlp_dim,
            state_dim,
            step_size,
            n_mem_blocks,
            dropout,
            dim_head,
            dt_rank,
            use_cls_token=use_cls_token,
            shift_memory=shift_memory,
            mem_layer_type=mem_layer_type,
            **kwargs,
        )

    def _build_transformer(
        self, depth, dim, heads, dim_head, dropout, mlp_dim, **kwargs
    ):
        self.layers = nn.ModuleList([])
        self.injection_layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads, dim_head, dropout),
                        FeedForward(dim, mlp_dim, dropout),
                    ]
                )
            )
            # Generate bias mask for cross-attention to prevent attending to future memory tokens
            bias = generate_diagonal_frame_mask(NUM_PATCHES, NUM_FRAMES)
            if i in self.mem_layer_idx:
                self.injection_layers.append(
                    CrossAttention(dim, heads, dim_head, dropout, bias=bias)
                )

    def forward(self, x):
        M_new = self._mem_blocks_forward(x)
        x = self.ln_in(x)

        for i, (attn, ff) in enumerate(self.layers):
            x = x + attn(x)
            if i in self.mem_layer_idx:
                x = x + self.injection_layers[self.mem_layer_idx.index(i)](x, M_new)
            x = x + ff(x)

        return self.ln_out(x), self.ln_mem_out(M_new)


class BasicMemCrossAttentionSSMTransformer(MemCrossAttentionSSMTransformer):
    def __init__(
        self,
        dim,
        num_patches,
        depth,
        heads,
        mlp_dim,
        state_dim,
        step_size,
        n_mem_blocks,
        dropout=0.0,
        dim_head=64,
        dt_rank: int = 16,
        use_cls_token: bool = False,
        shift_memory: bool = False,
        mem_layer_type: str = "all",
        **kwargs,
    ):
        super().__init__(
            dim,
            num_patches,
            depth,
            heads,
            mlp_dim,
            state_dim,
            step_size,
            n_mem_blocks,
            dropout,
            dim_head,
            dt_rank,
            use_cls_token=use_cls_token,
            shift_memory=shift_memory,
            mem_layer_type=mem_layer_type,
            **kwargs,
        )
    
    def _build_mem_blocks(
        self, n_mem_blocks, dim, state_dim, dropout, mlp_dim, dt_rank, **kwargs
    ):
        self.ln_mem_out = nn.Identity() # nn.LayerNorm(dim) #TODO: add back in
        self.mem_blocks = nn.ModuleList([])
        for _ in range(n_mem_blocks):
            self.mem_blocks.append(
                BasicMambaLayer(
                    d_model=dim,
                    n_state=state_dim,
                    step_size=self.step_size,
                    num_patches=self.num_patches if not self.use_cls_token else 1,
                    dt_rank=dt_rank,
                    dropout=dropout,
                ),
            )

    def _mem_blocks_forward(self, x):
        B, T, D = x.shape
        x = x.clone()
        x = rearrange(x, "b (t p) d -> b t p d", t=T // self.num_patches)
        if self.use_cls_token:
            x = x[:,:, 0, :].unsqueeze(2) # select only the cls token

        if self.shift_memory:
            H_0 = self.mem_blocks[0].H_cache
            if H_0 is None:
                H_0 = self.mem_blocks[0].init_state(B, device=x.device)

        for mem_block in self.mem_blocks:
            x, H_T = mem_block(x)
        
        if self.use_cls_token:
            # repeat the cls token to the number of patches
            x = x.repeat(1, 1, self.num_patches, 1)

        if self.shift_memory:
            H_T = H_T[:, :-1, :, :]
            H_T = torch.cat([H_0.unsqueeze(1), H_T], dim=1)

        return rearrange(x, "b t p d -> b (t p) d")

    def reset_memory(self):    
        for mem_block in self.mem_blocks:
            mem_block.reset_memory()

    def set_step_size(self, step_size):
        for mem_block in self.mem_blocks:
            mem_block.set_step_size(step_size)

class HiddenMemCrossAttentionSSMTransformer(StateSpaceTransformer):
    def __init__(
        self,
        dim,
        num_patches,
        depth,
        heads,
        mlp_dim,
        state_dim,
        step_size,
        n_mem_blocks,
        dropout=0.0,
        dim_head=64,
        dt_rank: int = 16,
        use_cls_token: bool = False,
        shift_memory: bool = True,
        mem_layer_type: str = "all",
        **kwargs,
    ):
        super().__init__(
            dim,
            num_patches,
            depth,
            heads,
            mlp_dim,
            state_dim,
            step_size,
            n_mem_blocks,
            dropout,
            dim_head,
            dt_rank,
            use_cls_token=use_cls_token,
            shift_memory=shift_memory,
            mem_layer_type=mem_layer_type,
            **kwargs,
        )
        self.shift_memory = shift_memory

    def _build_mem_blocks(
        self, n_mem_blocks, dim, state_dim, dropout, mlp_dim, dt_rank, **kwargs
    ):
        self.ln_mem_out = nn.Identity() # nn.LayerNorm(dim) #TODO: add back in
        self.mem_blocks = nn.ModuleList([])
        for _ in range(n_mem_blocks):
            self.mem_blocks.append(
                BasicHiddenMambaLayer(
                    d_model=dim,
                    n_state=state_dim,
                    step_size=self.step_size,
                    num_patches=self.num_patches if not self.use_cls_token else 1,
                    dt_rank=dt_rank,
                    dropout=dropout,
                ),
            )
    
    def _build_transformer(
        self, depth, dim, heads, dim_head, dropout, mlp_dim, **kwargs
    ):
        self.layers = nn.ModuleList([])
        self.injection_layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads, dim_head, dropout),
                        FeedForward(dim, mlp_dim, dropout),
                    ]
                )
            )
            # Generate bias mask for cross-attention to prevent attending to future memory tokens
            bias = generate_diagonal_frame_mask(NUM_PATCHES, NUM_FRAMES)
            if i in self.mem_layer_idx:
                self.injection_layers.append(
                    CrossAttentionInjection(dim, self.state_dim, heads, dim_head, dropout, bias=bias)
                )
    
    def _mem_blocks_forward(self, x):
        B, T, D = x.shape
        x = x.clone()
        x = rearrange(x, "b (t p) d -> b t p d", t=T // self.num_patches)
        if self.use_cls_token:
            x = x[:,:, 0, :].unsqueeze(2) # select only the cls token
        
        if self.shift_memory:
            H_0 = self.mem_blocks[0].H_cache
            if H_0 is None:
                H_0 = self.mem_blocks[0].init_state(B, device=x.device)
                  
        for mem_block in self.mem_blocks:
            x, H_T = mem_block(x)
        
        if self.use_cls_token:
            # repeat the cls token to the number of patches
            H_T = H_T.repeat(1, 1, self.num_patches, 1)
        
        if self.shift_memory:
            H_T = H_T[:, :-1, :, :]
            H_T = torch.cat([H_0.unsqueeze(1), H_T], dim=1)

        return rearrange(H_T, "b t p d -> b (t p) d")


    def forward(self, x):
        M_new = self._mem_blocks_forward(x)
        x = self.ln_in(x)

        for i, (attn, ff) in enumerate(self.layers):
            x = x + attn(x)
            if i in self.mem_layer_idx:
                x = x + self.injection_layers[self.mem_layer_idx.index(i)](x, M_new)
            x = x + ff(x)

        return self.ln_out(x), self.ln_mem_out(M_new)
    
    def reset_memory(self):
        for mem_block in self.mem_blocks:
            mem_block.reset_memory()

    def set_step_size(self, step_size):
        for mem_block in self.mem_blocks:
            mem_block.set_step_size(step_size)


class StateSpaceViTPredictor(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        state_dim,
        depth,
        heads,
        mlp_dim,
        injection_type,
        step_size,
        n_mem_blocks,
        alpha_init: float = 0.1,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=64,
        use_gate: bool = False,
        dt_rank: int = 16,
        zero_init: bool = False,
        lora_rank: int = 64,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        use_cls_token: bool = False,
        shift_memory: bool = True,
        both_injections: bool = False,
        mem_layer_type: str = "all",
    ):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
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
                dim=dim,
                num_patches=num_patches,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                state_dim=state_dim,
                step_size=step_size,
                n_mem_blocks=n_mem_blocks,
                dropout=dropout,
                dim_head=dim_head,
                dt_rank=dt_rank,
                use_gate=use_gate,
                use_cls_token=use_cls_token,
                shift_memory=shift_memory,
                mem_layer_type=mem_layer_type,
            )
        elif injection_type == "misst":
            self.transformer = MemoryInjectionSSMTransformer(
                dim=dim,
                num_patches=num_patches,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                state_dim=state_dim,
                step_size=step_size,
                n_mem_blocks=n_mem_blocks,
                dropout=dropout,
                dim_head=dim_head,
                dt_rank=dt_rank,
                alpha_init=alpha_init,
                use_cls_token=use_cls_token,
                shift_memory=shift_memory,
                mem_layer_type=mem_layer_type,
            )
        elif injection_type == "adamem":
            self.transformer = AdaMemSSMTransformer(
                dim=dim,
                num_patches=num_patches,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                state_dim=state_dim,
                step_size=step_size,
                n_mem_blocks=n_mem_blocks,
                dropout=dropout,
                dim_head=dim_head,
                dt_rank=dt_rank,
                zero_init=zero_init,
                use_cls_token=use_cls_token,
                shift_memory=shift_memory,
                both_injections=both_injections,
                mem_layer_type=mem_layer_type,
            )

        elif injection_type == "ssm_ca":
            self.transformer = MemCrossAttentionSSMTransformer(
                dim=dim,
                num_patches=num_patches,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                state_dim=state_dim,
                step_size=step_size,
                n_mem_blocks=n_mem_blocks,
                dropout=dropout,
                dim_head=dim_head,
                dt_rank=dt_rank,
                use_cls_token=use_cls_token,
                shift_memory=shift_memory,
                mem_layer_type=mem_layer_type,
            )
        elif injection_type == "ca_hidden":
            self.transformer = HiddenMemCrossAttentionSSMTransformer(
                dim=dim,
                num_patches=num_patches,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                state_dim=state_dim,
                step_size=step_size,
                n_mem_blocks=n_mem_blocks,
                dropout=dropout,
                dim_head=dim_head,
                dt_rank=dt_rank,
                use_cls_token=use_cls_token,
                shift_memory=shift_memory,
                mem_layer_type=mem_layer_type,
            )
        elif injection_type == "ca_basic":
            self.transformer = BasicMemCrossAttentionSSMTransformer(
                dim=dim,
                num_patches=num_patches,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                state_dim=state_dim,
                step_size=step_size,
                n_mem_blocks=n_mem_blocks,
                dropout=dropout,
                dim_head=dim_head,
                dt_rank=dt_rank,
                use_cls_token=use_cls_token,
                shift_memory=shift_memory,
                mem_layer_type=mem_layer_type,
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

    def set_step_size(self, step_size):
        self.transformer.set_step_size(step_size)

### CACHE MEMORY VIT PREDICTOR ###

class CacheMemoryTransformer(nn.Module):

    def __init__(
        self,
        dim,
        num_patches,
        depth,
        heads,
        mlp_dim,
        cache_size,
        step_size=1,
        dropout=0.0,
        dim_head=64,
        mem_layer_type: str = "all",  # "all", "first", "middle", "last", "alternate" (injection position)
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.dim_head = dim_head
        self.mem_layer_type = mem_layer_type
        self.step_size = step_size 
        self.num_patches = num_patches
        self.cache_size = cache_size

        if self.mem_layer_type == "all":
            self.mem_layer_idx = list(range(depth))
        elif self.mem_layer_type == "first":
            self.mem_layer_idx = [0]
        elif self.mem_layer_type == "middle":
            self.mem_layer_idx = list(range(1, depth - 1))
        elif self.mem_layer_type == "last":
            self.mem_layer_idx = [depth - 1]
        elif self.mem_layer_type == "alternate":
            self.mem_layer_idx = list(range(0, depth, 2))

        self.ln_in = nn.LayerNorm(dim)
        self.ln_out = nn.LayerNorm(dim)

        self.H_buffer = None # (b, cache_size, p, d)

        self._build_transformer(
            depth, dim, heads, dim_head, dropout, mlp_dim, **kwargs
        )

    def _build_transformer(
        self, depth, dim, heads, dim_head, dropout, mlp_dim, **kwargs
    ):
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
                            bias=generate_mask_with_memory(NUM_PATCHES, NUM_FRAMES)                      
                        ),
                        FeedForward(dim, mlp_dim, dropout),
                    ]
                )
            )

    def forward(self, x):
        B, T, D = x.shape
        x = self.ln_in(x)
        M_new = self._get_memory()
        M_T = M_new.size(1) if M_new is not None else 0

        ctx = self.ln_fuse(torch.cat([M_new, x], dim=1)) if M_new is not None else x
        for i, (attn, ff) in enumerate(self.layers):
            ctx = ctx + attn(ctx)
            ctx = ctx + ff(ctx)

        ctx = ctx[:, M_T:] if M_new is not None else ctx

        self._update_memory(ctx.detach().clone())
        return self.ln_out(ctx), self._get_memory()
        
    
    def reset_memory(self):
        self.H_buffer = None

    def set_step_size(self, step_size):
        self.step_size = step_size
    
    def _update_memory(self, mem):
        if mem is not None:
            mem = rearrange(mem, "b (t p) d -> b t p d", p=self.num_patches)
            mem = mem[:, :self.step_size]

            if self.H_buffer is None:
                self.H_buffer = mem[:, -self.cache_size:]
            else:
                self.H_buffer = torch.cat([self.H_buffer, mem], dim=1)[:, -self.cache_size:]
    
    def _get_memory(self):
        if self.H_buffer is None:
            return self.H_buffer
        else:
            return rearrange(self.H_buffer.clone(), "b t p d -> b (t p) d")


class CacheMemoryInjectionTransformer(CacheMemoryTransformer):
    def __init__(
        self,
        dim,
        num_patches,
        depth,
        heads,
        mlp_dim,
        cache_size,
        step_size=1,
        dropout=0.0,
        dim_head=64,
        mem_layer_type: str = "all",  # "all", "first", "middle", "last", "alternate" (injection position)
        alpha_init: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            num_patches=num_patches,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            cache_size=cache_size,
            step_size=step_size,
            dropout=dropout,
            dim_head=dim_head,
            mem_layer_type=mem_layer_type,
            alpha_init=alpha_init,
            **kwargs,
        )

    def _build_transformer(
        self, depth, dim, heads, dim_head, dropout, mlp_dim, **kwargs
    ):
        self.alphas = nn.ParameterList(
            [
                nn.Parameter(torch.ones(1) * kwargs.get("alpha_init", 0.1))
                for _ in self.mem_layer_idx
            ]
        )

        self.layers = nn.ModuleList([])
        self.injection_layers = nn.ModuleList([])
        for i in range(depth):
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
            if i in self.mem_layer_idx:
                self.injection_layers.append(nn.Linear(dim, dim))
    
    def forward(self, x):
        B, T, D = x.shape        
        x = self.ln_in(x)
        M_new = self._get_memory()
        if M_new is not None:
            M_new = M_new.mean(dim=1).unsqueeze(1).repeat(1, T, 1)

        for i, (attn, ff) in enumerate(self.layers):
            x = x + attn(x)
            if i in self.mem_layer_idx and M_new is not None:
                idx = self.mem_layer_idx.index(i)       
                injection = self.injection_layers[idx](M_new)      
                x = x + injection * self.alphas[idx]
            x = x + ff(x)

        self._update_memory(x.detach().clone())
        return self.ln_out(x), self._get_memory()


class CacheAdaMemTransformer(CacheMemoryTransformer):
    def __init__(
        self,
        dim,
        num_patches,
        depth,
        heads,
        mlp_dim,
        cache_size,
        step_size=1,
        dropout=0.0,
        dim_head=64,
        zero_init: bool = False,
        both_injections: bool = False,
        mem_layer_type: str = "all",
        **kwargs,
    ):
        self.both_injections = both_injections
        self.zero_init = zero_init
        super().__init__(
            dim=dim,
            num_patches=num_patches,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            cache_size=cache_size,
            step_size=step_size,
            dropout=dropout,
            dim_head=dim_head,
            mem_layer_type=mem_layer_type,
            zero_init=zero_init,
            both_injections=both_injections,
            **kwargs,   
        )

    def _build_transformer(
        self, depth, dim, heads, dim_head, dropout, mlp_dim, **kwargs
    ):
        self.layers = nn.ModuleList([])
        self.injection_layers = nn.ModuleList([])
        
        for i in range(depth):
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
            if i in self.mem_layer_idx:
                self.injection_layers.append(
                    nn.ModuleList(
                        [
                            AdaptiveLayerNorm(
                                dim, dim, zero_init=kwargs.get("zero_init", False)
                            ) if self.both_injections else TwoInputIdentity(),
                            AdaptiveLayerNorm(
                                dim, dim, zero_init=kwargs.get("zero_init", False)
                            )
                        ]
                    )                
                )
            else:
                self.injection_layers.append(nn.ModuleList([TwoInputIdentity(), TwoInputIdentity()]))
    
    def forward(self, x):
        M_new = self._get_memory()
        if M_new is not None:
            M = M_new.mean(dim=1).unsqueeze(1).repeat(1, x.size(1), 1)
        x = self.ln_in(x)

        for i, (attn, ff) in enumerate(self.layers):            
            if M_new is not None:
                injection_1, injection_2 = self.injection_layers[i]
                x = x + attn(injection_1(x, M))
                x = x + ff(injection_2(x, M))
            else:
                x = x + attn(x)
                x = x + ff(x)

        self._update_memory(x.detach().clone())
        return self.ln_out(x), self._get_memory()

class CacheLoRAAttentionTransformer(CacheMemoryTransformer):
    def __init__(
        self,
        dim,
        num_patches,
        depth,
        heads,
        mlp_dim,
        cache_size,
        step_size,
        dropout=0.0,
        dim_head=64,
        lora_rank: int = 16,
        lora_alpha: float = 2.0,
        use_qk: bool = True,
        use_vo: bool = False,
        gen_type: str = "A",
        mem_layer_type: str = "all",
        **kwargs,
    ):
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.use_qk = use_qk
        self.use_vo = use_vo
        self.gen_type = gen_type
        super().__init__(
            dim=dim,
            num_patches=num_patches,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            cache_size=cache_size,
            step_size=step_size,
            dropout=dropout,
            dim_head=dim_head,
            mem_layer_type=mem_layer_type,
            **kwargs,
        )
    
    def _build_transformer(self, depth, dim, heads, dim_head, dropout, mlp_dim, **kwargs):
        self.layers = nn.ModuleList([])
        for i in range(depth):
            block = nn.ModuleList([])
            if i in self.mem_layer_idx:
                block.append(
                    DynamicLoRAAttention(
                        dim=dim, 
                        heads=heads, 
                        dim_head=dim_head,
                        r=self.lora_rank, 
                        alpha=self.lora_alpha,
                        gen_type=self.gen_type,
                        use_qk=self.use_qk,
                        use_vo=self.use_vo,
                        dropout=dropout, 
                        inject_pooled=True
                    )
                )
            else:
                block.append(Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout))
            block.append(FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout))
            self.layers.append(block)

    def forward(self, x):
        M_new = self._get_memory()
        x = self.ln_in(x)
        for i, (attn, ff) in enumerate(self.layers):
            attn_out = attn(x, M_new) if i in self.mem_layer_idx else attn(x)
            x = x + attn_out            
            x = x + ff(x)
        self._update_memory(x.detach().clone())
        return self.ln_out(x), self._get_memory()

class CacheCrossAttentionTransformer(CacheMemoryTransformer):
    def __init__(
        self,
        dim,
        num_patches,
        depth,
        heads,
        mlp_dim,
        cache_size,
        step_size=1,
        dropout=0.0,
        dim_head=64,
        mem_layer_type: str = "all",
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            num_patches=num_patches,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            cache_size=cache_size,
            step_size=step_size,
            dropout=dropout,
            dim_head=dim_head,
            mem_layer_type=mem_layer_type,
            **kwargs,
        )
    
    def _build_transformer(
        self, depth, dim, heads, dim_head, dropout, mlp_dim, **kwargs
    ):
        self.layers = nn.ModuleList([])
        self.injection_layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads, dim_head, dropout),
                        FeedForward(dim, mlp_dim, dropout),
                    ]
                )
            )
            # since cache is always < T, we don't need to use a bias mask
            # bias = generate_diagonal_frame_mask(NUM_PATCHES, NUM_FRAMES)
            bias = generate_full_mask(NUM_PATCHES, NUM_FRAMES)
            if i in self.mem_layer_idx:
                self.injection_layers.append(
                    CrossAttention(dim, heads, dim_head, dropout, bias=bias)
                )

    def forward(self, x):
        M_new = self._get_memory()
        x = self.ln_in(x)

        for i, (attn, ff) in enumerate(self.layers):
            x = x + attn(x)
            if i in self.mem_layer_idx and M_new is not None:
                x = x + self.injection_layers[self.mem_layer_idx.index(i)](x, M_new)
            x = x + ff(x)

        self._update_memory(x.detach().clone())
        return self.ln_out(x), self._get_memory()


class CacheMemoryViTPredictor(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        depth,
        heads,
        mlp_dim,
        injection_type,
        step_size,
        cache_size,
        alpha_init: float = 0.1,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=64,
        use_gate: bool = False,
        dt_rank: int = 16,
        zero_init: bool = False,
        lora_rank: int = 64,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        use_cls_token: bool = False,
        shift_memory: bool = True,
        both_injections: bool = False,
        mem_layer_type: str = "all",
        gen_type: str = "A",
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.dim_head = dim_head
        self.use_gate = use_gate
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.cache_size = cache_size

        # update params for adding causal attention masks
        global NUM_FRAMES, NUM_PATCHES
        NUM_FRAMES = num_frames
        NUM_PATCHES = num_patches

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames * num_patches, dim)
        )
        self.dropout = nn.Dropout(emb_dropout)

        if injection_type == "cache_prepend":
            self.transformer = CacheMemoryTransformer(
                dim=dim,
                num_patches=num_patches,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                cache_size=cache_size,
                step_size=step_size,
                dropout=dropout,
                dim_head=dim_head,
                mem_layer_type=mem_layer_type,
            )
        elif injection_type == "misst":
            self.transformer = CacheMemoryInjectionTransformer(
                dim=dim,
                num_patches=num_patches,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                cache_size=cache_size,
                step_size=step_size,
                dropout=dropout,
                dim_head=dim_head,
                alpha_init=alpha_init,
                mem_layer_type=mem_layer_type,
            )
        elif injection_type == "adamem":
            self.transformer = CacheAdaMemTransformer(
                dim=dim,
                num_patches=num_patches,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                cache_size=cache_size,
                step_size=step_size,
                dropout=dropout,
                dim_head=dim_head,
                zero_init=zero_init,
                both_injections=both_injections,
                mem_layer_type=mem_layer_type,
            )
        elif injection_type == "lora":
            self.transformer = CacheLoRAAttentionTransformer(
                dim=dim,
                num_patches=num_patches,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                cache_size=cache_size,
                step_size=step_size,
                dropout=dropout,
                dim_head=dim_head,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                use_qk=True,
                use_vo=False,
                gen_type=gen_type,
                mem_layer_type=mem_layer_type,
            )
        elif injection_type == "ca_basic":
            self.transformer = CacheCrossAttentionTransformer(
                dim=dim,
                num_patches=num_patches,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                cache_size=cache_size,
                step_size=step_size,
                dropout=dropout,
                dim_head=dim_head,
                mem_layer_type=mem_layer_type,
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

    def set_step_size(self, step_size):
        self.transformer.set_step_size(step_size)


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
        step_size,
        heads=8,
        dim_head=64,
        dropout=0.0,
        n_patches=1,
        n_frames=1,
        fusion="sum",  # 'sum' | 'diff' | 'mul' | 'gate' | 'logit_diff'
        fusion_scale=0.1,
        bias=None,
    ):
        super().__init__()

        self.dim = dim
        self.step_size = step_size
        self.heads = heads
        self.dim_head = dim_head
        self.inner = heads * dim_head
        self.scale = dim_head**-0.5
        self.n_patches = n_patches
        self.n_frames = n_frames
        self.fusion = fusion
        self.fusion_scale = fusion_scale
        self.ssm = nn.ModuleList([
            MambaLayer(
                d_model=dim, n_state=dim // 2, step_size=step_size, num_patches=n_patches, dt_rank=16, dropout=dropout
            ),
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim*2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim*2, dim),
                nn.Dropout(dropout),
            )
        ])
        

        # projections
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, self.inner, bias=False)
        self.to_k_cnt = nn.Linear(dim, self.inner, bias=False)  # content keys
        self.to_v = nn.Linear(dim, self.inner, bias=False)

        # project SSM (C_t ⊙ h_t) -> key space (per token)
        # we will construct K_mem via: Proj( (C_t ⊙ H_seq) )
        self.proj_k_mem = nn.Linear(dim, self.inner, bias=False)

        # output projection
        self.to_out = nn.Sequential(
            nn.Linear(self.inner, dim), nn.Dropout(dropout)
        )

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
                nn.Sigmoid(),
            )

        # mask (expects T = n_frames * n_patches)
        if bias is None:
            bias = generate_mask_matrix(self.n_patches, self.n_frames)
        self.register_buffer("bias", bias)  # [1,1,T,T]

    def _reshape_for_ssm(self, x, num_frames=None):
        """
        x: [B, T, D] with T = n_frames * n_patches
        returns X_win: [B, T_frames, P, D]
        """
        B, T, D = x.shape
        P = self.n_patches
        if num_frames is None:
            F = T // P
        else:
            F = num_frames

        assert T == P * F, f"T ({T}) must equal n_patches*n_frames ({P*F})"
        X_win = rearrange(x, "b (t p) d -> b t p d", t=F, p=P)  # [B, F, P, D]
        return X_win

    def _heads(self, t):
        return rearrange(t, "b n (h d) -> b h n d", h=self.heads)

    def _unheads(self, t):
        return rearrange(t, "b h n d -> b n (h d)")

    def ssm_forward(self, x):
        for layer in self.ssm:
            x = layer(x) + x
        return x
    
    def forward(self, x, H0=None):
        """
        x:  [B, T, D]
        H0: [B, P, S] or None (zeros)
        returns: y [B,T,D], H_T [B,P,S]
        """
        B, T, D = x.shape
        P, F = self.n_patches, T // self.n_patches

        x = self.norm(x)

        # ---- Q, K_content, V from raw features
        Q = self._heads(self.to_q(x))  # [B,H,T,dh]
        K_cnt = self._heads(self.to_k_cnt(x))  # [B,H,T,dh]
        V = self._heads(self.to_v(x))  # [B,H,T,dh]

        Y_seq = self._reshape_for_ssm(x, num_frames=F)  # [B,F,P,D]
        Y_seq = self.ssm_forward(Y_seq).reshape(B, F*P, D)

        # Project to key space per token, then flatten time*patch -> tokens
        K_mem_tokens = self.proj_k_mem(Y_seq)  # [B,T,Inner]
        K_mem = self._heads(K_mem_tokens)  # [B,H,T,dh]

        # ---- Two attention maps (beta: content, alpha: memory)
        dots_beta = (
            torch.matmul(Q, K_cnt.transpose(-1, -2)) * self.scale
        )  # [B,H,T,T]
        dots_alpha = torch.matmul(Q, K_mem.transpose(-1, -2)) * self.scale

        # apply same causal/structured mask
        mask = self.bias[:, :, :T, :T] == 0  # [1,1,T,T] -> bool

        # logit fusion
        if self.fusion == "logit_diff":
            diff = dots_beta - self.fusion_scale * dots_alpha
            diff = diff.masked_fill(mask, float("-inf"))
            attn = self.dropout(self.attend(diff))
            out = torch.matmul(attn, V)

        # value fusion
        else:
            dots_beta = dots_beta.masked_fill(mask, float("-inf"))
            dots_alpha = dots_alpha.masked_fill(mask, float("-inf"))

            # softmax maps
            attn_beta = self.attend(dots_beta)  # [B,H,T,T]
            attn_alpha = self.attend(dots_alpha)  # [B,H,T,T]

            # optional dropout on maps
            attn_beta = self.dropout(attn_beta)
            attn_alpha = self.dropout(attn_alpha)

            # ---- Fuse attentions / outputs
            if self.fusion == "sum":
                # (betaV + scale * alphaV)
                out = torch.matmul(
                    attn_beta, V
                ) + self.fusion_scale * torch.matmul(attn_alpha, V)

            elif self.fusion == "diff":
                # (betaV - scale * alphaV)  (differential attention)
                out = torch.matmul(
                    attn_beta, V
                ) - self.fusion_scale * torch.matmul(attn_alpha, V)

            elif self.fusion == "mul":
                # multiplicative agreement: softmax(alpha ⊙ beta)
                attn_agree = attn_alpha * attn_beta
                # renormalize per head
                attn_agree = attn_agree / (
                    attn_agree.sum(dim=-1, keepdim=True) + 1e-9
                )
                out = torch.matmul(attn_agree, V)

            elif self.fusion == "gate":
                # token-wise gate g \in [0,1], shared across heads
                # use the input x to predict gate; broadcast to heads
                g = self.gate(x).clamp(0.0, 1.0)  # [B,T,1]
                g = g.transpose(1, 2)  # [B,1,T]
                g = g.unsqueeze(-1)  # [B,1,T,1]
                out_alpha = torch.matmul(attn_alpha, V)
                out_beta = torch.matmul(attn_beta, V)
                out = (
                    g * out_beta + (1.0 - g) * self.fusion_scale * out_alpha
                )  # broadcast over heads

        # ---- Merge heads and project out
        out = self._unheads(out)  # [B,T,Inner]
        y = self.to_out(out)  # [B,T,D]
        return y

    def reset_memory(self):
        self.ssm[0].reset_memory()

    def set_step_size(self, step_size):
        self.ssm[0].set_step_size(step_size)


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
        step_size,
        dropout=0.0,
        fusion="sum",
        fusion_scale=0.1,
        bias=None,
    ):
        super().__init__()
        self.attn = DualAttentionSSMKeys(
            dim=dim,
            step_size=step_size,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            n_patches=n_patches,
            n_frames=n_frames,
            fusion=fusion,
            fusion_scale=fusion_scale,
            bias=bias,
        )
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x, H0=None):
        # attention + residual
        attn_out = self.attn(x, H0=H0)
        x = x + attn_out
        # ff + residual
        x = x + self.ff(x)
        return x

    def reset_memory(self):
        self.attn.reset_memory()

    def set_step_size(self, step_size):
        self.attn.set_step_size(step_size)


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
        step_size,
        dropout=0.0,
        fusion="sum",
        fusion_scale=0.1,
        bias=None,
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
                step_size=step_size,
                dropout=dropout,
                fusion=fusion,
                fusion_scale=fusion_scale,
                bias=bias,
            )
            self.layers.append(layer)

    def forward(self, x):
        """
        x:  [B, T, D]  with T = n_frames * n_patches
        H0: Optional initial SSM state for the FIRST layer.
            If provided, it will be used for the first layer; subsequent
            layers will init zeros unless you choose to thread H across.
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x), None

    def reset_memory(self):
        for layer in self.layers:
            layer.reset_memory()

    def set_step_size(self, step_size):
        for layer in self.layers:
            layer.set_step_size(step_size)


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
        step_size,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        fusion="sum",
        fusion_scale=0.1,
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
            step_size=step_size,
            dropout=dropout,
            fusion=fusion,
            fusion_scale=fusion_scale,
        )
        self.pool = pool

    def forward(self, x):
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        return self.transformer(x)

    def reset_memory(self):
        self.transformer.reset_memory()

    def set_step_size(self, step_size):
        self.transformer.set_step_size(step_size)


class PerTokenCLSMemoryBlock(nn.Module):
    """
    Per-token CLS memory (Mamba-style) with parallel training and streaming step.

    Parallel mode:
      - X_seq: [B, T, P, D]  (frames × patches)
      - Build u_t from X_seq[:, t] (no future leakage)
      - SSM.scan over {u_t} -> c_t, shift to tilde{c}_t = c_{t-1}
      - Prepend tilde{c}_t as CLS to each frame, do masked attention once
      - Drop CLS outputs, FFN

    Streaming mode (step):
      - Input single frame X_t: [B, P, D] and H_prev: [B, 1, S]
      - Read with c_prev (from H_prev), attention, then write H_next with u_t
    """

    def __init__(
        self,
        dim,
        heads,
        dim_head,
        mlp_dim,
        n_patches,
        step_size,
        dropout=0.0,
    ):
        super().__init__()
        self.D = dim
        self.P = n_patches
        self.heads = heads
        self.dim_head = dim_head
        self.inner = heads * dim_head
        self.scale = dim_head**-0.5
        self.step_size = step_size
        self.ssm_state = dim // 2

        # norms
        self.attn_norm = nn.LayerNorm(dim)
        self.val_norm = nn.LayerNorm(dim)

        # Q/K/V projections for attention over [CLS, tokens]
        self.to_q = nn.Linear(dim, self.inner, bias=False)
        self.to_k = nn.Linear(dim, self.inner, bias=False)
        self.to_v = nn.Linear(dim, self.inner, bias=False)

        # output projection
        self.to_out = nn.Sequential(
            nn.Linear(self.inner, dim), nn.Dropout(dropout)
        )
        self.drop = nn.Dropout(dropout)

        # Mamba-like SSM and readout h_t -> c_t
        self.ssm = MambaSSMCell(d_model=dim, n_state=self.ssm_state)
        self.mem_readout = nn.Linear(self.ssm_state, dim, bias=False)

        self.write_norm = nn.LayerNorm(dim)

        # FFN
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)

        self.H_buffer = None

    # ---------- helpers ----------
    def _heads(self, t):
        return rearrange(t, "b n (h d) -> b h n d", h=self.heads)

    def _unheads(self, t):
        return rearrange(t, "b h n d -> b n (h d)")

    def _mask_for(self, T, P_plus, device, dtype):
        # build a block-causal mask for T frames with (P_plus) tokens per frame
        # this matches your style: full within-frame, causal across frames
        # generate_mask_matrix expects (npatch, nwindow), patch-major flattening.
        bias = generate_mask_matrix(P_plus, T).to(
            device=device, dtype=dtype
        )  # [1,1,TP,TP]
        return bias

    # ---------- PARALLEL (training/inference in batch) ----------
    def forward(self, X_seq):
        """
        X_seq: [B, (T * P), D]  (frames × patches)
        H0   : [B, 1, S] initial memory state (zeros if None)
        returns:
          Y_seq: [B, T, P, D] (tokens updated)
          H_T  : [B, 1, S] final memory state
        """
        B, T_P, D = X_seq.shape
        T, P = T_P // self.P, self.P

        # 1) ----- build per-frame summaries u_t -----
        # use pre-attn tokens for summaries (stable, single pass)
        X_norm = self.write_norm(X_seq)  # [B, T*P, D]
        X_reshaped = rearrange(
            X_norm, "b (t p) d -> b t p d", t=T, p=P
        )  # [B, T, P, D]
        U = X_reshaped.mean(
            dim=2
        )  # [B, T, D] - average over patches per timestep

        # 2) ----- SSM scan over u_t -> h_t -> c_t -----
        U = U.unsqueeze(2)  # [B, T, 1, D] to fit MambaSSMCell.scan API

        if self.H_buffer is None:
            self.H_buffer = self.ssm.init_state(
                B, 1, device=X_seq.device, dtype=X_seq.dtype
            )  # [B,1,S]
        H_seq, _, _ = self.ssm(
            U, self.H_buffer, mode="scan"
        )  # H_seq: [B,T,1,S]
        C_seq = self.mem_readout(
            H_seq.squeeze(2)
        )  # [B, T, D] - one CLS per timestep
        self.H_buffer = H_seq[
            :, min(self.step_size - 1, T - 1)
        ].detach()  # update buffer to carry over to next window

        # 4) ----- prepend per-frame CLS and do a single masked attention -----
        # Interleave CLS tokens with patches: [CLS1, patch1_1, patch1_2, ..., CLS2, patch2_1, ...]
        X_plus = []
        for t in range(T):
            X_plus.append(
                C_seq[:, t : t + 1, :]
            )  # [B, 1, D] - CLS for timestep t
            X_plus.append(
                X_seq[:, t * P : (t + 1) * P, :]
            )  # [B, P, D] - patches for timestep t
        X_plus = torch.cat(X_plus, dim=1)  # [B, T*(1+P), D]
        TPp = T * (1 + P)  # Total tokens: T CLS + T*P patches

        bias = self._mask_for(
            T=T, P_plus=1 + P, device=X_seq.device, dtype=X_seq.dtype
        )  # [1,1,TPp,TPp]

        Xf = self.attn_norm(X_plus)  # [B, TPp, D] - already in correct shape
        Q = self._heads(self.to_q(Xf))  # [B,H,TPp,dh]
        K = self._heads(self.to_k(Xf))  # [B,H,TPp,dh]
        V = self._heads(self.to_v(self.val_norm(Xf)))  # [B,H,TPp,dh]

        dots = torch.matmul(Q, K.transpose(-1, -2)) * (self.dim_head**-0.5)
        dots = dots.masked_fill(bias == 0, float("-inf"))
        A = torch.softmax(dots, dim=-1)
        A = self.drop(A)

        attn_out = torch.matmul(A, V)  # [B,H,TPp,dh]
        attn_out = self._unheads(attn_out)  # [B,TPp,Inner]
        attn_out = self.to_out(attn_out)  # [B,TPp,D]

        # Extract only the patch tokens from attention output (skip CLS tokens)
        attn_out_patches = []
        for t in range(T):
            # Skip CLS token at position t*(1+P), take patches at positions t*(1+P)+1 to (t+1)*(1+P)
            attn_out_patches.append(
                attn_out[:, t * (1 + P) + 1 : (t + 1) * (1 + P), :]
            )
        attn_out = torch.cat(attn_out_patches, dim=1)  # [B, T*P, D]

        X_seq = X_seq + attn_out

        return X_seq + self.ff(X_seq)

    @torch.no_grad()
    def init_state(self, B, device=None, dtype=None):
        return self.ssm.init_state(B, 1, device=device, dtype=dtype)  # [B,1,S]

    def set_step_size(self, step_size):
        self.step_size = step_size

    def reset_memory(self):
        self.H_buffer = None


class CLSMemoryTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        n_patches,
        n_frames,
        step_size,
        dropout=0.0,
        bias=None,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                PerTokenCLSMemoryBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=mlp_dim,
                    n_patches=n_patches,
                    step_size=step_size,
                    dropout=dropout,
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x), None

    def set_step_size(self, step_size):
        for layer in self.layers:
            layer.set_step_size(step_size)

    def reset_memory(self):
        for layer in self.layers:
            layer.reset_memory()


class CLSMemoryPredictor(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        depth,
        heads,
        mlp_dim,
        step_size,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.dim = dim
        self.pool = pool

        self.transformer = CLSMemoryTransformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            n_patches=num_patches,
            n_frames=num_frames,
            step_size=step_size,
            dropout=dropout,
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames * num_patches, dim)
        )
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x):
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        return self.transformer(x)

    def reset_memory(self):
        self.transformer.reset_memory()

    def set_step_size(self, step_size):
        self.transformer.set_step_size(step_size)


########### Dynamic Lora Attention Injection ###########


class DynamicLoRAAttention(nn.Module):
    """
    Multi-head attention with per-token DynamicLoRALinear injections:
      - QK injection (query & key)
      - VO injection (value & output)
    Each site can choose its own type: "mm" / "xm" / "mx".
    """

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        r=8,
        alpha=2.0,
        use_qk=True,
        use_vo=False,
        gen_type="A",
        dropout=0.0,
        bias=None,
        inject_pooled=False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.use_qk = use_qk
        self.use_vo = use_vo

        self.norm = nn.LayerNorm(dim)
        self.mem_norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        if use_qk:
            self.q_proj = DynamicLoRALinear(
                dim,
                inner_dim,
                r,
                alpha=alpha,
                gen_type=gen_type,
                use_bias=False,
                inject_pooled=inject_pooled,
            )
            self.k_proj = DynamicLoRALinear(
                dim,
                inner_dim,
                r,
                alpha=alpha,
                gen_type=gen_type,
                use_bias=False,
                inject_pooled=inject_pooled,
            )
        else:
            self.q_proj = nn.Linear(dim, inner_dim, bias=False)
            self.k_proj = nn.Linear(dim, inner_dim, bias=False)

        if use_vo and project_out:
            self.v_proj = DynamicLoRALinear(
                dim,
                inner_dim,
                r,
                alpha=alpha,
                gen_type=gen_type,
                use_bias=False,
                inject_pooled=inject_pooled,
            )
            self.o_proj = DynamicLoRALinear(
                inner_dim,
                dim,
                r,
                alpha=alpha,
                gen_type=gen_type,
                use_bias=False,
                mem_features=dim,
                inject_pooled=inject_pooled,
            )
        else:
            self.v_proj = nn.Linear(dim, inner_dim, bias=False)
            self.o_proj = nn.Linear(inner_dim, dim, bias=False)

        if bias is None:
            bias = generate_mask_matrix(NUM_PATCHES, NUM_FRAMES)
        self.register_buffer("bias", bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B,T,D] -> [B,h,T,dh]
        B, T, D = x.shape
        return x.view(B, T, self.heads, self.dim_head).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B,h,T,dh] -> [B,T,D]
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

    def forward(
        self,
        x: torch.Tensor,
        m_tok: torch.Tensor,
    ) -> torch.Tensor:
        """
        x:     [B,T, dim]
        m_tok: [B,T, dim] 
        """
        B, T, D = x.shape

        x = self.norm(x)

        if m_tok is not None:
            m_tok = self.mem_norm(m_tok)
        
        q = self.q_proj(x, m_tok) if self.use_qk else self.q_proj(x) 
        k = self.k_proj(x, m_tok) if self.use_qk else self.k_proj(x)
        v = self.v_proj(x, m_tok) if self.use_vo else self.v_proj(x)

        q = self._split_heads(q)  # [B,h,T,dh]
        k = self._split_heads(k)
        v = self._split_heads(v)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
    
        dots = dots.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = self._merge_heads(out)
        
        out = self.o_proj(out, m_tok) if self.use_vo else self.o_proj(out)

        return self.dropout(out)


class LoRAAttentionTransformer(StateSpaceTransformer):
    def __init__(
        self,
        dim,
        num_patches,
        depth,
        heads,
        mlp_dim,
        state_dim,
        step_size,
        n_mem_blocks,
        dropout=0.0,
        dim_head=64,
        dt_rank: int = 16,
        lora_rank: int = 16,
        lora_alpha: float = 2.0,
        use_qk: bool = True,
        use_vo: bool = False,
        gen_type: str = "A",
        use_cls_token: bool = False,
        shift_memory: bool = False,
        mem_layer_type: str = "all",
        **kwargs,
    ):
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.gen_type = gen_type
        self.use_qk = use_qk
        self.use_vo = use_vo
        
        super().__init__(
            dim=dim,
            num_patches=num_patches,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            state_dim=state_dim,
            step_size=step_size,
            n_mem_blocks=n_mem_blocks,
            dropout=dropout,
            dim_head=dim_head,
            dt_rank=dt_rank,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            gen_type=gen_type,
            use_qk=use_qk,
            use_vo=use_vo,
            use_cls_token=use_cls_token,
            shift_memory=shift_memory,
            mem_layer_type=mem_layer_type,
            **kwargs,
        )

    def _build_transformer(self, depth, dim, heads, dim_head, dropout, mlp_dim, **kwargs):
        self.layers = nn.ModuleList([])
        for i in range(depth):
            block = nn.ModuleList([])
            if i in self.mem_layer_idx:
                block.append(
                    DynamicLoRAAttention(
                        dim=dim, 
                        heads=heads, 
                        dim_head=dim_head,
                        r=self.lora_rank, 
                        alpha=self.lora_alpha,
                        gen_type=self.gen_type,
                        use_qk=self.use_qk,
                        use_vo=self.use_vo,
                        dropout=dropout, 
                        # bias=generate_mask_with_memory(NUM_PATCHES, NUM_FRAMES),
                    )
                )
            else:
                block.append(Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)) #, bias=generate_mask_with_memory(NUM_PATCHES, NUM_FRAMES)))
            block.append(FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout))
            self.layers.append(block)

    def forward(self, x):
        M_new = self._mem_blocks_forward(x)
        x = self.ln_in(x)
        for i, (attn, ff) in enumerate(self.layers):
            attn_out = attn(x, M_new) if i in self.mem_layer_idx else attn(x)
            x = x + attn_out            
            x = x + ff(x)
        return self.ln_out(x), self.ln_mem_out(M_new)


########### LoRA FFN with Memory ###########

class DynamicLoRAFFN(nn.Module):
    """
    SwiGLU FFN with three linears, each using DynamicLoRALinear:
    All per-token with memory-driven LoRA ("mm" by default).
    """
    def __init__(self, dim: int, hidden_dim: int, r: int = 16, alpha: float = 2.0,
                 gen_type: str = "A", dropout: float = 0.0):
        super().__init__()
        self.W1 = DynamicLoRALinear(dim, hidden_dim, r, alpha=alpha, gen_type=gen_type)
        self.W2 = DynamicLoRALinear(dim, hidden_dim, r, alpha=alpha, gen_type=gen_type)
        self.W3 = DynamicLoRALinear(hidden_dim, dim, r, alpha=alpha, gen_type=gen_type, mem_features=dim)

        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, m_tok: torch.Tensor) -> torch.Tensor:
        x1 = self.W1(x, m_tok)
        x2 = self.W2(x, m_tok) 
        return self.W3(self.dropout(self.silu(x1) * x2), m_tok)


class LoRAFFNTransformer(StateSpaceTransformer):
    def __init__(
        self,
        dim,
        num_patches,
        depth,
        heads,
        mlp_dim,
        state_dim,
        step_size,
        n_mem_blocks,
        dropout=0.0,
        dim_head=64,
        dt_rank: int = 16,
        lora_rank: int = 16,
        lora_alpha: float = 2.0,
        gen_type: str = "A",
        use_cls_token: bool = False,
        mem_layer_type: str = "all",
        **kwargs,
    ):
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.gen_type = gen_type
        
        super().__init__(
            dim=dim,
            num_patches=num_patches,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            state_dim=state_dim,
            step_size=step_size,
            n_mem_blocks=n_mem_blocks,
            dropout=dropout,
            dim_head=dim_head,
            dt_rank=dt_rank,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            gen_type=gen_type,
            use_cls_token=use_cls_token,
            mem_layer_type=mem_layer_type,
            **kwargs,
        )

    def _build_transformer(self, depth, dim, heads, dim_head, dropout, mlp_dim, **kwargs):
        self.layers = nn.ModuleList([])
        for i in range(depth):
            block = nn.ModuleList([])
            block.append(
                Attention(dim, heads, dim_head, dropout) #, bias=generate_mask_with_memory(NUM_PATCHES, NUM_FRAMES))
            )
            if i in self.mem_layer_idx:
                block.append(
                    DynamicLoRAFFN(
                        dim=dim,
                        hidden_dim=mlp_dim,
                        r=self.lora_rank,
                        alpha=self.lora_alpha,
                        gen_type=self.gen_type,
                        dropout=dropout,
                    )       
                )
            else:
                block.append(FeedForward(dim, mlp_dim, dropout))
            self.layers.append(block)


    def forward(self, x):
        M_new = self._mem_blocks_forward(x)
        x = self.ln_in(x)
        for i, (attn, ff) in enumerate(self.layers):
            x = x + attn(x)
            ff_out = ff(x, M_new) if i in self.mem_layer_idx else ff(x)
            x = x + ff_out
        return self.ln_out(x), self.ln_mem_out(M_new)


class LoRAInjectionViTPredictor(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        state_dim,
        depth,
        heads,
        mlp_dim,
        injection_type,
        step_size,
        n_mem_blocks,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=64,
        dt_rank: int = 16,
        lora_rank: int = 16,
        lora_alpha: float = 2.0,
        gen_type: str = "A",
        shift_memory: bool = False,
        mem_layer_type: str = "all",
    ):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.dim_head = dim_head
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.gen_type = gen_type
        self.shift_memory = shift_memory
        # update params for adding causal attention masks
        global NUM_FRAMES, NUM_PATCHES
        NUM_FRAMES = num_frames
        NUM_PATCHES = num_patches

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames * num_patches, dim)
        )
        self.dropout = nn.Dropout(emb_dropout)

        if injection_type in ["lora_qk", "lora_vo", "lora_qkvo"]:
            use_qk = injection_type == "lora_qk" or injection_type == "lora_qkvo"
            use_vo = injection_type == "lora_vo" or injection_type == "lora_qkvo"
            
            self.transformer = LoRAAttentionTransformer(
                dim=dim,
                num_patches=num_patches,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                state_dim=state_dim,
                step_size=step_size,
                n_mem_blocks=n_mem_blocks,
                dropout=dropout,
                dim_head=dim_head,
                dt_rank=dt_rank,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                gen_type=gen_type,
                use_qk=use_qk,
                use_vo=use_vo,
                shift_memory=shift_memory,
                mem_layer_type=mem_layer_type,
            )
        elif injection_type == "lora_ffn":
            self.transformer = LoRAFFNTransformer(
                dim=dim,
                num_patches=num_patches,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                state_dim=state_dim,
                step_size=step_size,
                n_mem_blocks=n_mem_blocks,
                dropout=dropout,
                dim_head=dim_head,
                dt_rank=dt_rank,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                gen_type=gen_type,
                mem_layer_type=mem_layer_type,
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

    def set_step_size(self, step_size):
        self.transformer.set_step_size(step_size)


########### Dynamic FFN with Memory ###########

class FFNDynamicMemories(nn.Module):
    """
    Dynamic FFN with Memory.
    W1: d_model -> hidden_dim
    W2: hidden_dim -> d_model
    W3: hidden_dim -> d_model
    W1_m: d_model -> r (generated by FFNMemGenerator)
    W2_m: r -> d_model (generated by FFNMemGenerator)
    W3_m: d_model -> r (generated by FFNMemGenerator)
    """
    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 d_m: int,            # memory token dim
                 r: int = 64,         # memory width
                 gen_hidden_mul = 2,
                 dropout: float = 0.0,
                 alpha_mem: float = 1.0):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.r = r
        self.alpha_mem = alpha_mem

        # Base FFN
        self.W1 = nn.Parameter(torch.empty(hidden_dim, dim))
        self.W2 = nn.Parameter(torch.empty(hidden_dim, dim))
        self.W3 = nn.Parameter(torch.empty(dim, hidden_dim))
        self.b1 = nn.Parameter(torch.empty(hidden_dim))
        self.b2 = nn.Parameter(torch.empty(hidden_dim))
        self.b3 = nn.Parameter(torch.empty(dim))
        self.silu = nn.SiLU()
        self.drop = nn.Dropout(dropout)

        # Memory generator F(m): produces per-token matrices
        self.mem_gen = FFNMemGenerator(in_features=d_m, out_features=dim, r=r, hidden_mul=gen_hidden_mul)

        # small regularizer gates
        self.mem_gate = nn.Parameter(torch.tensor(0.0))  # sigmoid gate for entire mem branch

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize base FFN parameters
        nn.init.kaiming_uniform_(self.W1, a=5**0.5)
        nn.init.kaiming_uniform_(self.W2, a=5**0.5)
        nn.init.kaiming_uniform_(self.W3, a=5**0.5)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)
        nn.init.zeros_(self.b3)

        nn.init.constant_(self.mem_gate, 0.0)
        
        self.mem_gen.reset_parameters()

    def forward(self, x: torch.Tensor, m_tok: torch.Tensor) -> torch.Tensor:
        """
        x:     [B,T,dim]
        m_tok: [B,T,d_m]  (per-token memory embeddings)
        """
        
        # ----- Base FFN -----
        x1 = torch.einsum("od,btd->bto", self.W1, x) + self.b1.view(1, 1, -1)  # [B,T,hidden_dim]
        x2 = torch.einsum("od,btd->bto", self.W2, x) + self.b2.view(1, 1, -1)  # [B,T,hidden_dim]
        h_base = self.drop(self.silu(x1) * x2)  # [B,T,hidden_dim]
        y_base = torch.einsum("do,bto->btd", self.W3, h_base) + self.b3.view(1, 1, -1)  # [B,T,dim]

        # ----- Memory FFN (dynamic, per token) -----
        W1_m, W2_m, W3_m = self.mem_gen(m_tok)  # TODO: should cache generated AB matrices

        # Memory FFN forward pass
        u1 = torch.einsum("btod,btd->bto", W1_m, x)  # [B,T,r]
        u2 = torch.einsum("btod,btd->bto", W2_m, x)  # [B,T,r]
        h_mem = self.drop(self.silu(u1) * u2)  # [B,T,r]
        y_mem = torch.einsum("btdo,bto->btd", W3_m, h_mem)  # [B,T,dim]

        # Combine base and memory outputs
        g = torch.sigmoid(self.mem_gate)
        y = y_base + g * self.alpha_mem * y_mem
        return y

class DynamicFFNTransformer(StateSpaceTransformer):
    def __init__(
        self,
        dim,
        num_patches,
        depth,
        heads,
        mlp_dim,
        state_dim,
        step_size,
        n_mem_blocks,
        dropout=0.0,
        dim_head=64,
        dt_rank: int = 16,
        gen_hidden_mul: int = 2,
        alpha_mem: float = 1.0,
        ffn_r: int = 64,
        **kwargs,
    ):
        self.gen_hidden_mul = gen_hidden_mul
        self.alpha_mem = alpha_mem
        self.ffn_r = ffn_r
        
        super().__init__(
            dim=dim,
            num_patches=num_patches,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            state_dim=state_dim,
            step_size=step_size,
            n_mem_blocks=n_mem_blocks,
            dropout=dropout,
            dim_head=dim_head,
            dt_rank=dt_rank,
            gen_hidden_mul=gen_hidden_mul,
            alpha_mem=alpha_mem,
            ffn_r=ffn_r,
            **kwargs,
        )
    
    def _build_transformer(self, depth, dim, heads, dim_head, dropout, mlp_dim, **kwargs):
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(dim, heads, dim_head, dropout), #, bias=generate_mask_with_memory(NUM_PATCHES, NUM_FRAMES)),
                    FFNDynamicMemories(
                        dim=dim,
                        hidden_dim=mlp_dim,
                        d_m=dim,
                        r=self.ffn_r,
                        gen_hidden_mul=self.gen_hidden_mul,
                        alpha_mem=self.alpha_mem,
                        dropout=dropout,

                    )
                ])
            )

    def forward(self, x):
        B, T, D = x.shape
        M_new = self._mem_blocks_forward(x)
        x = self.ln_in(x)

        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x, M_new) # could cache generated AB matrices

        return self.ln_out(x), self.ln_mem_out(M_new)


class DynamicFFNVitPredictor(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        state_dim,
        depth,
        heads,
        mlp_dim,
        step_size,
        n_mem_blocks,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=64,
        dt_rank: int = 16,
        ffn_rank: int = 64,
        gen_hidden_mul: int = 2,
        alpha_mem: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.dim_head = dim_head
        self.ffn_rank = ffn_rank
        self.gen_hidden_mul = gen_hidden_mul
        self.alpha_mem = alpha_mem
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

        self.transformer = DynamicFFNTransformer(
            dim=dim,
            num_patches=num_patches,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            state_dim=state_dim,
            step_size=step_size,
            n_mem_blocks=n_mem_blocks,
            dropout=dropout,
            dim_head=dim_head,
            dt_rank=dt_rank,
            gen_hidden_mul=gen_hidden_mul,
            alpha_mem=alpha_mem,
            ffn_r=ffn_rank,
        )
    
    def forward(self, x):
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        return self.transformer(x)
    
    def reset_memory(self):
        self.transformer.reset_memory()
    
    def set_step_size(self, step_size):
        self.transformer.set_step_size(step_size)


########### TransformerXL Implementation ###########

class TransformerXLAttention(nn.Module):
    """
    TransformerXL-style attention with segment-level recurrence.
    Uses relative positional encodings and maintains a memory buffer.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, mem_len=0, bias=None):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.mem_len = mem_len

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # Q, K, V projections
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        # Relative positional encoding projections
        self.r_w_bias = nn.Parameter(torch.zeros(heads, dim_head))
        # self.r_r_bias = nn.Parameter(torch.zeros(heads, dim_head))

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

        if bias is None:
            bias = generate_mask_matrix(NUM_PATCHES, NUM_FRAMES)
        self.register_buffer("bias", bias)

        

    def _rel_shift(self, x):
        """Relative positional shift for TransformerXL attention"""
        zero_pad = torch.zeros((*x.size()[:-2], x.size(-2), 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(*x.size()[:-2], x.size(-1) + 1, x.size(-2))
        x = x_padded[:, :, 1:].view_as(x)
        return x

    def forward(self, x, mems=None):
        B, T, C = x.size()
        
        x = self.norm(x)
        
        # Project to Q, K, V
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        
        # Reshape for multi-head attention
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)
        
        # Handle memory if provided
        if mems is not None and len(mems) > 0:
            # Concatenate memory with current keys and values
            # print(f"mems_shape: {mems.shape}", flush=True)
            mems = self.norm(mems)
            mem_k = self.to_k(mems)
            mem_v = self.to_v(mems)

            # print(f"mem_k_shape after projection: {mem_k.shape}", flush=True)

            mem_k = rearrange(mem_k, "b n (h d) -> b h n d", h=self.heads)
            mem_v = rearrange(mem_v, "b n (h d) -> b h n d", h=self.heads)

            # print(f"mem_k_shape after reshape: {mem_k.shape}", flush=True)
            
            k = torch.cat([mem_k, k], dim=2)  # [B, H, M+T, D]
            v = torch.cat([mem_v, v], dim=2)  # [B, H, M+T, D]  
        
        # Compute attention scores
        AC = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, H, T, M+T]
        
        # Add relative positional bias
        BD = torch.matmul(q + self.r_w_bias.unsqueeze(0).unsqueeze(2), 
                         k.transpose(-1, -2)) * self.scale
        BD = self._rel_shift(BD)
        
        # Combine content and positional attention
        attn = AC + BD

        
        # Apply causal mask
        if mems is not None:
            # For memory, we need to adjust the mask
            mem_len = mems.size(1) if mems is not None else 0
            mask = generate_frame_mask_with_memory(NUM_PATCHES, NUM_FRAMES, mem_len, device=x.device, dtype=torch.bool)
            mask = mask[:,:,mem_len:,:] == 1
        else:
            mask = self.bias[:, :, :T, :T] == 1
            mem_len = 0
        
        attn = attn.masked_fill(~mask, float("-inf"))
        
        attn = self.attend(attn)
        attn = self.dropout(attn)

        
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        
        return self.to_out(out)


class TransformerXLBlock(nn.Module):
    """
    Single TransformerXL block with attention and feedforward layers.
    """
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, mem_len=0):
        super().__init__()
        self.attention = TransformerXLAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            mem_len=mem_len
        )
        self.feed_forward = FeedForward(dim, mlp_dim, dropout=dropout)
        
    def forward(self, x, mems=None):
        # Self-attention with memory
        attn_out = self.attention(x, mems)
        x = x + attn_out
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = x + ff_out
        
        return x


class TransformerXL(nn.Module):
    """
    TransformerXL model with segment-level recurrence.
    Maintains a memory buffer across segments for long-range dependencies.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0, mem_len=0, step_size=1):
        super().__init__()
        self.mem_len = mem_len
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.step_size = step_size
        
        for _ in range(depth):
            self.layers.append(
                TransformerXLBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    mem_len=mem_len
                )
            )
        
        # Memory buffer to store previous segment representations
        self.memory = None
        
    def _update_memory(self, new_mem):
        if new_mem is not None:
            self.memory = new_mem[:, min(self.step_size - 1, new_mem.size(1) - 1):]
        else:
            self.memory = new_mem  
     
    
    def forward(self, x):
        """
        Forward pass with optional external memory.
        
        Args:
            x: Input tensor [B, T, D]
        """
        curr_memory = self.memory
        for i, layer in enumerate(self.layers):
            x = layer(x, curr_memory)
            
        new_mem = x.detach().clone()
        self._update_memory(new_mem) 
        
        return self.norm(x)
    
    def reset_memory(self):
        """Reset the internal memory buffer"""
        self.memory = None
    
    def set_step_size(self, step_size):
        self.step_size = step_size


class TransformerXLViTPredictor(nn.Module):
    """
    ViT Predictor using TransformerXL architecture.
    Combines the ViT structure with TransformerXL's segment-level recurrence.
    """
    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        depth,
        heads,
        step_size,
        mlp_dim,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        mem_len=0,
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
        self.mem_len = mem_len * num_patches

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames * num_patches, dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = TransformerXL(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            mem_len=mem_len,
            step_size=step_size,
        )

    def forward(self, x):
        """
        Forward pass with optional memory.
        
        Args:
            x: Input embeddings [B, T, D] where T = num_frames * num_patches
            H: Optional external memory (for compatibility with other models)
        """
        b, n, _ = x.shape
        
        # Add positional embeddings
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        
        # Pass through TransformerXL
        x = self.transformer(x)
        
        return x, None

    def reset_memory(self):
        """Reset the internal memory buffer"""
        self.transformer.reset_memory()
    
    def set_step_size(self, step_size):
        self.transformer.set_step_size(step_size * self.num_patches)


class BlockRecurrentTransformerLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, emb_dropout=0.0, attn_dropout=0.0, bias=None):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head**-0.5
        self.heads = heads

        # input processing
        self.to_qkv_input = nn.Linear(dim, 3 * inner_dim, bias=False)
        self.norm_input = nn.LayerNorm(dim)
        self.q_input_memory = nn.Linear(dim, inner_dim, bias=False) # for cross attention between input and memory

        # memory processing
        self.to_kv_memory = nn.Linear(dim, 2 * inner_dim, bias=False)
        self.norm_memory = nn.LayerNorm(dim)

        # shared
        self.attend = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.to_out = nn.Sequential(nn.Linear(2 * inner_dim, dim), nn.Dropout(emb_dropout))
        self.ff = FeedForward(dim, mlp_dim, dropout=emb_dropout)
        self.out_norm = nn.LayerNorm(dim)

        if bias is None:
            self_attention_bias = generate_mask_matrix(NUM_PATCHES, NUM_FRAMES)
        self.register_buffer("self_attention_bias", self_attention_bias)
        if bias is None:
            cross_attention_bias = generate_diagonal_frame_mask(NUM_PATCHES, NUM_FRAMES) # TODO: proper bias for cross attention
        self.register_buffer("cross_attention_bias", cross_attention_bias)

    def forward(self, x, m, s_pe=None):
        B, T, C = x.size()
        B_m, M, C_m = m.size()

        x = self.norm_input(x)

        qkv_input = self.to_qkv_input(x).chunk(3, dim=-1)
        q_in, k_in, v_in = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv_input)
        dots_in = torch.matmul(q_in, k_in.transpose(-1, -2)) * self.scale
        dots_in = dots_in.masked_fill(self.self_attention_bias[:, :, :T, :T] == 0, float("-inf"))
        attn_in = self.attn_dropout(self.attend(dots_in))

        z_in = torch.matmul(attn_in, v_in)
        z_in = rearrange(z_in, "b h n d -> b n (h d)")

        # memory processing
        if s_pe is not None:
            m = m + s_pe[:, :M, :] # context ids (memory positional encodings)

        m = self.norm_memory(m)
        q_mem = self.q_input_memory(x) 
        k_mem, v_mem = self.to_kv_memory(m).chunk(2, dim=-1)
        q_mem, k_mem, v_mem = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), [q_mem, k_mem, v_mem])
        dots_in_mem = torch.matmul(q_mem, k_mem.transpose(-1, -2)) * self.scale
        dots_in_mem = dots_in_mem.masked_fill(self.cross_attention_bias[:, :, :T, :M] == 0, float("-inf"))
        attn_in_mem = self.attn_dropout(self.attend(dots_in_mem))

        z_in_mem = torch.matmul(attn_in_mem, v_mem)
        z_in_mem = rearrange(z_in_mem, "b h n d -> b n (h d)")

        # concat input and memory 
        z = torch.cat([z_in, z_in_mem], dim=-1)
        x = x + self.to_out(z)

        # ff
        x = x + self.ff(x)

        return self.out_norm(x)


class BlockRecurrentTransformer(StateSpaceTransformer):
    def __init__(
        self,
        dim,
        num_patches,
        depth,
        heads,
        mlp_dim,
        state_dim,
        step_size,
        n_mem_blocks,
        dropout=0.0,
        dim_head=64,
        dt_rank: int = 16,
        use_gate: bool = False,
        mem_layer_type: str = "all", # 'all', 'first', 'middle', 'last', 'alternate'
        **kwargs,
    ):
        self.mem_layer_type = mem_layer_type
        assert mem_layer_type in {'all', 'first', 'middle', 'last', 'alternate'}, "mem_layer_type must be one of 'all', 'first', 'middle', 'last', 'alternate'"

        super().__init__(
            dim=dim,
            num_patches=num_patches,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            state_dim=state_dim,
            step_size=step_size,
            n_mem_blocks=n_mem_blocks,
            dropout=dropout,
            dim_head=dim_head,
            dt_rank=dt_rank,
            use_gate=use_gate,
            mem_layer_type=mem_layer_type,
            **kwargs,
        )
        self.s_pe = nn.Parameter(torch.randn(1, NUM_FRAMES * NUM_PATCHES, dim))


    def _build_transformer(self, depth, dim, heads, dim_head, dropout, mlp_dim, **kwargs):
        self.layers = nn.ModuleList([])
        if self.mem_layer_type == 'all':
            mem_layer_idx = list(range(depth))
        elif self.mem_layer_type == 'first':
            mem_layer_idx = [0]
        elif self.mem_layer_type == 'middle':
            mem_layer_idx = list(range(1, depth - 1))
        elif self.mem_layer_type == 'last':
            mem_layer_idx = [depth - 1]
        elif self.mem_layer_type == 'alternate':
            mem_layer_idx = list(range(0, depth, 2))
        
        self.mem_layer_idx = mem_layer_idx
        
        for i in range(depth):
            if i in mem_layer_idx:
                self.layers.append(
                    BlockRecurrentTransformerLayer(
                        dim=dim,
                        heads=heads,
                        dim_head=dim_head,
                        mlp_dim=mlp_dim,
                        emb_dropout=dropout,
                        attn_dropout=dropout,
                    )
                )
            else:
                self.layers.append(
                    nn.ModuleList([
                        Attention(
                            dim=dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            bias=generate_mask_matrix(NUM_PATCHES, NUM_FRAMES),
                        ),
                        FeedForward(
                            dim=dim,
                            hidden_dim=mlp_dim,
                            dropout=dropout,
                        )
                    ])
                )
    
    def _mem_blocks_forward(self, x):
        B, T, D = x.shape
        x = x.clone()
        x = rearrange(x, "b (t p) d -> b t p d", t=T // self.num_patches)
        for mem_block in self.mem_blocks:
            x, H_T = mem_block(x)

        return rearrange(x, "b t p d -> b (t p) d")

    def forward(self, x):
        B, T, D = x.size()
        M_new = self._mem_blocks_forward(x)
        x = self.ln_in(x)
        for i, layer in enumerate(self.layers):
            if i in self.mem_layer_idx:
                x = layer(x, M_new, self.s_pe) # residual connection inside layer
            else:
                x = x + layer[0](x)
                x = x + layer[1](x)
        return self.ln_out(x), self.ln_mem_out(M_new)


class BlockRecurrentViTPredictor(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        state_dim,
        depth,
        heads,
        mlp_dim,
        step_size,
        n_mem_blocks,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=64,
        dt_rank: int = 16,
        mem_layer_type: str = "all", # 'all', 'first', 'middle', 'last', 'alternate'
    ):
        super().__init__()
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.dim = dim
        self.state_dim = state_dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.step_size = step_size
        self.n_mem_blocks = n_mem_blocks
        
        global NUM_FRAMES, NUM_PATCHES
        NUM_FRAMES = num_frames
        NUM_PATCHES = num_patches

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = BlockRecurrentTransformer(
            dim=dim,
            num_patches=num_patches,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            state_dim=state_dim,
            step_size=step_size,
            n_mem_blocks=n_mem_blocks,
            dropout=dropout,
            dim_head=dim_head,
            dt_rank=dt_rank,
            mem_layer_type=mem_layer_type,
        )

    def forward(self, x):
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        return self.transformer(x)
    
    def reset_memory(self):
        self.transformer.reset_memory()
    
    def set_step_size(self, step_size):
        self.transformer.set_step_size(step_size)


class ViTConditionalPredictor(ViTPredictor):
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
        time_embed_dim=128,
        time_embed_n_freq=32,
        time_embed_sigma=1.0,
        use_delta_tau=False,
    ):
        dim = dim + time_embed_dim + (time_embed_dim if use_delta_tau else 0)
        self.dim = dim
        super().__init__(
            num_patches=num_patches,
            num_frames=num_frames,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool=pool,
        )
        self.norm_out = nn.LayerNorm(dim - time_embed_dim - (time_embed_dim if use_delta_tau else 0))
        self.time_embed_dim = time_embed_dim + (time_embed_dim if use_delta_tau else 0)
        self.use_delta_tau = use_delta_tau
        self.num_patches = num_patches
        self.time_embed = TimestepRFF(dim=time_embed_dim, m=time_embed_n_freq, sigma=time_embed_sigma)
        if use_delta_tau:
            self.delta_tau_embed = TimestepRFF(dim=time_embed_dim, m=time_embed_n_freq, sigma=time_embed_sigma)
        else:
            self.delta_tau_embed = None

        self.transformer = nn.ModuleList([])
        for _ in range(depth):
            self.transformer.append(nn.ModuleList([
                Attention(dim, heads, dim_head, dropout),
                FeedForward(dim, mlp_dim, dropout),
            ]))

    def forward(self, x, tau=None, delta_tau=None):
        if self.use_delta_tau:
            assert delta_tau is not None, "delta_tau must be provided if use_delta_tau is True"
            
        b, n, _ = x.shape
        x = rearrange(x, "b (t p) d -> b t p d", p=self.num_patches)

        if tau is not None:
            tau = self.time_embed(tau)
            tau_tiled = repeat(tau[:, None, None, :], "b 1 1 d -> b t p d", t=x.shape[1], p=self.num_patches)
        else:
            # tau_tiled = torch.zeros(b, x.shape[1], self.num_patches, self.time_embed_dim, device=x.device)
            tau_tiled = torch.tensor([], device=x.device)

        if delta_tau is not None:
            delta_tau = self.delta_tau_embed(delta_tau)
            delta_tau_tiled = repeat(delta_tau[:, None, None, :], "b 1 1 d -> b t p d", t=x.shape[1], p=self.num_patches)
        else:
            # delta_tau_tiled = torch.zeros(b, x.shape[1], self.num_patches, self.time_embed_dim, device=x.device)
            delta_tau_tiled = torch.tensor([], device=x.device)

        x = torch.cat([
            x, tau_tiled, delta_tau_tiled
        ], dim=3)
        x = rearrange(x, "b t p d -> b (t p) d")

        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)

        for attn, ff in self.transformer:
            x = x + attn(x)
            x = x + ff(x)

        out = x[:, :, :-self.time_embed_dim]
        return self.norm_out(out), None


class TimestepRFF(nn.Module):
    """
    Random Fourier Features for scalar timesteps.
    t: shape [B] or [B, 1]
    Returns: shape [B, 2*m] (cos and sin pairs)
    """

    def __init__(
        self,
        dim,
        m=64,
        sigma=1.0,
        learnable=True,
        include_t=False,
        t_scale=1.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.m = m
        self.include_t = include_t
        self.t_scale = float(t_scale)

        # Sample frequencies and phases
        w = torch.randn(m, **factory_kwargs) * float(
            sigma
        )  # ω ~ N(0, sigma^2)
        b = torch.rand(m, **factory_kwargs) * (
            2 * math.pi
        )  # b ~ Uniform[0, 2π)

        if learnable:
            self.w = nn.Parameter(w)  # make them trainable if you want
            self.b = nn.Parameter(b)
        else:
            self.register_buffer("w", w, persistent=False)
            self.register_buffer("b", b, persistent=False)

        in_dim = 2 * m + (1 if include_t else 0)        
        self.proj = nn.Sequential(nn.Linear(in_dim, dim), nn.SiLU())

        # Normalization factor for kernel approximation
        self.norm = math.sqrt(2.0 / m)

    def forward(self, t):
        # t: [B] or [B,1] -> [B,1]
        if t.dim() == 1:
            t = t[:, None]
        tt = t * self.t_scale  # rescale if your times are large/small
        # [B, m]
        phases = tt @ self.w[None, :]

        # Shift by random phase and make cos/sin
        phases = phases + self.b
        feats = (
            torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1)
            * self.norm
        )

        if self.include_t:
            feats = torch.cat([t, feats], dim=-1)

        return self.proj(feats)


class TimeEmbed(nn.Module):
    def __init__(self, emb_dim=128, n_freq=32, sigma=1.0):
        super().__init__()
        B = torch.randn(1, n_freq) * sigma
        self.register_buffer("B", B)
        self.norm = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_freq, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, tau):  
        x = tau @ self.B 
        fourier = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return self.norm(self.mlp(fourier))

class AdaLN(nn.Module):
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.to_scale_shift = nn.Linear(cond_dim, 2*hidden_dim)
        nn.init.zeros_(self.to_scale_shift.weight); nn.init.zeros_(self.to_scale_shift.bias)  # Zero-init (AdaLN-Zero)

    def forward(self, h, cond):
        h = self.ln(h)
        if cond is None:
            return h

        g, b = self.to_scale_shift(cond).chunk(2, dim=-1)   
        return h * (1 + g.unsqueeze(1)) + b.unsqueeze(1)
    
    def reset_parameters(self):
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)
        self.ln.reset_parameters()


class SecondOrderViTPredictor(ViTPredictor):
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
        dt=1.0,
        damping=-0.1,
        integration_method="rk4",
        inner_dim=256,
        action_dim=12,
        integration_func="odeint",
        force_orthogonal: bool = False,
        prior_scale: float = 1.0,
        integration_steps: int = 1,
        grad_ckpt: bool = False,
        random_steps: bool = False,
    ):
        super().__init__(
            num_patches=num_patches,
            num_frames=num_frames,
            dim=inner_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool=pool,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )
        self.grad_ckpt = grad_ckpt
        self.out_dim = dim
        self.inner_dim = dim
        self.action_dim = action_dim
        self.dt = dt
        self.integration_func = getattr(torchdiffeq, integration_func, None)
        assert self.integration_func is not None, f"Invalid integration function: {integration_func}. Options: odeint, odeint_adjoint"
        self.integration_steps = integration_steps
        self.force_orthogonal = force_orthogonal
        self.integration_method = integration_method
        self.random_steps = random_steps

        # projectors
        self.in_proj = nn.Linear(dim, inner_dim)
        self.out_proj = nn.Linear(inner_dim*2, dim*2)
        self.norm_x = nn.LayerNorm(inner_dim)
        self.norm_v = nn.LayerNorm(inner_dim)

        # velocity
        self.vel_head = nn.Sequential(nn.Linear(inner_dim * 2, inner_dim), nn.SiLU(), nn.Dropout(0.1), nn.Linear(inner_dim, inner_dim))

        # acceleration
        self.acc_fusion = nn.Sequential(nn.Linear(inner_dim * 2, inner_dim), nn.SiLU(), nn.Dropout(0.1), nn.Linear(inner_dim, inner_dim))
        self.damping = nn.Parameter(torch.tensor(damping))
       
    
    def extract_actions(self, x):
        x = x.clone()
        x = rearrange(x, "b (t p) d -> b t p d", p=NUM_PATCHES)
        actions = x[..., -self.action_dim:]
        return rearrange(actions, "b t p d -> b (t p) d")

    def inner_forward(self, x):
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x)
        return x
    
    def forward(self, x):
        x = self.in_proj(x) 

        # initial velocity
        v_0 = x - torch.cat([torch.zeros_like(x[:,:1], device=x.device), x[:, :-1]], dim=1)

        # integrate
        state_0 = torch.cat([x, v_0], dim=-1)
        t_span = torch.tensor([0.0, self.dt], device=x.device)

        if self.training and self.random_steps:
            step_options = [1, 2, 4]
            integration_steps = step_options[torch.randint(0, len(step_options), (1,))]
        else:
            integration_steps = self.integration_steps

        
        def dynamics(t, state):
            z, dxdt = state.chunk(2, dim=-1)
            z = self.norm_x(z)
            dxdt = self.norm_v(dxdt)

            # velocity correction
            dxdt = dxdt + self.vel_head(torch.cat([z, dxdt], dim=-1))

            # acceleration
            acc_in = self.acc_fusion(torch.cat([z, dxdt], dim=-1))
            acc = checkpoint.checkpoint(self.inner_forward, acc_in, use_reentrant=False)

            acc = acc + self.damping * dxdt

            return torch.cat([dxdt, acc], dim=-1)

        state_next = odeint(dynamics, state_0, t_span, method=self.integration_method, options={"step_size": self.dt / integration_steps})[-1]

        x_next, v_next  = self.out_proj(state_next).chunk(2, dim=-1)
        x_next = self.norm_x(x_next)
        v_next = self.norm_v(v_next)

        return x_next.contiguous(), v_next.contiguous()

    def set_dt(self, new_dt):
        self.dt = new_dt
    
    def get_dt(self):
        return self.dt