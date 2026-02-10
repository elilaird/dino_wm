# Predictor Architecture Documentation

This document provides detailed technical documentation for the predictor models in DINO-WM, including memory-augmented architectures and their underlying mechanisms.

## Overview

The predictor is the core component of DINO-WM that learns to forecast future visual embeddings given a history of observations, actions, and proprioceptive states. All predictors share a common interface but differ in their internal mechanisms for handling temporal dependencies and memory.

**Location**: `models/vit.py`, `models/memory.py`

---

## Base Predictor Models

### 1. ViTPredictor (Standard)

The standard Vision Transformer predictor with frame-level causal attention masking.

**Architecture**:
```
Input: (B, num_frames × num_patches, dim)
  ↓
Positional Embedding (learned, per-patch)
  ↓
Dropout (emb_dropout)
  ↓
Transformer Blocks (depth layers)
  ├─ Self-Attention (with causal mask)
  └─ FeedForward
  ↓
Output: (B, num_frames × num_patches, dim)
```

**Key Features**:
- **Frame-level causality**: Frame `i` can attend to all patches in frames `0...i` but not to future frames
- **Positional embeddings**: Learned embeddings for each patch position across all frames
- **Causal mask**: Generated via `generate_mask_matrix(npatch, nwindow)` — creates block-diagonal structure where each frame attends to current and previous frames

**Parameters**:
- `num_patches`: Number of patches per frame (e.g., 16×16 = 256 for 224×224 images with patch_size=14)
- `num_frames`: Number of frames in the input sequence (history + prediction)
- `dim`: Embedding dimension (384 for DINOv2-small)
- `depth`: Number of transformer layers (default: 6)
- `heads`: Number of attention heads (default: 6)
- `mlp_dim`: Hidden dimension for feedforward networks (default: 1536)
- `dropout`, `emb_dropout`: Dropout rates

**Usage Context**:
- Standard predictor for experiments without memory
- Baseline for comparison with memory-augmented variants

---

### 2. ViTPredictorWithPersistentTokens

Extends `ViTPredictor` with learnable persistent tokens that attend to all frames and provide a global memory mechanism.

**Architecture**:
```
Input: (B, num_frames × num_patches, dim)
  ↓
Positional Embedding (for visual patches only)
  ↓
Prepend Persistent Tokens: (B, n_persist, dim)
  ↓
  [P₁, P₂, ..., Pₙ | visual patches]
  ↓
Transformer (persistent tokens attend to everything)
  ↓
Drop Persistent Tokens
  ↓
Output: (B, num_frames × num_patches, dim)
```

**Key Differences from Standard ViT**:
- **Persistent tokens**: `n_persist` learnable tokens prepended to the sequence
- **Global attention**: Persistent tokens can attend to all frames (no causal masking for them)
- **Information aggregation**: Persistent tokens serve as a bottleneck for cross-frame communication

**Parameters**:
- All `ViTPredictor` parameters plus:
- `n_persist`: Number of persistent tokens (default: 0, disabled)

**Attention Pattern**:
```
        P₁  P₂  F₀  F₁  F₂
    P₁  ✓   ✓   ✓   ✓   ✓
    P₂  ✓   ✓   ✓   ✓   ✓
    F₀  ✓   ✓   ✓   ✗   ✗
    F₁  ✓   ✓   ✓   ✓   ✗
    F₂  ✓   ✓   ✓   ✓   ✓
```

---

### 3. AdditiveControlViTPredictor

Separates action processing from visual processing via additive injection at each transformer layer.

**Architecture**:
```
Input: (B, num_frames × num_patches + 1, dim)
       └─ last token is action embedding
  ↓
Positional Embedding
  ↓
Split: visual tokens, action token
  ↓
Transformer Layers (depth):
  ├─ Self-Attention (visual tokens only)
  ├─ FeedForward
  └─ Additive Action Injection:
       visual += α_i × Linear(action)
  ↓
Concatenate action token back
  ↓
Output: (B, num_frames × num_patches + 1, dim)
```

**Key Features**:
- **Gradient isolation**: Actions processed separately to avoid entanglement with visual gradients
- **Per-layer injection**: Each transformer layer has its own action projection and learnable scaling `α_i`
- **Learned influence**: `alpha_init` controls initial action influence (default: 0.1)

**Parameters**:
- All `ViTPredictor` parameters plus:
- `alpha_init`: Initial value for per-layer alpha parameters

**Use Case**:
- Experiments on action conditioning without cross-attention
- Better gradient flow for planning (actions not mixed into visual representation)

---

## Memory-Augmented Predictors

Memory-augmented predictors extend the standard ViT with external memory mechanisms that enable long-term information retention beyond the causal attention window.

### Memory Module Interface

All memory-augmented predictors use memory modules from `models/memory.py`:

#### NeuralMemory

A learnable MLP that associates keys with values via online gradient descent.

**Core Operations**:
```python
# Retrieve: y = M*(q)
output = memory.retrieve(query)

# Update: M ← M - θ∇||M(k) - v||² with momentum
memory.update_from_batch(keys, values)
```

**Learning Dynamics**:
- **Surprise-modulated momentum**:
  ```
  S_t = η × S_{t-1} - θ × grad
  W_t = (1 - α) × W_{t-1} + S_t
  ```
- **Parameters**:
  - `eta` (η): Momentum decay rate (default: 0.9)
  - `theta` (θ): Learning rate for online updates (default: 1e-3)
  - `alpha` (α): Weight decay / forgetting rate (default: 1e-5)
  - `hidden_scale`: MLP hidden dimension multiplier (default: 2)
  - `depth`: Number of MLP layers (default: 2)
  - `max_grad_norm`: Gradient clipping threshold (default: 1.0)
  - `momentum_clip`: Momentum buffer clipping (default: 1.0)
  - `weight_clip`: Weight norm clipping (default: 5.0)
  - `update_steps`: Number of gradient steps per update (default: 1)

**Architecture**:
```
Input (query/key): (B, T, d_model)
  ↓
MLP: [d_model → d_hidden → ... → d_model]
  └─ SiLU activations between layers
  ↓
Output (value): (B, T, d_model)
```

**Key Properties**:
- **No gradient flow**: Memory weights are updated via online learning, not backpropagation
- **Normalized inputs**: Keys/queries are L2-normalized before retrieval/update
- **Detached updates**: Keys and values are detached before memory update to prevent gradient interference

#### LookupMemory

A simple append-only memory bank for exact retrieval of past observations.

**Core Operations**:
```python
# Retrieve: return all stored memories
memories = memory.retrieve()  # (B, T_stored, d_model)

# Update: append new batch
memory.update(batch)  # concatenates along time dimension
```

**Use Case**:
- Baseline comparison with learned memory
- Perfect recall experiments
- No compression or learning

---

### 4. MAGViTPredictor (Memory-Augmented Gating)

Uses neural memory with gating mechanism to blend current representations with retrieved memories.

**Architecture**:
```
Input: (B, num_frames × num_patches, dim)
  ↓
Positional Embedding
  ↓
Transformer Blocks (depth):
  ├─ Pre-LayerNorm
  ├─ Self-Attention + Residual
  ├─ FeedForward + Residual
  └─ Memory Gating:
       Q ← W_Q(x)
       M ← NeuralMemory.retrieve(Q)
       gate ← σ(W_y(x)) ⊙ W_m(M)  [sigmoid gating]
       x ← gate  [or x + gate if use_residual]
       NeuralMemory.update(x_in, x_in)
  ↓
Output: (B, num_frames × num_patches, dim)
```

**Gating Modes**:

1. **Sigmoid** (`gate_type="sigmoid"`):
   ```
   gate = σ(W_y(x)) ⊙ W_m(M)
   ```

2. **Sigmoid Convex** (`gate_type="sigmoid_convex"`):
   ```
   g = σ(W_y(x))
   gate = (1 - g) ⊙ V_y(x) + g ⊙ W_m(M)
   ```
   - Convex combination of current representation and memory
   - Guarantees output is interpolation between x and M

**Parameters**:
- All `ViTPredictor` parameters plus:
- `hidden_scale`: Memory MLP hidden size multiplier (default: 1.0)
- `mem_eta`, `mem_theta`, `mem_alpha`: Memory learning hyperparameters
- `mem_depth`: Memory MLP depth (default: 1)
- `gate_type`: "sigmoid" or "sigmoid_convex"
- `use_residual`: If True, `x = x + gate`; else `x = gate`

**Memory Update Strategy**:
- Keys: `x_in.detach()` (input to the block, normalized)
- Values: `x_in.detach()` (same as keys — autoassociative memory)

---

### 5. MACViTPredictor (Memory-Augmented Context)

Prepends retrieved memory slots and persistent tokens to the input sequence, allowing attention-based memory integration.

**Architecture**:
```
Input: (B, T, dim)  where T = num_frames × num_patches
  ↓
Positional Embedding
  ↓
Transformer Blocks (depth):
  ├─ Memory Retrieval:
  │   Q ← W_Q(normalize(x))
  │   h ← NeuralMemory.retrieve(Q)  # (B, T, dim)
  │   h ← compress_to_slots(h)      # (B, n_retrieved, dim)
  ├─ Prepend Context:
  │   x_aug ← [P | h | x]  where P = persistent tokens
  │            └─n_persistent─┘└─n_retrieved─┘└─T─┘
  ├─ Self-Attention with Frame-Level Causality
  │   Attention Pattern:
  │     - Persistent: attend to everything
  │     - Memory slots: attend to everything
  │     - Frame i: attend to [P, h, frames 0...i]
  ├─ FeedForward
  └─ Memory Update:
       Extract memory slots from output
       NeuralMemory.update(keys, values)
  ↓
Strip [P | h], return only visual tokens
  ↓
Output: (B, T, dim)
```

**Key Components**:

1. **Persistent Tokens** (`n_persistent`):
   - Learnable parameters `P ∈ ℝ^(n_persistent × d_model)`
   - Attend to all positions (no causal masking)
   - Provide global context across frames

2. **Retrieved Memory Slots** (`n_retrieved`):
   - Query memory with current input
   - Compress per-token memory to fixed number of slots
   - If `use_slots=True`: `h = mean_pool(mem_slots(h))` → (B, n_retrieved, dim)
   - If `use_slots=False`: Use full per-token memory (B, T, dim)

3. **Attention Mask** (generated by `generate_mac_mask_matrix`):
   ```
   Rows: [n_persistent | n_retrieved | nwindow]
   Cols: [n_persistent | n_retrieved | nwindow]

   Persistent:  [  all ✓  |   all ✓   |  all ✓  ]
   Retrieved:   [  all ✓  |   all ✓   |  all ✓  ]
   Frame 0:     [  all ✓  |   all ✓   |  F₀ ✓   ]
   Frame 1:     [  all ✓  |   all ✓   | F₀,F₁ ✓ ]
   Frame 2:     [  all ✓  |   all ✓   |F₀,F₁,F₂✓]
   ```

4. **Memory Update Modes** (`update_type`):
   - **"selfattention"**: Update from memory slot outputs
     ```python
     k = W_Q(memory_tokens) if proj_k_eq_q else memory_tokens
     memory.update(k, memory_tokens)
     ```
   - **"crossattention"**: Update from visual output using memory slots as keys
     ```python
     k = W_Q(visual_out) if proj_k_eq_q else visual_out
     memory.update(k, memory_tokens)
     ```

**Parameters**:
- All `ViTPredictor` parameters plus:
- `n_persistent`: Number of persistent tokens (default: 4)
- `n_retrieved`: Number of memory slots (default: 4)
- `hidden_scale`: Memory MLP hidden scale (default: 2)
- `mem_depth`: Memory MLP layers (default: 2)
- `mem_eta`, `mem_theta`, `mem_alpha`: Memory learning rates
- `max_grad_norm`, `momentum_clip`, `weight_clip`: Memory optimization constraints
- `update_steps`: Gradient steps per memory update (default: 1)
- `use_slots`: If True, compress memory to slots; else use full sequence (default: True)
- `update_type`: "selfattention" or "crossattention" (default: "selfattention")
- `proj_k_eq_q`: If True, apply same projection to keys as queries (default: False)

**Design Rationale**:
- **Slots reduce computational cost**: Fixed number of memory tokens regardless of history length
- **Persistent tokens** act as global workspace for cross-frame reasoning
- **Normalized memory**: L2 normalization stabilizes retrieval and update dynamics

---

### 6. LayerMACViTPredictor

Similar to `MACViTPredictor` but uses **per-layer memory modules** instead of a shared memory.

**Key Difference**:
- Each transformer layer maintains its own `NeuralMemory` instance
- Allows hierarchical memory representation across layers
- More parameters but potentially more expressive

**Use Case**:
- Experiments on hierarchical memory
- Each layer can specialize in different temporal scales

---

### 7. LookupViTPredictor

Uses `LookupMemory` (append-only bank) instead of learnable neural memory.

**Architecture**:
```
Input: (B, T, dim)
  ↓
Transformer Blocks (depth):
  ├─ Lookup Memory:
  │   memory ← LookupMemory.retrieve()  # all past observations
  │   x_aug ← [memory | x]
  ├─ Self-Attention (with appropriate mask)
  └─ FeedForward
  ↓
Strip memory tokens
  ↓
Output: (B, T, dim)
```

**Key Features**:
- Perfect recall of all past observations
- No compression or learning
- Memory grows linearly with trajectory length

**Use Case**:
- Upper bound for memory-augmented methods
- Debugging and analysis

---

## State Space Models (Experimental)

### 8. StateSpaceViTPredictor

Integrates a linear state-space model (SSM) for maintaining a compressed hidden state across frames.

**SSM Cell**:
```
Discretization:
  τ = softplus(log_τ) + ε
  Ā = exp(-Δt / τ)
  B̄ = 1 - Ā

Forward (per frame):
  H_{t+1} = Ā ⊙ H_t + B̄ ⊙ U(X_t)
  M_{t+1} = C(H_{t+1})
```

**Architecture Variants**:

1. **StateSpaceTransformer** (`ssm_type="sst"`):
   ```
   For each frame:
     Update: H, M ← SSMCell(x_frame, H)

   Prepend M to x:
     ctx ← [M₀, M₁, ..., M_{n-1} | x₀, x₁, ..., x_{n-1}]

   Transformer with Memory-Aware Mask:
     - x_i attends to [M₀...M_i, x₀...x_i]
   ```

2. **MemoryInjectionSSMTransformer** (`ssm_type="misst"`):
   ```
   For each frame:
     Update: H, M ← SSMCell(x_frame, H)

   Transformer Layers:
     x ← x + Attention(x)
     x ← x + FeedForward(x)
     x ← x + αᵢ × Linear(M)  # additive memory injection
   ```

**Parameters**:
- All `ViTPredictor` parameters plus:
- `state_size`: Dimension of hidden state H (default: determined by config)
- `ssm_type`: "sst" or "misst"
- `dt`: Discretization timestep (default: 1.0, could be set to frameskip)
- `use_gate`: If True (misst only), use gating instead of concatenation
- `alpha_init`: Initial injection strength (misst only)

**State Persistence**:
- `H_buffer`: Hidden state is stored as a module buffer
- Detached after each update to prevent backprop through time
- Call `reset_memory()` to reinitialize

**Use Case**:
- Linear state-space models for efficient long-term memory
- Comparison with neural memory approaches

---

## Helper Functions

### Masking Functions

All masking functions generate boolean attention masks where `1` = attend, `0` = masked.

#### `generate_mask_matrix(npatch, nwindow)`
Standard frame-level causal mask:
```
Frame 0: [1 0 0]
Frame 1: [1 1 0]
Frame 2: [1 1 1]
```

#### `generate_mask_with_memory(npatch, nwindow)`
Causal mask with memory prepended (for SSM):
```
       M₀ M₁ M₂ | x₀ x₁ x₂
   M₀  [1  0  0 | 1  0  0]
   M₁  [1  1  0 | 1  1  0]
   M₂  [1  1  1 | 1  1  1]
   x₀  [1  0  0 | 1  0  0]
   x₁  [1  1  0 | 1  1  0]
   x₂  [1  1  1 | 1  1  1]
```

#### `generate_mac_mask_matrix(npatch, nwindow, n_persistent, n_retrieved)`
MAC-style mask with persistent and retrieved tokens:
```
         P₀ ... Pₙ | R₀ ... Rₘ | x₀ x₁ x₂
   P₀    [   all 1       all 1    all 1  ]
   ...
   R₀    [   all 1       all 1    all 1  ]
   ...
   x₀    [   all 1       all 1    1  0  0]
   x₁    [   all 1       all 1    1  1  0]
   x₂    [   all 1       all 1    1  1  1]
```

#### `generate_sliding_window_mask(seq_len, window_size)`
Sliding window attention (not frame-aware):
```
Position i attends to [max(0, i-window_size+1) : i+1]
```

---

## Common Patterns

### Memory Reset
All memory-augmented predictors implement `reset_memory()`:
```python
predictor.reset_memory()  # Clear memory state between episodes
```

This is **critical** for:
- Evaluation: prevent information leakage across episodes
- Training: optionally reset between trajectories (depends on config)

### Forward Pass
All predictors follow the same interface:
```python
# Input: concatenated embeddings
x = torch.cat([visual_emb, action_emb, proprio_emb], dim=1)
# Shape: (B, num_frames * num_patches, dim)

# Forward
output = predictor(x)
# Shape: (B, num_frames * num_patches, dim)

# Extract visual predictions (first num_patches tokens per frame)
pred_visual = output[:, :num_frames*num_patches, :]
```

### Global Variables
`NUM_FRAMES` and `NUM_PATCHES` are set globally at predictor initialization to configure mask generation:
```python
global NUM_FRAMES, NUM_PATCHES
NUM_FRAMES = num_frames
NUM_PATCHES = num_patches
```

---

## Configuration

Predictors are configured via Hydra YAML files in `conf/predictor/`:

**Example** (`conf/predictor/vit.yaml`):
```yaml
_target_: models.vit.ViTPredictor
num_patches: ${num_patches}
num_frames: ${num_hist}
dim: ${encoder.output_dim}
depth: 6
heads: 6
mlp_dim: 1536
dropout: 0.0
emb_dropout: 0.1
```

**Example** (`conf/predictor/mac_vit.yaml`):
```yaml
_target_: models.vit.MACViTPredictor
num_patches: ${num_patches}
num_frames: ${num_hist}
dim: ${encoder.output_dim}
depth: 6
heads: 6
mlp_dim: 1536
n_persistent: 4
n_retrieved: 4
hidden_scale: 2
mem_depth: 2
mem_eta: 0.9
mem_theta: 1e-3
mem_alpha: 1e-5
use_slots: true
update_type: selfattention
```

Override at command line:
```bash
python train.py predictor=mac_vit predictor.n_retrieved=8
```

---

## Training Considerations

### Gradient Flow
- **Standard ViT**: Gradients flow through all parameters via backprop
- **Memory-augmented**: Memory weights updated via online learning (no gradients)
  - Memory updates are **detached** from computational graph
  - Only query/key/value projection layers receive gradients

### Memory Persistence During Training
- `reset_memory()` can be called:
  - **Every trajectory**: Treats each trajectory independently
  - **Never**: Memory persists across entire training run
  - **Periodically**: Reset every N batches (experimental)

### Learning Rates
Typical settings (from `conf/train.yaml`):
```yaml
predictor_lr: 5e-4
predictor_weight_decay: 0.01
```

Memory modules have their own internal learning rates (`theta`) that are **not** affected by the optimizer.

---

## Planning Considerations

During planning, the predictor is used to roll out future trajectories:

```python
# Initialize
predictor.eval()
predictor.reset_memory()

# Rollout loop
obs = initial_observation
for t in range(horizon):
    # Encode
    visual_emb = encoder(obs)
    action_emb = action_encoder(action)

    # Predict
    x = torch.cat([visual_emb, action_emb], dim=1)
    pred_emb = predictor(x)

    # Decode (optional, for visualization)
    pred_obs = decoder(pred_emb)

    obs = pred_obs  # use prediction as next input
```

**Key Points**:
- **No gradient**: `torch.no_grad()` or `.eval()` mode
- **Memory accumulation**: Memory grows during rollout (for memory-augmented models)
- **Reset between plans**: Call `reset_memory()` before each planning episode

---

## Comparison Table

| Model | Memory Type | Shared Memory | Prepended Tokens | Additive Injection | Complexity |
|-------|-------------|---------------|------------------|-------------------|------------|
| ViTPredictor | None | - | No | No | Low |
| ViTPredictorWithPersistentTokens | Persistent | - | Yes (P) | No | Low |
| AdditiveControlViTPredictor | None | - | No | Yes (actions) | Low |
| MAGViTPredictor | Neural | Yes | No | Via gating | Medium |
| MACViTPredictor | Neural | Yes | Yes (P + M) | No | High |
| LayerMACViTPredictor | Neural | No (per-layer) | Yes (P + M) | No | Highest |
| LookupViTPredictor | Lookup | Yes | Yes (M) | No | Medium |
| StateSpaceViTPredictor | Linear SSM | Yes | Varies by mode | Optional | Medium |

**Legend**:
- P: Persistent tokens
- M: Memory slots/tokens

---

## References

- **ViT**: Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021
- **Persistent Memory**: Applicable to various memory-augmented architectures
- **Neural Memory**: Inspired by online learning and Hebbian-like updates
- **State Space Models**: Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces", ICLR 2022

---

## Debugging Tips

### Memory Not Learning
- Check `mem_theta` (learning rate) — try increasing
- Verify `update_from_batch()` is called during training
- Check gradient clipping (`max_grad_norm`) — might be too strict

### NaN/Inf in Memory
- Increase `momentum_clip` and `weight_clip`
- Verify input normalization
- Check `mem_alpha` (weight decay) — prevents unbounded growth

### Memory Growing Too Large (LookupMemory)
- LookupMemory stores all past observations — not suitable for long trajectories
- Use neural memory or limit context length

### Poor Planning Performance
- Ensure `reset_memory()` called between episodes
- Check if memory is being updated during planning (should only retrieve)
- Verify planning objective aligns with training setup

---

## Future Directions

Potential extensions and research directions:

1. **Hierarchical Memory**: Different memory modules for different temporal scales
2. **Sparse Memory**: Only store/retrieve task-relevant information
3. **Meta-learned Memory**: Learn memory update rules via meta-learning
4. **Episodic vs. Semantic**: Separate episodic (trajectory-specific) and semantic (task-general) memory
5. **Attention-based Retrieval**: Replace MLP memory with attention-based retrieval over stored embeddings
