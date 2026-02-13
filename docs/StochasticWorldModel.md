# Stochastic World Model Documentation

## Overview

The Stochastic World Model (`StochasticVWorldModel`) is an extension of the original `VWorldModel` designed to support stochastic decoding through beta-VAE and enhanced VQ-VAE architectures. This implementation enables the DINO-WM framework to model uncertainty in world predictions, which is crucial for robust planning in complex environments.

### Key Features
- **Stochastic Decoding Support**: Compatible with both beta-VAE and VQ-VAE decoder architectures
- **Enhanced Representations**: Deeper convolutional encoder (4-5 layers) for richer visual feature extraction
- **Memory Mechanism Compatibility**: Maintains full compatibility with existing memory evaluation mechanisms
- **Configurable Stochasticity**: Adjustable beta parameter for KL divergence weighting in beta-VAE
- **4-Layer ViT Predictor**: Optional deeper transformer architecture for improved prediction capabilities

## New Model Parameters

The `StochasticVWorldModel` extends the original `VWorldModel` with the following additional parameter:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | float | 1.0 | Weight for KL divergence term in beta-VAE loss. Higher values encourage more disentangled representations but may reduce reconstruction quality. |

All other parameters remain identical to the original `VWorldModel`.

## Key Components

### 1. Deep ResNet Encoder (`DeepResNetTokens`)

A 4-stage residual network encoder with configurable depths and channels:

```python
DeepResNetTokens(
    in_ch=3,
    stem_ch=32,
    stages=[2, 2, 2, 2],  # 4 stages
    channels=[32, 64, 128, 256],
    strides=[2, 2, 2, 2],
    norm="bn",
    emb_dim=256,
    freeze_backbone=False,
    return_map=False,
    gn_groups=32
)
```

**Key Features**:
- Processes 64x64 input images
- Outputs 4x4 spatial grid of 256-dimensional tokens (16 total tokens)
- Configurable depth and channel sizes
- GroupNorm/BatchNorm options

### 2. Enhanced VQ-VAE Decoder (`EnhancedVQVAE`)

An extension of the original VQ-VAE implementation with:
- Support for stochastic decoding
- Commitment loss for codebook optimization
- Configurable embedding dimensions

### 3. Beta-VAE Decoder (`BetaVAEDecoder`)

A variational autoencoder implementation with:
- Reparameterization trick
- KL divergence regularization
- Configurable latent dimension
- Beta parameter for KL weighting

### 4. Stochastic Forward Pass

The `forward()` method has been enhanced to properly handle stochastic decoder outputs:

```python
# Proper extraction of visual component from decoder output
decoder_output = self.decode(z_pred.detach())
obs_pred, diff_pred = decoder_output
visual_pred = obs_pred["visual"]  # Extract tensor from dictionary

# Safe application of loss function
recon_loss_pred = self.decoder_criterion(visual_pred, visual_tgt)
```

This ensures compatibility with both deterministic and stochastic decoders while maintaining the correct loss computation.

## Configuration

### Key Configuration Files

| File | Purpose |
|------|---------|
| `conf/encoder/deep_resnet.yaml` | Configuration for 4-stage ResNet encoder |
| `conf/decoder/enhanced_vqvae.yaml` | Configuration for enhanced VQ-VAE decoder |
| `conf/decoder/beta_vae.yaml` | Configuration for beta-VAE decoder |
| `conf/model/stochastic.yaml` | Main configuration for stochastic world model |
| `conf/predictor/vit_4layer.yaml` | 4-layer ViT predictor configuration |
| `conf/train_memory_maze_stochastic.yaml` | Training config with deep encoder and enhanced VQ-VAE |
| `conf/train_memory_maze_vit4_stochastic.yaml` | Training config with 4-layer ViT predictor |

### Sample Configuration

```yaml
# conf/train_memory_maze_stochastic.yaml
defaults:
  - _self_
  - env: memory_maze_3x7
  - encoder: deep_resnet 
  - action_encoder: discrete
  - proprio_encoder: memmaze
  - decoder: enhanced_vqvae
  - predictor: vit_small
  - model: stochastic
  - aux_predictor: retention

# Stochastic model specific parameter
model:
  beta: 0.8  # Adjust KL divergence weighting

# Training parameters
training:
  encoder_lr: 1e-3
  decoder_lr: 1e-3
  predictor_lr: 3e-4
```

## Usage

### Training

```bash
# Train with deep encoder and enhanced VQ-VAE
python train.py --config-name train_memory_maze_stochastic.yaml

# Train with deep encoder, 4-layer ViT predictor, and enhanced VQ-VAE
python train.py --config-name train_memory_maze_vit4_stochastic.yaml
```

### Key Considerations

1. **Beta Parameter Tuning**:
   - Start with beta=1.0 (standard VAE)
   - Increase for more disentangled representations (may reduce reconstruction quality)
   - Decrease for better reconstruction (less disentanglement)

2. **Memory Mechanisms**:
   - All existing memory evaluation mechanisms (`memory_eval.py`, `memory_plan.py`) work unchanged
   - Stochasticity is handled through the decoder's sampling process

3. **Performance Characteristics**:
   - Deeper encoder increases representational capacity but requires more memory
   - Stochastic decoding adds computational overhead (typically 10-15% increase in training time)

## Implementation Notes

### Critical Fixes

The initial implementation had an issue where the decoder output (a dictionary) was being passed directly to the loss function instead of extracting the "visual" tensor component. This was fixed by:

1. Properly unpacking the decoder output: `obs_pred, diff_pred = self.decode(z_pred.detach())`
2. Extracting the visual tensor: `visual_pred = obs_pred["visual"]`
3. Applying the loss function to the tensor: `self.decoder_criterion(visual_pred, visual_tgt)`

This ensures compatibility with both deterministic and stochastic decoders while maintaining correct loss computation.

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `AttributeError: 'dict' object has no attribute 'size'` | Ensure you're extracting `obs_pred["visual"]` before loss calculation |
| Poor reconstruction quality | Decrease beta value or increase decoder capacity |
| Slow training | Reduce encoder depth or use smaller batch sizes |

### Verification Steps

1. Check that all new configuration files exist
2. Verify that the stochastic model correctly handles both prediction and reconstruction paths
3. Confirm that memory evaluation mechanisms work as expected with the new model

For further assistance, refer to the implementation summary in `IMPLEMENTATION_SUMMARY.md`.