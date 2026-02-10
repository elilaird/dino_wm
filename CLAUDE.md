# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DINO-WM (World Models on Pre-trained Visual Features enable Zero-shot Planning) is a research project that builds visual world models using frozen DINOv2 features for zero-shot robotic planning. The core idea: encode observations with a frozen DINO encoder, predict future embeddings with a ViT predictor, and decode back to images with a VQ-VAE decoder. Planning is done by optimizing action sequences against objectives in the learned embedding space.

Paper: https://arxiv.org/abs/2411.04983

**Additional Documentation**:
- `PREDICTOR_ARCHITECTURE.md` — Comprehensive technical documentation of all predictor models (standard ViT, memory-augmented variants, state-space models) and memory modules (NeuralMemory, LookupMemory). Includes architecture diagrams, attention masking patterns, training/planning considerations, and debugging tips.

## Common Commands

### Environment Setup
```bash
conda env create -f environment.yaml
conda activate dino_wm
export DATASET_DIR=/path/to/data
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
```

### Training
```bash
# Single GPU
python train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3

# Multi-GPU via accelerate
accelerate launch --num_processes=4 train.py --config-name train.yaml env=point_maze
```

### Planning (evaluation)
```bash
python plan.py model_name=<model_name> n_evals=5 planner=cem goal_H=5 goal_source='random_state'
```

### Memory Evaluation
```bash
python memory_eval.py --config-name memory_eval --ckpt_base_path /path/to/ckpts --model_name <model_name>
python memory_plan.py --config-name memory_plan --ckpt_base_path /path/to/ckpts --model_name <model_name>
```

### SLURM Job Submission
```bash
GPU=8 TIME=2-00:00:00 PARTITION=batch TYPE=train ./make_sbatch.sh
GPU=4 TIME=1-00:00:00 PARTITION=batch TYPE=plan ./make_sbatch.sh
```
`make_sbatch.sh` clones the repo into a fresh working directory per job. TYPE can be: train, plan, jupyter, minigrid, memory_maze_download, memory_maze_chunk, habitat.

## Architecture

### Pipeline

```
Observations (224x224 RGB)
  → Encoder (frozen DINOv2, patch tokens 16x16 grid)
  → Predictor (ViT with frame-level causal masking)
  → Decoder (VQ-VAE transposed convolution)
  → Reconstructed images

Planning: CEM or gradient descent optimizes actions by rolling out
the world model and comparing predicted embeddings to goal embeddings.
```

### Key Components

- **`models/visual_world_model.py`** — `VWorldModel` class that composes encoder + predictor + decoder + action/proprio encoders
- **`models/dino.py`** — DINOv2 wrapper; extracts `x_norm_patchtokens` features; typically frozen during training (lr=1e-6)
- **`models/vit.py`** — ViT predictor with frame-level attention masking; predicts next-step embeddings from history of visual + action + proprio tokens
- **`models/vqvae.py`** — VQ-VAE decoder for embedding-to-image reconstruction
- **`models/memory.py`** — Lookup and neural memory modules for the predictor
- **`train.py`** — Main training loop using `accelerate` for distributed training, W&B logging, cosine LR with warmup
- **`plan.py`** — Loads trained model, runs CEM/GD planning, evaluates against environment ground truth
- **`planning/cem.py`, `planning/gd.py`** — Cross-Entropy Method and gradient descent planners
- **`planning/mpc.py`** — Model Predictive Control wrapper
- **`planning/evaluator.py`** — Rolls out plans in actual environment, computes metrics, generates comparison videos
- **`memory_eval.py`, `memory_plan.py`** — Memory evaluation suite (object recall, color memory, sequential memory, navigation)
- **`preprocessor.py`** — Normalization/denormalization for observations, actions, and proprioceptive states

### Configuration System

Uses Hydra with YAML configs under `conf/`. Key config files:
- `conf/train.yaml` — Master training config (epochs, LR, batch size, loss type)
- `conf/plan.yaml` — Planning config (planner type, objective, goal source)
- `conf/memory_eval.yaml`, `conf/memory_plan.yaml` — Memory evaluation configs
- Subconfig groups: `encoder/`, `decoder/`, `predictor/`, `planner/`, `env/`, `action_encoder/`, `proprio_encoder/`, `model/`

Override configs via command line: `python train.py env=pusht training.batch_size=128`

### Environments

Registered gymnasium environments: `pusht`, `point_maze`, `wall`, `deformable_env`, `four_rooms_*` (memory maze variants), `memory_maze_*`. Environment implementations live in `env/` with corresponding dataset classes in `datasets/`.

### Training Details

- Separate learning rates per component: encoder (1e-6), decoder (3e-4), predictor (5e-4), action_encoder (5e-4)
- Cosine annealing with warm restarts and decay; 5% warmup
- Gradient clipping at max_grad_norm=0.5
- Checkpoints saved to `${ckpt_base_path}/outputs/`; planning outputs to `./plan_outputs/`
- `num_hist` controls how many history frames the predictor sees; `frameskip` controls temporal stride in dataset
