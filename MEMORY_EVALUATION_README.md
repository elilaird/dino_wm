# Memory Evaluation System

This document describes the comprehensive memory evaluation system for world models that tests memory capabilities like object recall, spatial memory, and navigation by comparing world model rollouts to environment rollouts.

## Overview

The memory evaluation system extends your existing training and evaluation infrastructure to test world model memory capabilities through:

1. **Memory-specific environments** with object placement and recall tasks
2. **Scripted trajectory evaluation** that compares world model predictions to environment rollouts
3. **Memory-specific metrics** that measure recall accuracy, spatial memory, and navigation efficiency
4. **Integration with existing training pipeline** using the same model loading and evaluation framework

## System Architecture

```
Training Pipeline (train.py)
    ↓
Trained World Model
    ↓
Memory Evaluation (memory_eval.py)
    ↓
Memory-Specific Metrics
```

### Key Components

1. **Memory Environments** (`env/minigrid/minigrid_env.py`)
   - Extended FourRoomsMemoryEnv with memory testing capabilities
   - Object placement and recall tasks
   - Memory-specific evaluation metrics

2. **Memory Evaluator** (`planning/memory_evaluator.py`)
   - Compares world model rollouts to environment rollouts
   - Computes memory-specific metrics
   - Generates comparison videos and plots

3. **Memory Planning** (`memory_plan.py`)
   - Integrates with existing planning infrastructure
   - Uses scripted trajectories instead of MPC planning
   - Evaluates memory capabilities systematically

4. **Memory Evaluation Suite** (`memory_eval.py`)
   - Runs comprehensive evaluation across multiple configurations
   - Tests different memory modes, object counts, and planning horizons
   - Aggregates and summarizes results

## Memory Test Modes

### 1. Object Recall (`object_recall`)
- **Purpose**: Test spatial memory of object locations
- **Memory Challenge**: Remember where objects are placed after exploration
- **Evaluation**: 
  - Object discovery efficiency
  - Spatial memory accuracy
  - Navigation to remembered objects

### 2. Color Memory (`color_memory`)
- **Purpose**: Test associative memory of object colors
- **Memory Challenge**: Remember color-object associations
- **Evaluation**:
  - Color-object binding accuracy
  - Multi-modal memory retention
  - Association recall performance

### 3. Sequential Memory (`sequential_memory`)
- **Purpose**: Test temporal sequence memory
- **Memory Challenge**: Remember order of object placements
- **Evaluation**:
  - Sequence recall accuracy
  - Temporal memory retention
  - Order-dependent navigation

### 4. Navigation (`navigation`)
- **Purpose**: Test long-term spatial navigation
- **Memory Challenge**: Navigate to remembered locations
- **Evaluation**:
  - Path planning efficiency
  - Spatial memory accuracy
  - Navigation consistency

## Usage

### 1. Training a World Model

First, train a world model using your existing training pipeline:

```bash
# Train a world model on four rooms environment
python train.py --config-name train_four_rooms
```

### 2. Memory Evaluation

After training, evaluate the world model on memory tasks:

```bash
# Run comprehensive memory evaluation suite
python memory_eval.py \
    --ckpt_base_path /path/to/checkpoints \
    --model_name your_model_name \
    --memory_test_modes [object_recall,color_memory,sequential_memory,navigation] \
    --n_memory_objects_list [3,4,5] \
    --goal_H_list [5,10,15]
```

### 3. Single Memory Test

For testing a specific configuration:

```bash
# Run single memory test
python memory_plan.py \
    --ckpt_base_path /path/to/checkpoints \
    --model_name your_model_name \
    --memory_test_mode object_recall \
    --n_memory_objects 3 \
    --goal_H 10
```

### 4. Example Usage

Run the example script to see how the system works:

```bash
python example_memory_eval.py
```

## Configuration

### Memory Evaluation Configuration (`conf/memory_eval.yaml`)

```yaml
# Model configuration
ckpt_base_path: ""  # Base path to model checkpoints
model_name: ""      # Name of the trained model
model_epoch: final  # Epoch to load

# Memory test configuration
memory_test_modes: [object_recall, color_memory, sequential_memory, navigation]
n_memory_objects_list: [3, 4, 5]
goal_H_list: [5, 10, 15]

# Evaluation configuration
n_evals: 10         # Number of evaluation episodes per configuration
seed: 42
wandb_logging: true
```

### Memory Planning Configuration (`conf/memory_plan.yaml`)

```yaml
# Memory test configuration
memory_test_mode: object_recall
n_memory_objects: 3
memory_object_types: [ball, box, key]

# Model configuration
ckpt_base_path: ""
model_name: ""
model_epoch: final

# Evaluation configuration
n_evals: 10
goal_H: 10
goal_source: memory_exploration
```

## Memory Evaluation Metrics

### Basic Metrics
- **Success Rate**: Percentage of successful memory tasks
- **Visual Divergence**: L2 distance between predicted and actual visual observations
- **Proprio Divergence**: L2 distance between predicted and actual proprioceptive observations

### Memory-Specific Metrics

#### Object Recall Metrics
- **Object Discovery Efficiency**: Fraction of objects discovered during exploration
- **Spatial Memory Accuracy**: Accuracy of spatial memory recall
- **Navigation Efficiency**: Efficiency of navigation to remembered objects

#### Color Memory Metrics
- **Color Association Accuracy**: Accuracy of color-object associations
- **Multi-modal Memory Retention**: Retention of visual-proprioceptive associations
- **Color Memory Consistency**: Consistency between environment and world model color memory

#### Sequential Memory Metrics
- **Sequence Recall Accuracy**: Accuracy of temporal sequence recall
- **Temporal Memory Retention**: Retention of temporal information
- **Sequence Memory Consistency**: Consistency of sequence memory between environment and world model

#### Navigation Metrics
- **Path Planning Efficiency**: Efficiency of path planning to remembered locations
- **Spatial Memory Accuracy**: Accuracy of spatial memory for navigation
- **Navigation Consistency**: Consistency of navigation between environment and world model

## Integration with Existing Pipeline

### Model Loading
The memory evaluation system uses the same model loading infrastructure as your existing planning system:

```python
# Load trained world model
model = load_model_state_dict(model_ckpt, train_cfg, num_action_repeat, device)
```

### Environment Integration
Memory environments extend your existing environment infrastructure:

```python
# Create memory-enabled environment
env = SubprocVectorEnv([
    lambda: gym.make(
        "four_rooms_memory",
        memory_test_mode="object_recall",
        n_memory_objects=3,
        memory_object_types=["ball", "box", "key"]
    )
    for _ in range(n_evals)
])
```

### Evaluation Integration
Memory evaluation uses the same evaluation framework as your planning system:

```python
# Create memory evaluator
evaluator = MemoryEvaluator(
    obs_0=obs_0,
    obs_g=obs_g,
    state_0=state_0,
    state_g=state_g,
    env=env,
    wm=model,
    memory_test_mode="object_recall",
    memory_objects=memory_objects,
    memory_questions=memory_questions
)
```

## Output and Results

### Generated Files
- **Memory evaluation results**: `memory_evaluation_results.json`
- **Comparison videos**: `memory_output_*_success.mp4` and `memory_output_*_failure.mp4`
- **Comparison plots**: `memory_output_*.png`
- **Log files**: `memory_logs.json`

### Wandb Integration
Results are automatically logged to Wandb with the project name `memory_eval_{memory_test_mode}`.

### Result Interpretation
- **High success rate + low divergence**: Good memory capabilities
- **Low success rate + high divergence**: Poor memory capabilities
- **High success rate + high divergence**: Good task completion but poor world model accuracy
- **Low success rate + low divergence**: Poor task completion but good world model accuracy

## Advanced Usage

### Custom Memory Tests
You can create custom memory tests by extending the memory evaluation system:

```python
class CustomMemoryEvaluator(MemoryEvaluator):
    def _compute_custom_memory_metrics(self, env_rollout, wm_rollout, action_len):
        # Implement custom memory metrics
        return {"custom_metric": value}
```

### Custom Memory Objects
You can define custom memory objects and questions:

```python
memory_objects = [
    {"type": "ball", "color": "red", "position": (5, 3)},
    {"type": "box", "color": "blue", "position": (12, 8)},
    {"type": "key", "color": "green", "position": (8, 15)}
]

memory_questions = [
    {"type": "location", "question": "Where is the red ball?", "answer": (5, 3)},
    {"type": "color", "question": "What color is the box at (12, 8)?", "answer": "blue"}
]
```

### Batch Evaluation
For large-scale evaluation, you can run multiple configurations in parallel:

```python
# Run multiple memory evaluations in parallel
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = []
    for config in configs:
        future = executor.submit(memory_planning_main, config)
        futures.append(future)
    
    results = [future.result() for future in futures]
```

## Troubleshooting

### Common Issues

1. **Model loading errors**: Ensure checkpoint path and model name are correct
2. **Environment errors**: Check that memory-enabled environments are properly registered
3. **Memory errors**: Reduce batch size or number of evaluation episodes
4. **CUDA errors**: Ensure sufficient GPU memory for evaluation

### Debug Mode
Enable debug mode for detailed logging:

```python
# Set debug mode in configuration
cfg.debug = True
cfg.n_evals = 1  # Reduce number of evaluations for debugging
```

### Performance Optimization
- Use smaller batch sizes for memory evaluation
- Reduce number of evaluation episodes for faster testing
- Use CPU for small-scale evaluation
- Enable mixed precision for large models

## Future Extensions

### Planned Features
- **Dynamic memory tests**: Objects that move or change over time
- **Multi-room memory**: Objects spanning multiple rooms
- **Temporal memory**: Objects that appear/disappear over time
- **Complex memory reasoning**: Multi-step memory inference

### Research Directions
- **Memory capacity limits**: How many objects can be remembered
- **Memory interference**: Effect of similar objects on memory
- **Memory consolidation**: How memory improves with experience
- **Transfer learning**: Memory transfer across environments

## Citation

If you use this memory evaluation system in your research, please cite:

```bibtex
@software{memory_evaluation_system,
  title={Memory Evaluation System for World Models},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/dino_wm}
}
```
