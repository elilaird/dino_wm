# FourRoomsMemoryEnv Memory Testing Suite

This document describes the comprehensive memory testing capabilities added to the FourRoomsMemoryEnv for evaluating world model memory abilities.

## Overview

The memory testing suite extends the basic FourRoomsMemoryEnv with sophisticated memory evaluation procedures that test an agent's ability to:

1. **Remember object locations** after traveling out of frame
2. **Recall object colors and properties** from memory
3. **Remember sequential information** about object placements
4. **Navigate to remembered objects** efficiently
5. **Maintain spatial memory** across long navigation sequences

## Memory Test Modes

### 1. Navigation Mode (`navigation`)
- **Purpose**: Basic navigation to goal (baseline)
- **Memory Challenge**: Long-term navigation and path planning
- **Evaluation**: Success rate, path efficiency, exploration coverage

### 2. Object Recall Mode (`object_recall`)
- **Purpose**: Remember object locations after exploration
- **Memory Challenge**: Spatial memory of object positions
- **Evaluation**: 
  - Exploration efficiency (how many objects discovered)
  - Memory retention (ability to recall object locations)
  - Navigation to remembered objects

### 3. Color Memory Mode (`color_memory`)
- **Purpose**: Remember object colors and properties
- **Memory Challenge**: Associative memory (object type + color + location)
- **Evaluation**:
  - Color-object association accuracy
  - Spatial-color binding memory
  - Multi-modal memory retention

### 4. Sequential Memory Mode (`sequential_memory`)
- **Purpose**: Remember sequence of object placements
- **Memory Challenge**: Temporal sequence memory
- **Evaluation**:
  - Sequence recall accuracy
  - Temporal memory retention
  - Order-dependent navigation

## Environment Configuration

```python
env = FourRoomsMemoryEnv(
    world_size=17,                    # Grid size (must be odd >= 7)
    memory_test_mode="object_recall", # Memory test mode
    n_memory_objects=3,               # Number of objects to place
    memory_object_types=["ball", "box", "key"],  # Object types
    obs_mode="top_down",              # Observation mode
    agent_view_size=7,                # Agent field of view
    max_steps=200                     # Maximum episode length
)
```

## Memory Testing Phases

Each memory test episode consists of three phases:

### 1. Exploration Phase
- Agent explores the environment
- Discovers and observes memory objects
- Builds internal representation of object locations/properties
- **Metrics**: Exploration efficiency, objects discovered

### 2. Question Phase
- Agent is asked memory questions about observed objects
- Tests recall accuracy for locations, colors, or sequences
- **Metrics**: Recall accuracy, question answering performance

### 3. Navigation Phase
- Agent navigates to remembered object locations
- Tests spatial memory and navigation efficiency
- **Metrics**: Navigation efficiency, path optimality

## Memory Evaluation Metrics

### Exploration Metrics
- **Exploration Efficiency**: Fraction of objects discovered during exploration
- **Memory Retention**: Ability to maintain memory of objects after exploration
- **Objects Discovered**: Number of objects found during exploration

### Recall Metrics
- **Recall Accuracy**: Accuracy of memory recall tasks
- **Questions Answered**: Number of questions answered correctly
- **Navigation Efficiency**: Efficiency of navigation to remembered objects
- **Phase Transition Success**: Success rate of phase transitions

### Navigation Metrics
- **Path Efficiency**: Optimality of paths to remembered objects
- **Spatial Memory**: Accuracy of spatial memory recall
- **Temporal Memory**: Accuracy of sequence memory

## Usage Examples

### Basic Memory Testing

```python
from minigrid_env import make_four_rooms, run_memory_episode

# Create memory testing environment
env = make_four_rooms(
    memory_test_mode="object_recall",
    n_memory_objects=3,
    memory_object_types=["ball", "box", "key"]
)

# Run memory episode
traj = run_memory_episode(env, max_steps=200, seed=42, policy="random")

# Analyze results
print(f"Objects discovered: {traj.exploration_metrics['objects_discovered']}")
print(f"Memory retention: {traj.exploration_metrics['retention']}")
print(f"Memory phases: {set(traj.memory_phases)}")
```

### Memory Evaluation

```python
from minigrid_env import evaluate_memory_env

# Evaluate memory performance
stats = evaluate_memory_env(
    env_ctor=lambda: make_four_rooms(memory_test_mode="color_memory"),
    n_episodes=100,
    max_steps=200,
    policy="explore"
)

print(f"Exploration Efficiency: {stats.exploration_efficiency:.3f}")
print(f"Memory Retention: {stats.memory_retention:.3f}")
print(f"Recall Accuracy: {stats.recall_accuracy:.3f}")
```

### Dataset Generation

```bash
# Generate memory training dataset
python minigrid_env.py generate \
    --env four_rooms \
    --memory-test-mode object_recall \
    --n-memory-objects 3 \
    --episodes 1000 \
    --policy explore \
    --max-steps 200

# Generate memory evaluation dataset
python minigrid_env.py generate \
    --env four_rooms \
    --memory-test-mode color_memory \
    --n-memory-objects 4 \
    --episodes 500 \
    --policy bfs \
    --max-steps 150
```

### Memory Evaluation

```bash
# Evaluate memory capabilities
python minigrid_env.py memory-eval \
    --env four_rooms \
    --memory-test-mode sequential_memory \
    --n-memory-objects 3 \
    --episodes 100 \
    --policy explore \
    --max-steps 200
```

## Memory Object Types

The environment supports various object types for memory testing:

- **Ball**: Colored balls for color memory tasks
- **Box**: Colored boxes for spatial memory tasks  
- **Key**: Colored keys for associative memory tasks

Each object type can be colored with: `red`, `green`, `blue`, `yellow`, `purple`

## Memory Questions

The environment automatically generates memory questions based on the test mode:

### Object Recall Questions
- "Where is the red ball?"
- "Where is the green box?"
- "Where is the blue key?"

### Color Memory Questions
- "What color is the ball at position (5, 3)?"
- "What color is the box at position (12, 8)?"

### Sequential Memory Questions
- "What was the 1st object placed?"
- "What was the 2nd object placed?"
- "What was the 3rd object placed?"

## Advanced Memory Testing

### Custom Memory Object Types

```python
env = make_four_rooms(
    memory_test_mode="object_recall",
    n_memory_objects=5,
    memory_object_types=["ball", "box", "key", "ball", "box"]
)
```

### Multi-Modal Memory Testing

```python
# Test both spatial and color memory
env = make_four_rooms(
    memory_test_mode="color_memory",
    n_memory_objects=4,
    memory_object_types=["ball", "box", "key", "ball"]
)
```

### Sequential Memory Testing

```python
# Test temporal sequence memory
env = make_four_rooms(
    memory_test_mode="sequential_memory",
    n_memory_objects=3,
    memory_object_types=["ball", "box", "key"]
)
```

## Memory Dataset Structure

Memory datasets include additional fields beyond standard trajectories:

```python
@dataclass
class MemoryTrajectory:
    observations: np.ndarray          # [T, H, W, 3] - Visual observations
    actions: np.ndarray               # [T] - Actions taken
    proprio: np.ndarray               # [T, 3] - Proprioceptive info
    memory_objects: List[Tuple]       # Memory objects (pos, color, type)
    memory_questions: List[Dict]      # Memory questions
    memory_phases: List[str]          # Memory phase at each timestep
    exploration_metrics: Dict         # Exploration efficiency metrics
    recall_metrics: Dict              # Memory recall metrics
```

## Memory Evaluation Best Practices

### 1. Exploration Policies
- **Random**: Baseline exploration
- **BFS**: Optimal exploration (upper bound)
- **Explore**: Systematic exploration (realistic)

### 2. Memory Test Parameters
- **n_memory_objects**: 3-5 objects for good memory challenge
- **max_steps**: 150-300 steps for sufficient exploration
- **world_size**: 17x17 for good room separation

### 3. Evaluation Metrics
- Focus on exploration efficiency and memory retention
- Compare across different memory test modes
- Analyze phase transition success rates

### 4. Dataset Generation
- Generate balanced datasets across memory modes
- Include both training and evaluation splits
- Use diverse exploration policies

## Memory Testing Applications

### World Model Evaluation
- Test world model memory capabilities
- Evaluate spatial memory retention
- Assess multi-modal memory binding

### Navigation Research
- Test long-term spatial memory
- Evaluate path planning with memory
- Assess exploration efficiency

### Memory Architecture Research
- Compare different memory architectures
- Test memory capacity limits
- Evaluate memory consolidation

## Troubleshooting

### Common Issues

1. **No memory objects placed**: Check `n_memory_objects` and `memory_object_types`
2. **Memory questions not generated**: Ensure `memory_test_mode` is not "navigation"
3. **Phase transitions not working**: Check `max_steps` and exploration policy
4. **Memory metrics not calculated**: Verify memory objects are placed correctly

### Debug Mode

```python
# Enable debug information
env = make_four_rooms(memory_test_mode="object_recall")
traj = run_memory_episode(env, max_steps=200, seed=42, policy="random")

# Print memory information
memory_info = env.get_memory_test_info()
print(f"Memory objects: {memory_info['memory_objects']}")
print(f"Memory questions: {memory_info['memory_questions']}")
print(f"Memory phases: {memory_info['memory_phase']}")
```

## Future Extensions

### Planned Features
- **Dynamic object placement**: Objects that move or change
- **Multi-room memory**: Objects spanning multiple rooms
- **Temporal memory**: Objects that appear/disappear over time
- **Complex memory tasks**: Multi-step memory reasoning

### Research Directions
- **Memory capacity limits**: How many objects can be remembered
- **Memory interference**: Effect of similar objects on memory
- **Memory consolidation**: How memory improves with experience
- **Transfer learning**: Memory transfer across environments

## Citation

If you use this memory testing suite in your research, please cite:

```bibtex
@software{fourrooms_memory_testing,
  title={FourRoomsMemoryEnv Memory Testing Suite},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/dino_wm}
}
```
