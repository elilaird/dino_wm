#!/usr/bin/env python3
"""
Example script demonstrating the memory testing capabilities of FourRoomsMemoryEnv.

This script shows how to:
1. Create memory testing environments
2. Generate memory evaluation datasets
3. Run memory evaluations
4. Analyze memory performance metrics
"""

import os
import sys
import numpy as np
from minigrid_env import (
    FourRoomsMemoryEnv, 
    make_four_rooms,
    run_memory_episode,
    evaluate_memory_env,
    MemoryEvalStats
)

def demo_memory_testing():
    """Demonstrate different memory testing modes"""
    
    print("=== FourRoomsMemoryEnv Memory Testing Demo ===\n")
    
    # Test different memory modes
    memory_modes = ["navigation", "object_recall", "color_memory", "sequential_memory"]
    
    for mode in memory_modes:
        print(f"\n--- Testing {mode} mode ---")
        
        # Create environment
        env = make_four_rooms(
            world_size=17,
            memory_test_mode=mode,
            n_memory_objects=3,
            memory_object_types=["ball", "box", "key"],
            obs_mode="top_down"
        )
        
        # Run a single episode
        traj = run_memory_episode(env, max_steps=100, seed=42, policy="random")
        
        # Print memory information
        memory_info = env.get_memory_test_info()
        print(f"Memory objects placed: {len(memory_info['memory_objects'])}")
        for i, (pos, color, obj_type) in enumerate(memory_info['memory_objects']):
            print(f"  Object {i+1}: {color} {obj_type} at {pos}")
        
        print(f"Memory questions generated: {len(memory_info['memory_questions'])}")
        for i, question in enumerate(memory_info['memory_questions']):
            print(f"  Question {i+1}: {question['question']}")
        
        print(f"Memory phases: {set(traj.memory_phases)}")
        print(f"Exploration metrics: {traj.exploration_metrics}")
        print(f"Recall metrics: {traj.recall_metrics}")
        
        env.close()

def demo_memory_evaluation():
    """Demonstrate memory evaluation across multiple episodes"""
    
    print("\n\n=== Memory Evaluation Demo ===\n")
    
    # Test object recall mode
    print("Evaluating object_recall mode...")
    ctor = lambda: make_four_rooms(
        memory_test_mode="object_recall",
        n_memory_objects=3,
        memory_object_types=["ball", "box", "key"]
    )
    
    stats = evaluate_memory_env(ctor, n_episodes=10, max_steps=150, seed=42, policy="random")
    
    print(f"Memory Evaluation Results:")
    print(f"  Test Mode: {stats.test_mode}")
    print(f"  Episodes: {stats.n_episodes}")
    print(f"  Exploration Efficiency: {stats.exploration_efficiency:.3f}")
    print(f"  Memory Retention: {stats.memory_retention:.3f}")
    print(f"  Recall Accuracy: {stats.recall_accuracy:.3f}")
    print(f"  Navigation Efficiency: {stats.navigation_efficiency:.3f}")
    print(f"  Avg Objects Discovered: {stats.avg_objects_discovered:.1f}")
    print(f"  Avg Questions Answered: {stats.avg_questions_answered:.1f}")
    print(f"  Phase Transition Success: {stats.phase_transition_success:.3f}")

def demo_memory_dataset_generation():
    """Demonstrate memory dataset generation"""
    
    print("\n\n=== Memory Dataset Generation Demo ===\n")
    
    # Set up environment variables for dataset generation
    os.environ["DATASET_DIR"] = "/tmp/memory_datasets"
    
    # Create a small memory dataset
    print("Generating memory dataset...")
    
    # This would normally be run via CLI, but we can demonstrate the concept
    env = make_four_rooms(
        memory_test_mode="color_memory",
        n_memory_objects=4,
        memory_object_types=["ball", "box", "key", "ball"]
    )
    
    # Generate a few episodes
    trajectories = []
    for i in range(5):
        traj = run_memory_episode(env, max_steps=100, seed=42+i, policy="random")
        trajectories.append(traj)
        print(f"Generated episode {i+1}: {len(traj.observations)} steps, "
              f"{len(traj.memory_objects)} objects, {len(traj.memory_questions)} questions")
    
    env.close()
    
    print(f"\nGenerated {len(trajectories)} memory trajectories")
    print("Each trajectory contains:")
    print("  - Visual observations")
    print("  - Actions taken")
    print("  - Proprioceptive information")
    print("  - Memory objects and their properties")
    print("  - Memory questions")
    print("  - Memory phases throughout the episode")
    print("  - Exploration and recall metrics")

def demo_memory_analysis():
    """Demonstrate memory analysis capabilities"""
    
    print("\n\n=== Memory Analysis Demo ===\n")
    
    # Create environment and run episode
    env = make_four_rooms(
        memory_test_mode="sequential_memory",
        n_memory_objects=3,
        memory_object_types=["ball", "box", "key"]
    )
    
    traj = run_memory_episode(env, max_steps=200, seed=42, policy="explore")
    
    # Analyze memory performance
    print("Memory Analysis:")
    print(f"  Total steps: {len(traj.actions)}")
    print(f"  Memory objects: {len(traj.memory_objects)}")
    print(f"  Memory questions: {len(traj.memory_questions)}")
    
    # Analyze phase transitions
    phases = traj.memory_phases
    phase_counts = {}
    for phase in phases:
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    
    print(f"  Phase distribution: {phase_counts}")
    
    # Analyze exploration efficiency
    exploration_eff = traj.exploration_metrics.get('efficiency', 0.0)
    objects_discovered = traj.exploration_metrics.get('objects_discovered', 0)
    total_objects = traj.exploration_metrics.get('total_objects', 0)
    
    print(f"  Exploration efficiency: {exploration_eff:.3f}")
    print(f"  Objects discovered: {objects_discovered}/{total_objects}")
    
    # Analyze memory retention
    memory_retention = traj.exploration_metrics.get('retention', 0.0)
    print(f"  Memory retention: {memory_retention:.3f}")
    
    env.close()

if __name__ == "__main__":
    print("FourRoomsMemoryEnv Memory Testing Suite")
    print("======================================")
    
    try:
        demo_memory_testing()
        demo_memory_evaluation()
        demo_memory_dataset_generation()
        demo_memory_analysis()
        
        print("\n\n=== Demo Complete ===")
        print("The memory testing suite provides comprehensive evaluation of:")
        print("1. Object placement and recall")
        print("2. Color memory and spatial memory")
        print("3. Sequential memory tasks")
        print("4. Navigation to remembered objects")
        print("5. Exploration efficiency metrics")
        print("6. Memory retention analysis")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()
