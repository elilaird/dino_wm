import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["EGL_PLATFORM"] = "surfaceless"
import gymnasium as gym
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import random
from env.pointmaze.maze_model import U_MAZE, MEDIUM_MAZE, LARGE_MAZE
from env.pointmaze.point_maze_wrapper import PointMazeWrapper
import gc
import pickle
import argparse

def generate_point_maze_data_memory_efficient(
    output_dir="data/point_maze",
    n_rollouts=100,
    max_timesteps=300,
    maze_spec=U_MAZE,
    action_scale=1.0,
    seed=42,
    save_images=True,
    policy_type="random",
    batch_size=10,
    save_frequency=50,
    compress_images=True,
):
    """
    Memory-efficient version of point maze data generation.
    
    Args:
        output_dir: Directory to save the data
        n_rollouts: Number of trajectories to generate
        max_timesteps: Maximum timesteps per trajectory
        maze_spec: Maze specification
        action_scale: Scale factor for actions
        seed: Random seed
        save_images: Whether to save visual observations
        policy_type: Type of policy to use
        batch_size: Number of rollouts to process before saving
        save_frequency: Save data every N rollouts
        compress_images: Use compressed format for images
    """
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if save_images:
        (output_path / "obses").mkdir(exist_ok=True)
    
    # Initialize data containers for current batch
    batch_states = []
    batch_actions = []
    batch_seq_lengths = []
    
    # Track global statistics
    total_states = []
    total_actions = []
    total_seq_lengths = []
    
    print(f"Generating {n_rollouts} rollouts with {policy_type} policy...")
    
    # Process rollouts in batches
    for batch_start in range(0, n_rollouts, batch_size):
        batch_end = min(batch_start + batch_size, n_rollouts)
        batch_rollouts = batch_end - batch_start
        
        # Create environment for this batch using gym.make (same as plan.py)
        env = gym.make('point_maze')
        
        # Process batch
        for rollout_idx in tqdm(range(batch_start, batch_end), 
                               desc=f"Batch {batch_start//batch_size + 1}"):
            
            # Reset environment
            obs, state = env.reset()
            
            # Initialize trajectory data
            states = [state]
            actions = []
            images = [obs["visual"]]
            
            # Generate trajectory
            for timestep in range(max_timesteps):
                # Select action based on policy type
                if policy_type == "random":
                    action = np.random.uniform(-1.0, 1.0, size=env.action_space.shape[0])
                    distance = None
                elif policy_type == "goal_oriented":
                    current_pos = state[:2]
                    target_pos = env.get_target()
                    direction = target_pos - current_pos
                    distance = np.linalg.norm(direction)
                    if distance > 0.1:
                        action = direction / distance * 0.5
                    else:
                        action = np.random.uniform(-0.1, 0.1, size=env.action_space.shape[0])
                elif policy_type == "expert":
                    current_pos = state[:2]
                    target_pos = env.get_target()
                    direction = target_pos - current_pos
                    distance = np.linalg.norm(direction)
                    
                    if distance < 0.5:
                        action = np.random.uniform(-0.1, 0.1, size=env.action_space.shape[0])
                    else:
                        action = direction / distance * 0.8 + np.random.uniform(-0.2, 0.2, size=env.action_space.shape[0])
                        action = np.clip(action, -1.0, 1.0)
                else:
                    raise ValueError(f"Unknown policy type: {policy_type}")
                
                # Take step in environment
                obs, reward, done, info = env.step(action)
                
                # Store data
                states.append(info['state'])
                actions.append(action)
                images.append(obs["visual"])
                
                # Update state for next iteration
                state = info['state']
                
                # Check if episode is done
                if done or (distance is not None and distance < 0.5):
                    break
            
            # Convert to tensors
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)
            images = torch.stack(images, dim=0)
            
            # Pad to max_timesteps if necessary
            if len(states) < max_timesteps + 1:
                pad_length = max_timesteps + 1 - len(states)
                states = torch.cat([states, torch.zeros(pad_length, states.shape[1])], dim=0)
                actions = torch.cat([actions, torch.zeros(pad_length, actions.shape[1])], dim=0)
                images = torch.cat([images, torch.zeros(pad_length, *images.shape[1:])], dim=0)
            
            # Store in batch
            batch_states.append(states)
            batch_actions.append(actions)
            batch_seq_lengths.append(len(states) - 1)
            
            # Save images immediately to free memory
            if save_images:
                if compress_images:
                    # Save as compressed numpy array
                    np.savez_compressed(
                        output_path / "obses" / f"episode_{rollout_idx:03d}.npz",
                        images=images.numpy()
                    )
                else:
                    torch.save(images, output_path / "obses" / f"episode_{rollout_idx:03d}.pth")
            
            # Clear image memory
            del images
            gc.collect()
        
        # Save batch data
        batch_states_tensor = torch.stack(batch_states)
        batch_actions_tensor = torch.stack(batch_actions)
        batch_seq_lengths_tensor = torch.tensor(batch_seq_lengths, dtype=torch.long)
        
        # Scale actions
        batch_actions_tensor = batch_actions_tensor * action_scale
        
        # Append to total data
        total_states.append(batch_states_tensor)
        total_actions.append(batch_actions_tensor)
        total_seq_lengths.append(batch_seq_lengths_tensor)
        
        # Clear batch memory
        del batch_states, batch_actions, batch_seq_lengths
        del batch_states_tensor, batch_actions_tensor, batch_seq_lengths_tensor
        gc.collect()
        
        # Save intermediate results every save_frequency rollouts
        if (batch_end % save_frequency == 0) or (batch_end == n_rollouts):
            print(f"Saving intermediate results at rollout {batch_end}...")
            save_intermediate_results(
                total_states, total_actions, total_seq_lengths, 
                output_path, batch_end
            )
        
        # Close environment to free resources
        env.close()
        del env
        gc.collect()
    
    # Final save
    print("Saving final results...")
    save_final_results(total_states, total_actions, total_seq_lengths, output_path)
    
    print(f"Data saved to {output_path}")
    return output_path

def save_intermediate_results(states_list, actions_list, seq_lengths_list, output_path, rollout_count):
    """Save intermediate results to avoid losing progress."""
    # Concatenate all batches
    states = torch.cat(states_list, dim=0)
    actions = torch.cat(actions_list, dim=0)
    seq_lengths = torch.cat(seq_lengths_list, dim=0)
    
    # Save with intermediate suffix
    torch.save(states, output_path / f"states_intermediate_{rollout_count}.pth")
    torch.save(actions, output_path / f"actions_intermediate_{rollout_count}.pth")
    torch.save(seq_lengths, output_path / f"seq_lengths_intermediate_{rollout_count}.pth")
    
    print(f"Intermediate data shapes: states {states.shape}, actions {actions.shape}")

def save_final_results(states_list, actions_list, seq_lengths_list, output_path):
    """Save final concatenated results."""
    # Concatenate all batches
    states = torch.cat(states_list, dim=0)
    actions = torch.cat(actions_list, dim=0)
    seq_lengths = torch.cat(seq_lengths_list, dim=0)
    
    # Save final results
    torch.save(states, output_path / "states.pth")
    torch.save(actions, output_path / "actions.pth")
    torch.save(seq_lengths, output_path / "seq_lengths.pth")
    
    print(f"Final data shapes: states {states.shape}, actions {actions.shape}")
    print(f"Average sequence length: {seq_lengths.float().mean():.1f}")

def generate_large_dataset_with_multiprocessing(
    output_dir="data/point_maze_large",
    n_rollouts=1000,
    max_timesteps=300,
    maze_spec=U_MAZE,
    n_processes=4,
    rollouts_per_process=250,
    **kwargs
):
    """
    Generate large datasets using multiprocessing to distribute memory usage.
    """
    import multiprocessing as mp
    from functools import partial
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each process
    process_dirs = [output_path / f"process_{i}" for i in range(n_processes)]
    for dir_path in process_dirs:
        dir_path.mkdir(exist_ok=True)
        if kwargs.get('save_images', True):
            (dir_path / "obses").mkdir(exist_ok=True)
    
    # Prepare arguments for each process
    process_args = []
    for i in range(n_processes):
        process_seed = kwargs.get('seed', 42) + i * 1000  # Different seed per process
        process_args.append({
            'output_dir': str(process_dirs[i]),
            'n_rollouts': rollouts_per_process,
            'max_timesteps': max_timesteps,
            'maze_spec': maze_spec,
            'seed': process_seed,
            **{k: v for k, v in kwargs.items() if k not in ['output_dir', 'n_rollouts', 'seed']}
        })
    
    # Run processes
    with mp.Pool(n_processes) as pool:
        results = pool.map(generate_point_maze_data_memory_efficient, process_args)
    
    # Combine results
    combine_multiprocess_results(process_dirs, output_path)
    
    return output_path

def combine_multiprocess_results(process_dirs, final_output_dir):
    """Combine results from multiple processes."""
    print("Combining results from multiple processes...")
    
    all_states = []
    all_actions = []
    all_seq_lengths = []
    
    for process_dir in process_dirs:
        states = torch.load(process_dir / "states.pth")
        actions = torch.load(process_dir / "actions.pth")
        seq_lengths = torch.load(process_dir / "seq_lengths.pth")
        
        all_states.append(states)
        all_actions.append(actions)
        all_seq_lengths.append(seq_lengths)
    
    # Concatenate all data
    final_states = torch.cat(all_states, dim=0)
    final_actions = torch.cat(all_actions, dim=0)
    final_seq_lengths = torch.cat(all_seq_lengths, dim=0)
    
    # Save combined results
    torch.save(final_states, final_output_dir / "states.pth")
    torch.save(final_actions, final_output_dir / "actions.pth")
    torch.save(final_seq_lengths, final_output_dir / "seq_lengths.pth")
    
    # Copy images (if they exist)
    if (process_dirs[0] / "obses").exists():
        (final_output_dir / "obses").mkdir(exist_ok=True)
        episode_offset = 0
        for process_dir in process_dirs:
            obs_dir = process_dir / "obses"
            if obs_dir.exists():
                for img_file in obs_dir.glob("*.pth"):
                    new_name = f"episode_{episode_offset:03d}.pth"
                    import shutil
                    shutil.copy2(img_file, final_output_dir / "obses" / new_name)
                    episode_offset += 1
    
    print(f"Combined data shapes: states {final_states.shape}, actions {final_actions.shape}")

def get_maze_spec(maze_type):
    """Get maze specification from string."""
    maze_map = {
        'u_maze': U_MAZE,
        'medium_maze': MEDIUM_MAZE,
        'large_maze': LARGE_MAZE,
    }
    return maze_map.get(maze_type.lower(), U_MAZE)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate point maze dataset for training world models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic parameters
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='data/point_maze',
        help='Output directory for the dataset'
    )
    parser.add_argument(
        '--n-rollouts', 
        type=int, 
        default=100,
        help='Number of trajectories to generate'
    )
    parser.add_argument(
        '--max-timesteps', 
        type=int, 
        default=300,
        help='Maximum timesteps per trajectory'
    )
    parser.add_argument(
        '--maze-type', 
        type=str, 
        default='u_maze',
        choices=['u_maze', 'medium_maze', 'large_maze'],
        help='Type of maze to use'
    )
    parser.add_argument(
        '--action-scale', 
        type=float, 
        default=1.0,
        help='Scale factor for actions'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    
    # Policy and behavior
    parser.add_argument(
        '--policy-type', 
        type=str, 
        default='random',
        choices=['random', 'goal_oriented', 'expert'],
        help='Type of policy to use for data generation'
    )
    
    # Memory efficiency options
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=10,
        help='Number of rollouts to process before saving (memory efficiency)'
    )
    parser.add_argument(
        '--save-frequency', 
        type=int, 
        default=50,
        help='Save intermediate results every N rollouts'
    )
    parser.add_argument(
        '--compress-images', 
        action='store_true',
        help='Use compressed format for images (saves disk space)'
    )
    parser.add_argument(
        '--no-save-images', 
        action='store_true',
        help='Skip saving visual observations (saves disk space)'
    )
    
    # Multiprocessing options
    parser.add_argument(
        '--use-multiprocessing', 
        action='store_true',
        help='Use multiprocessing for large datasets'
    )
    parser.add_argument(
        '--n-processes', 
        type=int, 
        default=4,
        help='Number of processes to use for multiprocessing'
    )
    parser.add_argument(
        '--rollouts-per-process', 
        type=int, 
        default=None,
        help='Number of rollouts per process (auto-calculated if not specified)'
    )
    
    # Additional options
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Print configuration without running generation'
    )
    
    return parser.parse_args()

def main():
    """Main function to run the data generation."""
    args = parse_args()
    
    # Print configuration
    print("Point Maze Data Generation Configuration:")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Number of rollouts: {args.n_rollouts}")
    print(f"  Max timesteps: {args.max_timesteps}")
    print(f"  Maze type: {args.maze_type}")
    print(f"  Policy type: {args.policy_type}")
    print(f"  Action scale: {args.action_scale}")
    print(f"  Seed: {args.seed}")
    print(f"  Save images: {not args.no_save_images}")
    print(f"  Compress images: {args.compress_images}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Save frequency: {args.save_frequency}")
    
    if args.use_multiprocessing:
        print(f"  Using multiprocessing: {args.n_processes} processes")
        if args.rollouts_per_process:
            print(f"  Rollouts per process: {args.rollouts_per_process}")
        else:
            rollouts_per_process = args.n_rollouts // args.n_processes
            print(f"  Rollouts per process: {rollouts_per_process} (auto-calculated)")
    
    if args.dry_run:
        print("\nDry run - exiting without generating data")
        return
    
    # Get maze specification
    maze_spec = get_maze_spec(args.maze_type)
    
    # Calculate rollouts per process for multiprocessing
    if args.use_multiprocessing and args.rollouts_per_process is None:
        args.rollouts_per_process = args.n_rollouts // args.n_processes
    
    # Generate data
    if args.use_multiprocessing:
        generate_large_dataset_with_multiprocessing(
            output_dir=args.output_dir,
            n_rollouts=args.n_rollouts,
            max_timesteps=args.max_timesteps,
            maze_spec=maze_spec,
            n_processes=args.n_processes,
            rollouts_per_process=args.rollouts_per_process,
            action_scale=args.action_scale,
            seed=args.seed,
            save_images=not args.no_save_images,
            policy_type=args.policy_type,
            batch_size=args.batch_size,
            save_frequency=args.save_frequency,
            compress_images=args.compress_images,
        )
    else:
        generate_point_maze_data_memory_efficient(
            output_dir=args.output_dir,
            n_rollouts=args.n_rollouts,
            max_timesteps=args.max_timesteps,
            maze_spec=maze_spec,
            action_scale=args.action_scale,
            seed=args.seed,
            save_images=not args.no_save_images,
            policy_type=args.policy_type,
            batch_size=args.batch_size,
            save_frequency=args.save_frequency,
            compress_images=args.compress_images,
        )
    
    print(f"\nData generation completed successfully!")
    print(f"Dataset saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
