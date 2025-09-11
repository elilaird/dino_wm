#!/usr/bin/env python3
"""
Script to chunk memory maze data from individual npz files into memory-mappable npy files.

This script takes individual episode npz files and chunks them into larger npy files
that can be efficiently memory-mapped by the MiniGridMemmapDataset.

Usage:
    export DATASET_DIR=/path/to/dataset/root
    python chunk_memory_maze_data.py --input_dir memory_maze/raw --output_dir memory_maze/chunked --episodes_per_chunk 100
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
import glob
import os


def get_episode_files(input_dir: Path) -> List[Path]:
    """Get all episode npz files sorted by episode number."""
    pattern = str(input_dir / "episode-*.npz")
    files = glob.glob(pattern)
    files.sort(key=lambda x: int(Path(x).stem.split('-')[1]))
    return [Path(f) for f in files]


def load_episode_data(episode_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load image and action data from a single episode npz file."""
    with np.load(episode_path) as data:
        images = data['image']  # Shape: (501, 64, 64, 3)
        proprios = data['proprio']  # Shape: (501, 4)
        actions = data['action']  # Shape: (501, 6)
    return images, proprios, actions


def chunk_episodes(
    input_dir: Path,
    output_dir: Path,
    episodes_per_chunk: int,
    max_steps: int = 501
) -> None:
    """
    Chunk individual episode npz files into larger npy files.
    
    Args:
        input_dir: Directory containing episode-*.npz files
        output_dir: Directory to save chunked npy files
        episodes_per_chunk: Number of episodes per chunk
        max_steps: Maximum number of steps per episode (for index.json)
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all episode files
    episode_files = get_episode_files(input_dir)
    total_episodes = len(episode_files)
    
    print(f"Found {total_episodes} episode files")
    print(f"Chunking into groups of {episodes_per_chunk} episodes")
    
    # Calculate number of chunks needed
    n_chunks = (total_episodes + episodes_per_chunk - 1) // episodes_per_chunk
    
    print(f"Will create {n_chunks} chunks")
    
    # Process episodes in chunks
    chunk_idx = 0
    episode_idx = 0
    
    while episode_idx < total_episodes:
        # Determine episodes for this chunk
        end_idx = min(episode_idx + episodes_per_chunk, total_episodes)
        chunk_episodes = episode_files[episode_idx:end_idx]
        actual_episodes_in_chunk = len(chunk_episodes)
        
        print(f"Processing chunk {chunk_idx}: episodes {episode_idx}-{end_idx-1} ({actual_episodes_in_chunk} episodes)")
        
        # Load first episode to get shapes
        first_images, first_proprios, first_actions = load_episode_data(chunk_episodes[0])
        seq_len, height, width, channels = first_images.shape
        action_dim = first_actions.shape[1] if first_actions.ndim > 1 else 1
        proprio_dim = first_proprios.shape[1] if first_proprios.ndim > 1 else 1
        
        # Initialize arrays for this chunk
        chunk_images = np.zeros((actual_episodes_in_chunk, seq_len, height, width, channels), dtype=np.uint8)
        chunk_actions = np.zeros((actual_episodes_in_chunk, seq_len, action_dim), dtype=np.float32)
        chunk_proprios = np.zeros((actual_episodes_in_chunk, seq_len, proprio_dim), dtype=np.float32)

        # Load all episodes in this chunk
        for i, episode_file in enumerate(chunk_episodes):
            images, proprios, actions = load_episode_data(episode_file)
            
            # Ensure consistent shapes
            if images.shape != (seq_len, height, width, channels):
                print(f"Warning: Episode {episode_file.name} has unexpected image shape {images.shape}")
                # Pad or truncate as needed
                if images.shape[0] < seq_len:
                    # Pad with zeros
                    pad_shape = (seq_len - images.shape[0], height, width, channels)
                    images = np.concatenate([images, np.zeros(pad_shape, dtype=images.dtype)], axis=0)
                else:
                    # Truncate
                    images = images[:seq_len]
            
            if actions.shape != (seq_len, action_dim):
                print(f"Warning: Episode {episode_file.name} has unexpected action shape {actions.shape}")
                # Pad or truncate as needed
                if actions.shape[0] < seq_len:
                    # Pad with zeros
                    pad_shape = (seq_len - actions.shape[0], action_dim)
                    actions = np.concatenate([actions, np.zeros(pad_shape, dtype=actions.dtype)], axis=0)
                else:
                    # Truncate
                    actions = actions[:seq_len]
            
            if proprios.shape != (seq_len, proprio_dim):
                print(f"Warning: Episode {episode_file.name} has unexpected proprio shape {proprios.shape}")
                # Pad or truncate as needed
                if proprios.shape[0] < seq_len:
                    # Pad with zeros
                    pad_shape = (seq_len - proprios.shape[0], proprio_dim)
                    proprios = np.concatenate([proprios, np.zeros(pad_shape, dtype=proprios.dtype)], axis=0)
                else:
                    # Truncate
                    proprios = proprios[:seq_len]
            
            chunk_images[i] = images
            chunk_actions[i] = actions
            chunk_proprios[i] = proprios
        
        # Save chunk files
        observations_path = output_dir / f"observations_{chunk_idx:04d}.npy"
        actions_path = output_dir / f"actions_{chunk_idx:04d}.npy"
        proprios_path = output_dir / f"proprio_{chunk_idx:04d}.npy"
        
        print(f"Saving chunk {chunk_idx}: {observations_path.name}, {actions_path.name}")
        np.save(observations_path, chunk_images)
        np.save(actions_path, chunk_actions)
        np.save(proprios_path, chunk_proprios)
        
        # Clear memory
        del chunk_images, chunk_actions, chunk_proprios
        
        chunk_idx += 1
        episode_idx = end_idx
    
    # Create index.json
    index_data = {
        "episodes_per_chunk": episodes_per_chunk,
        "total_episodes": total_episodes,
        "n_chunks": n_chunks,
        "max_steps": max_steps,
        "image_shape": [height, width, channels],
        "action_dim": action_dim,
        "proprio_dim": proprio_dim
    }
    
    index_path = output_dir / "index.json"
    with open(index_path, 'w') as f:
        json.dump(index_data, f, indent=2)
    
    print(f"Created index.json with metadata:")
    print(f"  - Total episodes: {total_episodes}")
    print(f"  - Episodes per chunk: {episodes_per_chunk}")
    print(f"  - Number of chunks: {n_chunks}")
    print(f"  - Max steps per episode: {max_steps}")
    print(f"  - Image shape: {height}x{width}x{channels}")
    print(f"  - Action dimension: {action_dim}")
    print(f"  - Proprio dimension: {proprio_dim}")
    print(f"\nChunking complete! Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Chunk memory maze npz files into npy files")
    parser.add_argument("--input-dir", type=str, required=True,
                       help="Directory containing episode-*.npz files (relative to DATASET_DIR)")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save chunked npy files (relative to DATASET_DIR)")
    parser.add_argument("--episodes-per-chunk", type=int, default=100,
                       help="Number of episodes per chunk (default: 100)")
    parser.add_argument("--max-steps", type=int, default=501,
                       help="Maximum steps per episode (default: 501)")
    
    args = parser.parse_args()
    
    # Get base dataset directory from environment variable
    dataset_dir = os.environ.get('DATASET_DIR')
    if not dataset_dir:
        raise ValueError("DATASET_DIR environment variable must be set")
    
    dataset_dir = Path(dataset_dir)
    input_dir = dataset_dir / args.input_dir
    output_dir = dataset_dir / args.output_dir
    
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")
    
    # Check if input directory contains episode files
    episode_files = get_episode_files(input_dir)
    if not episode_files:
        raise ValueError(f"No episode-*.npz files found in {input_dir}")
    
    print(f"Using DATASET_DIR: {dataset_dir}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    chunk_episodes(
        input_dir=input_dir,
        output_dir=output_dir,
        episodes_per_chunk=args.episodes_per_chunk,
        max_steps=args.max_steps
    )


if __name__ == "__main__":
    main()
