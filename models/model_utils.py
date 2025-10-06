import torch

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def generate_mask_matrix(npatch, nwindow):
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)
    rows = []
    for i in range(nwindow):
        row = torch.cat([ones] * (i + 1) + [zeros] * (nwindow - i - 1), dim=1)
        rows.append(row)
    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask


def generate_mask_with_memory(npatch, nwindow):
    """
    M_i attends to all M_j and X_j for j < i
    X_i attends to all M_j and X_j for j < i
    """
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)
    rows = []
    for i in range(nwindow):
        row = torch.cat([ones] * (i + 1) + [zeros] * (nwindow - i - 1), dim=1)
        row = torch.cat([row, row], dim=1)
        rows.append(row)
    rows += rows
    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask


def generate_sliding_window_mask(seq_len, window_size):
    """Generate mask for sliding window attention"""
    mask = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, start : i + 1] = 1
    return mask.unsqueeze(0).unsqueeze(0)


def generate_mac_mask_matrix(npatch, nwindow, n_persistent, n_retrieved):
    """
    Generate frame-level mask for MAC transformer with the following attention patterns:
    - Persistent tokens: attend to everything
    - Memory frames: attend to memory frames behind them + main blocks at timesteps behind them
    - Main blocks: frame-level causality + access to memory blocks at current and previous time frames
    """
    total_frames = n_persistent + n_retrieved + nwindow

    # Create blocks for each frame type
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)

    rows = []

    # Persistent token rows (can attend to everything)
    for i in range(n_persistent):
        row = torch.cat([ones] * total_frames, dim=1)
        rows.append(row)

    # Memory frame rows (attend to memory frames behind them + main blocks at timesteps behind them)
    for i in range(n_retrieved):
        # Allow attention to persistent tokens (all frames can attend to persistent tokens)
        persistent_blocks = [ones] * n_persistent
        
        # Allow attention to memory frames behind current memory frame
        memory_blocks = [ones] * (i + 1) + [zeros] * (n_retrieved - i - 1)
        
        # Allow attention to main blocks at timesteps behind current memory frame
        # Memory frame i corresponds to main sequence timestep i
        main_blocks = [ones] * (i + 1) + [zeros] * (nwindow - i - 1)
        
        row = torch.cat(persistent_blocks + memory_blocks + main_blocks, dim=1)
        rows.append(row)

    # Main sequence rows (frame-level causality + access to memory blocks at current and previous time frames)
    for i in range(nwindow):
        # Allow attention to persistent tokens (all frames can attend to persistent tokens)
        persistent_blocks = [ones] * n_persistent

        # Allow attention to memory blocks at current and previous time frames
        # Main sequence timestep i can attend to memory frames 0 through i
        memory_blocks = [ones] * min(i + 1, n_retrieved) + [zeros] * max(0, n_retrieved - i - 1)

        # Allow attention to current and previous frames in main sequence (frame-level causality)
        main_blocks = [ones] * (i + 1) + [zeros] * (nwindow - i - 1)

        row = torch.cat(persistent_blocks + memory_blocks + main_blocks, dim=1)
        rows.append(row)

    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask


def generate_diagonal_frame_mask(num_patches, num_frames, device=None, dtype=None):
    """
    Generate a diagonal mask for frame-level cross-attention.
    Each frame's patches can only attend to that frame's memory tokens.
    
    Args:
        num_patches: Number of patches per frame
        num_frames: Number of frames
        device: Device for the tensor
        dtype: Data type for the tensor
    
    Returns:
        mask: [num_frames * num_patches, num_frames * num_patches] boolean mask
    """
    total_tokens = num_frames * num_patches
    mask = torch.zeros(total_tokens, total_tokens, device=device, dtype=torch.bool)
    
    for frame_idx in range(num_frames):
        start_patch = frame_idx * num_patches
        end_patch = (frame_idx + 1) * num_patches
        start_mem = frame_idx * num_patches
        end_mem = (frame_idx + 1) * num_patches
        
        # Allow all patches in this frame to attend to all memory tokens in this frame
        mask[start_patch:end_patch, start_mem:end_mem] = True
    
    return mask
