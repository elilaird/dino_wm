import torch

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def generate_block_causal_mask(num_frames, num_patches, device):
    """
    Returns a mask of shape (T*P, T*P)
    """
    # 1. Time Mask: Lower Triangular (T x T)
    # Frame T can see Frames 0...T
    time_mask = torch.tril(torch.ones(num_frames, num_frames, device=device))

    # 2. Spatial Mask: All Ones (P x P)
    # All patches in a frame can see each other
    space_mask = torch.ones(num_patches, num_patches, device=device)

    # 3. Kronecker Product -> Block Causal
    mask = torch.kron(time_mask, space_mask)

    # 4. Convert to Additive Mask (0.0 for keep, -inf for discard)
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float("-inf"))
    mask = mask.masked_fill(mask == 1, float(0.0))

    return mask

def generate_block_diagonal_mask(num_frames, num_patches, device):
    time_mask = torch.eye(num_frames, device=device)

    space_mask = torch.ones(num_patches, num_patches, device=device)

    mask = torch.kron(time_mask, space_mask)
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float("-inf"))
    mask = mask.masked_fill(mask == 1, float(0.0))

    return mask


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
    - Persistent tokens: attend to main sequence frames only
    - Memory frames: attend to main sequence frames only
    - Main blocks: attend to persistent tokens + memory tokens at same time step + previous frames
    """
    total_frames = n_persistent + n_retrieved + nwindow

    # Create blocks for each frame type
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)

    rows = []

    # Persistent token rows (attend to main sequence frames only)
    for i in range(n_persistent):
        # No attention to other persistent tokens
        persistent_blocks = [zeros] * n_persistent
        
        # No attention to memory tokens
        memory_blocks = [zeros] * n_retrieved
        
        # Allow attention to main blocks at current and previous time steps
        main_blocks = [ones] * (i + 1) + [zeros] * (nwindow - i - 1)
        
        row = torch.cat(persistent_blocks + memory_blocks + main_blocks, dim=1)
        rows.append(row)

    # Memory frame rows (attend to main sequence frames only)
    for i in range(n_retrieved):
        # No attention to persistent tokens
        persistent_blocks = [zeros] * n_persistent
        
        # No attention to other memory tokens
        memory_blocks = [zeros] * n_retrieved
        
        # Allow attention to main blocks at current and previous time steps
        main_blocks = [ones] * (i + 1) + [zeros] * (nwindow - i - 1)
        
        row = torch.cat(persistent_blocks + memory_blocks + main_blocks, dim=1)
        rows.append(row)

    # Main sequence rows (attend to persistent tokens + memory tokens at same time step + previous frames)
    for i in range(nwindow):
        # Allow attention to persistent tokens
        persistent_blocks = [ones] * n_persistent

        # Allow attention to memory tokens at same time step (if memory frame exists for this timestep)
        if i < n_retrieved:
            memory_blocks = [zeros] * i + [ones] + [zeros] * (n_retrieved - i - 1)
        else:
            memory_blocks = [zeros] * n_retrieved

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
        mask: [1, 1, num_frames * num_patches, num_frames * num_patches] boolean mask
    """
    zeros = torch.zeros(num_patches, num_patches, device=device, dtype=torch.bool)
    ones = torch.ones(num_patches, num_patches, device=device, dtype=torch.bool)
    rows = []
    
    for i in range(num_frames):
        row = torch.cat([zeros] * i + [ones] + [zeros] * (num_frames - i - 1), dim=1)
        rows.append(row)
    
    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask

def generate_full_mask(num_patches, num_frames, device=None, dtype=None):

    ones = torch.ones(num_patches, num_patches, device=device, dtype=torch.bool)
    rows = []
    for i in range(num_frames):
        rows.append(torch.cat([ones] * num_frames, dim=1))
    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask

def generate_frame_mask_with_memory(num_patches, num_frames, n_memory, device=None, dtype=None):
    """
    Generate a frame mask that allows all patches in all frames to attend to the n_memory tokens
    prepended at the beginning.
    
    Args:
        num_patches: Number of patches per frame
        num_frames: Number of frames
        n_memory: Number of memory tokens prepended at the beginning
        device: Device for the tensor
        dtype: Data type for the tensor
    
    Returns:
        mask: [1, 1, n_memory + num_frames * num_patches, n_memory + num_frames * num_patches] boolean mask
    """
    total_tokens = n_memory + num_frames * num_patches
    
    # Create memory tokens that can attend to themselves and all other tokens
    memory_zeros = torch.zeros(n_memory, n_memory, device=device, dtype=torch.bool)
    memory_ones = torch.ones(n_memory, n_memory, device=device, dtype=torch.bool)
    
    # Memory tokens attend to all tokens
    memory_to_all = torch.ones(n_memory, num_frames * num_patches, device=device, dtype=torch.bool)
    
    # Create frame patches that can attend to memory tokens and themselves
    frame_zeros = torch.zeros(num_patches, num_patches, device=device, dtype=torch.bool)
    frame_ones = torch.ones(num_patches, num_patches, device=device, dtype=torch.bool)
    
    rows = []
    
    # Memory token rows - can attend to everything
    for i in range(n_memory):
        memory_to_memory = memory_ones[i:i+1, :]
        memory_to_frames = memory_to_all[i:i+1, :]
        row = torch.cat([memory_to_memory, memory_to_frames], dim=1)
        rows.append(row)
    
    # Frame patch rows - can attend to memory tokens and past/current frames
    for i in range(num_frames):
        for j in range(num_patches):
            # Can attend to all memory tokens
            memory_attention = torch.ones(1, n_memory, device=device, dtype=torch.bool)
            
            # Can attend to patches in current and past frames (temporal causality)
            frame_attention = torch.zeros(1, num_frames * num_patches, device=device, dtype=torch.bool)
            for k in range(i + 1):  # Attend to frames 0 through i
                start_idx = k * num_patches
                end_idx = (k + 1) * num_patches
                frame_attention[0, start_idx:end_idx] = 1
            
            row = torch.cat([memory_attention, frame_attention], dim=1)
            rows.append(row)
    
    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask
