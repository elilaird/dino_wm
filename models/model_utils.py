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
    Generate frame-level mask for MAC transformer using the same pattern as generate_mask_matrix
    but accounting for persistent tokens and memory frames.
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

    # Memory frame rows (can attend to everything)
    for i in range(n_retrieved):
        row = torch.cat([ones] * total_frames, dim=1)
        rows.append(row)

    # Main sequence rows (frame-level causality + access to persistent/memory)
    for i in range(nwindow):
        # Allow attention to persistent tokens (all frames can attend to persistent tokens)
        persistent_blocks = [ones] * n_persistent

        # Allow attention to memory frames (all frames can attend to memory frames)
        memory_blocks = [ones] * n_retrieved

        # Allow attention to current and previous frames in main sequence (frame-level causality)
        main_blocks = [ones] * (i + 1) + [zeros] * (nwindow - i - 1)

        row = torch.cat(persistent_blocks + memory_blocks + main_blocks, dim=1)
        rows.append(row)

    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask
