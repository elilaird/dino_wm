import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional
from einops import rearrange, repeat
import math

class StateSpaceModel(nn.Module):
    """
    State Space Model predictor that can replace ViT in the visual world model.
    Uses a linear state space model with learnable parameters.
    """
    def __init__(self, *, num_patches, num_frames, dim, depth, heads, mlp_dim, 
                 state_dim=None, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.pool = pool
        
        # State space parameters
        if state_dim is None:
            state_dim = dim
        self.state_dim = state_dim
        
        # Position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # State space model components
        self.state_projection = nn.Linear(dim, state_dim)
        self.output_projection = nn.Linear(state_dim, dim)
        
        # Learnable state transition matrix (A) and input matrix (B)
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.1)
        self.B = nn.Parameter(torch.randn(state_dim, dim) * 0.1)
        
        # Initial state
        self.initial_state = nn.Parameter(torch.randn(1, 1, state_dim) * 0.1)
        
        # Optional: Add some non-linearity to the state transition
        self.state_norm = nn.LayerNorm(state_dim)
        self.state_activation = nn.GELU()
        
        # Optional: Add residual connections and normalization
        self.input_norm = nn.LayerNorm(dim)
        self.output_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        """
        x: (b, num_frames * num_patches, dim)
        """
        b, n, _ = x.shape
        
        # Add position embeddings
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        
        # Reshape to (batch, time, patches, dim)
        x = rearrange(x, 'b (t p) d -> b t p d', t=self.num_frames, p=self.num_patches)
        
        # Initialize state
        state = self.initial_state.repeat(b, 1, 1)  # (b, 1, state_dim)
        
        # Process each time step
        outputs = []
        for t in range(self.num_frames):
            # Current input
            current_input = x[:, t, :, :]  # (b, patches, dim)
            
            # Project input to state space
            input_projected = self.state_projection(current_input)  # (b, patches, state_dim)
            
            # State transition: state_{t+1} = A * state_t + B * input_t
            # We process each patch independently
            batch_size, num_patches, _ = input_projected.shape
            
            # Reshape for batch processing
            state_expanded = state.unsqueeze(2).expand(-1, -1, num_patches, -1)  # (b, 1, patches, state_dim)
            state_expanded = state_expanded.squeeze(1)  # (b, patches, state_dim)
            
            # State transition
            state_next = torch.einsum('sd,bps->bpd', self.A, state_expanded) + \
                        torch.einsum('sd,bps->bpd', self.B, input_projected)
            
            # Apply non-linearity and normalization
            state_next = self.state_norm(state_next)
            state_next = self.state_activation(state_next)
            
            # Project back to output dimension
            output = self.output_projection(state_next)  # (b, patches, dim)
            output = self.output_norm(output)
            
            # Store output
            outputs.append(output)
            
            # Update state for next iteration
            state = state_next.mean(dim=1, keepdim=True)  # Average across patches
        
        # Concatenate outputs
        output = torch.stack(outputs, dim=1)  # (b, time, patches, dim)
        
        # Reshape back to original format
        output = rearrange(output, 'b t p d -> b (t p) d')
        
        return output


class LinearStateSpaceModel(nn.Module):
    """
    Simplified linear state space model for comparison.
    """
    def __init__(self, *, num_patches, num_frames, dim, depth, heads, mlp_dim, 
                 state_dim=None, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.dim = dim
        self.pool = pool
        
        # State space parameters
        if state_dim is None:
            state_dim = dim
        self.state_dim = state_dim
        
        # Position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # Linear state space model
        self.input_projection = nn.Linear(dim, state_dim)
        self.output_projection = nn.Linear(state_dim, dim)
        
        # State transition matrix (A) - constrained to be stable
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        
        # Input matrix (B)
        self.B = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        
        # Initial state
        self.initial_state = nn.Parameter(torch.randn(1, state_dim) * 0.01)
        
    def forward(self, x):
        """
        x: (b, num_frames * num_patches, dim)
        """
        b, n, _ = x.shape
        
        # Add position embeddings
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        
        # Reshape to (batch, time, patches, dim)
        x = rearrange(x, 'b (t p) d -> b t p d', t=self.num_frames, p=self.num_patches)
        
        # Initialize state for each batch
        state = self.initial_state.unsqueeze(0).expand(b, -1)  # (b, state_dim)
        
        # Process each time step
        outputs = []
        for t in range(self.num_frames):
            # Current input
            current_input = x[:, t, :, :]  # (b, patches, dim)
            
            # Project input
            input_projected = self.input_projection(current_input)  # (b, patches, state_dim)
            
            # State transition for each patch
            batch_size, num_patches, _ = input_projected.shape
            state_expanded = state.unsqueeze(1).expand(-1, num_patches, -1)  # (b, patches, state_dim)
            
            # Linear state space update
            state_next = torch.einsum('sd,bps->bpd', self.A, state_expanded) + \
                        torch.einsum('sd,bps->bpd', self.B, input_projected)
            
            # Project back to output dimension
            output = self.output_projection(state_next)  # (b, patches, dim)
            
            # Store output
            outputs.append(output)
            
            # Update state (average across patches)
            state = state_next.mean(dim=1)  # (b, state_dim)
        
        # Concatenate outputs
        output = torch.stack(outputs, dim=1)  # (b, time, patches, dim)
        
        # Reshape back to original format
        output = rearrange(output, 'b t p d -> b (t p) d')
        
        return output


class RecurrentStateSpaceModel(nn.Module):
    """
    Recurrent state space model with GRU-like gating.
    """
    def __init__(self, *, num_patches, num_frames, dim, depth, heads, mlp_dim, 
                 state_dim=None, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.dim = dim
        self.pool = pool
        
        # State space parameters
        if state_dim is None:
            state_dim = dim
        self.state_dim = state_dim
        
        # Position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # Recurrent state space components
        self.input_projection = nn.Linear(dim, state_dim)
        self.output_projection = nn.Linear(state_dim, dim)
        
        # Gated state transition
        self.update_gate = nn.Linear(state_dim + state_dim, state_dim)
        self.reset_gate = nn.Linear(state_dim + state_dim, state_dim)
        self.candidate_gate = nn.Linear(state_dim + state_dim, state_dim)
        
        # State transition matrices
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        
        # Initial state
        self.initial_state = nn.Parameter(torch.randn(1, state_dim) * 0.01)
        
        # Normalization
        self.state_norm = nn.LayerNorm(state_dim)
        self.output_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        """
        x: (b, num_frames * num_patches, dim)
        """
        b, n, _ = x.shape
        
        # Add position embeddings
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        
        # Reshape to (batch, time, patches, dim)
        x = rearrange(x, 'b (t p) d -> b t p d', t=self.num_frames, p=self.num_patches)
        
        # Initialize state for each batch
        state = self.initial_state.unsqueeze(0).expand(b, -1)  # (b, state_dim)
        
        # Process each time step
        outputs = []
        for t in range(self.num_frames):
            # Current input
            current_input = x[:, t, :, :]  # (b, patches, dim)
            
            # Project input
            input_projected = self.input_projection(current_input)  # (b, patches, state_dim)
            
            # State transition for each patch
            batch_size, num_patches, _ = input_projected.shape
            state_expanded = state.unsqueeze(1).expand(-1, num_patches, -1)  # (b, patches, state_dim)
            
            # Concatenate state and input
            combined = torch.cat([state_expanded, input_projected], dim=-1)  # (b, patches, 2*state_dim)
            
            # Gated update
            update = torch.sigmoid(self.update_gate(combined))
            reset = torch.sigmoid(self.reset_gate(combined))
            candidate = torch.tanh(self.candidate_gate(combined))
            
            # Linear state space component
            linear_update = torch.einsum('sd,bps->bpd', self.A, state_expanded) + \
                           torch.einsum('sd,bps->bpd', self.B, input_projected)
            
            # Combine gated and linear updates
            state_next = update * candidate + (1 - update) * linear_update
            state_next = self.state_norm(state_next)
            
            # Project back to output dimension
            output = self.output_projection(state_next)  # (b, patches, dim)
            output = self.output_norm(output)
            
            # Store output
            outputs.append(output)
            
            # Update state (average across patches)
            state = state_next.mean(dim=1)  # (b, state_dim)
        
        # Concatenate outputs
        output = torch.stack(outputs, dim=1)  # (b, time, patches, dim)
        
        # Reshape back to original format
        output = rearrange(output, 'b t p d -> b (t p) d')
        
        return output
