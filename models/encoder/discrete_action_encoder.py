import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


class DiscreteActionEncoder(nn.Module):
    """
    A learnable encoder for discrete actions that maps each action index to a learned embedding.
    
    Args:
        num_actions: Number of discrete actions
        emb_dim: Dimension of the action embeddings
        padding_idx: Optional padding index for variable-length sequences
        max_norm: Optional max norm for embedding weights
        dropout: Dropout rate applied to embeddings
    """
    
    def __init__(
        self,
        num_actions: int,
        emb_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        dropout: float = 0.0,
        in_chans: int = 0
    ):
        super().__init__()
        
        self.num_actions = num_actions
        self.emb_dim = emb_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.dropout = dropout
        
        # Learnable embedding table
        self.embedding = nn.Embedding(
            num_embeddings=num_actions,
            embedding_dim=emb_dim,
            padding_idx=padding_idx,
            max_norm=max_norm
        )
        
        # Optional dropout layer
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
    
    def forward(
        self, 
        actions: Union[torch.Tensor, int, list]
    ) -> torch.Tensor:
        """
        Encode discrete actions into embeddings.
        
        Args:
            actions: Action indices as tensor, int, list, or [B, T] sequence
            
        Returns:
            Action embeddings of shape [batch_size, emb_dim] or [emb_dim] for single action,
            or [B, T, embedding_dim] for sequence input
        """
        # Handle different input types
        if isinstance(actions, int):
            actions = torch.tensor([actions], device=self.embedding.weight.device)
        elif isinstance(actions, list):
            actions = torch.tensor(actions, device=self.embedding.weight.device)
        
        # Ensure actions are long/int64 for embedding lookup
        if actions.dtype != torch.long:
            actions = actions.long()
        
        # Validate action indices
        if torch.any(actions < 0) or torch.any(actions >= self.num_actions):
            raise ValueError(f"Action indices must be in range [0, {self.num_actions-1}]")
        
        if actions.dim() != 2:
            actions = actions.squeeze(-1)
        
        # Get embeddings
        embeddings = self.embedding(actions)
        
        # Apply dropout if specified
        if self.dropout_layer is not None:
            embeddings = self.dropout_layer(embeddings)
        
        return embeddings
    
    def get_embedding_weights(self) -> torch.Tensor:
        """Get the current embedding weights matrix."""
        return self.embedding.weight.clone()
    
    def set_embedding_weights(self, weights: torch.Tensor):
        """Set the embedding weights matrix."""
        if weights.shape != (self.num_actions, self.emb_dim):
            raise ValueError(f"Expected weights shape ({self.num_actions}, {self.emb_dim}), got {weights.shape}")
        with torch.no_grad():
            self.embedding.weight.copy_(weights)
    
    def get_action_embedding(self, action_idx: int) -> torch.Tensor:
        """Get embedding for a specific action index."""
        if not 0 <= action_idx < self.num_actions:
            raise ValueError(f"Action index {action_idx} out of range [0, {self.num_actions-1}]")
        return self.embedding.weight[action_idx].clone()
    
    def similarity_matrix(self) -> torch.Tensor:
        """Compute cosine similarity matrix between all action embeddings."""
        embeddings = F.normalize(self.embedding.weight, p=2, dim=1)
        return torch.mm(embeddings, embeddings.t())
    
    # def forward_sequence(self, actions: torch.Tensor) -> torch.Tensor:
    #     """
    #     Encode a sequence of discrete actions in [B, T] format to [B, T, emb_dim].
        
    #     Args:
    #         actions: Action indices of shape [B, T] where B=batch_size, T=sequence_length
            
    #     Returns:
    #         Action embeddings of shape [B, T, emb_dim]
    #     """
    #     if actions.dim() != 2:
    #         raise ValueError(f"Expected 2D input [B, T], got shape {actions.shape}")
        
    #     B, T = actions.shape
    #     # Reshape to [B*T] for embedding lookup
    #     actions_flat = actions.view(-1)
        
    #     # Get embeddings and reshape back to [B, T, emb_dim]
    #     embeddings = self.forward(actions_flat)
    #     embeddings = embeddings.view(B, T, -1)
        
    #     return embeddings
    
    def __repr__(self):
        return (f"DiscreteActionEncoder(num_actions={self.num_actions}, "
                f"emb_dim={self.emb_dim}, dropout={self.dropout})")


