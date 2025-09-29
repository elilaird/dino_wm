import torch
import torch.nn as nn

class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN) module.

    This module extends standard Layer Normalization by dynamically generating
    the scaling (γ) and shifting (β) parameters based on a conditioning input.
    This allows the model to adapt its normalization process to varying inputs.

    Args:
        normalized_shape (int or tuple): The shape of the input tensor to normalize.
            If int, it's treated as the last dimension.
        cond_dim (int): The dimension of the conditioning input.
        eps (float): A small value added to the denominator for numerical stability.
        elementwise_affine (bool): If True, applies learnable affine transformation.
            If False, only uses adaptive parameters from conditioning input.
        bias (bool): If True, includes bias in the adaptive parameters.

    Shape:
        - Input: (N, ..., L) where L is normalized_shape
        - Conditioning: (N, cond_dim) or (N, ..., cond_dim)
        - Output: (N, ..., L) same shape as input

    Example:
        >>> ada_ln = AdaptiveLayerNorm(64, 128)
        >>> x = torch.randn(32, 10, 64)  # (batch, seq_len, hidden_dim)
        >>> cond = torch.randn(32, 128)  # (batch, cond_dim)
        >>> output = ada_ln(x, cond)
    """

    def __init__(
        self,
        normalized_shape,
        cond_dim,
        eps=1e-5,
        elementwise_affine=True,
        bias=True,
        zero_init=False,
    ):
        super(AdaptiveLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.cond_dim = cond_dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape

        # Adaptive parameter generators
        self.ada_gamma = nn.Linear(cond_dim, normalized_shape[0], bias=bias)
        self.ada_beta = nn.Linear(cond_dim, normalized_shape[0], bias=bias)

        if zero_init:
            nn.init.zeros_(self.ada_gamma.weight)
            nn.init.zeros_(self.ada_beta.weight)
            if bias:
                nn.init.zeros_(self.ada_gamma.bias)
                nn.init.zeros_(self.ada_beta.bias)

        # Optional learnable affine parameters (like standard LayerNorm)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias_param = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias_param", None)

    def forward(self, x, cond_input):
        """
        Forward pass of adaptive layer normalization.

        Args:
            x (torch.Tensor): Input tensor to normalize
            cond_input (torch.Tensor): Conditioning input for adaptive parameters

        Returns:
            torch.Tensor: Normalized and adaptively scaled/shifted tensor
        """
        # Compute mean and variance over the last len(normalized_shape) dimensions
        mean = x.mean(
            dim=tuple(range(-len(self.normalized_shape), 0)), keepdim=True
        )
        var = x.var(
            dim=tuple(range(-len(self.normalized_shape), 0)),
            keepdim=True,
            unbiased=False,
        )

        # Normalize input
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Generate adaptive parameters from conditioning input
        # Handle different conditioning input shapes
        if cond_input.dim() == 2:
            # (batch, cond_dim) -> (batch, 1, ..., 1, normalized_shape[0])
            gamma = self.ada_gamma(cond_input)
            beta = self.ada_beta(cond_input)
            # Expand to match input dimensions
            for _ in range(x.dim() - 2):
                gamma = gamma.unsqueeze(-2)
                beta = beta.unsqueeze(-2)
        else:
            # (batch, ..., cond_dim) -> (batch, ..., normalized_shape[0])
            gamma = self.ada_gamma(cond_input)
            beta = self.ada_beta(cond_input)

        # Apply adaptive scaling and shifting
        output = gamma * x_norm + beta

        # Apply optional learnable affine transformation
        if self.elementwise_affine:
            output = output * self.weight + self.bias_param

        return output
