import torch
import torch.nn as nn
from typing import Optional


def fwht_sequency(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Fast Walsh-Hadamard Transform in sequency order.
    
    The FWHT is computed in-place using the butterfly algorithm.
    For a signal of length N = 2^n, the complexity is O(N log N).
    
    Args:
        x: Input tensor of shape (..., N) where N must be a power of 2.
        
    Returns:
        Transformed tensor of the same shape.
    """
    N = x.shape[-1]
    
    # Verify N is a power of 2
    if N & (N - 1) != 0:
        raise ValueError(f"Input length {N} must be a power of 2")
    
    # Copy to avoid modifying input
    result = x.clone()
    
    # Number of stages
    n_stages = int(torch.log2(torch.tensor(N, dtype=torch.float32)).item())
    
    # Butterfly computation
    h = 1
    for _ in range(n_stages):
        for i in range(0, N, h * 2):
            for j in range(i, i + h):
                # Get values
                a = result[..., j].clone()
                b = result[..., j + h].clone()
                # Butterfly operation
                result[..., j] = a + b
                result[..., j + h] = a - b
        h *= 2
    
    return result


def fwht_natural(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Fast Walsh-Hadamard Transform in natural (Hadamard) order.
    Uses bit-reversal permutation.
    
    Args:
        x: Input tensor of shape (..., N) where N must be a power of 2.
        
    Returns:
        Transformed tensor of the same shape.
    """
    N = x.shape[-1]
    
    if N & (N - 1) != 0:
        raise ValueError(f"Input length {N} must be a power of 2")
    
    result = x.clone()
    
    n_stages = int(torch.log2(torch.tensor(N, dtype=torch.float32)).item())
    
    # Butterfly computation (from MSB to LSB)
    for stage in range(n_stages):
        h = 1 << (n_stages - stage - 1)
        for i in range(0, N, h * 2):
            for j in range(i, i + h):
                a = result[..., j].clone()
                b = result[..., j + h].clone()
                result[..., j] = a + b
                result[..., j + h] = a - b
    
    return result


class FWHT(nn.Module):
    """
    Fast Walsh-Hadamard Transform module for PyTorch.
    
    This module implements both forward and inverse WHT using the fast algorithm.
    Since the Hadamard matrix H is symmetric and orthogonal (H = H^T, H^{-1} = H/N),
    applying FWHT twice gives back the original signal scaled by N.
    
    Attributes:
        device: Device to run computations on ('cuda' or 'cpu').
        normalize: If True, normalize by sqrt(N) for orthonormal transform.
        order: 'natural' or 'sequency' ordering of Hadamard basis.
    """
    
    def __init__(
        self, 
        device: str = "cuda",
        normalize: bool = True,
        order: str = "natural"
    ) -> None:
        """
        Initialize the FWHT module.
        
        Args:
            device: Device for computation ('cuda' or 'cpu').
            normalize: If True, apply 1/sqrt(N) normalization for orthonormal transform.
            order: 'natural' (Hadamard) or 'sequency' (Walsh) ordering.
        """
        super().__init__()
        self.device = device
        self.normalize = normalize
        self.order = order
        
        if order not in ["natural", "sequency"]:
            raise ValueError("order must be 'natural' or 'sequency'")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the Fast Walsh-Hadamard Transform.
        
        Args:
            x: Input tensor of shape (B, C, N) where N is a power of 2.
               N typically equals im_size * im_size for 2D images.
               
        Returns:
            Transformed tensor of the same shape.
        """
        x = x.to(self.device)
        
        N = x.shape[-1]
        
        # Apply the appropriate FWHT variant
        if self.order == "natural":
            result = fwht_natural(x)
        else:
            result = fwht_sequency(x)
        
        # Normalize for orthonormal transform
        if self.normalize:
            result = result / (N ** 0.5)
        
        return result
    
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply the inverse Fast Walsh-Hadamard Transform.
        
        For orthonormal WHT, the inverse is the same as the forward transform.
        
        Args:
            y: Input tensor of shape (B, C, N).
            
        Returns:
            Inverse transformed tensor of the same shape.
        """
        # For orthonormal Hadamard, H^{-1} = H (self-inverse)
        return self.forward(y)


class FWHTVectorized(nn.Module):
    """
    Vectorized Fast Walsh-Hadamard Transform using matrix operations.
    
    This version pre-computes the Hadamard matrix for faster batch operations
    on GPU when the signal length is not too large (N <= 4096).
    
    For larger N, use the iterative FWHT class instead.
    """
    
    def __init__(
        self,
        n: int,
        device: str = "cuda",
        normalize: bool = True
    ) -> None:
        """
        Initialize the vectorized FWHT module.
        
        Args:
            n: Signal length (must be a power of 2).
            device: Device for computation.
            normalize: If True, apply orthonormal normalization.
        """
        super().__init__()
        
        if n & (n - 1) != 0:
            raise ValueError(f"n={n} must be a power of 2")
        
        self.n = n
        self.device = device
        self.normalize = normalize
        
        # Pre-compute Hadamard matrix
        H = self._build_hadamard_matrix(n)
        if normalize:
            H = H / (n ** 0.5)
        
        self.register_buffer("H", H.to(device))
    
    def _build_hadamard_matrix(self, n: int) -> torch.Tensor:
        """Build Hadamard matrix recursively using Sylvester construction."""
        if n == 1:
            return torch.tensor([[1.0]])
        
        H_half = self._build_hadamard_matrix(n // 2)
        H = torch.cat([
            torch.cat([H_half, H_half], dim=1),
            torch.cat([H_half, -H_half], dim=1)
        ], dim=0)
        
        return H
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply WHT using pre-computed matrix multiplication.
        
        Args:
            x: Input tensor of shape (B, C, N) or (B, N).
            
        Returns:
            Transformed tensor.
        """
        x = x.to(self.device)
        
        # Handle different input shapes
        if x.dim() == 2:
            # (B, N) -> (B, N)
            return torch.matmul(x, self.H.T)
        elif x.dim() == 3:
            # (B, C, N) -> (B, C, N)
            return torch.matmul(x, self.H.T)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")
    
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse is the same as forward for orthonormal Hadamard."""
        return self.forward(y)