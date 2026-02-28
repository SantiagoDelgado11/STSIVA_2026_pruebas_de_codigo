import torch
import torch.nn as nn
from typing import Optional

# Importamos las operaciones y el archivo de utilidades numpy
from . import hadamard_ops 
from . import hadamard # Tu archivo de generación de índices 2D

class SPCModel(nn.Module):
    """SPC forward and adjoint operator using Hadamard patterns."""

    def __init__(
        self,
        im_size: int,
        compression_ratio: float,
        sampling_method: str = "random",
        mask: Optional[torch.Tensor] = None,
        apply_mask: bool = True,
        device: str = "cuda" # Recibimos el device dinámicamente
    ) -> None:
        super().__init__()

        self.im_size = im_size
        self.num_pixels = im_size * im_size 
        self.compression_ratio = compression_ratio
        self.num_patterns = self.num_pixels 
        self.apply_mask = apply_mask
        self.device = device

        n_select = max(1, int(round(self.num_patterns * compression_ratio)))

        # 1. Configurar el orden de la FWHT basado en el método de muestreo
        fwht_order = "sequency" if sampling_method == "sequency" else "natural"
        self.hadamard_op = hadamard_ops.FWHT(device=self.device, order=fwht_order)

        # 2. Construir la máscara de patrones (M)
        if mask is not None:
            if mask.numel() != self.num_patterns:
                raise ValueError("mask must have length equal to num_pixels")
            self.mask = mask.float().to(self.device).flatten()
            
        else:
            if sampling_method == "random":
                mask_full = torch.zeros(self.num_patterns, dtype=torch.float32)
                indices = torch.randperm(self.num_patterns)[:n_select]
                mask_full[indices] = 1.0
                self.mask = mask_full.to(self.device)
                
            elif sampling_method == "sequency":
                # Como FWHT está en modo sequency 1D, tomar los primeros N sí 
                # representa bajas sequencies 1D (aunque en imágenes 2D el zig-zag es mejor)
                mask_full = torch.zeros(self.num_patterns, dtype=torch.float32)
                indices = torch.arange(n_select)
                mask_full[indices] = 1.0
                self.mask = mask_full.to(self.device)
                
            elif sampling_method == "low_frequency":
                # Aquí usamos tu hadamard.py para el verdadero Zig-Zag 2D!
                # hadamard_ops opera en natural, por lo que usamos el index_matrix
                # que mapea el espacio 2D sobre el orden natural.
                index_matrix = hadamard.get_index_matrix(im_size)
                mask_np = hadamard.get_mask(index_matrix, im_size, m=n_select)
                
                # Convertimos de NumPy (CPU) a PyTorch (GPU respectiva) de forma segura
                self.mask = torch.tensor(mask_np, dtype=torch.float32, device=self.device).flatten()
                
            else:
                raise ValueError("sampling_method must be 'random', 'low_frequency', or 'sequency'")

    # ------------------------------------------------------------------
    def _mask_patterns(self, measurements: torch.Tensor) -> torch.Tensor:
        """Aplica la máscara a las mediciones de Hadamard (Operador diagonal M)."""
        if self.apply_mask:
            return measurements * self.mask.view(1, 1, -1)
        return measurements

    # ------------------------------------------------------------------
    def forward_pass(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica la transformada de Hadamard y la máscara de compresión."""
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        measurements = self.hadamard_op(x_flat) 
        return self._mask_patterns(measurements)

    # ------------------------------------------------------------------
    def transpose_pass(self, y: torch.Tensor) -> torch.Tensor:
        """Aplica el operador adjunto exacto (H^T M y)."""
        y_masked = self._mask_patterns(y)
        B, C, _ = y_masked.shape
        x_reconstructed_flat = self.hadamard_op(y_masked) 
        return x_reconstructed_flat.view(B, C, self.im_size, self.im_size)

    # ------------------------------------------------------------------
    def direct_inverse(self, y: torch.Tensor) -> torch.Tensor:
        return self.transpose_pass(y) 

    # ------------------------------------------------------------------
    def pseudoinverse_cgls(
        self,
        y: torch.Tensor,
        max_iter: int = 10,
        tol: float = 1e-6,
        x0: Optional[torch.Tensor] = None,
    ):
        pass