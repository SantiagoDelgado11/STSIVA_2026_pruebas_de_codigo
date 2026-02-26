import torch
import torch.nn as nn
from typing import Optional

# Asumimos que tienes un módulo que implementa la Transformada Rápida de Hadamard
from . import hadamard_ops 

class SPCModel(nn.Module):
    """SPC forward and adjoint operator using Hadamard patterns."""

    def __init__(
        self,
        im_size: int,
        compression_ratio: float, # Reemplaza sampling_ratio
        sampling_method: str = "random", # 'random' o 'low_frequency'
        mask: Optional[torch.Tensor] = None,
        apply_mask: bool = True,
    ) -> None:
        super().__init__()

        self.im_size = im_size
        self.num_pixels = im_size * im_size # N total de pixeles
        self.compression_ratio = compression_ratio
        
        # El número total de patrones en Hadamard completo es igual al número de píxeles
        self.num_patterns = self.num_pixels 

        # Construir la máscara de patrones (M)
        if mask is not None:
            if mask.numel() != self.num_patterns:
                raise ValueError("mask must have length equal to num_pixels")
            self.mask = mask.float().to(device="cuda")
            self.apply_mask = apply_mask
        else:
            n_select = max(1, int(round(self.num_patterns * compression_ratio)))
            mask_full = torch.zeros(self.num_patterns, dtype=torch.float32)
            
            if sampling_method in ["random"]:
                # Selecciona patrones aleatorios (muy común en compressive sensing)
                indices = torch.randperm(self.num_patterns)[:n_select]
            elif sampling_method in ["low_frequency", "sequency"]:
                # Selecciona los primeros N patrones (los de menor frecuencia/sequency)
                indices = torch.arange(n_select)
            else:
                raise ValueError("sampling_method must be 'random' or 'low_frequency'")
            
            mask_full[indices] = 1.0
            self.mask = mask_full.to(device="cuda")
            self.apply_mask = apply_mask

        # Operadores Forward (H) y Adjoint (H^T) de Hadamard
        # En la práctica, FWHT (Fast Walsh-Hadamard Transform) hace ambas cosas
        self.hadamard_op = hadamard_ops.FWHT(device="cuda")

    # ------------------------------------------------------------------
    def _mask_patterns(self, measurements: torch.Tensor) -> torch.Tensor:
        """Aplica la máscara a las mediciones de Hadamard (Operador diagonal M)."""
        if self.apply_mask:
            # Multiplica las mediciones por 0 en los patrones no muestreados
            return measurements * self.mask.view(1, 1, -1).to(device=measurements.device)
        return measurements

    # ------------------------------------------------------------------
    def forward_pass(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica la transformada de Hadamard y la máscara de compresión."""
        # x suele venir como (B, C, H, W). Lo aplanamos para Hadamard (B, C, N)
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        
        # y = H x
        measurements = self.hadamard_op(x_flat) 
        
        # y_masked = M H x
        return self._mask_patterns(measurements)

    # ------------------------------------------------------------------
    def transpose_pass(self, y: torch.Tensor) -> torch.Tensor:
        """
        Aplica el operador adjunto exacto (H^T M y).
        Como Hadamard es ortogonal, aplicar H nuevamente actúa como la inversa/transpuesta
        (asumiendo la normalización correcta en hadamard_op).
        """
        y_masked = self._mask_patterns(y)
        B, C, _ = y_masked.shape
        
        # x' = H^T (M y)
        x_reconstructed_flat = self.hadamard_op(y_masked) 
        
        # Devolver a formato imagen 2D
        return x_reconstructed_flat.view(B, C, self.im_size, self.im_size)

    # ------------------------------------------------------------------
    def direct_inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Equivalente al FBP de CT. Transpuesta directa con normalización."""
        # Si la compresión no es del 100%, esto generará artefactos de 'aliasing',
        # al igual que el FBP en CT con pocos ángulos.
        return self.transpose_pass(y) # Con Hadamard, transpuesta = inversa.

    # ------------------------------------------------------------------
    def pseudoinverse_cgls(
        self,
        y: torch.Tensor,
        max_iter: int = 10,
        tol: float = 1e-6,
        x0: Optional[torch.Tensor] = None,
    ):
        """
        Resuelve x ≈ argmin ||M H x - M y||_2^2 vía CGLS.
        El solver de CGLS recibe forward_pass y transpose_pass para iterar.
        """
        B, C = y.shape[:2]
        x_shape = (B, C, self.im_size, self.im_size)

        if x0 is None:
            # Inicialización rápida con la inversa directa
            x0 = self.direct_inverse(y)
            
        # Aquí llamarías a tu solver cgls genérico pasándole 
        # self.forward_pass y self.transpose_pass en lugar de radon_op
        # x, history = cgls_solver(...) 
        # return x
        pass
        
        