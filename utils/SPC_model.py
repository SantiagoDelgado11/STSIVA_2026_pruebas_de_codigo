import torch
import torch.nn as nn
from typing import Optional

class SPCModel(nn.Module):

    def __init__(
        self,
        im_size: int,
        sampling_ratio: float,
        sampling_method: str = "cake", # "cake", "russian_dolls", "zigzag", etc.
        device: str = "cuda",
    ) -> None:
        super().__init__()

        self.im_size = im_size
        self.sampling_ratio = sampling_ratio
        self.device = device

        self.total_pixels = im_size * im_size
        self.num_measurements = max(1, int(round(self.total_pixels * sampling_ratio)))