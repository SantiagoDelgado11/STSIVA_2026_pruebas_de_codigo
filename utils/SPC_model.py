import torch
import torch.nn as nn
from typing import Optional

class SPCModel(nn.Module):

    def __init__(
        self,
        im_size: int,
        sampling_ratio: float,
        sampling_method: int = 180,
        
        