"""
Scaled MSE loss for Transferable Tactile Transformer (T3) pre-training

Author: Jialiang (Alan) Zhao
Email: alanzhao@csail.mit.edu
MIT License
"""

import torch
import torch.nn as nn

class VarianceScaledLoss(nn.Module):
    def __init__(self, scale=5.):
        super().__init__()
        self.scale = scale
        self.mse = nn.MSELoss()

    def forward(self, pred, Y):
        mse_loss = self.mse(pred, Y)
        return torch.sqrt(mse_loss) / self.scale