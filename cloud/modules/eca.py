# cloud/modules/eca.py
"""ECA (Efficient Channel Attention) - 高效通道注意力，一维卷积实现"""
import torch
import torch.nn as nn


class ECA(nn.Module):
    """高效通道注意力：用1D卷积替代全连接，参数量极小，提升闪络等细微特征响应"""
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)
