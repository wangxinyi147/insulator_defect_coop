# cloud/modules/ghostconv.py
"""GhostConv - 轻量化卷积模块，通过廉价操作生成冗余特征"""
import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv


class GhostConv(nn.Module):
    """Ghost 卷积：用一半通道做标准卷积 + 廉价深度卷积生成另一半，减少50%参数"""
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        self.c_ghost = c2 // 2
        self.conv1 = Conv(c1, self.c_ghost, k, s, None, g, act)
        self.conv2 = Conv(self.c_ghost, self.c_ghost, 5, 1, None, self.c_ghost, act)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return torch.cat([x1, x2], dim=1)
