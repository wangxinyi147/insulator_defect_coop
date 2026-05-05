# cloud/modules/asff.py
"""ASFF (Adaptive Spatial Feature Fusion) - 自适应空间特征融合"""
import torch
import torch.nn as nn


class ASFF(nn.Module):
    """自适应空间特征融合：学习各层融合权重，增强小目标特征表达"""
    def __init__(self, level, channels, rfb=False):
        super().__init__()
        self.level = level
        self.inter_dim = channels[level]
        compress_c = max(8, self.inter_dim // 16)

        self.weight_level_0 = nn.Sequential(
            nn.Conv2d(channels[0], compress_c, 1, bias=False),
            nn.BatchNorm2d(compress_c),
            nn.ReLU(inplace=True)
        )
        self.weight_level_1 = nn.Sequential(
            nn.Conv2d(channels[1], compress_c, 1, bias=False),
            nn.BatchNorm2d(compress_c),
            nn.ReLU(inplace=True)
        )
        self.weight_level_2 = nn.Sequential(
            nn.Conv2d(channels[2], compress_c, 1, bias=False),
            nn.BatchNorm2d(compress_c),
            nn.ReLU(inplace=True)
        )
        self.weight_levels = nn.Sequential(
            nn.Conv2d(compress_c * 3, 3, 1, bias=False),
            nn.Sigmoid()
        )

    def _resize(self, x, target_h, target_w):
        if x.shape[2] != target_h or x.shape[3] != target_w:
            return nn.functional.interpolate(x, size=(target_h, target_w), mode='nearest')
        return x

    def forward(self, x_level_0, x_level_1, x_level_2):
        _, _, h, w = x_level_0.shape

        w0 = self.weight_level_0(x_level_0)
        w1 = self.weight_level_1(self._resize(x_level_1, h, w))
        w2 = self.weight_level_2(self._resize(x_level_2, h, w))

        weights = self.weight_levels(torch.cat([w0, w1, w2], dim=1))
        alpha, beta, gamma = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]

        x1_resized = self._resize(x_level_1, h, w)
        x2_resized = self._resize(x_level_2, h, w)

        return alpha * x_level_0 + beta * x1_resized + gamma * x2_resized
