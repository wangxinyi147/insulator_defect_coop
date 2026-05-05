# cloud/modules/lska.py
"""LSKA (Large Separable Kernel Attention) - 大核可分离注意力模块"""
import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv


class LSKA(nn.Module):
    """大核可分离注意力：通过深度可分离大卷积核增强小目标感受野"""
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // reduction, 1)
        self.conv2 = nn.Conv2d(dim // reduction, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        attn = self.act(attn)
        attn = self.conv2(attn)
        return x * attn
