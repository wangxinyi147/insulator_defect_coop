# cloud/modules/ca.py
"""CA (Coordinate Attention) - 坐标注意力模块"""
import torch
import torch.nn as nn


class CA(nn.Module):
    """坐标注意力：沿水平和垂直方向编码位置信息，增强小目标定位能力"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool_x = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, None))
        self.conv = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.bn = nn.BatchNorm2d(channel // reduction)
        self.act = nn.Hardswish()
        self.conv_x = nn.Conv2d(channel // reduction, channel, 1, bias=False)
        self.conv_y = nn.Conv2d(channel // reduction, channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)
        x_cat = self.act(self.bn(self.conv(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(x_cat, [h, w], dim=2)
        x_h = x_h.permute(0, 1, 3, 2)
        out = x * self.sigmoid(self.conv_x(x_h)) * self.sigmoid(self.conv_y(x_w))
        return out
