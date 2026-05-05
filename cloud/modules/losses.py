# cloud/modules/losses.py
"""改进损失函数：WIoU + Focal Loss + Slide Loss"""
import torch
import torch.nn as nn
import math


class WIoULoss(nn.Module):
    """Wise-IoU v3：动态非单调聚焦机制，更适合小目标和低质量标注"""
    def __init__(self, monotonic=False):
        super().__init__()
        self.monotonic = monotonic

    def forward(self, pred, target, iou_mean=0.5):
        """
        pred: [N, 4] 预测框 (x1, y1, x2, y2)
        target: [N, 4] 目标框 (x1, y1, x2, y2)
        """
        # 计算交集面积
        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        # 并集面积
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union_area = pred_area + target_area - inter_area

        iou = inter_area / torch.clamp(union_area, min=1e-7)

        # 最小闭包面积
        enclose_x1 = torch.min(pred[:, 0], target[:, 0])
        enclose_y1 = torch.min(pred[:, 1], target[:, 1])
        enclose_x2 = torch.max(pred[:, 2], target[:, 2])
        enclose_y2 = torch.max(pred[:, 3], target[:, 3])
        enclose_area = torch.clamp(enclose_x2 - enclose_x1, min=0) * torch.clamp(enclose_y2 - enclose_y1, min=0)

        # 中心点距离
        pred_cx = (pred[:, 0] + pred[:, 2]) / 2
        pred_cy = (pred[:, 1] + pred[:, 3]) / 2
        target_cx = (target[:, 0] + target[:, 2]) / 2
        target_cy = (target[:, 1] + target[:, 3]) / 2
        center_dist = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

        # 动态离群度 r
        r = (iou_mean - iou) / torch.clamp(iou_mean, min=1e-7)
        beta = r / (1 + r) if not self.monotonic else r

        # WIoU v3 损失
        loss_iou = 1 - iou
        loss_dist = center_dist / torch.clamp(enclose_area, min=1e-7)
        wiou = loss_iou + loss_dist

        return (beta.detach() * wiou).mean()


class FocalLoss(nn.Module):
    """Focal Loss：解决类别不平衡，针对性提升闪络类别权重"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        """
        pred: [N, C] 预测logits
        target: [N] 目标类别索引
        """
        ce_loss = nn.functional.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()


class SlideLoss(nn.Module):
    """Slide Loss：自适应阈值聚焦小目标/难样本"""
    def __init__(self, mu=0.5):
        super().__init__()
        self.mu = mu

    def forward(self, pred, target, iou_thresholds=None):
        """
        对IoU处于 μ 附近的样本施加更高权重，使模型更关注模糊边界（小目标通常处于此范围）
        """
        if iou_thresholds is None:
            return nn.functional.binary_cross_entropy_with_logits(pred, target)

        with torch.no_grad():
            # 计算每个预测的IoU得分
            weights = torch.ones_like(target)
            for i, iou in enumerate(iou_thresholds):
                mask = (iou > self.mu - 0.2) & (iou < self.mu + 0.2)
                weights[mask] = 2.0  # 加权难样本

        bce = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
        return (weights * bce).mean()
