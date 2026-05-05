# cloud/data_collector/normalize.py
"""多源数据归一化 - 分辨率统一、色彩标准化、命名规范"""
import cv2
import numpy as np
from datetime import datetime


def normalize_image(img, target_size=None, auto_white_balance=True):
    """图像归一化：统一分辨率 + 色彩空间标准化"""
    if img is None:
        return None

    # 分辨率归一化（保持宽高比）
    if target_size is not None:
        h, w = img.shape[:2]
        tw, th = target_size
        scale = min(tw / w, th / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 色彩空间标准化
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if auto_white_balance:
        img = _simple_white_balance(img)

    return img


def _simple_white_balance(img):
    """简单的自动白平衡（灰度世界假设）"""
    b, g, r = cv2.split(img)
    b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
    gray_mean = (b_mean + g_mean + r_mean) / 3.0
    if gray_mean == 0:
        return img
    b = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    r = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.merge([b, g, r])


def normalize_naming(source_type, source_id, timestamp=None):
    """统一命名规范: {source_type}_{source_id}_{timestamp}.jpg"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_id = source_id.replace(" ", "_").replace("/", "_")
    return f"{source_type}_{safe_id}_{timestamp}.jpg"


def get_source_type(filename):
    """从文件名推断来源类型"""
    filename = filename.lower()
    if "drone" in filename or "uav" in filename or filename.endswith("d.jpg"):
        return "drone"
    elif "fixed" in filename or "camera" in filename:
        return "fixed"
    elif "h.jpg" in filename or "h_" in filename:
        return "fixed"  # 水平视角 → 固定摄像机
    elif "v.jpg" in filename or "v_" in filename:
        return "fixed"  # 垂直视角 → 固定摄像机
    elif "d.jpg" in filename or "d_" in filename:
        return "drone"
    return "unknown"
