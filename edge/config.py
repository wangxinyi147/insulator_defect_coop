import os

class EdgeConfig:
    # 模型路径（从云端同步）
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "current.onnx")
    # 缓存目录
    CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
    # 模型检查间隔（秒），用于热加载
    MODEL_CHECK_INTERVAL = 10
    # 推理参数
    CONF_THRESH = 0.25
    IOU_THRESH = 0.45
    IMG_SIZE = 640
    # 小目标优化（多尺度）
    MULTI_SCALE = True
    # 类别映射（与云端一致）
    CLASS_NAMES = {
        0: "闪络",
        1: "绝缘子",
        2: "掉片",
        3: "破损",
    }