# cloud/data_collector/__init__.py
"""多源数据采集模块 - 无人机 + 固定摄像机"""
from .capture import DroneCapture, FixedCameraCapture
from .normalize import normalize_image, normalize_naming, get_source_type
