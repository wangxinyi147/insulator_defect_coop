# cloud/cloud_api.py
import os
import sys
import json
import shutil
from flask import Flask, request, jsonify
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from cloud.model_registry import ModelRegistry

app = Flask(__name__)

# 配置
BASE_DIR = Path(__file__).parent
FEEDBACK_DIR = BASE_DIR / "feedback_data"
INCREMENTAL_IMAGES_DIR = BASE_DIR / "data" / "images" / "train"
INCREMENTAL_LABELS_DIR = BASE_DIR / "data" / "labels" / "train"
EDGE_MODELS_DIR = BASE_DIR.parent / "edge" / "models"
os.makedirs(FEEDBACK_DIR, exist_ok=True)
os.makedirs(INCREMENTAL_IMAGES_DIR, exist_ok=True)
os.makedirs(INCREMENTAL_LABELS_DIR, exist_ok=True)

# 模型注册表
registry = ModelRegistry()

def convert_boxes_to_yolo(boxes, img_width, img_height):
    """将绝对坐标框转换为YOLO归一化格式"""
    yolo_lines = []
    for box in boxes:
        cls_id = box["cls"]
        x1, y1, x2, y2 = box["xyxy"]
        x_center = (x1 + x2) / 2.0 / img_width
        y_center = (y1 + y2) / 2.0 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    return yolo_lines

@app.route('/upload', methods=['POST'])
def upload_feedback():
    data = request.get_json()
    if not isinstance(data, list):
        data = [data]
    saved_count = 0
    for item in data:
        try:
            img_path = Path(item["image_path"])
            feedback_type = item["feedback_type"]
            boxes = item["boxes"]
            # 复制图像到增量训练目录
            if not img_path.exists():
                continue
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            new_img_name = f"feedback_{timestamp}.jpg"
            new_img_path = INCREMENTAL_IMAGES_DIR / new_img_name
            shutil.copy(img_path, new_img_path)
            # 获取图像尺寸（从图像文件读取）
            import cv2
            img = cv2.imread(str(new_img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            # 根据反馈类型生成标注
            if feedback_type == "false_positive":
                # 误检：不应有框，生成空标注文件
                label_lines = []
            else:  # false_negative
                # 漏检：需要人工补充真实框，这里简单使用原框（实际应让用户标注）
                # 为演示，我们假设原框是正确的（实际需要人工修正）
                label_lines = convert_boxes_to_yolo(boxes, w, h)
            # 保存标注文件
            label_path = INCREMENTAL_LABELS_DIR / (new_img_name.replace('.jpg', '.txt'))
            with open(label_path, 'w') as f:
                f.writelines(label_lines)
            saved_count += 1
        except Exception as e:
            print(f"处理反馈失败: {e}")
    return jsonify({"status": "ok", "saved": saved_count})


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    onnx_exists = (EDGE_MODELS_DIR / "current.onnx").exists()
    edge_models = list(EDGE_MODELS_DIR.glob("current.*")) if EDGE_MODELS_DIR.exists() else []
    return jsonify({
        "status": "healthy",
        "edge_model_ready": onnx_exists,
        "edge_models_count": len(edge_models),
        "timestamp": datetime.now().isoformat()
    })


@app.route('/models', methods=['GET'])
def list_models():
    """列出所有训练模型"""
    models = registry.list_all(limit=20)
    return jsonify({"models": models, "count": len(models)})


@app.route('/models/current', methods=['GET'])
def current_model():
    """获取当前激活模型信息"""
    active = registry.get_active()
    if active is None:
        return jsonify({"error": "无激活模型"}), 404
    return jsonify(active)


@app.route('/models/activate', methods=['POST'])
def activate_model():
    """切换激活模型"""
    data = request.get_json()
    model_id = data.get("model_id")
    if model_id is None:
        return jsonify({"error": "缺少 model_id"}), 400
    registry.promote(model_id)
    return jsonify({"status": "ok", "activated": model_id})


@app.route('/feedback/stats', methods=['GET'])
def feedback_stats():
    """反馈统计"""
    summary = registry.get_feedback_summary(days=30)
    total = sum(s["total_received"] for s in summary)
    return jsonify({
        "total_30d": total,
        "daily_breakdown": summary
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)