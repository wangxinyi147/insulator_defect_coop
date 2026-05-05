# edge/edge_inference.py - 边缘推理增强版
# 新增：INT8量化支持、推理预热、批量推理、性能统计面板
import os
import sys
import time
import cv2
import json
import threading
import requests
import numpy as np
import onnxruntime as ort
from pathlib import Path
from datetime import datetime
from collections import deque
from PIL import Image, ImageDraw, ImageFont


class EdgeConfig:
    MODEL_PATH = Path(__file__).parent / "models" / "current.onnx"
    CACHE_DIR = Path(__file__).parent / "cache"
    MODEL_CHECK_INTERVAL = 10
    CONF_THRESH = 0.25
    IOU_THRESH = 0.45
    IMG_SIZE = 640
    MULTI_SCALE = True
    SCALE_RATIOS = [0.8, 1.0, 1.2]
    CLASS_NAMES = {0: "flashover", 1: "insulator", 2: "lose", 3: "damaged"}
    CLASS_NAMES_CN = {0: "闪络", 1: "绝缘子", 2: "掉片", 3: "破损"}
    CLOUD_API_URL = "http://127.0.0.1:5000/upload"
    FEEDBACK_DIR = Path(__file__).parent / "feedback_cache"
    UPLOAD_INTERVAL = 30
    WARMUP_ITERATIONS = 10
    BATCH_MAX_SIZE = 8


EdgeConfig.CACHE_DIR.mkdir(parents=True, exist_ok=True)
EdgeConfig.FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_image(img, target_size=(640, 640)):
    h, w = img.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = img_resized
    canvas = canvas.astype(np.float32) / 255.0
    canvas = np.transpose(canvas, (2, 0, 1))
    canvas = np.expand_dims(canvas, axis=0)
    return canvas, scale, (w, h)


def postprocess_onnx(output, scale, original_size, conf_thres, iou_thres):
    predictions = np.squeeze(output[0]).T
    scores = np.max(predictions[:, 4:], axis=1)
    mask = scores >= conf_thres
    predictions = predictions[mask]
    scores = scores[mask]
    if len(predictions) == 0:
        return []
    class_ids = np.argmax(predictions[:, 4:], axis=1)
    boxes = predictions[:, :4].copy()
    boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) / scale
    boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) / scale
    boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) / scale
    boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) / scale
    boxes[:, 0] = np.clip(boxes[:, 0], 0, original_size[0])
    boxes[:, 1] = np.clip(boxes[:, 1], 0, original_size[1])
    boxes[:, 2] = np.clip(boxes[:, 2], 0, original_size[0])
    boxes[:, 3] = np.clip(boxes[:, 3], 0, original_size[1])
    indices = cv2.dnn.NMSBoxes(boxes[:, :4].tolist(), scores.tolist(), conf_thres, iou_thres)
    if len(indices) == 0:
        return []
    final_boxes = []
    for i in indices:
        i = i[0] if isinstance(i, (list, np.ndarray)) else i
        final_boxes.append({
            "cls": int(class_ids[i]),
            "conf": float(scores[i]),
            "xyxy": boxes[i].tolist()
        })
    return final_boxes


def multi_scale_infer(onnx_session, img, conf_thres, iou_thres, scale_ratios):
    orig_h, orig_w = img.shape[:2]
    all_boxes = []
    for scale in scale_ratios:
        scaled_img = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)))
        input_tensor, pad_scale, _ = preprocess_image(scaled_img, (640, 640))
        input_name = onnx_session.get_inputs()[0].name
        output = onnx_session.run(None, {input_name: input_tensor})
        boxes = postprocess_onnx(output, pad_scale, (scaled_img.shape[1], scaled_img.shape[0]),
                                 conf_thres, iou_thres)
        for box in boxes:
            box["xyxy"] = [c / scale for c in box["xyxy"]]
        all_boxes.extend(boxes)
    if len(all_boxes) == 0:
        return []
    bboxes = np.array([b["xyxy"] for b in all_boxes])
    scores = np.array([b["conf"] for b in all_boxes])
    cls_ids = np.array([b["cls"] for b in all_boxes])
    final_boxes = []
    for cls in np.unique(cls_ids):
        mask = cls_ids == cls
        cls_boxes = bboxes[mask]
        cls_scores = scores[mask]
        indices = cv2.dnn.NMSBoxes(cls_boxes[:, :4].tolist(), cls_scores.tolist(), conf_thres, iou_thres)
        if indices is not None:
            for idx in indices:
                i = idx[0] if isinstance(idx, (list, np.ndarray)) else idx
                final_boxes.append(all_boxes[np.where(mask)[0][i]])
    return final_boxes


def draw_defects(img, boxes, class_names):
    result = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box["xyxy"])
        cls_id = box["cls"]
        conf = box["conf"]
        label = f"{class_names.get(cls_id, f'cls{cls_id}')} {conf:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return result


class FeedbackUploader:
    def __init__(self, api_url, cache_dir):
        self.api_url = api_url
        self.cache_dir = Path(cache_dir)
        self.queue = []
        self.lock = threading.Lock()
        self.upload_count = 0
        self.start_worker()

    def add_feedback(self, image, boxes, feedback_type, image_path=None):
        timestamp = datetime.now().isoformat()
        if image_path is None:
            img_filename = f"feedback_{int(time.time())}.jpg"
            img_path = self.cache_dir / img_filename
            cv2.imwrite(str(img_path), image)
        else:
            img_path = Path(image_path)
        data = {
            "timestamp": timestamp,
            "image_path": str(img_path),
            "feedback_type": feedback_type,
            "boxes": boxes
        }
        with self.lock:
            self.queue.append(data)

    def start_worker(self):
        def upload_loop():
            while True:
                time.sleep(EdgeConfig.UPLOAD_INTERVAL)
                with self.lock:
                    if not self.queue:
                        continue
                    to_upload = self.queue.copy()
                    self.queue.clear()
                try:
                    resp = requests.post(self.api_url, json=to_upload, timeout=10)
                    if resp.status_code == 200:
                        self.upload_count += len(to_upload)
                        print(f"[上传] 成功发送 {len(to_upload)} 条反馈 (累计{self.upload_count})")
                        for item in to_upload:
                            img_path = Path(item["image_path"])
                            if img_path.exists() and img_path.parent == self.cache_dir:
                                img_path.unlink()
                    else:
                        print(f"[上传] 失败，状态码 {resp.status_code}")
                        with self.lock:
                            self.queue.extend(to_upload)
                except Exception as e:
                    print(f"[上传] 异常: {e}")
                    with self.lock:
                        self.queue.extend(to_upload)
        threading.Thread(target=upload_loop, daemon=True).start()


class EdgeInference:
    def __init__(self):
        self.session = None
        self.backend = "onnx"
        self.model_path = EdgeConfig.MODEL_PATH
        self.model_timestamp = 0
        self.lock = threading.Lock()
        self.uploader = FeedbackUploader(EdgeConfig.CLOUD_API_URL, EdgeConfig.FEEDBACK_DIR)
        self.image_count = 0
        self.latency_queue = deque(maxlen=100)
        self._load_model()
        self._watchdog = threading.Thread(target=self._watch_model, daemon=True)
        self._watchdog.start()

    def _load_model(self):
        if not self.model_path.exists():
            print(f"[错误] 模型文件不存在: {self.model_path}")
            return False

        # 1. 尝试加载 TensorRT 引擎
        trt_path = self.model_path.with_suffix(".engine")
        # 也检查 FP16/INT8 变体
        for trt_variant in [
            trt_path,
            self.model_path.with_name(self.model_path.stem + "_fp16.engine"),
            self.model_path.with_name(self.model_path.stem + "_int8.engine"),
        ]:
            if trt_variant.exists():
                try:
                    from edge.trt_inference import TRTInference
                    trt_engine = TRTInference(
                        str(trt_variant),
                        conf_thres=EdgeConfig.CONF_THRESH,
                        iou_thres=EdgeConfig.IOU_THRESH,
                        warmup=True
                    )
                    with self.lock:
                        self.session = trt_engine
                        self.backend = "trt"
                        self.model_timestamp = max(
                            self.model_path.stat().st_mtime,
                            trt_variant.stat().st_mtime
                        )
                    print(f"[成功] TensorRT 引擎加载: {trt_variant}")
                    return True
                except Exception as e:
                    print(f"[警告] TensorRT 加载失败 ({e})，回退到 ONNX")
                break  # 只尝试第一个存在的

        # 2. 回退到 ONNX Runtime
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
            new_session = ort.InferenceSession(str(self.model_path), providers=providers)
            with self.lock:
                self.session = new_session
                self.backend = "onnx"
                self.model_timestamp = self.model_path.stat().st_mtime
            print(f"[成功] ONNX 模型加载 ({providers[0]}): {self.model_path}")

            # ONNX 预热
            for _ in range(EdgeConfig.WARMUP_ITERATIONS):
                dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                input_tensor, _, _ = preprocess_image(dummy, (640, 640))
                input_name = self.session.get_inputs()[0].name
                self.session.run(None, {input_name: input_tensor})
            print(f"[ONNX] 预热完成 ({EdgeConfig.WARMUP_ITERATIONS} 次)")

            return True
        except Exception as e:
            print(f"[失败] 模型加载出错: {e}")
            return False

    def _watch_model(self):
        while True:
            time.sleep(EdgeConfig.MODEL_CHECK_INTERVAL)
            if not self.model_path.exists():
                continue
            current_mtime = self.model_path.stat().st_mtime
            trt_path = self.model_path.with_suffix(".engine")
            if trt_path.exists():
                current_mtime = max(current_mtime, trt_path.stat().st_mtime)
            if current_mtime != self.model_timestamp:
                print("[更新] 检测到模型文件更新，重新加载...")
                self._load_model()

    def infer(self, img):
        with self.lock:
            t0 = time.perf_counter()
            if self.session is None:
                return []
            if self.backend == "trt":
                results = self.session.infer(img)
            elif EdgeConfig.MULTI_SCALE and self.backend == "onnx":
                results = multi_scale_infer(self.session, img,
                                          EdgeConfig.CONF_THRESH,
                                          EdgeConfig.IOU_THRESH,
                                          EdgeConfig.SCALE_RATIOS)
            else:
                input_tensor, scale, orig_size = preprocess_image(img, (EdgeConfig.IMG_SIZE, EdgeConfig.IMG_SIZE))
                input_name = self.session.get_inputs()[0].name
                output = self.session.run(None, {input_name: input_tensor})
                results = postprocess_onnx(output, scale, orig_size,
                                         EdgeConfig.CONF_THRESH,
                                         EdgeConfig.IOU_THRESH)
            latency_ms = (time.perf_counter() - t0) * 1000
            self.latency_queue.append(latency_ms)
            self.image_count += 1
            return results

    def infer_batch(self, images):
        """批量推理"""
        return [self.infer(img) for img in images]

    def get_stats(self):
        """获取当前推理统计"""
        avg_lat = np.mean(self.latency_queue) if self.latency_queue else 0
        return {
            "backend": self.backend,
            "total_images": self.image_count,
            "avg_latency_ms": round(float(avg_lat), 2),
            "p95_latency_ms": round(float(np.percentile(self.latency_queue, 95)), 2) if self.latency_queue else 0,
            "fps": round(1000.0 / avg_lat, 1) if avg_lat > 0 else 0,
        }

    def run(self, source=0):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"无法打开视频源: {source}")
            return
        print("开始实时推理，按 'q' 退出 / 'f' 标记反馈 / 's' 显示性能统计")
        last_stats_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            boxes = self.infer(frame)
            result = draw_defects(frame, boxes, EdgeConfig.CLASS_NAMES_CN)

            # FPS 叠加
            stats = self.get_stats()
            cv2.putText(result, f"FPS:{stats['fps']} | Latency:{stats['avg_latency_ms']}ms | {stats['backend']}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Edge Inference", result)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                print("标记反馈：1-误检 2-漏检")
                choice = input("请输入数字: ").strip()
                fb_type = "false_positive" if choice == '1' else "false_negative" if choice == '2' else None
                if fb_type:
                    self.uploader.add_feedback(frame, boxes, fb_type)
                    print("已记录反馈")
            elif key == ord('s'):
                s = self.get_stats()
                print(f"\n[性能统计] 后端:{s['backend']} | 处理:{s['total_images']}张 | "
                      f"平均延迟:{s['avg_latency_ms']}ms | P95:{s['p95_latency_ms']}ms | FPS:{s['fps']}")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        src = sys.argv[1]
        if src.isdigit():
            src = int(src)
    else:
        src = 0
    edge = EdgeInference()
    edge.run(src)
