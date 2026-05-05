import sys
import os
import cv2
import time
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QMessageBox, QTabWidget, QTextEdit, QCheckBox, QSlider,
                             QListWidget, QProgressBar, QGroupBox, QRadioButton)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PIL import Image, ImageDraw, ImageFont
import glob

# ==================== 配置类 ====================
class DetConfig:
    CLASS_NAMES_CN = {
        0: "闪络",
        1: "绝缘子",
        2: "掉片",
        3: "破损",
    }
    DEFECT_CLASS_IDS = [0, 2, 3]  # 0:闪络, 2:掉片, 3:破损

    if os.path.exists("C:/Windows/Fonts/simhei.ttf"):
        CHINESE_FONT_PATH = "C:/Windows/Fonts/simhei.ttf"
    elif os.path.exists("C:/Windows/Fonts/msyh.ttc"):
        CHINESE_FONT_PATH = "C:/Windows/Fonts/msyh.ttc"
    else:
        CHINESE_FONT_PATH = None

    SMALL_DEFECT_THRESHOLD = 32 * 32
    CACHE_DIR = "edge_cache"
    LABEL_FONT_SIZE = 14
    LABEL_OFFSET_Y = 30
    BATCH_SAVE_DIR = "batch_detect_results"
    VIDEO_FRAME_INTERVAL = 1
    CAMERA_INDEX = 0

# ==================== 模型类别验证工具 ====================
def check_model_classes(model_path):
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        print("=" * 50)
        print("模型类别数：", model.model.yaml.get('nc', 'unknown'))
        names = model.model.yaml.get('names', {})
        print("模型真实类别映射：", names)
        print("=" * 50)
        return {int(k): v for k, v in names.items()}
    except Exception as e:
        print(f"验证模型类别失败：{e}")
        return {}

# ==================== 工具类 ====================
class DetUtils:
    _font = None

    @staticmethod
    def get_font():
        if DetUtils._font is None:
            try:
                if DetConfig.CHINESE_FONT_PATH and os.path.exists(DetConfig.CHINESE_FONT_PATH):
                    DetUtils._font = ImageFont.truetype(DetConfig.CHINESE_FONT_PATH, DetConfig.LABEL_FONT_SIZE)
                else:
                    DetUtils._font = ImageFont.load_default()
            except:
                DetUtils._font = ImageFont.load_default()
        return DetUtils._font

    @staticmethod
    def cv2qt(img):
        h, w, c = img.shape
        bytes_per_line = 3 * w
        return QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)

    @staticmethod
    def assess_severity(total, max_conf):
        if total == 0:
            return "正常", "green"
        elif total <= 2 and max_conf < 0.5:
            return "轻微", "orange"
        elif total <= 5:
            return "中等", "darkorange"
        else:
            return "严重", "red"

    @staticmethod
    def draw_chinese_text_force(img, text, pos, color=(0, 255, 0)):
        try:
            x, y = pos
            font = DetUtils.get_font()
            temp_img = Image.new('RGB', (1, 1))
            draw_temp = ImageDraw.Draw(temp_img)
            bbox = draw_temp.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            bg_x1 = x
            bg_y1 = y - text_h - 5
            bg_x2 = x + text_w
            bg_y2 = y
            bg_x1, bg_x2 = min(bg_x1, bg_x2), max(bg_x1, bg_x2)
            bg_y1, bg_y2 = min(bg_y1, bg_y2), max(bg_y1, bg_y2)
            cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
            cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(img_pil)
            draw.text((x + 2, y - text_h - 3), text, font=font, fill=color)
            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            return img_bgr
        except Exception as e:
            print(f"❌ 强制绘制失败：{e}")
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            return img

    @staticmethod
    def preprocess_image(img, input_size=(640, 640)):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = min(input_size[0] / w, input_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        canvas[:new_h, :new_w, :] = img_resized
        canvas = canvas.astype(np.float32) / 255.0
        canvas = np.transpose(canvas, (2, 0, 1))
        canvas = np.expand_dims(canvas, axis=0)
        return canvas, scale, (w, h)

    @staticmethod
    def postprocess_onnx(output, scale, original_size, conf_thres=0.25, iou_thres=0.45):
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
            x1, y1, x2, y2 = boxes[i]
            final_boxes.append({
                "cls": class_ids[i],
                "conf": scores[i],
                "xyxy": np.array([x1, y1, x2, y2])
            })
        return final_boxes

    @staticmethod
    def onnx_multi_scale_infer(onnx_session, img, conf_thres, iou_thres):
        scale_ratios = [0.8, 1.0, 1.2]
        model_fixed_size = (640, 640)
        all_boxes = []
        origin_h, origin_w = img.shape[:2]
        for scale in scale_ratios:
            scaled_h, scaled_w = int(origin_h * scale), int(origin_w * scale)
            scaled_img = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            input_tensor, pad_scale, _ = DetUtils.preprocess_image(scaled_img, model_fixed_size)
            input_name = onnx_session.get_inputs()[0].name
            output = onnx_session.run(None, {input_name: input_tensor})
            scaled_boxes = DetUtils.postprocess_onnx(output, pad_scale, (scaled_w, scaled_h), conf_thres, iou_thres)
            for box in scaled_boxes:
                box["xyxy"] = box["xyxy"] / scale
                box["xyxy"][0] = np.clip(box["xyxy"][0], 0, origin_w)
                box["xyxy"][1] = np.clip(box["xyxy"][1], 0, origin_h)
                box["xyxy"][2] = np.clip(box["xyxy"][2], 0, origin_w)
                box["xyxy"][3] = np.clip(box["xyxy"][3], 0, origin_h)
                all_boxes.append(box)
        if len(all_boxes) == 0:
            return []
        boxes = np.array([b["xyxy"] for b in all_boxes])
        scores = np.array([b["conf"] for b in all_boxes])
        class_ids = np.array([b["cls"] for b in all_boxes])
        final_boxes = []
        for cls in np.unique(class_ids):
            cls_mask = class_ids == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            indices = cv2.dnn.NMSBoxes(cls_boxes[:, :4].tolist(), cls_scores.tolist(), conf_thres, iou_thres)
            if indices is not None:
                for i in indices:
                    i = i[0] if isinstance(i, (list, np.ndarray)) else i
                    final_boxes.append(all_boxes[np.where(cls_mask)[0][i]])
        return final_boxes

    @staticmethod
    def pt_multi_scale_infer(model, img_path, conf, iou):
        res_640 = model.predict(img_path, imgsz=640, conf=conf, iou=iou, verbose=False)
        res_1280 = model.predict(img_path, imgsz=1280, conf=conf * 0.9, iou=iou, verbose=False)
        boxes_640 = res_640[0].boxes.data.cpu().numpy() if len(res_640[0].boxes) > 0 else np.array([])
        boxes_1280 = res_1280[0].boxes.data.cpu().numpy() if len(res_1280[0].boxes) > 0 else np.array([])
        all_boxes = np.vstack([boxes_640, boxes_1280]) if len(boxes_640) > 0 and len(boxes_1280) > 0 else (boxes_640 if len(boxes_640) > 0 else boxes_1280)
        if len(all_boxes) == 0:
            return []
        final_boxes = []
        for box in all_boxes:
            x1, y1, x2, y2, conf, cls = box[:6]
            final_boxes.append({
                "cls": int(cls),
                "conf": float(conf),
                "xyxy": np.array([x1, y1, x2, y2])
            })
        cls_ids = np.array([b["cls"] for b in final_boxes])
        bboxes = np.array([b["xyxy"] for b in final_boxes])
        scores = np.array([b["conf"] for b in final_boxes])
        keep_boxes = []
        for cls in np.unique(cls_ids):
            mask = cls_ids == cls
            cls_boxes = bboxes[mask]
            cls_scores = scores[mask]
            indices = cv2.dnn.NMSBoxes(cls_boxes[:, :4].tolist(), cls_scores.tolist(), conf, iou)
            if indices is not None:
                for i in indices:
                    i = i[0] if isinstance(i, (list, np.ndarray)) else i
                    keep_boxes.append(final_boxes[np.where(mask)[0][i]])
        return keep_boxes

    @staticmethod
    def draw_defects(frame, boxes, model=None, is_video=False):
        result_img = frame.copy() if is_video else cv2.imread(frame)
        small_count = 0
        model_names = {}
        if model is not None:
            try:
                if hasattr(model, 'model') and hasattr(model.model, 'yaml'):
                    model_names = {int(k): v for k, v in model.model.yaml.get('names', {}).items()}
                else:
                    model_names = DetConfig.CLASS_NAMES_CN
            except:
                model_names = DetConfig.CLASS_NAMES_CN
        for box in boxes:
            if hasattr(box, 'cls'):
                cls_id = int(box.cls.cpu().numpy()[0]) if hasattr(box.cls, 'cpu') else int(box.cls)
                conf = float(box.conf.cpu().numpy()[0]) if hasattr(box.conf, 'cpu') else float(box.conf)
                xyxy = box.xyxy.cpu().numpy()[0] if hasattr(box.xyxy, 'cpu') else box.xyxy
                x1, y1, x2, y2 = xyxy.astype(int)
            elif isinstance(box, dict):
                cls_id = int(box["cls"])
                conf = float(box["conf"])
                x1, y1, x2, y2 = box["xyxy"].astype(int)
            else:
                continue
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            if x1 >= x2 or y1 >= y2:
                continue
            area = (x2 - x1) * (y2 - y1)
            if area < DetConfig.SMALL_DEFECT_THRESHOLD and area > 0:
                color, thickness = (0, 0, 255), 3
                small_count += 1
            else:
                color, thickness = (0, 255, 0), 2
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, thickness)
            class_name = model_names.get(cls_id, DetConfig.CLASS_NAMES_CN.get(cls_id, f"类别{cls_id}"))
            label = f"{class_name} {conf:.2f}"
            text_x = max(0, x1 - 5)
            text_y = max(DetConfig.LABEL_FONT_SIZE + 10, y1 - 10)
            result_img = DetUtils.draw_chinese_text_force(result_img, label, (text_x, text_y), color)
        return result_img, small_count


# ==================== 批量检测线程 ====================
class BatchDetThread(QThread):
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    finish_signal = pyqtSignal(str)

    def __init__(self, img_paths, model, conf, iou, multi_scale, save_dir):
        super().__init__()
        self.img_paths = img_paths
        self.model = model
        self.conf = conf
        self.iou = iou
        self.multi_scale = multi_scale
        self.save_dir = save_dir
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        total = len(self.img_paths)
        if total == 0:
            self.finish_signal.emit("未选择任何图片！")
            return
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        for idx, img_path in enumerate(self.img_paths):
            if not self.is_running:
                break
            try:
                img = cv2.imread(img_path)
                is_trt = hasattr(self.model, 'engine')
                if self.multi_scale:
                    if is_trt:
                        boxes = self.model.infer(img)
                    elif 'onnxruntime' in str(type(self.model)):
                        boxes = DetUtils.onnx_multi_scale_infer(self.model, img, self.conf, self.iou)
                    else:
                        boxes = DetUtils.pt_multi_scale_infer(self.model, img_path, self.conf, self.iou)
                else:
                    if is_trt:
                        boxes = self.model.infer(img)
                    elif 'onnxruntime' in str(type(self.model)):
                        input_tensor, scale, original_size = DetUtils.preprocess_image(img, (640, 640))
                        input_name = self.model.get_inputs()[0].name
                        output = self.model.run(None, {input_name: input_tensor})
                        boxes = DetUtils.postprocess_onnx(output, scale, original_size, self.conf, self.iou)
                    else:
                        res = self.model.predict(img_path, imgsz=640, conf=self.conf, iou=self.iou, verbose=False)
                        boxes = []
                        for box in res[0].boxes:
                            cls_id = int(box.cls.cpu().numpy()[0]) if hasattr(box.cls, 'cpu') else int(box.cls)
                            conf = float(box.conf.cpu().numpy()[0]) if hasattr(box.conf, 'cpu') else float(box.conf)
                            xyxy = box.xyxy.cpu().numpy()[0] if hasattr(box.xyxy, 'cpu') else box.xyxy
                            boxes.append({"cls": cls_id, "conf": conf, "xyxy": xyxy})
                res_img, small_cnt = DetUtils.draw_defects(img_path, boxes, model=self.model)
                save_name = os.path.basename(img_path)
                save_path = os.path.join(self.save_dir, save_name)
                cv2.imwrite(save_path, res_img)
                count = {k: 0 for k in DetConfig.CLASS_NAMES_CN.keys()}
                defect_count = 0
                for box in boxes:
                    cls_id = box["cls"]
                    if cls_id in count:
                        count[cls_id] += 1
                        if cls_id in DetConfig.DEFECT_CLASS_IDS:
                            defect_count += 1
                log_msg = f"✅ {save_name} | 缺陷{defect_count}个 | 小目标{small_cnt}个"
                self.log_signal.emit(log_msg)
                progress = int((idx + 1) / total * 100)
                self.progress_signal.emit(progress)
            except Exception as e:
                self.log_signal.emit(f"❌ {os.path.basename(img_path)} 检测失败：{str(e)[:30]}")
                progress = int((idx + 1) / total * 100)
                self.progress_signal.emit(progress)
        if self.is_running:
            self.finish_signal.emit(f"批量检测完成！结果已保存至：{self.save_dir}")
        else:
            self.finish_signal.emit("批量检测已取消！")

# ==================== 视频检测线程（改进版：支持暂停/继续） ====================
class VideoDetThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)
    log_signal = pyqtSignal(str)
    stop_signal = pyqtSignal()

    def __init__(self, source_type, source_path, model, conf, iou, multi_scale):
        super().__init__()
        self.source_type = source_type
        self.source_path = source_path
        self.model = model
        self.conf = conf
        self.iou = iou
        self.multi_scale = multi_scale
        self.cap = None
        self.is_running = False
        self.stop_requested = False
        self.paused = False          # 新增：暂停标志
        self.frame_count = 0

    def start_detect(self):
        self.stop_requested = False
        self.paused = False
        self.is_running = True
        self.start()

    def pause_detect(self):
        self.paused = True
        self.log_signal.emit("⏸ 视频检测已暂停")

    def resume_detect(self):
        self.paused = False
        self.log_signal.emit("▶️ 视频检测已继续")

    def stop_detect(self):
        self.stop_requested = True
        self.paused = False
        self.is_running = False

    def run(self):
        try:
            if self.source_type == "camera":
                self.cap = cv2.VideoCapture(int(self.source_path))
                if not self.cap.isOpened():
                    self.log_signal.emit("❌ 无法打开摄像头！")
                    self.stop_signal.emit()
                    return
                self.log_signal.emit("✅ 摄像头已打开，开始实时检测...")
            else:
                self.cap = cv2.VideoCapture(self.source_path)
                if not self.cap.isOpened():
                    self.log_signal.emit(f"❌ 无法打开视频：{self.source_path}")
                    self.stop_signal.emit()
                    return
                self.log_signal.emit(f"✅ 视频已打开：{os.path.basename(self.source_path)}")

            while not self.stop_requested:
                if self.paused:
                    time.sleep(0.05)
                    continue
                ret, frame = self.cap.read()
                if not ret:
                    if self.source_type == "video":
                        self.log_signal.emit("✅ 视频播放完毕")
                    break

                if self.frame_count % DetConfig.VIDEO_FRAME_INTERVAL == 0:
                    is_trt = hasattr(self.model, 'engine')
                    if self.multi_scale:
                        if is_trt:
                            boxes = self.model.infer(frame)
                        elif 'onnxruntime' in str(type(self.model)):
                            boxes = DetUtils.onnx_multi_scale_infer(self.model, frame, self.conf, self.iou)
                        else:
                            res = self.model.predict(frame, imgsz=640, conf=self.conf, iou=self.iou, verbose=False)
                            boxes = []
                            for box in res[0].boxes:
                                cls_id = int(box.cls.cpu().numpy()[0]) if hasattr(box.cls, 'cpu') else int(box.cls)
                                conf = float(box.conf.cpu().numpy()[0]) if hasattr(box.conf, 'cpu') else float(box.conf)
                                xyxy = box.xyxy.cpu().numpy()[0] if hasattr(box.xyxy, 'cpu') else box.xyxy
                                boxes.append({"cls": cls_id, "conf": conf, "xyxy": xyxy})
                    else:
                        if is_trt:
                            boxes = self.model.infer(frame)
                        elif 'onnxruntime' in str(type(self.model)):
                            input_tensor, scale, original_size = DetUtils.preprocess_image(frame, (640, 640))
                            input_name = self.model.get_inputs()[0].name
                            output = self.model.run(None, {input_name: input_tensor})
                            boxes = DetUtils.postprocess_onnx(output, scale, (frame.shape[1], frame.shape[0]),
                                                              self.conf, self.iou)
                        else:
                            res = self.model.predict(frame, imgsz=640, conf=self.conf, iou=self.iou, verbose=False)
                            boxes = []
                            for box in res[0].boxes:
                                cls_id = int(box.cls.cpu().numpy()[0]) if hasattr(box.cls, 'cpu') else int(box.cls)
                                conf = float(box.conf.cpu().numpy()[0]) if hasattr(box.conf, 'cpu') else float(box.conf)
                                xyxy = box.xyxy.cpu().numpy()[0] if hasattr(box.xyxy, 'cpu') else box.xyxy
                                boxes.append({"cls": cls_id, "conf": conf, "xyxy": xyxy})

                    try:
                        res_frame, _ = DetUtils.draw_defects(frame, boxes, model=self.model, is_video=True)
                        self.frame_signal.emit(res_frame)
                    except Exception as e:
                        self.log_signal.emit(f"❌ 绘制失败：{str(e)[:50]}")
                        self.frame_signal.emit(frame)

                    defect_count = 0
                    for box in boxes:
                        if box["cls"] in DetConfig.DEFECT_CLASS_IDS:
                            defect_count += 1
                    if defect_count > 0:
                        self.log_signal.emit(f"⚠️ 检测到缺陷：{defect_count}个")

                self.frame_count += 1

            self.cap.release()
            self.stop_signal.emit()
            if self.source_type == "video":
                self.log_signal.emit("✅ 视频检测已停止")
            else:
                self.log_signal.emit("✅ 摄像头检测已停止")

        except Exception as e:
            self.log_signal.emit(f"❌ 视频检测出错：{str(e)}")
            if self.cap is not None:
                self.cap.release()
            self.stop_signal.emit()

# ==================== 主窗口类 ====================
class InsulatorDetWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.current_img_path = None
        self.current_result = []
        self.current_img_size = None
        self.conf = 0.25
        self.iou = 0.45
        self.multi_scale = False

        self.batch_thread = None
        self.video_thread = None

        self.init_ui()
        self.check_device()

    def init_ui(self):
        self.setWindowTitle("基于边缘计算的输电线路绝缘子缺陷检测系统 | 学生：王新壹")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # 顶部状态栏
        status_layout = QHBoxLayout()
        self.device_lb = QLabel("边缘设备：检测中...")
        self.mode_lb = QLabel("算力模式：标准")
        self.cache_lb = QLabel(f"缓存目录：{DetConfig.CACHE_DIR}")
        status_layout.addWidget(self.device_lb)
        status_layout.addWidget(self.mode_lb)
        status_layout.addStretch(1)
        status_layout.addWidget(self.cache_lb)
        main_layout.addLayout(status_layout)

        # 模型与参数栏
        param_layout = QHBoxLayout()
        self.model_lb = QLabel("未加载模型")
        self.model_lb.setStyleSheet("color:red; font-weight:bold")
        self.model_lb.setMinimumWidth(100)
        load_model_btn = QPushButton("加载模型")
        load_model_btn.clicked.connect(self.load_model)

        param_layout.addWidget(QLabel("检测模型："))
        param_layout.addWidget(self.model_lb)
        param_layout.addWidget(load_model_btn)
        param_layout.addStretch(1)

        param_layout.addWidget(QLabel("置信度："))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(25)
        self.conf_slider.setFixedWidth(150)
        self.conf_slider.valueChanged.connect(self.update_conf)
        self.conf_val_lb = QLabel("0.25")
        self.conf_val_lb.setFixedWidth(40)
        param_layout.addWidget(self.conf_slider)
        param_layout.addWidget(self.conf_val_lb)

        param_layout.addWidget(QLabel("IOU："))
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(1, 100)
        self.iou_slider.setValue(45)
        self.iou_slider.setFixedWidth(150)
        self.iou_slider.valueChanged.connect(self.update_iou)
        self.iou_val_lb = QLabel("0.45")
        self.iou_val_lb.setFixedWidth(40)
        param_layout.addWidget(self.iou_slider)
        param_layout.addWidget(self.iou_val_lb)

        self.multi_scale_cb = QCheckBox("小目标优化(多尺度)")
        self.multi_scale_cb.stateChanged.connect(self.update_multi_scale)
        param_layout.addWidget(self.multi_scale_cb)

        main_layout.addLayout(param_layout)

        # 标签页
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget, stretch=1)

        self.single_tab = QWidget()
        self.tab_widget.addTab(self.single_tab, "单张图片检测")
        self.init_single_tab()

        self.batch_tab = QWidget()
        self.tab_widget.addTab(self.batch_tab, "批量图片检测")
        self.init_batch_tab()

        self.video_tab = QWidget()
        self.tab_widget.addTab(self.video_tab, "视频/实时流检测")
        self.init_video_tab()

        self.log_tab = QTextEdit()
        self.log_tab.setReadOnly(True)
        self.tab_widget.addTab(self.log_tab, "检测日志")

    def init_single_tab(self):
        main_layout = QHBoxLayout(self.single_tab)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(left_widget, stretch=3)

        self.img_container = QWidget()
        self.img_container.setMinimumSize(600, 400)
        self.img_container.setStyleSheet("border:1px solid #ccc;")
        img_layout = QVBoxLayout(self.img_container)
        img_layout.setContentsMargins(0, 0, 0, 0)

        self.img_lb = QLabel("请上传图片")
        self.img_lb.setStyleSheet("border:2px solid blue; background-color:#f0f0f0")
        self.img_lb.setAlignment(Qt.AlignCenter)
        self.img_lb.setScaledContents(True)
        img_layout.addWidget(self.img_lb)
        left_layout.addWidget(self.img_container, stretch=1)

        btn_layout = QHBoxLayout()
        upload_btn = QPushButton("上传图片")
        detect_btn = QPushButton("开始检测")
        save_btn = QPushButton("保存结果")
        export_btn = QPushButton("导出标注")

        upload_btn.setFixedSize(100, 30)
        detect_btn.setFixedSize(100, 30)
        save_btn.setFixedSize(100, 30)
        export_btn.setFixedSize(100, 30)

        btn_layout.addWidget(upload_btn)
        btn_layout.addWidget(detect_btn)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(export_btn)
        btn_layout.setAlignment(Qt.AlignCenter)
        left_layout.addLayout(btn_layout)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(right_widget, stretch=1)

        right_layout.addWidget(QLabel("检测报告："), alignment=Qt.AlignTop)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumWidth(300)
        right_layout.addWidget(self.result_text, stretch=1)

        upload_btn.clicked.connect(self.upload_img)
        detect_btn.clicked.connect(self.single_detect)
        save_btn.clicked.connect(self.save_result)
        export_btn.clicked.connect(self.export_label)

    def init_batch_tab(self):
        main_layout = QVBoxLayout(self.batch_tab)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        top_layout = QHBoxLayout()
        self.batch_list = QListWidget()
        self.batch_list.setMinimumHeight(200)
        top_layout.addWidget(self.batch_list, stretch=3)

        btn_widget = QWidget()
        btn_layout = QVBoxLayout(btn_widget)
        select_dir_btn = QPushButton("选择文件夹")
        select_files_btn = QPushButton("选择多张图片")
        clear_list_btn = QPushButton("清空列表")

        select_dir_btn.setFixedSize(120, 30)
        select_files_btn.setFixedSize(120, 30)
        clear_list_btn.setFixedSize(120, 30)

        btn_layout.addWidget(select_dir_btn)
        btn_layout.addWidget(select_files_btn)
        btn_layout.addWidget(clear_list_btn)
        btn_layout.addStretch()
        top_layout.addWidget(btn_widget)

        main_layout.addLayout(top_layout)

        self.batch_progress = QProgressBar()
        self.batch_progress.setValue(0)
        main_layout.addWidget(self.batch_progress)

        bottom_layout = QHBoxLayout()
        self.batch_start_btn = QPushButton("开始批量检测")
        self.batch_stop_btn = QPushButton("停止检测")
        self.batch_save_btn = QPushButton("选择保存目录")

        self.batch_start_btn.setFixedSize(120, 30)
        self.batch_stop_btn.setFixedSize(120, 30)
        self.batch_save_btn.setFixedSize(120, 30)
        self.batch_stop_btn.setEnabled(False)

        bottom_layout.addWidget(self.batch_start_btn)
        bottom_layout.addWidget(self.batch_stop_btn)
        bottom_layout.addWidget(self.batch_save_btn)
        bottom_layout.addStretch()
        main_layout.addLayout(bottom_layout)

        self.batch_save_dir = DetConfig.BATCH_SAVE_DIR
        self.batch_save_lb = QLabel(f"结果保存至：{self.batch_save_dir}")
        main_layout.addWidget(self.batch_save_lb)

        select_dir_btn.clicked.connect(self.batch_select_dir)
        select_files_btn.clicked.connect(self.batch_select_files)
        clear_list_btn.clicked.connect(self.batch_clear_list)
        self.batch_start_btn.clicked.connect(self.batch_start_detect)
        self.batch_stop_btn.clicked.connect(self.batch_stop_detect)
        self.batch_save_btn.clicked.connect(self.batch_select_save_dir)

    def init_video_tab(self):
        main_layout = QVBoxLayout(self.video_tab)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        source_group = QGroupBox("检测源")
        source_layout = QHBoxLayout(source_group)

        self.video_radio = QRadioButton("本地视频")
        self.camera_radio = QRadioButton("摄像头")
        self.camera_radio.setChecked(True)

        source_layout.addWidget(self.video_radio)
        source_layout.addWidget(self.camera_radio)

        self.video_path_btn = QPushButton("选择视频文件")
        self.video_path_btn.setEnabled(False)
        source_layout.addWidget(self.video_path_btn)
        source_layout.addStretch()

        main_layout.addWidget(source_group)

        self.video_container = QWidget()
        self.video_container.setMinimumSize(600, 400)
        self.video_container.setStyleSheet("border:1px solid #ccc;")
        video_layout = QVBoxLayout(self.video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)

        self.video_lb = QLabel("等待检测...")
        self.video_lb.setStyleSheet("border:2px solid blue; background-color:#f0f0f0")
        self.video_lb.setAlignment(Qt.AlignCenter)
        self.video_lb.setScaledContents(True)
        video_layout.addWidget(self.video_lb)
        main_layout.addWidget(self.video_container, stretch=1)

        ctrl_layout = QHBoxLayout()
        self.video_start_btn = QPushButton("开始检测")
        self.video_stop_btn = QPushButton("停止检测")
        self.video_save_btn = QPushButton("保存视频")

        self.video_start_btn.setFixedSize(120, 30)
        self.video_stop_btn.setFixedSize(120, 30)
        self.video_save_btn.setFixedSize(120, 30)
        self.video_stop_btn.setEnabled(False)
        self.video_save_btn.setEnabled(False)

        ctrl_layout.addWidget(self.video_start_btn)
        ctrl_layout.addWidget(self.video_stop_btn)
        ctrl_layout.addWidget(self.video_save_btn)
        ctrl_layout.addStretch()
        main_layout.addLayout(ctrl_layout)

        self.video_radio.toggled.connect(self.video_radio_toggle)
        self.camera_radio.toggled.connect(self.video_radio_toggle)
        self.video_path_btn.clicked.connect(self.video_select_file)
        self.video_start_btn.clicked.connect(self.video_start_detect)
        self.video_stop_btn.clicked.connect(self.video_stop_detect)
        self.video_save_btn.clicked.connect(self.video_save_result)

        self.current_video_path = ""
        self.video_writer = None
        self.save_video_flag = False

    # ==================== 通用方法 ====================
    def _model_backend(self):
        """检测当前模型后端类型: 'pt' | 'onnx' | 'trt'"""
        t = str(type(self.model))
        if 'onnxruntime' in t:
            return 'onnx'
        elif hasattr(self.model, 'engine'):
            return 'trt'
        return 'pt'

    def _run_single_inference(self, img_path, img):
        """统一推理接口：根据模型后端执行推理，返回 boxes 列表"""
        backend = self._model_backend()
        if self.multi_scale:
            if backend == 'onnx':
                return DetUtils.onnx_multi_scale_infer(self.model, img, self.conf, self.iou)
            elif backend == 'trt':
                return self.model.infer(img)
            else:
                return DetUtils.pt_multi_scale_infer(self.model, img_path, self.conf, self.iou)
        else:
            if backend == 'onnx':
                input_tensor, scale, original_size = DetUtils.preprocess_image(img, (640, 640))
                input_name = self.model.get_inputs()[0].name
                output = self.model.run(None, {input_name: input_tensor})
                return DetUtils.postprocess_onnx(output, scale, original_size, self.conf, self.iou)
            elif backend == 'trt':
                return self.model.infer(img)
            else:
                res = self.model.predict(img_path, imgsz=640, conf=self.conf, iou=self.iou, verbose=False)
                boxes = []
                for box in res[0].boxes:
                    cls_id = int(box.cls.cpu().numpy()[0]) if hasattr(box.cls, 'cpu') else int(box.cls)
                    conf = float(box.conf.cpu().numpy()[0]) if hasattr(box.conf, 'cpu') else float(box.conf)
                    xyxy = box.xyxy.cpu().numpy()[0] if hasattr(box.xyxy, 'cpu') else box.xyxy
                    boxes.append({"cls": cls_id, "conf": conf, "xyxy": xyxy})
                return boxes

    def check_device(self):
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                self.device_lb.setText(f"设备：GPU(NVIDIA {device_name})")
                self.device_lb.setStyleSheet("color:green")
            else:
                self.device_lb.setText("设备：CPU")
                self.device_lb.setStyleSheet("color:orange")
        except Exception as e:
            self.device_lb.setText(f"设备：检测失败 ({str(e)[:20]})")
            self.device_lb.setStyleSheet("color:red")

    def write_log(self, msg):
        log_msg = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.log_tab.append(log_msg)
        print(log_msg)

    def update_conf(self):
        self.conf = self.conf_slider.value() / 100
        self.conf_val_lb.setText(f"{self.conf:.2f}")

    def update_iou(self):
        self.iou = self.iou_slider.value() / 100
        self.iou_val_lb.setText(f"{self.iou:.2f}")

    def update_multi_scale(self, state):
        self.multi_scale = (state == Qt.Checked)
        self.mode_lb.setText("算力模式：小目标优化" if self.multi_scale else "算力模式：标准")

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型", "", "Model (*.pt *.onnx *.engine)")
        if not path:
            return
        try:
            if path.endswith(".onnx"):
                import onnxruntime as ort
                self.write_log(f"⏳ 加载ONNX模型：{os.path.basename(path)}")
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
                self.model = ort.InferenceSession(path, providers=providers)
                self.model_lb.setText(os.path.basename(path))
                self.model_lb.setStyleSheet("color:green; font-weight:bold")
                QMessageBox.information(self, "成功", "ONNX模型加载完成！")
            elif path.endswith(".engine"):
                self.write_log(f"⏳ 加载TensorRT引擎：{os.path.basename(path)}")
                try:
                    from edge.trt_inference import TRTInference
                    self.model = TRTInference(path, conf_thres=self.conf, iou_thres=self.iou)
                    self.model_lb.setText(os.path.basename(path))
                    self.model_lb.setStyleSheet("color:green; font-weight:bold")
                    QMessageBox.information(self, "成功", "TensorRT引擎加载完成！")
                except ImportError as e:
                    self.write_log(f"❌ TensorRT不可用：{str(e)[:50]}")
                    QMessageBox.warning(self, "警告",
                        "TensorRT仅在 Jetson/Linux+CUDA 环境可用，请使用 ONNX 或 PT 模型")
            elif path.endswith(".pt"):
                import torch
                from ultralytics import YOLO
                from ultralytics.nn.tasks import DetectionModel

                self.write_log(f"⏳ 加载PT模型：{os.path.basename(path)}")
                model_classes = check_model_classes(path)
                if model_classes:
                    self.write_log(f"✅ 模型类别映射：{model_classes}")

                try:
                    import ultralytics.serialization as ult_serial
                    ult_serial.add_safe_globals(DetectionModel)
                except ImportError:
                    import warnings
                    warnings.filterwarnings("ignore")
                    torch_version = [int(x) for x in torch.__version__.split('.')[:2]]
                    if torch_version >= [2, 6]:
                        original_load = torch.load
                        def safe_load(*args, **kwargs):
                            kwargs['weights_only'] = False
                            return original_load(*args, **kwargs)
                        torch.load = safe_load

                self.model = YOLO(path)
                self.model_lb.setText(os.path.basename(path))
                self.model_lb.setStyleSheet("color:green; font-weight:bold")
                QMessageBox.information(self, "成功", "PT模型加载完成！")
            else:
                self.write_log(f"❌ 不支持的模型格式：{os.path.splitext(path)[1]}")
                QMessageBox.warning(self, "警告", "仅支持PT/ONNX格式模型！")
        except Exception as e:
            self.write_log(f"❌ 模型加载失败：{str(e)}")
            QMessageBox.critical(self, "错误", f"加载失败：{str(e)}")

    # ==================== 单张检测方法 ====================
    def upload_img(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image (*.jpg *.jpeg *.png *.bmp)")
        if not path:
            return
        self.current_img_path = path
        pixmap = QPixmap(path).scaled(self.img_container.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_lb.setPixmap(pixmap)
        self.write_log(f"✅ 上传图片：{os.path.basename(path)}")

    def single_detect(self):
        if not self.model or not self.current_img_path:
            QMessageBox.warning(self, "警告", "请先加载模型并上传图片！")
            return
        try:
            start = time.time()
            img = cv2.imread(self.current_img_path)
            self.current_img_size = (img.shape[1], img.shape[0])
            self.current_result = self._run_single_inference(self.current_img_path, img)
            infer_time = (time.time() - start) * 1000
            res_img, small_cnt = DetUtils.draw_defects(self.current_img_path, self.current_result, model=self.model, is_video=False)

            count = {k: 0 for k in DetConfig.CLASS_NAMES_CN.keys()}
            defect_count = 0
            max_conf = 0
            for box in self.current_result:
                cls_id = box["cls"]
                conf = box["conf"]
                if cls_id in count:
                    count[cls_id] += 1
                    if cls_id in DetConfig.DEFECT_CLASS_IDS:
                        defect_count += 1
                max_conf = max(max_conf, conf)

            severity, color = DetUtils.assess_severity(defect_count, max_conf)

            qt_img = DetUtils.cv2qt(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
            self.img_lb.setPixmap(QPixmap.fromImage(qt_img).scaled(self.img_container.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            report = f"""
            <h3 align='center'>边缘推理检测报告</h3>
            图片：{os.path.basename(self.current_img_path)}<br>
            耗时：{infer_time:.2f}ms<br>
            小目标优化：{'开启' if self.multi_scale else '关闭'}<br>
            <hr>
            {''.join([f"{DetConfig.CLASS_NAMES_CN[k]}：{v} 个<br>" for k, v in count.items()])}
            小目标：{small_cnt} 个<br>
            总缺陷：{defect_count} 个<br>
            <hr>
            严重程度：<span style='color:{color};font-weight:bold'>{severity}</span>
            """
            self.result_text.setHtml(report)
            self.write_log(f"✅ 检测完成 | 缺陷{defect_count}个 | 耗时{infer_time:.1f}ms")
        except Exception as e:
            self.write_log(f"❌ 检测失败：{str(e)}")
            QMessageBox.critical(self, "错误", f"检测失败：{str(e)}")

    def save_result(self):
        if not self.current_img_path or not self.current_result:
            QMessageBox.warning(self, "警告", "请先完成检测！")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存结果", "", "Image (*.jpg *.png)")
        if not path:
            return
        res_img, _ = DetUtils.draw_defects(self.current_img_path, self.current_result, model=self.model)
        cv2.imwrite(path, res_img)
        QMessageBox.information(self, "成功", "结果保存成功！")
        self.write_log(f"✅ 结果保存至：{path}")

    def export_label(self):
        if not self.current_img_path or not self.current_result:
            QMessageBox.warning(self, "警告", "请先完成检测！")
            return
        if self.current_img_size is None:
            QMessageBox.warning(self, "警告", "图片尺寸信息缺失，请重新检测！")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出标注", "", "Label (*.txt)")
        if not path:
            return
        img_width, img_height = self.current_img_size
        try:
            with open(path, 'w', encoding='utf-8') as f:
                written_count = 0
                for box in self.current_result:
                    cls_id = box.get("cls")
                    xyxy = box.get("xyxy")
                    if cls_id is None or xyxy is None:
                        continue
                    x1, y1, x2, y2 = map(float, xyxy)
                    x_center = (x1 + x2) / 2.0 / img_width
                    y_center = (y1 + y2) / 2.0 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    written_count += 1
                if written_count == 0:
                    QMessageBox.warning(self, "警告", "没有有效的检测结果可导出！")
                    if os.path.exists(path):
                        os.remove(path)
                    return
            QMessageBox.information(self, "成功", f"标注导出成功！共导出 {written_count} 个目标。")
            self.write_log(f"✅ 标注导出至：{path}，共 {written_count} 个目标")
        except Exception as e:
            self.write_log(f"❌ 导出标注失败：{str(e)}")
            QMessageBox.critical(self, "错误", f"导出失败：{str(e)}")

    # ==================== 批量检测方法 ====================
    def batch_select_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not dir_path:
            return
        img_ext = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        img_paths = []
        for ext in img_ext:
            img_paths.extend(glob.glob(os.path.join(dir_path, ext)))
        if len(img_paths) == 0:
            QMessageBox.warning(self, "警告", "该文件夹下未找到图片！")
            return
        self.batch_list.clear()
        for path in img_paths:
            self.batch_list.addItem(path)
        self.write_log(f"✅ 加载文件夹：{dir_path} | 图片数量：{len(img_paths)}")

    def batch_select_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "选择多张图片", "", "Image (*.jpg *.jpeg *.png *.bmp)")
        if not paths:
            return
        current_items = [self.batch_list.item(i).text() for i in range(self.batch_list.count())]
        for path in paths:
            if path not in current_items:
                self.batch_list.addItem(path)
        self.write_log(f"✅ 选择图片：{len(paths)} 张")

    def batch_clear_list(self):
        self.batch_list.clear()
        self.write_log("✅ 清空图片列表")

    def batch_select_save_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not dir_path:
            return
        self.batch_save_dir = dir_path
        self.batch_save_lb.setText(f"结果保存至：{self.batch_save_dir}")
        self.write_log(f"✅ 批量保存目录已设置为：{dir_path}")

    def batch_start_detect(self):
        if not self.model:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return
        img_paths = [self.batch_list.item(i).text() for i in range(self.batch_list.count())]
        if len(img_paths) == 0:
            QMessageBox.warning(self, "警告", "请先选择图片！")
            return
        self.batch_thread = BatchDetThread(
            img_paths=img_paths, model=self.model, conf=self.conf, iou=self.iou,
            multi_scale=self.multi_scale, save_dir=self.batch_save_dir
        )
        self.batch_thread.progress_signal.connect(self.batch_progress.setValue)
        self.batch_thread.log_signal.connect(self.write_log)
        self.batch_thread.finish_signal.connect(self.batch_detect_finish)
        self.batch_start_btn.setEnabled(False)
        self.batch_stop_btn.setEnabled(True)
        self.batch_progress.setValue(0)
        self.batch_thread.start()
        self.write_log("🚀 开始批量检测...")

    def batch_stop_detect(self):
        if self.batch_thread and self.batch_thread.isRunning():
            self.batch_thread.stop()
            self.batch_start_btn.setEnabled(True)
            self.batch_stop_btn.setEnabled(False)
            self.write_log("🛑 停止批量检测")

    def batch_detect_finish(self, msg):
        self.batch_start_btn.setEnabled(True)
        self.batch_stop_btn.setEnabled(False)
        self.write_log(msg)
        QMessageBox.information(self, "完成", msg)

    # ==================== 视频检测方法（支持暂停/继续） ====================
    def video_radio_toggle(self):
        if self.video_radio.isChecked():
            self.video_path_btn.setEnabled(True)
        else:
            self.video_path_btn.setEnabled(False)

    def video_select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video (*.mp4 *.avi *.mov *.mkv)")
        if not path:
            return
        self.current_video_path = path
        self.write_log(f"✅ 选择视频：{os.path.basename(path)}")

    def video_start_detect(self):
        if not self.model:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return

        source_type = "camera" if self.camera_radio.isChecked() else "video"
        source_path = str(DetConfig.CAMERA_INDEX) if source_type == "camera" else self.current_video_path

        if source_type == "video" and not self.current_video_path:
            QMessageBox.warning(self, "警告", "请先选择视频文件！")
            return

        # 如果已有线程且处于暂停状态，则恢复
        if self.video_thread is not None and self.video_thread.isRunning() and self.video_thread.paused:
            self.video_thread.resume_detect()
            self.video_start_btn.setEnabled(False)
            self.video_stop_btn.setEnabled(True)
            self.video_save_btn.setEnabled(True)
            self.write_log("▶️ 继续视频检测")
            return

        # 否则新建线程（先停止之前的）
        if self.video_thread is not None and self.video_thread.isRunning():
            self.video_stop_detect()

        self.video_thread = VideoDetThread(
            source_type=source_type, source_path=source_path, model=self.model,
            conf=self.conf, iou=self.iou, multi_scale=self.multi_scale
        )
        self.video_thread.frame_signal.connect(self.update_video_frame)
        self.video_thread.log_signal.connect(self.write_log)
        self.video_thread.stop_signal.connect(self.video_detect_stop)

        self.video_start_btn.setEnabled(False)
        self.video_stop_btn.setEnabled(True)
        self.video_save_btn.setEnabled(True)
        self.video_thread.start_detect()
        self.write_log(f"🚀 开始{source_type}检测...")

    def video_stop_detect(self):
        if self.video_thread and self.video_thread.isRunning():
            # 暂停而不是彻底停止
            self.video_thread.pause_detect()
            self.video_start_btn.setEnabled(True)
            self.video_stop_btn.setEnabled(False)
            # 视频保存状态不变
            self.write_log("⏸ 暂停视频检测")

    def video_detect_stop(self):
        # 线程完全停止时的清理（例如视频播放完毕或出错）
        self.video_start_btn.setEnabled(True)
        self.video_stop_btn.setEnabled(False)
        self.video_save_btn.setEnabled(False)
        self.video_lb.setText("等待检测...")
        # 释放视频写入器
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.save_video_flag = False
            self.video_save_btn.setText("保存视频")

    def update_video_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_frame = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_lb.setPixmap(QPixmap.fromImage(qt_frame).scaled(
            self.video_container.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # 保存视频（如果开启）
        if self.save_video_flag:
            if self.video_writer is None:
                os.makedirs(DetConfig.CACHE_DIR, exist_ok=True)
                save_path = os.path.join(DetConfig.CACHE_DIR, f"detect_result_{int(time.time())}.avi")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(save_path, fourcc, 25.0, (w, h))
                if not self.video_writer.isOpened():
                    self.write_log("❌ 无法初始化视频写入器，保存失败")
                    self.save_video_flag = False
                    self.video_save_btn.setText("保存视频")
                    return
                self.write_log(f"✅ 开始保存视频：{save_path}")

            if self.video_writer is not None:
                try:
                    self.video_writer.write(frame)
                except Exception as e:
                    self.write_log(f"❌ 写入视频帧失败: {e}")
                    self.save_video_flag = False
                    self.video_save_btn.setText("保存视频")
                    if self.video_writer:
                        self.video_writer.release()
                        self.video_writer = None

    def video_save_result(self):
        if not self.save_video_flag:
            self.save_video_flag = True
            self.video_save_btn.setText("停止保存")
            self.write_log("📹 开始保存检测视频（等待视频帧写入）")
        else:
            self.save_video_flag = False
            self.video_save_btn.setText("保存视频")
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
                self.write_log("📹 停止保存视频，文件已生成")

# ==================== 运行程序 ====================
if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
        from PIL import Image, ImageDraw, ImageFont

    app = QApplication(sys.argv)
    window = InsulatorDetWindow()
    window.show()
    sys.exit(app.exec_())