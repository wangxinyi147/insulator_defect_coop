import cv2
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont

def cv2_to_qt(cv_img):
    """OpenCV图像转QImage"""
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    from PyQt5.QtGui import QImage
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

def draw_defects(img, boxes, class_names, font_path=None):
    """绘制检测框和标签"""
    result = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box['xyxy'])
        cls_id = box['cls']
        conf = box['conf']
        label = f"{class_names[cls_id]} {conf:.2f}"
        color = (0, 255, 0)  # 绿色
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        # 简单绘制文字（PIL支持中文需字体）
        cv2.putText(result, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return result

def preprocess_image(img, target_size=(640,640)):
    """预处理：保持长宽比填充"""
    h, w = img.shape[:2]
    scale = min(target_size[0]/w, target_size[1]/h)
    new_w, new_h = int(w*scale), int(h*scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = img_resized
    # 归一化并转为CHW
    canvas = canvas.astype(np.float32) / 255.0
    canvas = np.transpose(canvas, (2,0,1))
    canvas = np.expand_dims(canvas, axis=0)
    return canvas, scale, (w, h)

def postprocess_onnx(output, scale, original_size, conf_thres, iou_thres):
    """ONNX输出后处理"""
    predictions = np.squeeze(output[0]).T
    scores = np.max(predictions[:, 4:], axis=1)
    mask = scores >= conf_thres
    predictions = predictions[mask]
    scores = scores[mask]
    if len(predictions) == 0:
        return []
    class_ids = np.argmax(predictions[:, 4:], axis=1)
    boxes = predictions[:, :4].copy()
    # 还原到原图坐标
    boxes[:, 0] = (boxes[:, 0] - boxes[:, 2]/2) / scale
    boxes[:, 1] = (boxes[:, 1] - boxes[:, 3]/2) / scale
    boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) / scale
    boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) / scale
    boxes[:, 0] = np.clip(boxes[:, 0], 0, original_size[0])
    boxes[:, 1] = np.clip(boxes[:, 1], 0, original_size[1])
    boxes[:, 2] = np.clip(boxes[:, 2], 0, original_size[0])
    boxes[:, 3] = np.clip(boxes[:, 3], 0, original_size[1])
    # NMS
    indices = cv2.dnn.NMSBoxes(boxes[:, :4].tolist(), scores.tolist(), conf_thres, iou_thres)
    if len(indices) == 0:
        return []
    final_boxes = []
    for i in indices:
        i = i[0] if isinstance(i, (list, np.ndarray)) else i
        final_boxes.append({
            "cls": class_ids[i],
            "conf": scores[i],
            "xyxy": boxes[i]
        })
    return final_boxes

def multi_scale_infer(onnx_session, img, conf_thres, iou_thres):
    """多尺度推理（小目标优化）"""
    scales = [0.8, 1.0, 1.2]
    all_boxes = []
    orig_h, orig_w = img.shape[:2]
    for scale in scales:
        scaled = cv2.resize(img, (int(orig_w*scale), int(orig_h*scale)))
        input_tensor, pad_scale, (orig_w_scaled, orig_h_scaled) = preprocess_image(scaled, (640,640))
        input_name = onnx_session.get_inputs()[0].name
        output = onnx_session.run(None, {input_name: input_tensor})
        boxes = postprocess_onnx(output, pad_scale, (orig_w_scaled, orig_h_scaled), conf_thres, iou_thres)
        for box in boxes:
            box['xyxy'] = box['xyxy'] / scale
            all_boxes.append(box)
    # 合并后NMS
    if len(all_boxes) == 0:
        return []
    bboxes = np.array([b['xyxy'] for b in all_boxes])
    scores = np.array([b['conf'] for b in all_boxes])
    cls_ids = np.array([b['cls'] for b in all_boxes])
    final_boxes = []
    for cls in np.unique(cls_ids):
        mask = cls_ids == cls
        cls_boxes = bboxes[mask]
        cls_scores = scores[mask]
        indices = cv2.dnn.NMSBoxes(cls_boxes[:,:4].tolist(), cls_scores.tolist(), conf_thres, iou_thres)
        for i in indices:
            i = i[0] if isinstance(i, (list, np.ndarray)) else i
            final_boxes.append(all_boxes[np.where(mask)[0][i]])
    return final_boxes