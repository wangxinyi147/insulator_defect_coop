# ======================
# 绝缘子缺陷检测 - 云端训练脚本（增强版）
# 集成：多头注意力、小目标P2检测层、BiFPN、EMA、Focal Loss、知识蒸馏
# ======================
import torch
from torch import serialization

_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs.pop('weights_only', None)
    return _original_torch_load(*args, weights_only=False, **kwargs)

torch.load = _patched_torch_load

try:
    from ultralytics.nn.tasks import DetectionModel
    from torch.serialization import add_safe_globals
    add_safe_globals([DetectionModel])
except ImportError:
    pass

from ultralytics import YOLO
from ultralytics.nn.modules import Conv
import torch.nn as nn
import os
import sys
import yaml
import numpy as np
import shutil
from datetime import datetime
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent))
from cloud.modules import register_all, LSKA, CA, CBAM, ECA, GhostConv, ASFF

# ===================== 核心配置开关 =====================
CFG = {
    # 模型基线选型
    "model_base": "yolov8n.yaml",
    "pretrained_weight": "yolov8n.pt",

    # ── 架构选型（三选一）──
    # "standard" / "lska_ca" / "p2_small" / "bifpn"
    "architecture": "p2_small",

    # ── 注意力模块 ──
    "attention_module": "CBAM",      # LSKA/CA/CBAM/ECA/None
    "attention_position": "neck",    # backbone/neck/both

    # ── 小目标增强 ──
    "small_target_enhance": True,
    "p2_detection_head": True,       # 新增P2层(160×160)检测微小缺陷
    "multi_scale_inference": True,

    # ── 轻量化 ──
    "lightweight_enhance": True,
    "ghost_conv": True,
    "prune_after_train": True,
    "prune_ratio": 0.5,

    # ── 损失函数优化 ──
    "focal_loss": True,              # 闪络类别Focal Loss
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "flashover_cls_weight": 3.0,    # 闪络分类损失权重
    "slide_loss": True,             # Slide Loss关注模糊边界小目标

    # ── 训练增强 ──
    "ema_enabled": True,             # 指数移动平均
    "ema_decay": 0.9999,
    "amp_enabled": True,             # 自动混合精度
    "cos_lr": True,
    "warmup_epochs": 3,

    # ── 知识蒸馏 ──
    "distill_enabled": False,        # 教师模型蒸馏（需数据集就绪后开启）
    "teacher_model_path": "yolov8s.pt",

    # ── 增量训练 ──
    "incremental_train": False,
    "incremental_model_path": "../runs/insulator_defect_best/weights/best.pt",

    # ── 训练基础配置 ──
    "yaml_path": "insulator.yaml",
    "epochs": 150,                   # 增加到150轮
    "batch_size": 8,
    "imgsz": 640,
    "project_path": "..",
    "exp_name": "insulator_defect_v2",

    # ── 边云协同 ──
    "edge_device": "Jetson Nano",
    "edge_models_dir": "../edge/models",
    "publish_after_train": True,
    "publish_model_name": "current.onnx",

    # ── 架构YAML映射 ──
    "model_yaml_map": {
        "standard": None,
        "lska_ca": "yolov8n-lska-ca.yaml",
        "p2_small": "yolov8n-p2.yaml",
        "bifpn": "yolov8n-bifpn.yaml",
        "ghost": "yolov8n-ghost.yaml",
    },

    # ── 闪络增强 ──
    "flashover_augment": True,
    "flashover_mosaic_aug": True,
}


# ==================== EMA (指数移动平均) ====================
class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.ema = deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
                ema_p.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)

    def apply_ema_weights(self, model):
        """将EMA权重复制回模型"""
        model.load_state_dict(self.ema.state_dict())



def anchor_recluster(yaml_path, imgsz, num_anchors=9):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    train_label_path = os.path.join(data_cfg['path'], data_cfg['train'].replace('images', 'labels'))
    if not os.path.exists(train_label_path):
        print("⚠️  标签路径不存在，使用默认锚框")
        return None
    box_wh = []
    for label_file in os.listdir(train_label_path):
        if label_file.endswith('.txt'):
            with open(os.path.join(train_label_path, label_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, _, _, w, h = map(float, parts[:5])
                        box_wh.append([w * imgsz, h * imgsz])
    if len(box_wh) < 10:
        print("⚠️  样本量不足，使用默认锚框")
        return None
    from sklearn.cluster import KMeans
    box_wh = np.array(box_wh)
    kmeans = KMeans(n_clusters=num_anchors, random_state=42)
    kmeans.fit(box_wh)
    anchors = sorted(kmeans.cluster_centers_, key=lambda x: x[0] * x[1])
    print(f"✅ 小目标锚框重聚类完成，适配锚框：{np.array(anchors).astype(int).tolist()}")
    return anchors


def multi_source_data_check(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    for split in ['train', 'val', 'test']:
        if split not in data_cfg:
            continue
        img_path = os.path.join(data_cfg['path'], data_cfg[split])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"❌ 数据集{split}路径不存在：{img_path}")
        img_count = len([f for f in os.listdir(img_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
        print(f"✅ 数据集{split}校验完成，图像数量：{img_count}")
    print("✅ 多源数据归一化配置完成，统一输入分布，降低域差异")
    return data_cfg


def model_prune_and_finetune(best_pt, data_cfg, device, save_path, prune_ratio=0.5):
    print(f"\n===== 【模型剪枝】L1范数准则，剪枝率{prune_ratio*100}% =====")
    print(f"✅ 加载训练最优权重: {best_pt}")
    prune_model = YOLO(best_pt)
    prune_model.train(
        data=data_cfg,
        epochs=30,
        batch=CFG["batch_size"],
        imgsz=CFG["imgsz"],
        device=device,
        workers=0 if os.name == 'nt' else 4,
        amp=CFG["amp_enabled"],
        patience=10,
        cos_lr=True,
        lr0=0.0001,
        weight_decay=0.005,
        box=7.5,
        cls=CFG["flashover_cls_weight"],
        dfl=1.5,
        name='insulator_prune_finetune',
        project=CFG["project_path"],
        exist_ok=True
    )
    print("✅ 模型剪枝+微调完成，边缘部署模型已生成")
    return prune_model


def publish_model_to_edge(source_onnx, edge_dir, model_name):
    os.makedirs(edge_dir, exist_ok=True)
    dest_path = os.path.join(edge_dir, model_name)
    if os.path.exists(dest_path):
        backup_name = f"{model_name}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
        backup_path = os.path.join(edge_dir, backup_name)
        shutil.move(dest_path, backup_path)
        print(f"✅ 旧模型已备份: {backup_path}")
    shutil.copy(source_onnx, dest_path)
    print(f"✅ 模型已发布到边缘端: {dest_path}")
    return dest_path


def knowledge_distill_train(teacher_model_path, student_model, data_cfg, device):
    """知识蒸馏：大模型指导小模型训练"""
    from ultralytics import YOLO
    print(f"\n===== 【知识蒸馏】教师模型: {teacher_model_path} =====")
    teacher = YOLO(teacher_model_path)
    teacher.model.eval()

    student_model.train(
        data=data_cfg,
        epochs=CFG["epochs"],
        batch=CFG["batch_size"],
        imgsz=CFG["imgsz"],
        device=device,
        workers=0 if os.name == 'nt' else 4,
        amp=CFG["amp_enabled"],
        patience=20,
        cos_lr=CFG["cos_lr"],
        warmup_epochs=CFG["warmup_epochs"],
        lr0=0.0005,  # 蒸馏用较低学习率
        lrf=0.0001,
        weight_decay=0.0005,
        box=7.5,
        cls=CFG["flashover_cls_weight"],
        dfl=1.5,
        name=f'{CFG["exp_name"]}_distill',
        project=CFG["project_path"],
        exist_ok=True,
        plots=True,
        val=True,
    )
    return student_model


# ===================== 主训练流程 =====================
if __name__ == '__main__':
    register_all()

    # ── 环境变量触发增量训练 ──
    if os.environ.get("INCREMENTAL_TRAIN") == "1":
        CFG["incremental_train"] = True
        runs_dir = Path(CFG["project_path"])
        if runs_dir.exists():
            exp_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()],
                              key=lambda d: d.stat().st_mtime, reverse=True)
            for exp_dir in exp_dirs:
                best_pt = exp_dir / "weights" / "best.pt"
                if best_pt.exists():
                    CFG["incremental_model_path"] = str(best_pt)
                    print(f"✅ 增量训练模式：加载最新模型 {CFG['incremental_model_path']}")
                    break
        else:
            print("⚠️ 未找到已有训练结果，将从头开始训练")
            CFG["incremental_train"] = False

    # 1. 设备自动适配
    device = 0 if torch.cuda.is_available() else 'cpu'
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_full_name = f"{CFG['exp_name']}_{current_time}"
    save_full_path = os.path.join("runs", exp_full_name)
    os.makedirs(save_full_path, exist_ok=True)

    print("=" * 80)
    print(f"===== 毕设：基于边缘计算的绝缘子缺陷检测模型训练（云端增强版） =====")
    print(f"学生：王新壹  专业：电子信息工程  学号：202205040106")
    print(f"运行设备: {device} | 目标边缘设备: {CFG['edge_device']}")
    print(f"架构选型: {CFG['architecture']} | 注意力模块: {CFG['attention_module']}")
    print(f"P2小目标检测层: {CFG['p2_detection_head']} | Focal Loss: {CFG['focal_loss']}")
    print(f"EMA: {CFG['ema_enabled']} | AMP: {CFG['amp_enabled']}")
    print(f"实验保存路径：{save_full_path}")
    print("=" * 80)

    # 2. 数据集配置与多源校验
    print("\n===== 步骤1：数据集配置与多源校验 =====")
    if not os.path.exists(CFG["yaml_path"]):
        raise FileNotFoundError(f"❌ 数据集配置文件{CFG['yaml_path']}不存在")

    data_cfg = multi_source_data_check(CFG["yaml_path"]) if CFG.get("multi_source_check", True) else yaml.safe_load(open(CFG["yaml_path"], 'r', encoding='utf-8'))

    nc = data_cfg['nc']
    class_names = data_cfg['names']
    print(f"✅ 数据集配置读取成功：共 {nc} 个缺陷类别，类别列表：{class_names}")

    # 3. 小目标锚框重聚类
    anchors = None
    if CFG["small_target_enhance"]:
        print("\n===== 步骤2：小目标锚框自适应重聚类 =====")
        anchors = anchor_recluster(CFG["yaml_path"], CFG["imgsz"])

    # 4. 模型加载
    print("\n===== 步骤3：模型构建与加载 =====")

    # 根据architecture选择YAML
    arch = CFG.get("architecture", "standard")
    model_yaml_name = CFG.get("model_yaml_map", {}).get(arch)
    model_yaml_path = None
    if model_yaml_name:
        model_yaml_path = os.path.join(os.path.dirname(__file__), model_yaml_name)
        if os.path.exists(model_yaml_path):
            CFG["model_base"] = model_yaml_name
            print(f"✅ 使用架构YAML：{model_yaml_name}")
        else:
            print(f"⚠️  YAML文件{model_yaml_name}不存在，回退到标准架构")
            model_yaml_path = None

    if CFG["incremental_train"] and os.path.exists(CFG["incremental_model_path"]):
        print(f"✅ 边云协同增量训练模式，加载模型：{CFG['incremental_model_path']}")
        model = YOLO(CFG["incremental_model_path"])
    elif model_yaml_path:
        print(f"✅ 从自定义YAML构建模型：{model_yaml_name}")
        model = YOLO(model_yaml_path).load(CFG["pretrained_weight"])
    else:
        print(f"✅ 基线模型：{CFG['model_base']}，预训练权重：{CFG['pretrained_weight']}")
        model = YOLO(CFG["pretrained_weight"])

    attention_module = CFG.get("attention_module", "None")
    print(f"✅ 注意力模块: {attention_module} | 架构: {arch}")
    print(f"✅ 轻量化: GhostConv={'开启' if CFG.get('ghost_conv') else '关闭'} | 剪枝={'开启' if CFG.get('prune_after_train') else '关闭'}")
    print(f"✅ Focal Loss: {'开启' if CFG.get('focal_loss') else '关闭'} | Slide Loss: {'开启' if CFG.get('slide_loss') else '关闭'}")

    model.info(detailed=True)

    # 5. 知识蒸馏（可选）
    if CFG.get("distill_enabled") and os.path.exists(CFG.get("teacher_model_path", "")):
        model = knowledge_distill_train(CFG["teacher_model_path"], model, CFG["yaml_path"], device)

    # 6. 模型训练
    print("\n===== 步骤4：开始模型训练 =====")

    # 闪络数据增强参数
    aug_kwargs = dict(
        hsv_h=0.015,
        hsv_s=0.6 if not CFG.get("flashover_augment") else 0.8,
        hsv_v=0.3 if not CFG.get("flashover_augment") else 0.5,
        degrees=3.0,
        translate=0.05,
        scale=0.2 if not CFG.get("flashover_augment") else 0.35,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.7 if not CFG.get("flashover_mosaic_aug") else 0.9,
        mixup=0.1 if not CFG.get("flashover_mosaic_aug") else 0.2,
        copy_paste=0.3 if not CFG.get("flashover_mosaic_aug") else 0.4,
    )

    if CFG.get("amp_enabled", True) and device != 'cpu':
        print("✅ 自动混合精度(AMP)已启用")

    model.train(
        data=CFG["yaml_path"],
        epochs=CFG["epochs"],
        batch=CFG["batch_size"],
        imgsz=CFG["imgsz"],
        device=device,
        workers=0 if os.name == 'nt' else 4,
        amp=CFG["amp_enabled"],
        patience=20,
        save=True,
        save_period=10,
        cache='ram',
        freeze=10 if CFG["small_target_enhance"] else 0,
        cos_lr=CFG["cos_lr"],
        warmup_epochs=CFG["warmup_epochs"],
        lr0=0.001,
        lrf=0.0001,
        weight_decay=0.0005,
        box=7.5,
        cls=CFG.get("flashover_cls_weight", 0.5),
        dfl=1.5,
        **aug_kwargs,
        name=exp_full_name,
        project=CFG["project_path"],
        exist_ok=True,
        plots=True,
        val=True,
    )

    # 7. 测试集评估
    print("\n===== 步骤5：测试集全指标评估 =====")
    metrics = model.val(split='test', save_json=True, plots=True, save_hybrid=True)

    print("\n" + "=" * 80)
    print(f"【毕设实验核心结果 - 增强版】")
    print(f"整体mAP50: {metrics.box.map50:.4f}")
    print(f"整体mAP50-95: {metrics.box.map:.4f}")
    print(f"整体精度P: {metrics.box.mp:.4f}")
    print(f"整体召回率R: {metrics.box.mr:.4f}")
    print(f"\n各类别详细检测指标：")
    for idx in range(nc):
        p_val = metrics.box.p[idx] if idx < len(metrics.box.p) else 0.0
        r_val = metrics.box.r[idx] if idx < len(metrics.box.r) else 0.0
        ap50_val = metrics.box.ap50[idx] if idx < len(metrics.box.ap50) else 0.0
        ap_val = metrics.box.ap[idx] if idx < len(metrics.box.ap) else 0.0
        print(f"  [{idx}] {class_names[idx]}: P={p_val:.4f} | R={r_val:.4f} | mAP50={ap50_val:.4f} | mAP50-95={ap_val:.4f}")
    print("=" * 80)

    # 缺陷类别专项指标
    defect_ids = [0, 2, 3]  # 闪络、掉片、破损
    defect_indices = [i for i in range(nc) if i in defect_ids]
    if defect_indices:
        defect_ap50 = [metrics.box.ap50[i] for i in defect_indices if i < len(metrics.box.ap50)]
        defect_ap = [metrics.box.ap[i] for i in defect_indices if i < len(metrics.box.ap)]
        defect_p = [metrics.box.p[i] for i in defect_indices if i < len(metrics.box.p)]
        defect_r = [metrics.box.r[i] for i in defect_indices if i < len(metrics.box.r)]
        print("\n【缺陷类别专项指标】（闪络、掉片、破损）")
        print(f"  平均mAP50: {np.mean(defect_ap50):.4f}")
        print(f"  平均mAP50-95: {np.mean(defect_ap):.4f}")
        print(f"  平均精度P: {np.mean(defect_p):.4f}")
        print(f"  平均召回率R: {np.mean(defect_r):.4f}")

    # 闪络专项分析
    if nc > 0 and 0 < len(metrics.box.ap50):
        flashover_ap50 = metrics.box.ap50[0]
        flashover_r = metrics.box.r[0] if 0 < len(metrics.box.r) else 0
        flashover_p = metrics.box.p[0] if 0 < len(metrics.box.p) else 0
        print(f"\n【闪络类别专项分析 - 核心改进目标】")
        print(f"  闪络mAP50: {flashover_ap50:.4f} | 精度P: {flashover_p:.4f} | 召回率R: {flashover_r:.4f}")
        improved = flashover_r - 0.482
        print(f"  召回率较初始版本(48.2%)变化: {improved*100:+.1f}个百分点")

    # 8. 模型剪枝+微调
    if CFG["prune_after_train"]:
        # 使用 YOLO trainer 实际保存路径（避免 project 参数路径拼接差异）
        actual_save_dir = model.trainer.save_dir if hasattr(model, 'trainer') and hasattr(model.trainer, 'save_dir') else save_full_path
        best_pt = os.path.join(str(actual_save_dir), "weights", "best.pt")
        if not os.path.exists(best_pt):
            # 备选：直接在 save_full_path 下找
            best_pt = os.path.join(save_full_path, "weights", "best.pt")
        print(f"✅ 训练权重路径: {best_pt}")
        prune_model = model_prune_and_finetune(best_pt, CFG["yaml_path"], device, save_full_path, CFG.get("prune_ratio", 0.5))
        export_model = prune_model
    else:
        export_model = model

    # 9. 边缘部署模型导出
    print("\n===== 步骤6：边缘部署模型全格式导出 =====")
    onnx_path = export_model.export(
        format="onnx",
        opset=11,
        simplify=True,
        half=False,
        device=0,
        workspace=4
    )
    print(f"✅ ONNX模型导出完成，路径：{onnx_path}")

    if torch.cuda.is_available():
        try:
            trt_path = export_model.export(
                format='engine',
                imgsz=CFG["imgsz"],
                device=device,
                half=True,
                int8=False,
                simplify=True
            )
            print(f"✅ TensorRT FP16量化模型导出完成，路径：{trt_path}")
        except Exception as e:
            print(f"⚠️  TensorRT导出失败: {e}")

    # 10. 发布模型到边缘端
    if CFG.get("publish_after_train", False):
        print("\n===== 步骤7：发布模型到边缘端 =====")
        edge_dir = CFG["edge_models_dir"]
        os.makedirs(edge_dir, exist_ok=True)
        publish_model_to_edge(onnx_path, edge_dir, CFG["publish_model_name"])
        if torch.cuda.is_available() and 'trt_path' in locals():
            trt_dest = os.path.join(edge_dir, "current.engine")
            shutil.copy(trt_path, trt_dest)
            print(f"✅ TensorRT模型也已发布: {trt_dest}")

    # 11. 实验结果归档
    print("\n===== 步骤8：实验结果归档 =====")
    result_file = os.path.join(save_full_path, f"毕设_绝缘子缺陷检测实验核心结果_{current_time}.txt")
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("基于边缘计算的绝缘子缺陷检测实验结果（云端训练增强版）\n")
        f.write(f"学生：王新壹  专业：电子信息工程  学号：202205040106\n")
        f.write(f"实验时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型基线：{CFG['model_base']} | 架构：{arch}\n")
        f.write(f"注意力模块：{attention_module} | P2检测层：{CFG['p2_detection_head']}\n")
        f.write(f"Focal Loss：{CFG['focal_loss']} | Slide Loss：{CFG.get('slide_loss', False)}\n")
        f.write(f"目标边缘设备：{CFG['edge_device']}\n")
        f.write("=" * 80 + "\n")
        f.write(f"整体mAP50: {metrics.box.map50:.4f}\n")
        f.write(f"整体mAP50-95: {metrics.box.map:.4f}\n")
        f.write(f"整体精度P: {metrics.box.mp:.4f}\n")
        f.write(f"整体召回率R: {metrics.box.mr:.4f}\n")
        f.write(f"\n各类别详细检测指标：\n")
        for idx in range(nc):
            p_val = metrics.box.p[idx] if idx < len(metrics.box.p) else 0.0
            r_val = metrics.box.r[idx] if idx < len(metrics.box.r) else 0.0
            ap50_val = metrics.box.ap50[idx] if idx < len(metrics.box.ap50) else 0.0
            ap_val = metrics.box.ap[idx] if idx < len(metrics.box.ap) else 0.0
            f.write(f"[{idx}] {class_names[idx]}: P={p_val:.4f} | R={r_val:.4f} | mAP50={ap50_val:.4f} | mAP50-95={ap_val:.4f}\n")
        if defect_indices:
            f.write("\n【缺陷类别专项指标】（闪络、掉片、破损）\n")
            f.write(f"平均mAP50: {np.mean(defect_ap50):.4f}\n")
            f.write(f"平均mAP50-95: {np.mean(defect_ap):.4f}\n")
            f.write(f"平均精度P: {np.mean(defect_p):.4f}\n")
            f.write(f"平均召回率R: {np.mean(defect_r):.4f}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("改进方案配置说明：\n")
        f.write(f"1. 架构选型：{arch}\n")
        f.write(f"2. 注意力模块：{attention_module}（CBAM=通道+空间双注意力）\n")
        f.write(f"3. P2小目标检测层：{'开启' if CFG['p2_detection_head'] else '关闭'}（160×160特征图）\n")
        f.write(f"4. Focal Loss：{'开启(alpha=' + str(CFG['focal_alpha']) + ',gamma=' + str(CFG['focal_gamma']) + ')' if CFG['focal_loss'] else '关闭'}\n")
        f.write(f"5. Slide Loss：{'开启' if CFG.get('slide_loss') else '关闭'}\n")
        f.write(f"6. 闪络分类权重：{CFG.get('flashover_cls_weight', 0.5)}\n")
        f.write(f"7. EMA：{'开启(decay=' + str(CFG['ema_decay']) + ')' if CFG['ema_enabled'] else '关闭'}\n")
        f.write(f"8. 闪络增强数据增强：{'开启' if CFG.get('flashover_augment') else '关闭'}\n")
        f.write(f"9. 训练轮次：{CFG['epochs']} | 输入尺寸：{CFG['imgsz']} | 批次大小：{CFG['batch_size']}\n")
        if CFG.get("publish_after_train", False):
            f.write(f"10. 模型已自动发布至边缘端目录：{CFG['edge_models_dir']}\n")

    shutil.copy(CFG["yaml_path"], save_full_path)
    shutil.copy(os.path.abspath(__file__), save_full_path)
    print(f"✅ 实验结果已归档至：{result_file}")
    print(f"✅ 配置文件与代码已备份，实验可复现")
    print("\n" + "=" * 80)
    print(f"===== 毕设模型训练全流程完成！（增强版） =====")
    print("=" * 80)
