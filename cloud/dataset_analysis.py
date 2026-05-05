# cloud/dataset_analysis.py
"""
数据集分析工具 - 统计类别分布、边界框尺寸分布
用于识别数据不平衡问题，辅助毕设论文实验分析
"""
import os
import sys
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_yaml(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def analyze_labels(label_dir, class_names):
    """分析标签目录，返回各类别实例数量和边界框尺寸分布"""
    label_files = list(Path(label_dir).glob("*.txt"))
    if not label_files:
        return {}, [], {}

    class_counts = defaultdict(int)
    bbox_sizes = {i: [] for i in range(len(class_names))}
    images_per_class = defaultdict(set)
    boxes_per_image = []

    for lf in label_files:
        img_name = lf.stem
        box_count = 0
        with open(lf, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                w, h = float(parts[3]), float(parts[4])
                class_counts[cls_id] += 1
                area = w * h
                bbox_sizes[cls_id].append(area)
                images_per_class[cls_id].add(img_name)
                box_count += 1
        boxes_per_image.append(box_count)

    return class_counts, bbox_sizes, images_per_class, boxes_per_image


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main(yaml_path, class_names_cn=None):
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        print(f"配置文件不存在: {yaml_path}")
        return

    cfg = load_yaml(yaml_path)
    nc = cfg.get("nc", 0)
    names = cfg.get("names", [f"class_{i}" for i in range(nc)])
    names_cn = class_names_cn or {
        0: "闪络(flashover)",
        1: "绝缘子(insulator)",
        2: "掉片(lose)",
        3: "破损(damaged)"
    }

    data_root = Path(cfg["path"])
    print_header("绝缘子缺陷数据集分析报告")
    print(f"  数据根目录: {data_root}")
    print(f"  类别数: {nc}")

    total_images = 0
    total_boxes = 0
    all_class_counts = defaultdict(int)
    all_bbox_sizes = {i: [] for i in range(nc)}
    all_images_per_class = defaultdict(set)

    for split in ["train", "val", "test"]:
        if split not in cfg:
            continue
        label_dir = data_root / cfg[split].replace("images", "labels")
        img_dir = data_root / cfg[split]
        if not label_dir.exists():
            print(f"\n  [{split}] 标签目录不存在: {label_dir}")
            continue

        img_count = len(list(Path(img_dir).glob("*.[jp][np][g]")) + list(Path(img_dir).glob("*.jpeg")))
        class_counts, bbox_sizes, imgs_per_cls, boxes_per_img = analyze_labels(label_dir, names)

        for k, v in class_counts.items():
            all_class_counts[k] += v
        for k, v in bbox_sizes.items():
            all_bbox_sizes[k].extend(v)
        for k, v in imgs_per_cls.items():
            all_images_per_class[k].update(v)

        total_images += img_count
        total_boxes += sum(class_counts.values())

        print(f"\n  [{split.upper()}] 图片: {img_count} | 标注框: {sum(class_counts.values())}")
        for cls_id in sorted(class_counts.keys()):
            print(f"    类别{cls_id} [{names_cn.get(cls_id, names[cls_id])}]: "
                  f"{class_counts[cls_id]} 个实例 ({len(imgs_per_cls[cls_id])} 张图片)")

        if boxes_per_img:
            print(f"    每图平均框数: {np.mean(boxes_per_img):.1f} | "
                  f"中位框数: {np.median(boxes_per_img):.0f} | "
                  f"最大框数: {np.max(boxes_per_img)}")

    # 全局汇总
    print_header("全局汇总")
    print(f"  总图片数: {total_images}")
    print(f"  总标注框数: {total_boxes}")

    print(f"\n  类别分布:")
    for cls_id in sorted(all_class_counts.keys()):
        count = all_class_counts[cls_id]
        ratio = count / total_boxes * 100 if total_boxes > 0 else 0
        name_cn = names_cn.get(cls_id, names[cls_id] if cls_id < len(names) else f"class_{cls_id}")
        print(f"    类别{cls_id} [{name_cn}]: {count} ({ratio:.1f}%)"
              f" | 涉及 {len(all_images_per_class[cls_id])} 张图片")

    # 边界框尺寸分析
    print_header("边界框尺寸分析（相对面积 = w*h，归一化到 [0, 1]）")
    size_bins = {"极小 (<0.01)": (0, 0.01), "小 (0.01-0.04)": (0.01, 0.04),
                 "中 (0.04-0.09)": (0.04, 0.09), "大 (>0.09)": (0.09, 1.0)}

    for cls_id in sorted(all_bbox_sizes.keys()):
        sizes = np.array(all_bbox_sizes[cls_id])
        if len(sizes) == 0:
            continue
        name_cn = names_cn.get(cls_id, f"class_{cls_id}")
        print(f"\n  [{name_cn}] (共 {len(sizes)} 个框):")
        print(f"    平均面积: {np.mean(sizes):.4f} | 中位面积: {np.median(sizes):.4f}")
        print(f"    最小: {np.min(sizes):.4f} | 最大: {np.max(sizes):.4f}")
        for bin_name, (lo, hi) in size_bins.items():
            cnt = np.sum((sizes >= lo) & (sizes < hi))
            pct = cnt / len(sizes) * 100
            bar = "█" * int(pct / 2)
            print(f"    {bin_name:20s}: {cnt:5d} ({pct:5.1f}%) {bar}")

    # 数据不平衡分析
    print_header("数据平衡分析")
    if all_class_counts:
        max_cls = max(all_class_counts, key=all_class_counts.get)
        min_cls = min(all_class_counts, key=all_class_counts.get)
        imbalance_ratio = all_class_counts[max_cls] / all_class_counts[min_cls]
        print(f"  最多类别: {names_cn.get(max_cls, f'class_{max_cls}')} ({all_class_counts[max_cls]} 个)")
        print(f"  最少类别: {names_cn.get(min_cls, f'class_{min_cls}')} ({all_class_counts[min_cls]} 个)")
        print(f"  不平衡比: {imbalance_ratio:.2f}:1")
        if imbalance_ratio > 3:
            print(f"  ⚠️  数据严重不平衡，建议对少数类别进行过采样或数据增强")

    # 小目标分析
    print_header("小目标分析（面积 < 0.04）")
    for cls_id in sorted(all_bbox_sizes.keys()):
        sizes = np.array(all_bbox_sizes[cls_id])
        small_pct = np.sum(sizes < 0.04) / len(sizes) * 100
        name_cn = names_cn.get(cls_id, f"class_{cls_id}")
        tag = " ⚠️ 小目标占比高" if small_pct > 30 else ""
        print(f"  [{name_cn}]: {small_pct:.1f}% 为小/极小目标{tag}")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    yaml_path = script_dir / "insulator.yaml"
    main(yaml_path)
