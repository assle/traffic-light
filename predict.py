#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 批量推理 + 混淆矩阵生成（含背景类）
根据用户配置：7个信号灯类别 + Background
"""

import os
import time

# Avoid a Windows OpenMP runtime conflict between PyTorch/Ultralytics and
# NumPy/Matplotlib when the confusion matrix is rendered.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import yaml
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ========== 配置参数（请根据实际情况修改）==========
MODEL_PATH = "runs/detect/train-2/weights/best.pt"   # 训练好的模型路径
INPUT_DIR = "dataset_signal/images/train"            # 待推理的图片文件夹
OUTPUT_DIR = "inference_results_6"                    # 结果保存文件夹
LABEL_DIR = "dataset_signal/labels/train"            # 标签文件夹（与图片同名txt）
CONF_THRES = 0.1                                     # 置信度阈值
IOU_THRES = 0.01                                       # NMS 的 IoU 阈值
MATCH_IOU = 0.01                                       # 匹配预测框与真实框的 IoU 阈值
CLASS_NAMES_PATH = "ultraLytics/cfg/datasets/signal.yaml"  # 类别名称文件（YAML）
# ====================================================

def load_ground_truth(label_path, img_shape):
    """加载YOLO格式标签，返回类别id列表和归一化框"""
    classes = []
    boxes_norm = []
    if not os.path.exists(label_path):
        return classes, boxes_norm
    # 使用 utf-8 编码打开，避免中文路径或内容报错
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            x_c = float(parts[1])
            y_c = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            classes.append(cls_id)
            boxes_norm.append([x_c, y_c, w, h])
    return classes, boxes_norm

def denormalize_box(box_norm, img_w, img_h):
    """将归一化框转换为像素坐标 (x1,y1,x2,y2)"""
    x_c, y_c, w, h = box_norm
    x1 = int((x_c - w/2) * img_w)
    y1 = int((y_c - h/2) * img_h)
    x2 = int((x_c + w/2) * img_w)
    y2 = int((y_c + h/2) * img_h)
    return [x1, y1, x2, y2]

def iou(box1, box2):
    """计算两个框的IoU，输入为[x1,y1,x2,y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 > x1 and y2 > y1:
        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter)
    return 0.0

def get_matched_pairs(gt_classes, gt_boxes_norm, pred_boxes, pred_classes, img_w, img_h, iou_thres):
    """
    匹配真实框和预测框
    返回：
        matched_pairs: list of (gt_class, pred_class)
        unmatched_gt: list of gt_class (漏检)
        unmatched_pred: list of pred_class (误检)
    """
    gt_boxes_pixel = [denormalize_box(b, img_w, img_h) for b in gt_boxes_norm]
    n_gt = len(gt_classes)
    n_pred = len(pred_classes)
    matched_gt = set()
    matched_pred = set()
    pairs = []
    # 为每个真实框找最佳匹配
    for i, gt_cls in enumerate(gt_classes):
        best_iou = 0
        best_j = -1
        for j, pred_cls in enumerate(pred_classes):
            if j in matched_pred:
                continue
            iou_val = iou(gt_boxes_pixel[i], pred_boxes[j])
            if iou_val > best_iou and iou_val >= iou_thres and gt_cls == pred_cls:
                best_iou = iou_val
                best_j = j
        if best_j != -1:
            matched_gt.add(i)
            matched_pred.add(best_j)
            pairs.append((gt_cls, pred_classes[best_j]))
    # 未匹配的真实框
    unmatched_gt = [gt_classes[i] for i in range(n_gt) if i not in matched_gt]
    # 未匹配的预测框
    unmatched_pred = [pred_classes[j] for j in range(n_pred) if j not in matched_pred]
    return pairs, unmatched_gt, unmatched_pred

def load_class_names(yaml_path):
    """从YAML文件加载类别名称列表（只返回实际定义的类别）"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    nc = data.get('nc', 0)
    names = data.get('names', {})
    if isinstance(names, dict):
        # 确保按索引顺序生成列表
        class_list = [names[i] for i in range(nc) if i in names]
    elif isinstance(names, list):
        class_list = names[:nc]
    else:
        raise ValueError("无法解析类别名称")
    return class_list

def plot_confusion_matrix(conf_matrix, class_names, output_path, normalize=False):
    """绘制混淆矩阵（包含背景类）"""
    if normalize:
        # 按行归一化
        conf_norm = conf_matrix.astype('float') / (conf_matrix.sum(axis=1, keepdims=True) + 1e-9)
        title = "Normalized Confusion Matrix (with Background)"
        cmap = plt.cm.Blues
        data = conf_norm
    else:
        title = "Confusion Matrix (with Background)"
        cmap = plt.cm.Oranges
        data = conf_matrix

    plt.figure(figsize=(10, 8))
    plt.imshow(data, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    # 添加数值文本
    thresh = data.max() / 2.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if val > 0:
                plt.text(j, i, f"{val:.2f}" if normalize else f"{int(val)}",
                         ha="center", va="center",
                         color="white" if val > thresh else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"混淆矩阵已保存: {output_path}")

def main():
    # 1. 检查模型文件
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path.absolute()}")

    print(f"加载模型: {model_path}")
    model = YOLO(str(model_path))

    # 2. 检查输入目录
    input_dir = Path(INPUT_DIR)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入目录不存在: {input_dir.absolute()}")

    # 3. 创建输出目录
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. 标签目录
    label_dir = Path(LABEL_DIR)
    if not label_dir.is_dir():
        print(f"警告：标签目录 {label_dir} 不存在，无法计算混淆矩阵")

    # 5. 读取类别名称（7个信号灯类别）
    class_names = load_class_names(CLASS_NAMES_PATH)
    print(f"真实类别数: {len(class_names)} -> {class_names}")
    # 添加背景类（用于漏检/误检）
    class_names_with_bg = class_names + ["Background"]
    num_classes = len(class_names)          # 7
    bg_idx = num_classes                    # 背景索引 = 7

    # 6. 收集所有图片文件
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
    img_files = []
    for ext in img_exts:
        img_files.extend(input_dir.glob(f'*{ext}'))
        img_files.extend(input_dir.glob(f'*{ext.upper()}'))
    img_files = list(set(img_files))   # 去重
    print(f"找到 {len(img_files)} 张图片，开始推理...")

    # 7. 初始化混淆矩阵 (8x8)
    conf_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    # 8. 逐张图片处理
    for idx, img_path in enumerate(img_files, 1):
        # 读取图片获取尺寸
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"警告：无法读取图片 {img_path}，跳过")
            continue
        h, w = img_bgr.shape[:2]

        # 模型推理
        inference_start = time.perf_counter()
        results = model(str(img_path), conf=CONF_THRES, iou=IOU_THRES, verbose=False)
        inference_time_ms = (time.perf_counter() - inference_start) * 1000
        print(f"[{idx}/{len(img_files)}] {img_path.name} | inference time: {inference_time_ms:.2f} ms")
        pred_boxes = []      # 像素坐标列表
        pred_classes = []    # 类别id列表
        if results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0])
                pred_boxes.append(xyxy.tolist())
                pred_classes.append(cls_id)

        # 保存带预测框和数量标注的结果图片
        plot_img = results[0].plot()   # BGR格式
        num_pred = len(pred_classes)
        cv2.putText(plot_img, f"Predictions: {num_pred}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), plot_img)
        print(f"[{idx}/{len(img_files)}] {img_path.name} | 预测框数: {num_pred}")

        # 加载对应的真实标签
        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            # 没有标签：所有预测框都是误检（背景->预测类别）
            for pred_cls in pred_classes:
                if pred_cls < num_classes:   # 确保类别有效
                    conf_matrix[bg_idx, pred_cls] += 1
            continue

        gt_classes, gt_boxes_norm = load_ground_truth(label_path, (h, w))

        # 匹配预测框和真实框
        pairs, unmatched_gt, unmatched_pred = get_matched_pairs(
            gt_classes, gt_boxes_norm, pred_boxes, pred_classes, w, h, MATCH_IOU)

        # 更新混淆矩阵
        # 匹配上的对 (真实类别, 预测类别)
        for gt_cls, pred_cls in pairs:
            if gt_cls < num_classes and pred_cls < num_classes:
                conf_matrix[gt_cls, pred_cls] += 1
        # 漏检：真实类别 -> 背景
        for gt_cls in unmatched_gt:
            if gt_cls < num_classes:
                conf_matrix[gt_cls, bg_idx] += 1
        # 误检：背景 -> 预测类别
        for pred_cls in unmatched_pred:
            if pred_cls < num_classes:
                conf_matrix[bg_idx, pred_cls] += 1

    # 9. 生成混淆矩阵图像
    if conf_matrix.sum() > 0:
        plot_confusion_matrix(conf_matrix, class_names_with_bg,
                              output_dir / "confusion_matrix_raw.png", normalize=False)
        plot_confusion_matrix(conf_matrix, class_names_with_bg,
                              output_dir / "confusion_matrix_norm.png", normalize=True)
        tp = np.trace(conf_matrix)           # 对角线元素和（包括背景匹配，但背景通常不会自匹配）
        total = conf_matrix.sum()
        print(f"\n总样本数（框数）: {total}")
        print(f"匹配正确的样本数（对角线）: {tp}")
        print(f"总体准确率: {tp/total:.3f}")
    else:
        print("没有有效的标签或预测数据，无法生成混淆矩阵。")

    print(f"\n推理完成！结果保存至: {output_dir.absolute()}")

if __name__ == "__main__":
    main()
