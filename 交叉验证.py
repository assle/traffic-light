import os
import cv2
import torch
import numpy as np
import yaml
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

# ==================== 配置参数（请修改这里） ====================
POS_MODEL_PATH = "/Users/xiongere/Desktop/信号灯检测/best.pt"       # 位置模型路径
COLOR_MODEL_PATH = "/Users/xiongere/Desktop/信号灯检测/best-26.pt"        # 颜色模型路径
IMAGE_DIR = "/Users/xiongere/Desktop/信号灯检测/a系数据(已筛选)"                       # 验证图片文件夹
CLASS_NAMES_PATH = "/Users/xiongere/Desktop/信号灯检测/signal.yaml"           # 颜色类别yaml文件
CONF_THRES = 0.5                                         # 置信度阈值
OUTPUT_DIR = "cross_validation_results"                  # 输出文件夹
# ============================================================

# ==================== 颜色英文转中文映射 ====================
COLOR_EN_TO_CN = {
    "Signallight_None": "无",
    "Signallight_White": "白",
    "Signallight_Blue": "蓝",
    "Signallight_Red": "红",
    "Signallight_Yellow": "黄",
    "Signallight_Green": "绿",
    "Signallight_Red_Yellow": "红黄"  # 特殊：红+黄同时亮
}

# ==================== 交叉验证规则 ====================
# 颜色 -> 允许的位置列表
VALID_MAPPINGS = {
    "绿": ["上", "中"],       # 绿色可以对应上或中
    "红": ["下"],              # 红色只能对应下
    "黄": ["上"],              # 黄色只能对应上
    "无": ["无"],              # 无只能对应无
    "白": ["上", "中", "下"],  # 白色可对应任意位置
    "蓝": ["上", "中", "下"]   # 蓝色可对应任意位置
}


# ==================== 解析位置模型输出 ====================
def parse_position_output(pos_class_name):
    """
    解析位置模型输出，提取每个信号机的亮灯位置

    规则：
    - 第1个字：第一个信号机在图片中的位置（上/下）
    - 第2个字：第一个信号机亮的灯位（上/中/下/无）
    - 第3个字：第二个信号机在图片中的位置（上/下）
    - 第4个字：第二个信号机亮的灯位（上/中/下/无）
    - 以此类推...

    示例：
    "上下下中" → 第2字"下" + 第4字"中" → ["下", "中"]
    "上"      → ["上"]
    "上下"    → ["下"] (第1字"上"是信号机位置，第2字"下"是亮灯位)
    "上下下无" → ["下", "无"]
    """
    if not pos_class_name or len(pos_class_name) == 0:
        return []

    # 单字输出，直接返回
    if len(pos_class_name) == 1:
        return [pos_class_name]

    # 从位置模型输出中提取亮灯位置
    # 偶数索引(0,2,4...)是信号机位置，奇数索引(1,3,5...)是亮灯位置
    positions = []
    for i in range(1, len(pos_class_name), 2):
        if i < len(pos_class_name):
            positions.append(pos_class_name[i])

    return positions


# ==================== 解析颜色模型输出 ====================
def parse_color_output(colors_en):
    """
    解析颜色模型输出，将每个检测到的颜色转为中文
    特殊处理：红黄同时亮拆分为红和黄两个独立颜色

    参数:
        colors_en: list[str] - 英文颜色列表，如 ["Signallight_Red", "Signallight_Yellow"]

    返回:
        list[str] - 中文颜色列表，如 ["红", "黄"]
    """
    colors_cn = []
    for en in colors_en:
        cn = COLOR_EN_TO_CN.get(en, en)

        # 红黄同时亮 → 拆成红和黄
        if cn == "红黄":
            colors_cn.append("红")
            colors_cn.append("黄")
        else:
            colors_cn.append(cn)

    return colors_cn


# ==================== 调用位置模型（YOLO26分类） ====================
def run_position_model(model, image_path, conf_threshold=0.5):
    """
    调用位置模型，返回解析后的亮灯位置列表和原始输出
    """
    try:
        results = model(image_path)

        if results[0].probs is not None:
            probs = results[0].probs
            top1 = probs.top1
            class_names = results[0].names
            raw_output = class_names[top1]

            # 解析位置
            positions = parse_position_output(raw_output)

            # 打印置信度信息
            conf = float(probs.data[top1])
            print(f"[位置模型] 原始输出: {raw_output} (conf={conf:.3f})")
            print(f"[位置模型] 解析亮灯位置: {positions}")

            return positions, raw_output
        else:
            print(f"[位置模型] 未检测到结果")
            return [], ""

    except Exception as e:
        print(f"[位置模型] 推理出错: {e}")
        return [], ""


# ==================== 调用颜色模型（YOLOv8检测） ====================
def run_color_model(model, image_path, conf_threshold=0.5):
    """
    调用颜色模型，返回中文颜色列表和原始英文颜色列表
    """
    try:
        results = model(image_path, conf=conf_threshold)

        colors_en = []
        colors_cn = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            class_ids = boxes.cls.cpu().numpy().astype(int)
            class_names = results[0].names
            confs = boxes.conf.cpu().numpy()

            for cls_id, conf in zip(class_ids, confs):
                en_name = class_names[cls_id]
                colors_en.append(en_name)
                print(f"  [颜色] {en_name} (conf={conf:.3f})")

        # 解析颜色（红黄拆分）
        colors_cn = parse_color_output(colors_en)

        print(f"[颜色模型] 英文: {colors_en}")
        print(f"[颜色模型] 中文(拆分后): {colors_cn}")

        return colors_cn, colors_en

    except Exception as e:
        print(f"[颜色模型] 推理出错: {e}")
        return [], []


# ==================== 单对交叉验证 ====================
def cross_validate_pair(position, color):
    """
    验证单个 (位置, 颜色) 对是否合法

    参数:
        position: str - 亮灯位置 "上"/"中"/"下"/"无"
        color: str - 颜色 "红"/"绿"/"黄"/"无"/...

    返回:
        dict - {"passed": bool, "reason": str}
    """
    # 无对应无
    if color == "无" and position == "无":
        return {"passed": True, "reason": "无对应无，验证通过"}

    if color == "无" and position != "无":
        return {"passed": False, "reason": f"颜色为无，但位置为{position}"}

    if position == "无" and color != "无":
        return {"passed": False, "reason": f"位置为无，但颜色为{color}"}

    # 检查颜色是否在验证规则中
    if color not in VALID_MAPPINGS:
        return {"passed": False, "reason": f"未知颜色: {color}"}

    # 检查位置是否合法
    allowed = VALID_MAPPINGS[color]
    if position in allowed:
        return {"passed": True, "reason": f"{color}→{position} 合法"}
    else:
        return {"passed": False, "reason": f"{color}不能对应{position}，应匹配{allowed}"}


# ==================== 全部交叉验证 ====================
def cross_validate_all(positions, colors):
    """
    对所有灯对进行交叉验证

    参数:
        positions: list[str] - 亮灯位置列表，如 ["下", "中"]
        colors: list[str] - 颜色列表（已拆分红黄），如 ["红", "黄"]

    返回:
        dict - 完整的验证结果
    """
    pair_results = []

    n_pos = len(positions)
    n_col = len(colors)

    # 数量不一致的情况
    if n_pos != n_col:
        min_len = min(n_pos, n_col)

        # 能配对的先配对
        for i in range(min_len):
            res = cross_validate_pair(positions[i], colors[i])
            pair_results.append({
                "index": i,
                "position": positions[i],
                "color": colors[i],
                "passed": res["passed"],
                "reason": res["reason"]
            })

        # 多余的位置（没有颜色对应）
        for i in range(min_len, n_pos):
            pair_results.append({
                "index": i,
                "position": positions[i],
                "color": "无",
                "passed": False,
                "reason": f"位置{positions[i]}无对应颜色"
            })

        # 多余的颜色（没有位置对应）
        for i in range(min_len, n_col):
            pair_results.append({
                "index": i,
                "position": "无",
                "color": colors[i],
                "passed": False,
                "reason": f"颜色{colors[i]}无对应位置"
            })
    else:
        # 数量一致，逐对验证
        for i, (pos, col) in enumerate(zip(positions, colors)):
            res = cross_validate_pair(pos, col)
            pair_results.append({
                "index": i,
                "position": pos,
                "color": col,
                "passed": res["passed"],
                "reason": res["reason"]
            })

    # 整体是否通过
    all_passed = all(p["passed"] for p in pair_results)
    passed_count = sum(1 for p in pair_results if p["passed"])

    return {
        "all_passed": all_passed,
        "pair_results": pair_results,
        "total_pairs": len(pair_results),
        "passed_pairs": passed_count,
        "failed_pairs": len(pair_results) - passed_count
    }


# ==================== 处理单张图片 ====================
def process_single_image(pos_model, color_model, image_path, conf_threshold=0.5):
    """
    处理单张图片：位置推理 + 颜色推理 + 交叉验证
    """
    print(f"\n{'='*60}")
    print(f"图片: {Path(image_path).name}")
    print(f"{'='*60}")

    # 1. 位置模型
    positions, raw_pos = run_position_model(pos_model, image_path, conf_threshold)

    # 2. 颜色模型
    colors_cn, colors_en = run_color_model(color_model, image_path, conf_threshold)

    # 3. 交叉验证
    validation = cross_validate_all(positions, colors_cn)

    # 4. 打印结果
    print(f"\n{'─'*40}")
    print(f"验证结果:")
    print(f"  位置模型原始输出: {raw_pos}")
    print(f"  解析亮灯位置:    {positions}")
    print(f"  颜色模型输出:    {colors_en}")
    print(f"  颜色(拆分后):    {colors_cn}")
    print(f"  整体状态: {'✓ 全部通过' if validation['all_passed'] else '✗ 存在失败'}")

    for p in validation["pair_results"]:
        status = "✓" if p["passed"] else "✗"
        print(f"  {status} 灯{p['index']+1}: 位置={p['position']}, 颜色={p['color']} → {p['reason']}")

    return {
        "image_path": str(image_path),
        "raw_position": raw_pos,
        "parsed_positions": positions,
        "raw_colors_en": colors_en,
        "parsed_colors": colors_cn,
        "validation": validation
    }


# ==================== 可视化结果（修复中文显示问题） ====================
def visualize_result(image_path, result, output_dir):
    """
    在图片上标注验证结果（使用英文避免中文乱码）
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    # 整体状态
    all_pass = result["validation"]["all_passed"]
    status_text = "ALL PASS" if all_pass else "FAIL"
    color = (0, 255, 0) if all_pass else (0, 0, 255)

    cv2.rectangle(img, (0, 0), (img.shape[1], 120), (0, 0, 0), -1)
    cv2.putText(img, f"Status: {status_text}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # 原始位置输出（直接用英文显示原始类别名）
    cv2.putText(img, f"Position: {result['raw_position']}", (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 颜色输出
    color_text = ", ".join(result['raw_colors_en'])
    cv2.putText(img, f"Colors: {color_text}", (15, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 每个灯对的详细结果（用英文缩写）
    y_offset = 140
    for p in result["validation"]["pair_results"]:
        status = "PASS" if p["passed"] else "FAIL"
        # 位置用英文：UP/MID/DOWN/NONE
        pos_en = {"上": "UP", "中": "MID", "下": "DOWN", "无": "NONE"}.get(p["position"], p["position"])
        # 颜色用英文
        col_en = {"红": "RED", "绿": "GREEN", "黄": "YELLOW", "无": "NONE", 
                  "白": "WHITE", "蓝": "BLUE"}.get(p["color"], p["color"])
        
        text = f"#{p['index']+1} Pos={pos_en} Col={col_en} [{status}]"
        text_color = (0, 255, 0) if p["passed"] else (0, 0, 255)
        cv2.putText(img, text, (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        y_offset += 28

    # 保存
    viz_dir = Path(output_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    out_path = viz_dir / f"{Path(image_path).stem}_result.jpg"
    cv2.imwrite(str(out_path), img)

    return str(out_path)


# ==================== 生成混淆矩阵和统计图表 ====================
def generate_reports(all_results, output_dir):
    """
    生成混淆矩阵、统计图表
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== 统计汇总 =====
    total_images = len(all_results)
    passed_images = sum(1 for r in all_results if r["validation"]["all_passed"])
    failed_images = total_images - passed_images

    total_pairs = sum(r["validation"]["total_pairs"] for r in all_results)
    passed_pairs = sum(r["validation"]["passed_pairs"] for r in all_results)
    failed_pairs = total_pairs - passed_pairs

    # ===== 统计 (位置, 颜色) 组合 =====
    pair_counts = defaultdict(int)       # 总出现次数
    pair_pass_counts = defaultdict(int)  # 通过次数
    pair_fail_reasons = defaultdict(list) # 失败原因

    for result in all_results:
        for p in result["validation"]["pair_results"]:
            key = (p["position"], p["color"])
            pair_counts[key] += 1
            if p["passed"]:
                pair_pass_counts[key] += 1
            else:
                pair_fail_reasons[key].append(p["reason"])

    # ===== 构建混淆矩阵 =====
    all_positions = sorted(set(k[0] for k in pair_counts.keys()))
    all_colors = sorted(set(k[1] for k in pair_counts.keys()))

    # 确保包含所有可能的类别
    for pos in ["上", "中", "下", "无"]:
        if pos not in all_positions:
            all_positions.append(pos)
    for col in ["红", "绿", "黄", "红黄", "无", "白", "蓝"]:
        if col not in all_colors:
            all_colors.append(col)

    all_positions.sort()
    all_colors.sort()

    n_pos = len(all_positions)
    n_col = len(all_colors)

    conf_matrix = np.zeros((n_pos, n_col), dtype=np.int64)
    pass_matrix = np.zeros((n_pos, n_col), dtype=np.int64)

    for (pos, col), count in pair_counts.items():
        if pos in all_positions and col in all_colors:
            i = all_positions.index(pos)
            j = all_colors.index(col)
            conf_matrix[i, j] = count
            pass_matrix[i, j] = pair_pass_counts.get((pos, col), 0)

    # ===== 图1: 混淆矩阵 + 通过率热力图 =====
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # 左：总出现次数
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=all_colors, yticklabels=all_positions,
                ax=axes[0], linewidths=0.5, linecolor='gray')
    axes[0].set_title('Position × Color Pair Count', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Color', fontsize=12)
    axes[0].set_ylabel('Position', fontsize=12)

    # 右：通过率
    with np.errstate(divide='ignore', invalid='ignore'):
        pass_rate = np.divide(pass_matrix, conf_matrix, where=conf_matrix > 0)
        pass_rate[conf_matrix == 0] = np.nan

    sns.heatmap(pass_rate, annot=True, fmt='.0%', cmap='RdYlGn',
                xticklabels=all_colors, yticklabels=all_positions,
                ax=axes[1], vmin=0, vmax=1, mask=np.isnan(pass_rate),
                linewidths=0.5, linecolor='gray')
    axes[1].set_title('Pass Rate by Position × Color', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Color', fontsize=12)
    axes[1].set_ylabel('Position', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存: {output_dir / 'confusion_matrix.png'}")

    # ===== 图2: 汇总统计 =====
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))

    # 图片级通过率饼图
    axes2[0, 0].pie([passed_images, failed_images],
                    labels=[f'PASS\n{passed_images} ({passed_images/total_images*100:.1f}%)',
                            f'FAIL\n{failed_images} ({(total_images-passed_images)/total_images*100:.1f}%)'],
                    autopct='', colors=['#2ecc71', '#e74c3c'], startangle=90,
                    textprops={'fontsize': 11})
    axes2[0, 0].set_title(f'Image Level ({total_images} images)', fontsize=13, fontweight='bold')

    # 灯对级柱状图
    bars = axes2[0, 1].bar(['PASS', 'FAIL'], [passed_pairs, failed_pairs],
                           color=['#2ecc71', '#e74c3c'], alpha=0.8, width=0.5)
    axes2[0, 1].set_title(f'Pair Level ({total_pairs} pairs)', fontsize=13, fontweight='bold')
    axes2[0, 1].set_ylabel('Count')
    for bar, v in zip(bars, [passed_pairs, failed_pairs]):
        axes2[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         f'{v}\n({v/total_pairs*100:.1f}%)', ha='center', fontsize=10)

    # 各位置通过率
    pos_pass_rates = []
    pos_labels = []
    for pos in all_positions:
        i = all_positions.index(pos)
        total = conf_matrix[i, :].sum()
        passed = pass_matrix[i, :].sum()
        rate = passed / total * 100 if total > 0 else 0
        pos_pass_rates.append(rate)
        pos_labels.append(f"{pos}\n(n={total})")

    bars2 = axes2[1, 0].bar(range(len(pos_labels)), pos_pass_rates, color='#3498db', alpha=0.8)
    axes2[1, 0].set_title('Pass Rate by Position', fontsize=13, fontweight='bold')
    axes2[1, 0].set_ylabel('Pass Rate (%)')
    axes2[1, 0].set_xticks(range(len(pos_labels)))
    axes2[1, 0].set_xticklabels(pos_labels)
    axes2[1, 0].set_ylim(0, 110)
    for bar, rate in zip(bars2, pos_pass_rates):
        axes2[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                         f'{rate:.1f}%', ha='center', fontsize=10)

    # 各颜色通过率
    col_pass_rates = []
    col_labels = []
    for col in all_colors:
        j = all_colors.index(col)
        total = conf_matrix[:, j].sum()
        passed = pass_matrix[:, j].sum()
        rate = passed / total * 100 if total > 0 else 0
        col_pass_rates.append(rate)
        col_labels.append(f"{col}\n(n={total})")

    bars3 = axes2[1, 1].bar(range(len(col_labels)), col_pass_rates, color='#e67e22', alpha=0.8)
    axes2[1, 1].set_title('Pass Rate by Color', fontsize=13, fontweight='bold')
    axes2[1, 1].set_ylabel('Pass Rate (%)')
    axes2[1, 1].set_xticks(range(len(col_labels)))
    axes2[1, 1].set_xticklabels(col_labels)
    axes2[1, 1].set_ylim(0, 110)
    for bar, rate in zip(bars3, col_pass_rates):
        axes2[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                         f'{rate:.1f}%', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "validation_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"汇总统计已保存: {output_dir / 'validation_summary.png'}")

    # ===== 返回统计信息 =====
    return {
        "total_images": total_images,
        "passed_images": passed_images,
        "failed_images": failed_images,
        "total_pairs": total_pairs,
        "passed_pairs": passed_pairs,
        "failed_pairs": failed_pairs,
        "pair_counts": dict(pair_counts),
        "pair_pass_counts": dict(pair_pass_counts)
    }


# ==================== 保存详细JSON结果 ====================
def save_json_results(all_results, stats, output_dir):
    """
    保存详细验证结果到JSON文件
    """
    output_dir = Path(output_dir)

    json_data = {
        "summary": {
            "total_images": stats["total_images"],
            "passed_images": stats["passed_images"],
            "failed_images": stats["failed_images"],
            "image_pass_rate": f"{stats['passed_images']/stats['total_images']*100:.1f}%",
            "total_pairs": stats["total_pairs"],
            "passed_pairs": stats["passed_pairs"],
            "failed_pairs": stats["failed_pairs"],
            "pair_pass_rate": f"{stats['passed_pairs']/stats['total_pairs']*100:.1f}%"
        },
        "images": []
    }

    for r in all_results:
        img_data = {
            "filename": Path(r["image_path"]).name,
            "raw_position": r["raw_position"],
            "parsed_positions": r["parsed_positions"],
            "raw_colors": r["raw_colors_en"],
            "parsed_colors": r["parsed_colors"],
            "all_passed": r["validation"]["all_passed"],
            "lights": []
        }

        for p in r["validation"]["pair_results"]:
            img_data["lights"].append({
                "light_index": p["index"] + 1,
                "position": p["position"],
                "color": p["color"],
                "passed": p["passed"],
                "reason": p["reason"]
            })

        json_data["images"].append(img_data)

    json_path = output_dir / "detailed_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"详细结果已保存: {json_path}")
    return json_path


# ==================== 主程序 ====================
def main():
    print("=" * 60)
    print("  灯位 × 颜色 双模型交叉验证系统")
    print("=" * 60)

    # 1. 加载模型
    print(f"\n[1/4] 加载模型...")

    print(f"  位置模型: {POS_MODEL_PATH}")
    try:
        # 先尝试新版加载
        pos_model = YOLO(POS_MODEL_PATH, task='classify')
    except:
        # 如果失败，用旧版兼容方式
        import torch
        from ultralytics.nn.tasks import ClassificationModel

        # 加载权重
        ckpt = torch.load(POS_MODEL_PATH, map_location='cpu')

        # 创建模型实例
        model = ClassificationModel(cfg='ultralytics/cfg/models/v8/yolov8-cls.yaml', ch=3, nc=7)
        model.load_state_dict(ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model'])

        # 包装成 YOLO 对象
        pos_model = YOLO()
        pos_model.model = model

    print(f"  颜色模型: {COLOR_MODEL_PATH}")
    color_model = YOLO(COLOR_MODEL_PATH)

    # 2. 获取图片列表
    print(f"\n[2/4] 扫描图片文件夹: {IMAGE_DIR}")
    img_dir = Path(IMAGE_DIR)
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
    img_files = []
    for ext in img_exts:
        img_files.extend(img_dir.glob(f'*{ext}'))
        img_files.extend(img_dir.glob(f'*{ext.upper()}'))
    img_files = sorted(set(img_files))
    print(f"  找到 {len(img_files)} 张图片")

    if len(img_files) == 0:
        print("  错误：没有找到图片文件！请检查 IMAGE_DIR 路径。")
        return

    # 3. 逐张处理
    print(f"\n[3/4] 开始交叉验证...")
    all_results = []

    for idx, img_path in enumerate(img_files, 1):
        print(f"\n--- 进度 [{idx}/{len(img_files)}] ---")
        result = process_single_image(pos_model, color_model, str(img_path), CONF_THRES)
        all_results.append(result)

        # 可视化
        viz_path = visualize_result(img_path, result, OUTPUT_DIR)
        if viz_path:
            print(f"  → 可视化: {Path(viz_path).name}")

    # 4. 生成报告
    print(f"\n[4/4] 生成统计报告...")
    stats = generate_reports(all_results, OUTPUT_DIR)
    json_path = save_json_results(all_results, stats, OUTPUT_DIR)

    # 5. 打印最终汇总
    print(f"\n{'='*60}")
    print(f"  最终汇总")
    print(f"{'='*60}")
    print(f"  图片总数:   {stats['total_images']}")
    print(f"  图片通过:   {stats['passed_images']} ({stats['passed_images']/stats['total_images']*100:.1f}%)")
    print(f"  图片失败:   {stats['failed_images']} ({stats['failed_images']/stats['total_images']*100:.1f}%)")
    print(f"  灯对总数:   {stats['total_pairs']}")
    print(f"  灯对通过:   {stats['passed_pairs']} ({stats['passed_pairs']/stats['total_pairs']*100:.1f}%)")
    print(f"  灯对失败:   {stats['failed_pairs']} ({(stats['total_pairs']-stats['passed_pairs'])/stats['total_pairs']*100:.1f}%)")
    print(f"{'='*60}")
    print(f"  输出目录: {Path(OUTPUT_DIR).absolute()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
