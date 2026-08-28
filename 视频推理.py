import os
import cv2
import torch
import numpy as np
import yaml
from pathlib import Path
from ultralytics import YOLO
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

# ==================== 配置参数 ====================
ROOT = Path(__file__).resolve().parent
POS_MODEL_PATH = ROOT / "best.pt"
COLOR_MODEL_PATH = ROOT / "best-26.pt"
VIDEO_INPUT = (
    ROOT.parent / "taps-ga-yolo-traffic-light" / "dataset"
    / "信号灯数据" / "output_a.mp4"
)
OUTPUT_DIR = ROOT / "cross_validation_results_revised"
VIDEO_OUTPUT = OUTPUT_DIR / "output_a_revised_statistics.mp4"
CLASS_NAMES_PATH = ROOT / "signal.yaml"
CONF_THRES = 0.5                                                    # 置信度阈值
SAMPLE_INTERVAL = 1                                                 # 采样间隔（每N帧处理一帧，1表示每帧都处理）
MAX_FRAMES = 0                                                      # 最大处理帧数（0表示处理全部）
# ================================================================

# ==================== 颜色英文转中文映射 ====================
COLOR_EN_TO_CN = {
    "Signallight_None": "无",
    "Signallight_White": "白",
    "Signallight_Blue": "蓝",
    "Signallight_Red": "红",
    "Signallight_Yellow": "黄",
    "Signallight_Green": "绿",
    "Signallight_Red_Yellow": "红黄"
}

# ==================== 交叉验证规则 ====================
VALID_MAPPINGS = {
    "绿": ["上", "中"],
    "红": ["下"],
    "黄": ["上"],
    "无": ["无"],
    "白": ["上", "中", "下"],
    "蓝": ["上", "中", "下"]
}


# ==================== 解析位置模型输出 ====================
def parse_position_output(pos_class_name):
    if not pos_class_name or len(pos_class_name) == 0:
        return []
    if len(pos_class_name) == 1:
        return [pos_class_name]
    positions = []
    for i in range(1, len(pos_class_name), 2):
        if i < len(pos_class_name):
            positions.append(pos_class_name[i])
    return positions


# ==================== 解析颜色模型输出 ====================
def parse_color_output(colors_en):
    colors_cn = []
    for en in colors_en:
        cn = COLOR_EN_TO_CN.get(en, en)
        if cn == "红黄":
            colors_cn.append("红")
            colors_cn.append("黄")
        else:
            colors_cn.append(cn)
    return colors_cn


# ==================== 调用位置模型 ====================
def run_position_model(model, image, conf_threshold=0.5):
    try:
        results = model(image, verbose=False)
        if results[0].probs is not None:
            probs = results[0].probs
            top1 = probs.top1
            class_names = results[0].names
            raw_output = class_names[top1]
            positions = parse_position_output(raw_output)
            conf = float(probs.data[top1])
            return positions, raw_output, conf
        else:
            return [], "", 0.0
    except Exception as e:
        print(f"[位置模型] 推理出错: {e}")
        return [], "", 0.0


# ==================== 调用颜色模型 ====================
def run_color_model(model, image, conf_threshold=0.5):
    try:
        results = model(image, conf=conf_threshold, verbose=False)
        colors_en = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            class_ids = boxes.cls.cpu().numpy().astype(int)
            class_names = results[0].names
            for cls_id in class_ids:
                en_name = class_names[cls_id]
                colors_en.append(en_name)
        colors_cn = parse_color_output(colors_en)
        return colors_cn, colors_en
    except Exception as e:
        print(f"[颜色模型] 推理出错: {e}")
        return [], []


# ==================== 单对交叉验证 ====================
def cross_validate_pair(position, color):
    if color == "无" and position == "无":
        return {"passed": True, "reason": "无对应无"}
    if color == "无" and position != "无":
        return {"passed": False, "reason": f"颜色为无，但位置为{position}"}
    if position == "无" and color != "无":
        return {"passed": False, "reason": f"位置为无，但颜色为{color}"}
    if color not in VALID_MAPPINGS:
        return {"passed": False, "reason": f"未知颜色: {color}"}
    allowed = VALID_MAPPINGS[color]
    if position in allowed:
        return {"passed": True, "reason": f"{color}→{position} 合法"}
    else:
        return {"passed": False, "reason": f"{color}不能对应{position}，应匹配{allowed}"}


# ==================== 全部交叉验证 ====================
def cross_validate_all(positions, colors):
    pair_results = []
    n_pos = len(positions)
    n_col = len(colors)

    if n_pos != n_col:
        min_len = min(n_pos, n_col)
        for i in range(min_len):
            res = cross_validate_pair(positions[i], colors[i])
            pair_results.append({"index": i, "position": positions[i], "color": colors[i],
                                "passed": res["passed"], "reason": res["reason"]})
        for i in range(min_len, n_pos):
            pair_results.append({"index": i, "position": positions[i], "color": "无",
                                "passed": False, "reason": f"位置{positions[i]}无对应颜色"})
        for i in range(min_len, n_col):
            pair_results.append({"index": i, "position": "无", "color": colors[i],
                                "passed": False, "reason": f"颜色{colors[i]}无对应位置"})
    else:
        for i, (pos, col) in enumerate(zip(positions, colors)):
            res = cross_validate_pair(pos, col)
            pair_results.append({"index": i, "position": pos, "color": col,
                                "passed": res["passed"], "reason": res["reason"]})

    strict_all_passed = all(p["passed"] for p in pair_results)
    # 最终输出口径：每个灯位都必须有一个同序颜色且规则一致。
    # 颜色模型多出的结果不再拖累整帧统计，但仍保留在pair_results中审计。
    final_all_passed = (
        n_pos > 0
        and n_col >= n_pos
        and all(pair_results[i]["passed"] for i in range(n_pos))
    )
    for i, pair in enumerate(pair_results):
        pair["included_in_final"] = (
            final_all_passed and i < n_pos and pair["passed"]
        )
    passed_count = sum(1 for p in pair_results if p["passed"])
    ignored_extra_colors = max(0, n_col - n_pos) if final_all_passed else 0

    return {
        "all_passed": final_all_passed,
        "strict_all_passed": strict_all_passed,
        "corrected_by_surplus_filter": (
            final_all_passed and not strict_all_passed
        ),
        "ignored_extra_colors": ignored_extra_colors,
        "pair_results": pair_results,
        "total_pairs": len(pair_results),
        "passed_pairs": passed_count,
        "failed_pairs": len(pair_results) - passed_count
    }


# ==================== 处理单帧 ====================
def process_frame(pos_model, color_model, frame, frame_idx, conf_threshold=0.5):
    """
    处理视频中的单帧图像
    """
    # 1. 位置模型
    positions, raw_pos, pos_conf = run_position_model(pos_model, frame, conf_threshold)

    # 2. 颜色模型
    colors_cn, colors_en = run_color_model(color_model, frame, conf_threshold)

    # 3. 交叉验证
    validation = cross_validate_all(positions, colors_cn)

    return {
        "frame_idx": frame_idx,
        "raw_position": raw_pos,
        "pos_conf": pos_conf,
        "parsed_positions": positions,
        "raw_colors_en": colors_en,
        "parsed_colors": colors_cn,
        "validation": validation
    }


# ==================== 在帧上绘制结果 ====================
def draw_on_frame(frame, result):
    """
    在视频帧上绘制验证结果
    """
    h, w = frame.shape[:2]
    
    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # 整体状态
    all_pass = result["validation"]["all_passed"]
    corrected = result["validation"]["corrected_by_surplus_filter"]
    status_text = (
        "PASS (SURPLUS IGNORED)" if corrected
        else "ALL PASS" if all_pass
        else "FAIL"
    )
    color = (0, 255, 0) if all_pass else (0, 0, 255)
    cv2.putText(frame, f"Frame: {result['frame_idx']} | Status: {status_text}", 
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 位置和颜色信息
    cv2.putText(frame, f"Position: {result['raw_position']} (conf={result['pos_conf']:.2f})", 
                (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    color_text = ", ".join(result['raw_colors_en']) if result['raw_colors_en'] else "None"
    cv2.putText(frame, f"Colors: {color_text}", 
                (15, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 每个灯对的详细结果
    y_offset = 112
    for p in result["validation"]["pair_results"]:
        status = "PASS" if p["passed"] else "FAIL"
        pos_en = {"上": "UP", "中": "MID", "下": "DOWN", "无": "NONE"}.get(p["position"], p["position"])
        col_en = {"红": "RED", "绿": "GREEN", "黄": "YELLOW", "无": "NONE",
                  "白": "WHITE", "蓝": "BLUE"}.get(p["color"], p["color"])
        
        text = f"#{p['index']+1} Pos={pos_en} Col={col_en} [{status}]"
        text_color = (0, 255, 0) if p["passed"] else (0, 0, 255)
        cv2.putText(frame, text, (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1)
        y_offset += 25

    return frame


# ==================== 生成视频统计报告 ====================
def generate_video_reports(all_results, output_dir):
    """
    生成视频处理的统计报告
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_frames = len(all_results)
    passed_frames = sum(1 for r in all_results if r["validation"]["all_passed"])
    failed_frames = total_frames - passed_frames
    strict_passed_frames = sum(
        1 for r in all_results if r["validation"]["strict_all_passed"])
    corrected_frames = sum(
        1 for r in all_results
        if r["validation"]["corrected_by_surplus_filter"]
    )
    ignored_extra_colors = sum(
        r["validation"]["ignored_extra_colors"] for r in all_results
    )

    total_pairs = sum(r["validation"]["total_pairs"] for r in all_results)
    passed_pairs = sum(r["validation"]["passed_pairs"] for r in all_results)
    failed_pairs = total_pairs - passed_pairs

    # 统计 (位置, 颜色) 组合
    pair_counts = defaultdict(int)
    pair_pass_counts = defaultdict(int)

    for result in all_results:
        for p in result["validation"]["pair_results"]:
            key = (p["position"], p["color"])
            pair_counts[key] += 1
            if p["passed"]:
                pair_pass_counts[key] += 1

    # 构建混淆矩阵
    all_positions = sorted(set(k[0] for k in pair_counts.keys()))
    all_colors = sorted(set(k[1] for k in pair_counts.keys()))

    for pos in ["上", "中", "下", "无"]:
        if pos not in all_positions:
            all_positions.append(pos)
    for col in ["红", "绿", "黄", "无", "白", "蓝"]:
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

    # 绘制混淆矩阵
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=all_colors, yticklabels=all_positions,
                ax=axes[0], linewidths=0.5, linecolor='gray')
    axes[0].set_title('Position × Color Pair Count (Video)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Color', fontsize=12)
    axes[0].set_ylabel('Position', fontsize=12)

    with np.errstate(divide='ignore', invalid='ignore'):
        pass_rate = np.divide(pass_matrix, conf_matrix, where=conf_matrix > 0)
        pass_rate[conf_matrix == 0] = np.nan

    sns.heatmap(pass_rate, annot=True, fmt='.0%', cmap='RdYlGn',
                xticklabels=all_colors, yticklabels=all_positions,
                ax=axes[1], vmin=0, vmax=1, mask=np.isnan(pass_rate),
                linewidths=0.5, linecolor='gray')
    axes[1].set_title('Pass Rate by Position × Color (Video)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Color', fontsize=12)
    axes[1].set_ylabel('Position', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / "video_confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存: {output_dir / 'video_confusion_matrix.png'}")

    # 绘制帧级通过率趋势图
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))

    # 饼图
    axes2[0, 0].pie([passed_frames, failed_frames],
                    labels=[f'PASS\n{passed_frames} ({passed_frames/total_frames*100:.1f}%)',
                            f'FAIL\n{failed_frames} ({failed_frames/total_frames*100:.1f}%)'],
                    autopct='', colors=['#2ecc71', '#e74c3c'], startangle=90)
    axes2[0, 0].set_title(f'Frame Level ({total_frames} frames)', fontsize=13, fontweight='bold')

    # 柱状图
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
    plt.savefig(output_dir / "video_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"汇总统计已保存: {output_dir / 'video_summary.png'}")

    return {
        "total_frames": total_frames,
        "passed_frames": passed_frames,
        "failed_frames": failed_frames,
        "strict_passed_frames": strict_passed_frames,
        "corrected_frames": corrected_frames,
        "ignored_extra_colors": ignored_extra_colors,
        "total_pairs": total_pairs,
        "passed_pairs": passed_pairs,
        "failed_pairs": failed_pairs
    }


# ==================== 保存JSON结果 ====================
def save_json_results(all_results, stats, output_dir):
    output_dir = Path(output_dir)
    json_data = {
        "summary": {
            "total_frames": stats["total_frames"],
            "passed_frames": stats["passed_frames"],
            "failed_frames": stats["failed_frames"],
            "frame_pass_rate": f"{stats['passed_frames']/stats['total_frames']*100:.1f}%",
            "strict_passed_frames": stats["strict_passed_frames"],
            "strict_frame_pass_rate": (
                f"{stats['strict_passed_frames']/stats['total_frames']*100:.1f}%"
            ),
            "corrected_frames": stats["corrected_frames"],
            "ignored_extra_colors": stats["ignored_extra_colors"],
            "total_pairs": stats["total_pairs"],
            "passed_pairs": stats["passed_pairs"],
            "failed_pairs": stats["failed_pairs"],
            "pair_pass_rate": f"{stats['passed_pairs']/stats['total_pairs']*100:.1f}%"
        },
        "frames": []
    }

    for r in all_results:
        frame_data = {
            "frame_idx": r["frame_idx"],
            "raw_position": r["raw_position"],
            "parsed_positions": r["parsed_positions"],
            "raw_colors": r["raw_colors_en"],
            "parsed_colors": r["parsed_colors"],
            "all_passed": r["validation"]["all_passed"],
            "strict_all_passed": r["validation"]["strict_all_passed"],
            "corrected_by_surplus_filter": (
                r["validation"]["corrected_by_surplus_filter"]
            ),
            "ignored_extra_colors": r["validation"]["ignored_extra_colors"],
            "lights": []
        }
        for p in r["validation"]["pair_results"]:
            frame_data["lights"].append({
                "light_index": p["index"] + 1,
                "position": p["position"],
                "color": p["color"],
                "passed": p["passed"],
                "included_in_final": p["included_in_final"],
                "reason": p["reason"]
            })
        json_data["frames"].append(frame_data)

    json_path = output_dir / "video_detailed_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"详细结果已保存: {json_path}")
    return json_path


# ==================== 主程序（视频模式） ====================
def main():
    print("=" * 80)
    print("  灯位 × 颜色 双模型交叉验证系统 - 视频模式")
    print("=" * 80)

    # 1. 加载模型
    print(f"\n[1/5] 加载模型...")
    print(f"  位置模型: {POS_MODEL_PATH}")
    pos_model = YOLO(POS_MODEL_PATH, task='classify')
    print(f"  颜色模型: {COLOR_MODEL_PATH}")
    color_model = YOLO(COLOR_MODEL_PATH)

    # 2. 打开视频
    print(f"\n[2/5] 打开视频: {VIDEO_INPUT}")
    cap = cv2.VideoCapture(str(VIDEO_INPUT))
    if not cap.isOpened():
        print("  错误：无法打开视频文件！")
        return

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  FPS: {fps:.2f}")
    print(f"  分辨率: {width}x{height}")
    print(f"  总帧数: {total_frames}")
    print(f"  采样间隔: {SAMPLE_INTERVAL}帧")
    print(f"  最大处理帧数: {MAX_FRAMES if MAX_FRAMES > 0 else '全部'}")

    # 3. 准备输出视频
    print(f"\n[3/5] 准备输出视频: {VIDEO_OUTPUT}")
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(VIDEO_OUTPUT), fourcc, fps, (width, height))

    # 4. 逐帧处理
    print(f"\n[4/5] 开始处理视频帧...")
    all_results = []
    frame_count = 0
    processed_count = 0
    passed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 采样：每隔 SAMPLE_INTERVAL 帧处理一次
        if frame_count % SAMPLE_INTERVAL != 0:
            # 不处理的帧直接写入（不做标注）
            out.write(frame)
            continue

        # 限制处理帧数
        if MAX_FRAMES > 0 and processed_count >= MAX_FRAMES:
            break

        # 处理当前帧
        result = process_frame(pos_model, color_model, frame, frame_count, CONF_THRES)
        all_results.append(result)

        # 在帧上绘制结果
        annotated_frame = draw_on_frame(frame, result)

        # 写入输出视频
        out.write(annotated_frame)

        # 统计
        processed_count += 1
        if result["validation"]["all_passed"]:
            passed_count += 1

        # 打印进度
        if processed_count % 10 == 0 or processed_count == 1:
            progress = processed_count / (total_frames / SAMPLE_INTERVAL) * 100
            print(f"  进度: {processed_count}帧 ({progress:.1f}%) | "
                  f"通过: {passed_count}/{processed_count} "
                  f"({passed_count/processed_count*100:.1f}%)")

    # 清理
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 5. 生成报告
    print(f"\n[5/5] 生成统计报告...")
    stats = generate_video_reports(all_results, output_dir)
    save_json_results(all_results, stats, output_dir)

    # 打印最终汇总
    print(f"\n{'='*80}")
    print(f"  最终汇总")
    print(f"{'='*80}")
    print(f"  视频总帧数:   {frame_count}")
    print(f"  处理帧数:     {processed_count}")
    print(f"  帧通过率:     {passed_count}/{processed_count} ({passed_count/processed_count*100:.1f}%)")
    print(f"  灯对总数:     {stats['total_pairs']}")
    print(f"  灯对通过率:   {stats['passed_pairs']}/{stats['total_pairs']} ({stats['passed_pairs']/stats['total_pairs']*100:.1f}%)")
    print(f"{'='*80}")
    print(f"  输出视频: {VIDEO_OUTPUT}")
    print(f"  输出目录: {output_dir.absolute()}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
