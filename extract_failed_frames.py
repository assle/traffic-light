#!/usr/bin/env python3
"""Render every revised-statistics failure with boxes and audit labels."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "cross_validation_results_revised/video_detailed_results.json"
VIDEO = (
    ROOT.parent / "taps-ga-yolo-traffic-light/dataset/信号灯数据/output_a.mp4"
)
MODEL = ROOT / "best-26.pt"
OUTPUT = ROOT / "cross_validation_results_revised/failed_frames_annotated"

POSITION_EN = {"上": "UP", "中": "MID", "下": "DOWN", "无": "NONE"}
COLOR_EN = {
    "红": "RED",
    "绿": "GREEN",
    "黄": "YELLOW",
    "无": "NONE",
    "白": "WHITE",
    "蓝": "BLUE",
}


def classify_failure(frame):
    positions = frame["parsed_positions"]
    colors = frame["parsed_colors"]
    if not colors:
        return "no_color"
    if len(colors) < len(positions):
        return "missing_color"
    if any(not pair["passed"] for pair in frame["lights"][:len(positions)]):
        return "rule_conflict"
    return "other"


def put_label(image, text, origin, color, scale=0.55, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(
        image,
        (x, max(0, y - height - baseline - 4)),
        (x + width + 6, y + 3),
        (0, 0, 0),
        -1,
    )
    cv2.putText(image, text, (x + 3, y), font, scale, color, thickness,
                cv2.LINE_AA)


def annotate(frame, record, result, category):
    image = frame.copy()
    positions = record["parsed_positions"]
    lights = record["lights"]

    if result.boxes is not None:
        boxes = result.boxes
        for index, (xyxy, cls_id, confidence) in enumerate(zip(
                boxes.xyxy.cpu().numpy(),
                boxes.cls.cpu().numpy().astype(int),
                boxes.conf.cpu().numpy())):
            x1, y1, x2, y2 = (int(value) for value in xyxy)
            if index >= len(positions):
                status = "SURPLUS"
                color = (0, 165, 255)
            elif index < len(lights) and lights[index]["passed"]:
                status = "PAIR PASS"
                color = (0, 220, 0)
            else:
                status = "PAIR FAIL"
                color = (0, 0, 255)
            class_name = result.names[cls_id].replace("Signallight_", "")
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            put_label(
                image,
                f"#{index + 1} {class_name} {confidence:.3f} {status}",
                (x1, max(22, y1 - 3)),
                color,
            )

    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 155), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.72, image, 0.28, 0)
    pos_text = ",".join(POSITION_EN.get(value, value) for value in positions)
    color_text = ",".join(
        COLOR_EN.get(value, value) for value in record["parsed_colors"])
    pair_text = " | ".join(
        f"#{pair['light_index']} "
        f"{POSITION_EN.get(pair['position'], pair['position'])}+"
        f"{COLOR_EN.get(pair['color'], pair['color'])} "
        f"{'PASS' if pair['passed'] else 'FAIL'}"
        for pair in lights
    )
    put_label(
        image,
        f"Frame {record['frame_idx']}  FINAL FAIL  category={category}",
        (14, 32),
        (0, 0, 255),
        scale=0.75,
        thickness=2,
    )
    put_label(
        image,
        f"Lamp label={record['raw_position']}  slots=[{pos_text}]  colors=[{color_text}]",
        (14, 70),
        (255, 255, 255),
        scale=0.62,
    )
    put_label(image, pair_text or "Pairs: NONE", (14, 108),
              (255, 255, 255), scale=0.55)
    put_label(
        image,
        "Box legend: GREEN=pair pass  RED=pair conflict  ORANGE=surplus",
        (14, 143),
        (180, 220, 255),
        scale=0.52,
    )
    return image


def evenly_spaced(items, limit):
    if len(items) <= limit:
        return list(items)
    return [items[round(i * (len(items) - 1) / (limit - 1))]
            for i in range(limit)]


def build_contact_sheet(paths, destination, columns=4, tile_size=(480, 270)):
    if not paths:
        return
    rows = (len(paths) + columns - 1) // columns
    sheet = 255 * __import__("numpy").ones(
        (rows * tile_size[1], columns * tile_size[0], 3), dtype="uint8")
    for index, path in enumerate(paths):
        image = cv2.imread(str(path))
        if image is None:
            continue
        tile = cv2.resize(image, tile_size, interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        y, x = row * tile_size[1], column * tile_size[0]
        sheet[y:y + tile_size[1], x:x + tile_size[0]] = tile
    cv2.imwrite(str(destination), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    failures = {
        frame["frame_idx"]: frame
        for frame in report["frames"] if not frame["all_passed"]
    }
    categories = defaultdict(list)
    for frame in failures.values():
        categories[classify_failure(frame)].append(frame["frame_idx"])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for category in categories:
        (OUTPUT / "all" / category).mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL)
    capture = cv2.VideoCapture(str(VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO}")

    manifest_rows = []
    written = defaultdict(list)
    frame_idx = 0
    processed = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_idx += 1
        record = failures.get(frame_idx)
        if record is None:
            continue
        category = classify_failure(record)
        result = model(frame, conf=0.5, imgsz=640, device=0, verbose=False)[0]
        annotated = annotate(frame, record, result, category)
        path = OUTPUT / "all" / category / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
        written[category].append(path)
        processed += 1
        manifest_rows.append({
            "frame_idx": frame_idx,
            "timestamp_seconds": f"{(frame_idx - 1) / 10.0:.1f}",
            "category": category,
            "lamp_label": record["raw_position"],
            "lamp_slots": "|".join(record["parsed_positions"]),
            "colors": "|".join(record["parsed_colors"]),
            "color_boxes": 0 if result.boxes is None else len(result.boxes),
            "image": str(path.relative_to(OUTPUT)),
        })
        if processed == 1 or processed % 250 == 0:
            print(f"rendered {processed}/{len(failures)} failed frames", flush=True)
    capture.release()

    with (OUTPUT / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    representative_root = OUTPUT / "representative"
    representative_root.mkdir(exist_ok=True)
    for category, paths in sorted(written.items()):
        selected = evenly_spaced(paths, 24)
        build_contact_sheet(
            selected,
            representative_root / f"{category}_contact_sheet.jpg",
        )
        category_dir = representative_root / category
        category_dir.mkdir(exist_ok=True)
        for path in selected:
            target = category_dir / path.name
            if not target.exists():
                target.symlink_to(path.resolve())

    counts = Counter(row["category"] for row in manifest_rows)
    readme = [
        "# A系失败帧标注集",
        "",
        f"- 总失败帧: {len(manifest_rows)}",
        *[f"- {name}: {count}" for name, count in sorted(counts.items())],
        "",
        "框颜色：绿色=同序灯位规则通过；红色=同序灯位规则冲突；橙色=超出灯位数量。",
        "`all/` 包含全部失败帧，`representative/` 包含均匀抽样和联系表。",
    ]
    (OUTPUT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps({"total": len(manifest_rows), "categories": counts},
                     ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
