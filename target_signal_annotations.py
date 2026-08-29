"""Canonical annotations and derived datasets for current target signals."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class AnnotationValidationError(ValueError):
    """Raised when acquisition metadata or manual annotations are invalid."""


class Camera(str, Enum):
    A = "A"
    B = "B"


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VAL = "val"


class DisplayConfiguration(str, Enum):
    ULH = "ULH"
    LH = "LH"
    EMPTY_H = "空H"


class Observation(str, Enum):
    U = "U"
    L = "L"
    H = "H"
    UNLIT = "UNLIT"
    UNREADABLE = "UNREADABLE"


DETECTOR_CLASSES = {
    DisplayConfiguration.ULH: (0, "TARGET_ULH"),
    DisplayConfiguration.LH: (1, "TARGET_LH"),
    DisplayConfiguration.EMPTY_H: (2, "TARGET_EMPTY_H"),
}

COLOR_TARGETS = {
    Observation.U: (1, 0, 0),
    Observation.L: (0, 1, 0),
    Observation.H: (0, 0, 1),
    Observation.UNLIT: (0, 0, 0),
}

ALLOWED_OBSERVATIONS = {
    DisplayConfiguration.ULH: {
        Observation.U,
        Observation.L,
        Observation.H,
        Observation.UNLIT,
        Observation.UNREADABLE,
    },
    DisplayConfiguration.LH: {
        Observation.L,
        Observation.H,
        Observation.UNLIT,
        Observation.UNREADABLE,
    },
    DisplayConfiguration.EMPTY_H: {
        Observation.H,
        Observation.UNLIT,
        Observation.UNREADABLE,
    },
}

_SAFE_FRAME_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _required(mapping: Mapping[str, Any], field: str, context: str) -> Any:
    if field not in mapping:
        raise AnnotationValidationError(
            f"{context}: missing required field '{field}'"
        )
    return mapping[field]


def _string(value: Any, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnnotationValidationError(
            f"{context}: field '{field}' must be a non-empty string"
        )
    return value.strip()


def _integer(value: Any, field: str, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AnnotationValidationError(
            f"{context}: field '{field}' must be an integer >= {minimum}"
        )
    return value


def _enum(enum_type: type[Enum], value: Any, field: str, context: str) -> Enum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        allowed = ", ".join(str(item.value) for item in enum_type)
        raise AnnotationValidationError(
            f"{context}: field '{field}' must be one of: {allowed}"
        ) from error


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        image_width: int,
        image_height: int,
        context: str,
    ) -> "BoundingBox":
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != 4
        ):
            raise AnnotationValidationError(
                f"{context}: bbox_xyxy must contain four finite numbers"
            )

        coordinates = []
        for coordinate in value:
            if (
                isinstance(coordinate, bool)
                or not isinstance(coordinate, (int, float))
                or not math.isfinite(coordinate)
            ):
                raise AnnotationValidationError(
                    f"{context}: bbox_xyxy must contain four finite numbers"
                )
            coordinates.append(float(coordinate))

        x1, y1, x2, y2 = coordinates
        if not (0 <= x1 < x2 <= image_width):
            raise AnnotationValidationError(
                f"{context}: bbox x coordinates must satisfy "
                f"0 <= x1 < x2 <= {image_width}"
            )
        if not (0 <= y1 < y2 <= image_height):
            raise AnnotationValidationError(
                f"{context}: bbox y coordinates must satisfy "
                f"0 <= y1 < y2 <= {image_height}"
            )
        return cls(x1=x1, y1=y1, x2=x2, y2=y2)

    def as_list(self) -> list[float | int]:
        return [_compact(value) for value in (self.x1, self.y1, self.x2, self.y2)]

    def as_yolo(self, image_width: int, image_height: int) -> tuple[float, ...]:
        width = self.x2 - self.x1
        height = self.y2 - self.y1
        return (
            (self.x1 + self.x2) / 2 / image_width,
            (self.y1 + self.y2) / 2 / image_height,
            width / image_width,
            height / image_height,
        )


def _compact(value: float) -> float | int:
    return int(value) if value.is_integer() else value


@dataclass(frozen=True)
class AcquisitionFrame:
    frame_id: str
    image_path: str
    camera: Camera
    video_id: str
    frame_index: int
    timestamp_ms: int
    section_id: str
    split: DatasetSplit
    image_width: int
    image_height: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AcquisitionFrame":
        if not isinstance(value, Mapping):
            raise AnnotationValidationError("metadata row must be an object")
        context = "metadata"
        frame_id = _string(
            _required(value, "frame_id", context), "frame_id", context
        )
        context = f"metadata[{frame_id}]"
        if not _SAFE_FRAME_ID.fullmatch(frame_id):
            raise AnnotationValidationError(
                f"{context}: frame_id may contain only letters, digits, '.', '_' and '-'"
            )

        return cls(
            frame_id=frame_id,
            image_path=_string(
                _required(value, "image_path", context), "image_path", context
            ),
            camera=_enum(
                Camera,
                _required(value, "camera", context),
                "camera",
                context,
            ),
            video_id=_string(
                _required(value, "video_id", context), "video_id", context
            ),
            frame_index=_integer(
                _required(value, "frame_index", context),
                "frame_index",
                context,
            ),
            timestamp_ms=_integer(
                _required(value, "timestamp_ms", context),
                "timestamp_ms",
                context,
            ),
            section_id=_string(
                _required(value, "section_id", context), "section_id", context
            ),
            split=_enum(
                DatasetSplit,
                _required(value, "split", context),
                "split",
                context,
            ),
            image_width=_integer(
                _required(value, "image_width", context),
                "image_width",
                context,
                minimum=1,
            ),
            image_height=_integer(
                _required(value, "image_height", context),
                "image_height",
                context,
                minimum=1,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "image_path": self.image_path,
            "camera": self.camera.value,
            "video_id": self.video_id,
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "section_id": self.section_id,
            "split": self.split.value,
            "image_width": self.image_width,
            "image_height": self.image_height,
        }


@dataclass(frozen=True)
class TargetAnnotation:
    bbox: BoundingBox
    configuration: DisplayConfiguration
    observation: Observation

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], source: AcquisitionFrame
    ) -> "TargetAnnotation":
        context = f"annotation[{source.frame_id}].target"
        if not isinstance(value, Mapping):
            raise AnnotationValidationError(f"{context}: target must be an object")

        configuration = _enum(
            DisplayConfiguration,
            _required(value, "configuration", context),
            "configuration",
            context,
        )
        observation = _enum(
            Observation,
            _required(value, "observation", context),
            "observation",
            context,
        )
        if observation not in ALLOWED_OBSERVATIONS[configuration]:
            raise AnnotationValidationError(
                f"{context}: observation '{observation.value}' is not valid for "
                f"configuration '{configuration.value}'"
            )

        return cls(
            bbox=BoundingBox.from_value(
                _required(value, "bbox_xyxy", context),
                image_width=source.image_width,
                image_height=source.image_height,
                context=context,
            ),
            configuration=configuration,
            observation=observation,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_xyxy": self.bbox.as_list(),
            "configuration": self.configuration.value,
            "observation": self.observation.value,
        }


@dataclass(frozen=True)
class AnnotationRecord:
    source: AcquisitionFrame
    target: TargetAnnotation | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "target": None if self.target is None else self.target.to_dict(),
        }


def _manual_target(
    value: Mapping[str, Any], source: AcquisitionFrame
) -> TargetAnnotation | None:
    context = f"annotation[{source.frame_id}]"
    if not isinstance(value, Mapping):
        raise AnnotationValidationError(f"{context}: annotation row must be an object")
    targets = _required(value, "targets", context)
    if not isinstance(targets, list):
        raise AnnotationValidationError(f"{context}: targets must be a list")
    if len(targets) > 1:
        raise AnnotationValidationError(
            f"{context}: at most one current target is allowed"
        )
    if not targets:
        return None
    return TargetAnnotation.from_mapping(targets[0], source)


def build_annotation_records(
    metadata_rows: Iterable[Mapping[str, Any]],
    annotation_rows: Iterable[Mapping[str, Any]],
) -> list[AnnotationRecord]:
    """Merge generated acquisition metadata with minimal manual annotations."""

    sources: dict[str, AcquisitionFrame] = {}
    source_order: list[str] = []
    for row in metadata_rows:
        source = AcquisitionFrame.from_mapping(row)
        if source.frame_id in sources:
            raise AnnotationValidationError(
                f"duplicate acquisition metadata for frame_id '{source.frame_id}'"
            )
        sources[source.frame_id] = source
        source_order.append(source.frame_id)

    manual: dict[str, Mapping[str, Any]] = {}
    for row in annotation_rows:
        if not isinstance(row, Mapping):
            raise AnnotationValidationError("manual annotation row must be an object")
        frame_id = _string(
            _required(row, "frame_id", "manual annotation"),
            "frame_id",
            "manual annotation",
        )
        if frame_id in manual:
            raise AnnotationValidationError(
                f"duplicate manual annotation for frame_id '{frame_id}'"
            )
        manual[frame_id] = row

    unknown_frames = sorted(set(manual) - set(sources))
    if unknown_frames:
        raise AnnotationValidationError(
            "manual annotation has no acquisition metadata for frame_id: "
            + ", ".join(unknown_frames)
        )

    missing_frames = sorted(set(sources) - set(manual))
    if missing_frames:
        raise AnnotationValidationError(
            "missing manual annotation for frame_id: " + ", ".join(missing_frames)
        )

    return [
        AnnotationRecord(
            source=sources[frame_id],
            target=_manual_target(manual[frame_id], sources[frame_id]),
        )
        for frame_id in source_order
    ]


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise AnnotationValidationError(
                        f"{path}:{line_number}: invalid JSON: {error.msg}"
                    ) from error
                if not isinstance(row, Mapping):
                    raise AnnotationValidationError(
                        f"{path}:{line_number}: each JSONL row must be an object"
                    )
                rows.append(row)
    except OSError as error:
        raise AnnotationValidationError(f"cannot read {path}: {error}") from error
    return rows


def load_annotation_records(
    metadata_path: Path, annotations_path: Path
) -> list[AnnotationRecord]:
    return build_annotation_records(
        _load_jsonl(metadata_path), _load_jsonl(annotations_path)
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def annotation_summary(records: Sequence[AnnotationRecord]) -> dict[str, int]:
    detector_targets = sum(record.target is not None for record in records)
    no_target_frames = len(records) - detector_targets
    unreadable_color_samples = sum(
        record.target is not None
        and record.target.observation is Observation.UNREADABLE
        for record in records
    )
    color_samples = sum(
        record.target is not None
        and record.target.observation is not Observation.UNREADABLE
        for record in records
    )
    return {
        "records": len(records),
        "detector_targets": detector_targets,
        "no_target_frames": no_target_frames,
        "color_samples": color_samples,
        "unreadable_color_samples": unreadable_color_samples,
    }


def export_annotation_dataset(
    records: Sequence[AnnotationRecord], output_dir: Path
) -> dict[str, int]:
    """Export detector labels and an ROI color manifest from canonical records."""

    if output_dir.exists():
        if not output_dir.is_dir():
            raise AnnotationValidationError(
                f"output path must be a directory: {output_dir}"
            )
        if any(output_dir.iterdir()):
            raise AnnotationValidationError(
                f"output directory must be empty: {output_dir}"
            )

    detector_dir = output_dir / "detector"
    labels_dir = detector_dir / "labels"
    color_dir = output_dir / "color"
    labels_dir.mkdir(parents=True, exist_ok=True)
    color_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        detector_dir / "classes.json",
        {
            str(class_id): class_name
            for class_id, class_name in (
                DETECTOR_CLASSES[configuration]
                for configuration in DisplayConfiguration
            )
        },
    )
    _write_jsonl(
        output_dir / "canonical_manifest.jsonl",
        (record.to_dict() for record in records),
    )

    detector_manifest = []
    color_manifest = []
    for record in records:
        source = record.source
        label_name = f"{source.frame_id}.txt"
        label_path = labels_dir / label_name
        label = ""
        if record.target is not None:
            class_id, _ = DETECTOR_CLASSES[record.target.configuration]
            x_center, y_center, width, height = record.target.bbox.as_yolo(
                source.image_width, source.image_height
            )
            label = (
                f"{class_id} {x_center:.6f} {y_center:.6f} "
                f"{width:.6f} {height:.6f}\n"
            )
        label_path.write_text(label, encoding="utf-8")

        detector_manifest.append(
            {
                **source.to_dict(),
                "has_target": record.target is not None,
                "label_path": f"labels/{label_name}",
            }
        )

        if (
            record.target is not None
            and record.target.observation is not Observation.UNREADABLE
        ):
            color_manifest.append(
                {
                    "frame_id": source.frame_id,
                    "image_path": source.image_path,
                    "camera": source.camera.value,
                    "video_id": source.video_id,
                    "frame_index": source.frame_index,
                    "timestamp_ms": source.timestamp_ms,
                    "section_id": source.section_id,
                    "split": source.split.value,
                    "crop_xyxy": record.target.bbox.as_list(),
                    "configuration": record.target.configuration.value,
                    "observation": record.target.observation.value,
                    "target": list(COLOR_TARGETS[record.target.observation]),
                }
            )

    _write_jsonl(detector_dir / "manifest.jsonl", detector_manifest)
    _write_jsonl(color_dir / "manifest.jsonl", color_manifest)
    summary = annotation_summary(records)
    _write_json(output_dir / "export_summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and export current target signal annotations."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("validate", "export"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--metadata", required=True, type=Path)
        subparser.add_argument("--annotations", required=True, type=Path)
        if command == "export":
            subparser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        records = load_annotation_records(args.metadata, args.annotations)
        if args.command == "validate":
            result = annotation_summary(records)
        else:
            result = export_annotation_dataset(records, args.output)
    except AnnotationValidationError as error:
        print(f"annotation error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
