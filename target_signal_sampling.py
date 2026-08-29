"""Station grouping and keyframe sampling for target signal annotation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from target_signal_annotations import (
    AnnotationValidationError,
    BoundingBox,
    Camera,
    DatasetSplit,
    Observation,
)


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


def _score(value: Any, field: str, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise AnnotationValidationError(
            f"{context}: field '{field}' must be a finite number from 0 to 1"
        )
    return float(value)


def _enum(enum_type: type, value: Any, field: str, context: str):
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        allowed = ", ".join(str(item.value) for item in enum_type)
        raise AnnotationValidationError(
            f"{context}: field '{field}' must be one of: {allowed}"
        ) from error


@dataclass(frozen=True)
class SamplingConfig:
    max_gap_frames: int = 30
    distance_change_ratio: float = 0.5
    occlusion_threshold: float = 0.7
    exposure_threshold: float = 0.7

    def __post_init__(self) -> None:
        if self.max_gap_frames < 1:
            raise AnnotationValidationError("max_gap_frames must be >= 1")
        if self.distance_change_ratio <= 0:
            raise AnnotationValidationError("distance_change_ratio must be > 0")
        for field in ("occlusion_threshold", "exposure_threshold"):
            value = getattr(self, field)
            if not 0 <= value <= 1:
                raise AnnotationValidationError(f"{field} must be from 0 to 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_gap_frames": self.max_gap_frames,
            "distance_change_ratio": self.distance_change_ratio,
            "occlusion_threshold": self.occlusion_threshold,
            "exposure_threshold": self.exposure_threshold,
        }


@dataclass(frozen=True)
class SectionAssignment:
    station_group_id: str
    station_id: str
    split: DatasetSplit
    section_id: str
    camera: Camera
    video_id: str
    start_frame: int
    end_frame: int

    def contains(self, camera: Camera, video_id: str, frame_index: int) -> bool:
        return (
            self.camera is camera
            and self.video_id == video_id
            and self.start_frame <= frame_index <= self.end_frame
        )


@dataclass(frozen=True)
class StationPlan:
    sections: tuple[SectionAssignment, ...]


def parse_station_plan(value: Mapping[str, Any]) -> StationPlan:
    if not isinstance(value, Mapping):
        raise AnnotationValidationError("station plan must be an object")
    raw_groups = _required(value, "station_groups", "station plan")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise AnnotationValidationError(
            "station plan: station_groups must be a non-empty list"
        )

    sections: list[SectionAssignment] = []
    group_ids: set[str] = set()
    section_ids: set[str] = set()
    station_splits: dict[str, DatasetSplit] = {}

    for group_index, raw_group in enumerate(raw_groups):
        context = f"station_groups[{group_index}]"
        if not isinstance(raw_group, Mapping):
            raise AnnotationValidationError(f"{context}: group must be an object")
        group_id = _string(
            _required(raw_group, "station_group_id", context),
            "station_group_id",
            context,
        )
        if group_id in group_ids:
            raise AnnotationValidationError(
                f"duplicate station_group_id '{group_id}'"
            )
        group_ids.add(group_id)
        station_id = _string(
            _required(raw_group, "station_id", context), "station_id", context
        )
        split = _enum(
            DatasetSplit,
            _required(raw_group, "split", context),
            "split",
            context,
        )
        previous_split = station_splits.get(station_id)
        if previous_split is not None and previous_split is not split:
            raise AnnotationValidationError(
                f"station '{station_id}' is assigned to multiple splits: "
                f"{previous_split.value}, {split.value}"
            )
        station_splits[station_id] = split

        raw_sections = _required(raw_group, "sections", context)
        if not isinstance(raw_sections, list) or not raw_sections:
            raise AnnotationValidationError(
                f"{context}: sections must be a non-empty list"
            )
        for section_index, raw_section in enumerate(raw_sections):
            section_context = f"{context}.sections[{section_index}]"
            if not isinstance(raw_section, Mapping):
                raise AnnotationValidationError(
                    f"{section_context}: section must be an object"
                )
            section_id = _string(
                _required(raw_section, "section_id", section_context),
                "section_id",
                section_context,
            )
            if section_id in section_ids:
                raise AnnotationValidationError(
                    f"duplicate section_id '{section_id}'"
                )
            section_ids.add(section_id)
            camera = _enum(
                Camera,
                _required(raw_section, "camera", section_context),
                "camera",
                section_context,
            )
            video_id = _string(
                _required(raw_section, "video_id", section_context),
                "video_id",
                section_context,
            )
            start_frame = _integer(
                _required(raw_section, "start_frame", section_context),
                "start_frame",
                section_context,
            )
            end_frame = _integer(
                _required(raw_section, "end_frame", section_context),
                "end_frame",
                section_context,
            )
            if end_frame < start_frame:
                raise AnnotationValidationError(
                    f"{section_context}: end_frame must be >= start_frame"
                )
            sections.append(
                SectionAssignment(
                    station_group_id=group_id,
                    station_id=station_id,
                    split=split,
                    section_id=section_id,
                    camera=camera,
                    video_id=video_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            )

    by_source: dict[tuple[Camera, str], list[SectionAssignment]] = defaultdict(list)
    for item in sections:
        by_source[(item.camera, item.video_id)].append(item)
    for (camera, video_id), source_sections in by_source.items():
        ordered = sorted(source_sections, key=lambda item: item.start_frame)
        for previous, current in zip(ordered, ordered[1:]):
            if current.start_frame <= previous.end_frame:
                if current.split is not previous.split:
                    raise AnnotationValidationError(
                        f"frame range for {camera.value}/{video_id} is assigned "
                        f"to multiple splits: {previous.split.value}, "
                        f"{current.split.value}"
                    )
                raise AnnotationValidationError(
                    f"continuous section overlap for {camera.value}/{video_id}: "
                    f"'{previous.section_id}' and '{current.section_id}'"
                )

    return StationPlan(sections=tuple(sections))


@dataclass(frozen=True)
class TargetHint:
    track_id: str
    bbox: BoundingBox
    state_hint: Observation | None
    occlusion_score: float
    exposure_score: float

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        image_width: int,
        image_height: int,
        context: str,
    ) -> "TargetHint":
        if not isinstance(value, Mapping):
            raise AnnotationValidationError(f"{context}: target_hint must be an object")
        state_value = value.get("state_hint")
        return cls(
            track_id=_string(
                _required(value, "track_id", context), "track_id", context
            ),
            bbox=BoundingBox.from_value(
                _required(value, "bbox_xyxy", context),
                image_width=image_width,
                image_height=image_height,
                context=context,
            ),
            state_hint=(
                None
                if state_value is None
                else _enum(Observation, state_value, "state_hint", context)
            ),
            occlusion_score=_score(
                value.get("occlusion_score", 0.0),
                "occlusion_score",
                context,
            ),
            exposure_score=_score(
                value.get("exposure_score", 0.0),
                "exposure_score",
                context,
            ),
        )

    @property
    def area(self) -> float:
        return (self.bbox.x2 - self.bbox.x1) * (self.bbox.y2 - self.bbox.y1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox_xyxy": self.bbox.as_list(),
            "state_hint": None if self.state_hint is None else self.state_hint.value,
            "occlusion_score": self.occlusion_score,
            "exposure_score": self.exposure_score,
        }


@dataclass(frozen=True)
class InventoryFrame:
    frame_id: str
    image_path: str
    camera: Camera
    video_id: str
    frame_index: int
    timestamp_ms: int
    image_width: int
    image_height: int
    target_hint: TargetHint | None

    def source_key(self) -> tuple[Camera, str, int]:
        return self.camera, self.video_id, self.frame_index

    def to_source_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "image_path": self.image_path,
            "camera": self.camera.value,
            "video_id": self.video_id,
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "image_width": self.image_width,
            "image_height": self.image_height,
        }


def parse_frame_inventory(
    rows: Iterable[Mapping[str, Any]],
) -> list[InventoryFrame]:
    frames: list[InventoryFrame] = []
    frame_ids: set[str] = set()
    source_keys: set[tuple[Camera, str, int]] = set()
    for row_index, row in enumerate(rows):
        context = f"frame_inventory[{row_index}]"
        if not isinstance(row, Mapping):
            raise AnnotationValidationError(f"{context}: frame must be an object")
        frame_id = _string(
            _required(row, "frame_id", context), "frame_id", context
        )
        if frame_id in frame_ids:
            raise AnnotationValidationError(f"duplicate frame_id '{frame_id}'")
        camera = _enum(
            Camera, _required(row, "camera", context), "camera", context
        )
        video_id = _string(
            _required(row, "video_id", context), "video_id", context
        )
        frame_index = _integer(
            _required(row, "frame_index", context), "frame_index", context
        )
        image_width = _integer(
            _required(row, "image_width", context),
            "image_width",
            context,
            minimum=1,
        )
        image_height = _integer(
            _required(row, "image_height", context),
            "image_height",
            context,
            minimum=1,
        )
        raw_hint = row.get("target_hint")
        hint = (
            None
            if raw_hint is None
            else TargetHint.from_mapping(
                raw_hint,
                image_width=image_width,
                image_height=image_height,
                context=f"{context}.target_hint",
            )
        )
        item = InventoryFrame(
            frame_id=frame_id,
            image_path=_string(
                _required(row, "image_path", context), "image_path", context
            ),
            camera=camera,
            video_id=video_id,
            frame_index=frame_index,
            timestamp_ms=_integer(
                _required(row, "timestamp_ms", context),
                "timestamp_ms",
                context,
            ),
            image_width=image_width,
            image_height=image_height,
            target_hint=hint,
        )
        source_key = item.source_key()
        if source_key in source_keys:
            raise AnnotationValidationError(
                f"duplicate source frame {camera.value}/{video_id}/{frame_index}"
            )
        frame_ids.add(frame_id)
        source_keys.add(source_key)
        frames.append(item)
    if not frames:
        raise AnnotationValidationError("frame inventory must contain at least one frame")
    return frames


@dataclass(frozen=True)
class AssignedFrame:
    frame: InventoryFrame
    station_group_id: str
    station_id: str
    section_id: str
    split: DatasetSplit

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            **self.frame.to_source_dict(),
            "station_group_id": self.station_group_id,
            "station_id": self.station_id,
            "section_id": self.section_id,
            "split": self.split.value,
            "target_hint": (
                None
                if self.frame.target_hint is None
                else self.frame.target_hint.to_dict()
            ),
        }

    def to_acquisition_dict(self) -> dict[str, Any]:
        return {
            **self.frame.to_source_dict(),
            "section_id": self.section_id,
            "split": self.split.value,
            "station_group_id": self.station_group_id,
            "station_id": self.station_id,
        }


def assign_inventory_frames(
    station_plan: StationPlan,
    inventory: Sequence[InventoryFrame],
) -> list[AssignedFrame]:
    assigned: list[AssignedFrame] = []
    for frame in inventory:
        matches = [
            section
            for section in station_plan.sections
            if section.contains(frame.camera, frame.video_id, frame.frame_index)
        ]
        if not matches:
            raise AnnotationValidationError(
                f"frame '{frame.frame_id}' is not assigned to a station section"
            )
        if len(matches) > 1:
            splits = ", ".join(sorted({item.split.value for item in matches}))
            raise AnnotationValidationError(
                f"frame '{frame.frame_id}' is assigned to multiple sections or "
                f"splits: {splits}"
            )
        section = matches[0]
        assigned.append(
            AssignedFrame(
                frame=frame,
                station_group_id=section.station_group_id,
                station_id=section.station_id,
                section_id=section.section_id,
                split=section.split,
            )
        )
    return assigned


@dataclass(frozen=True)
class KeyframeCandidate:
    assigned: AssignedFrame
    reasons: tuple[str, ...]

    @property
    def frame_id(self) -> str:
        return self.assigned.frame.frame_id

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.assigned.to_manifest_dict(),
            "selection_reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class InterpolationSegment:
    station_group_id: str
    section_id: str
    track_id: str
    start_frame_id: str
    start_frame_index: int
    end_frame_id: str
    end_frame_index: int
    intermediate_frame_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "station_group_id": self.station_group_id,
            "section_id": self.section_id,
            "track_id": self.track_id,
            "start_frame_id": self.start_frame_id,
            "start_frame_index": self.start_frame_index,
            "end_frame_id": self.end_frame_id,
            "end_frame_index": self.end_frame_index,
            "intermediate_frame_ids": list(self.intermediate_frame_ids),
        }


@dataclass(frozen=True)
class ReviewItem:
    frame_id: str
    station_group_id: str
    section_id: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "station_group_id": self.station_group_id,
            "section_id": self.section_id,
            "review_reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class KeyframeSelection:
    candidates: tuple[KeyframeCandidate, ...]
    interpolation_segments: tuple[InterpolationSegment, ...]
    review_items: tuple[ReviewItem, ...]


def _mark_peak_runs(
    frames: Sequence[AssignedFrame],
    *,
    score_name: str,
    threshold: float,
    reason: str,
    reasons: dict[str, set[str]],
) -> None:
    run: list[AssignedFrame] = []

    def flush() -> None:
        if not run:
            return
        peak = max(
            run,
            key=lambda item: getattr(item.frame.target_hint, score_name),
        )
        reasons[peak.frame.frame_id].add(reason)
        run.clear()

    for item in frames:
        score = getattr(item.frame.target_hint, score_name)
        if score >= threshold:
            run.append(item)
        else:
            flush()
    flush()


def _mark_stable_intervals(
    frames: Sequence[AssignedFrame],
    reasons: dict[str, set[str]],
    max_gap_frames: int,
    reason: str,
) -> None:
    if not frames:
        return
    last_selected = frames[0]
    for item in frames[1:]:
        if item.frame.frame_id in reasons:
            last_selected = item
            continue
        if item.frame.frame_index - last_selected.frame.frame_index >= max_gap_frames:
            reasons[item.frame.frame_id].add(reason)
            last_selected = item


def _contiguous_runs(
    frames: Sequence[AssignedFrame], predicate
) -> list[list[AssignedFrame]]:
    runs: list[list[AssignedFrame]] = []
    current: list[AssignedFrame] = []
    for item in frames:
        if predicate(item):
            current.append(item)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def select_keyframes(
    assigned_frames: Sequence[AssignedFrame],
    config: SamplingConfig | None = None,
) -> KeyframeSelection:
    config = config or SamplingConfig()
    reasons: dict[str, set[str]] = defaultdict(set)
    by_section: dict[str, list[AssignedFrame]] = defaultdict(list)
    for item in assigned_frames:
        by_section[item.section_id].append(item)

    track_frames_by_key: dict[tuple[str, str], list[AssignedFrame]] = {}
    for section_id, unordered in by_section.items():
        frames = sorted(unordered, key=lambda item: item.frame.frame_index)
        tracks: dict[str, list[AssignedFrame]] = defaultdict(list)
        for item in frames:
            hint = item.frame.target_hint
            if hint is not None:
                tracks[hint.track_id].append(item)

        for track_id, track_frames in tracks.items():
            track_frames_by_key[(section_id, track_id)] = track_frames
            reasons[track_frames[0].frame.frame_id].add("TRACK_START")
            reasons[track_frames[-1].frame.frame_id].add("TRACK_END")
            nearest = max(track_frames, key=lambda item: item.frame.target_hint.area)
            reasons[nearest.frame.frame_id].add("NEAR")

            reference_area = track_frames[0].frame.target_hint.area
            previous_state = track_frames[0].frame.target_hint.state_hint
            for item in track_frames[1:]:
                hint = item.frame.target_hint
                ratio = abs(hint.area - reference_area) / max(reference_area, 1.0)
                if ratio >= config.distance_change_ratio:
                    reasons[item.frame.frame_id].add("DISTANCE_CHANGE")
                    reference_area = hint.area
                if hint.state_hint != previous_state:
                    reasons[item.frame.frame_id].add("STATE_CHANGE")
                previous_state = hint.state_hint

            _mark_peak_runs(
                track_frames,
                score_name="occlusion_score",
                threshold=config.occlusion_threshold,
                reason="OCCLUSION",
                reasons=reasons,
            )
            _mark_peak_runs(
                track_frames,
                score_name="exposure_score",
                threshold=config.exposure_threshold,
                reason="EXPOSURE",
                reasons=reasons,
            )
            _mark_stable_intervals(
                track_frames,
                reasons,
                config.max_gap_frames,
                "STABLE_INTERVAL",
            )

        hinted = [item for item in frames if item.frame.target_hint is not None]
        for previous, current in zip(hinted, hinted[1:]):
            if (
                previous.frame.target_hint.track_id
                != current.frame.target_hint.track_id
            ):
                reasons[previous.frame.frame_id].add("HANDOVER")
                reasons[current.frame.frame_id].add("HANDOVER")

        no_target_runs = _contiguous_runs(
            frames, lambda item: item.frame.target_hint is None
        )
        for run in no_target_runs:
            reasons[run[0].frame.frame_id].add("NO_TARGET_INTERVAL")
            reasons[run[-1].frame.frame_id].add("NO_TARGET_INTERVAL")
            _mark_stable_intervals(
                run,
                reasons,
                config.max_gap_frames,
                "NO_TARGET_INTERVAL",
            )

    assigned_by_id = {item.frame.frame_id: item for item in assigned_frames}
    candidates = tuple(
        KeyframeCandidate(
            assigned=assigned_by_id[frame_id],
            reasons=tuple(sorted(frame_reasons)),
        )
        for frame_id, frame_reasons in sorted(
            reasons.items(),
            key=lambda item: (
                assigned_by_id[item[0]].station_group_id,
                assigned_by_id[item[0]].section_id,
                assigned_by_id[item[0]].frame.camera.value,
                assigned_by_id[item[0]].frame.video_id,
                assigned_by_id[item[0]].frame.frame_index,
            ),
        )
    )
    selected_ids = {candidate.frame_id for candidate in candidates}

    interpolation_segments: list[InterpolationSegment] = []
    for (section_id, track_id), track_frames in track_frames_by_key.items():
        selected_track_frames = [
            item for item in track_frames if item.frame.frame_id in selected_ids
        ]
        for start, end in zip(selected_track_frames, selected_track_frames[1:]):
            intermediates = tuple(
                item.frame.frame_id
                for item in track_frames
                if start.frame.frame_index
                < item.frame.frame_index
                < end.frame.frame_index
            )
            if not intermediates:
                continue
            interpolation_segments.append(
                InterpolationSegment(
                    station_group_id=start.station_group_id,
                    section_id=section_id,
                    track_id=track_id,
                    start_frame_id=start.frame.frame_id,
                    start_frame_index=start.frame.frame_index,
                    end_frame_id=end.frame.frame_id,
                    end_frame_index=end.frame.frame_index,
                    intermediate_frame_ids=intermediates,
                )
            )

    review_reasons: dict[str, set[str]] = defaultdict(set)
    review_context: dict[str, AssignedFrame] = {}
    risky_reasons = {"STATE_CHANGE", "HANDOVER", "OCCLUSION", "EXPOSURE"}
    for candidate in candidates:
        risky = risky_reasons.intersection(candidate.reasons)
        if risky:
            review_reasons[candidate.frame_id].update(risky)
            review_context[candidate.frame_id] = candidate.assigned
    for segment in interpolation_segments:
        midpoint = segment.intermediate_frame_ids[
            len(segment.intermediate_frame_ids) // 2
        ]
        review_reasons[midpoint].add("INTERPOLATION_MIDPOINT")
        review_context[midpoint] = assigned_by_id[midpoint]

    review_items = tuple(
        ReviewItem(
            frame_id=frame_id,
            station_group_id=review_context[frame_id].station_group_id,
            section_id=review_context[frame_id].section_id,
            reasons=tuple(sorted(item_reasons)),
        )
        for frame_id, item_reasons in sorted(
            review_reasons.items(),
            key=lambda item: (
                review_context[item[0]].station_group_id,
                review_context[item[0]].section_id,
                review_context[item[0]].frame.frame_index,
            ),
        )
    )

    return KeyframeSelection(
        candidates=candidates,
        interpolation_segments=tuple(interpolation_segments),
        review_items=review_items,
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


def assignment_summary(assigned_frames: Sequence[AssignedFrame]) -> dict[str, Any]:
    return {
        "frames": len(assigned_frames),
        "train_frames": sum(
            item.split is DatasetSplit.TRAIN for item in assigned_frames
        ),
        "val_frames": sum(item.split is DatasetSplit.VAL for item in assigned_frames),
        "station_groups": len({item.station_group_id for item in assigned_frames}),
        "stations": len({item.station_id for item in assigned_frames}),
        "sections": len({item.section_id for item in assigned_frames}),
        "a_frames": sum(item.frame.camera is Camera.A for item in assigned_frames),
        "b_frames": sum(item.frame.camera is Camera.B for item in assigned_frames),
    }


def build_sampling_package(
    assigned_frames: Sequence[AssignedFrame],
    output_dir: Path,
    config: SamplingConfig | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise AnnotationValidationError(
                f"output path must be a directory: {output_dir}"
            )
        if any(output_dir.iterdir()):
            raise AnnotationValidationError(
                f"output directory must be empty: {output_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config or SamplingConfig()
    selection = select_keyframes(assigned_frames, config)
    candidates_by_id = {
        candidate.frame_id: candidate for candidate in selection.candidates
    }

    data_manifest = []
    for item in assigned_frames:
        candidate = candidates_by_id.get(item.frame.frame_id)
        data_manifest.append(
            {
                **item.to_manifest_dict(),
                "selected": candidate is not None,
                "selection_reasons": (
                    [] if candidate is None else list(candidate.reasons)
                ),
            }
        )

    keyframes = [candidate.to_dict() for candidate in selection.candidates]
    acquisition_metadata = [
        candidate.assigned.to_acquisition_dict()
        for candidate in selection.candidates
    ]
    manual_templates = [
        {"frame_id": candidate.frame_id, "targets": None}
        for candidate in selection.candidates
    ]

    _write_jsonl(output_dir / "data-manifest.jsonl", data_manifest)
    _write_jsonl(output_dir / "keyframes.jsonl", keyframes)
    _write_jsonl(output_dir / "acquisition-metadata.jsonl", acquisition_metadata)
    _write_jsonl(
        output_dir / "manual-annotations-template.jsonl", manual_templates
    )
    _write_jsonl(
        output_dir / "interpolation-segments.jsonl",
        (item.to_dict() for item in selection.interpolation_segments),
    )
    _write_jsonl(
        output_dir / "review-checklist.jsonl",
        (item.to_dict() for item in selection.review_items),
    )

    reason_counts = Counter(
        reason
        for candidate in selection.candidates
        for reason in candidate.reasons
    )
    summary = {
        **assignment_summary(assigned_frames),
        "selected_frames": len(selection.candidates),
        "interpolation_segments": len(selection.interpolation_segments),
        "review_items": len(selection.review_items),
        "selection_reasons": dict(sorted(reason_counts.items())),
        "sampling_config": config.to_dict(),
    }
    _write_json(output_dir / "sampling-summary.json", summary)
    return summary


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise AnnotationValidationError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise AnnotationValidationError(
            f"{path}: invalid JSON: {error.msg}"
        ) from error
    if not isinstance(value, Mapping):
        raise AnnotationValidationError(f"{path}: JSON root must be an object")
    return value


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


def _sampling_config(args: argparse.Namespace) -> SamplingConfig:
    return SamplingConfig(
        max_gap_frames=args.max_gap_frames,
        distance_change_ratio=args.distance_change_ratio,
        occlusion_threshold=args.occlusion_threshold,
        exposure_threshold=args.exposure_threshold,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assign station splits and build target annotation keyframes."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "build"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--plan", required=True, type=Path)
        subparser.add_argument("--inventory", required=True, type=Path)
        subparser.add_argument("--max-gap-frames", type=int, default=30)
        subparser.add_argument("--distance-change-ratio", type=float, default=0.5)
        subparser.add_argument("--occlusion-threshold", type=float, default=0.7)
        subparser.add_argument("--exposure-threshold", type=float, default=0.7)
        if command == "build":
            subparser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        station_plan = parse_station_plan(_load_json(args.plan))
        inventory = parse_frame_inventory(_load_jsonl(args.inventory))
        assigned = assign_inventory_frames(station_plan, inventory)
        config = _sampling_config(args)
        if args.command == "validate":
            selection = select_keyframes(assigned, config)
            result = {
                **assignment_summary(assigned),
                "selected_frames": len(selection.candidates),
            }
        else:
            result = build_sampling_package(assigned, args.output, config)
    except AnnotationValidationError as error:
        print(f"sampling error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
