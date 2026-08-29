import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from target_signal_annotations import (
    AnnotationValidationError,
    build_annotation_records,
)
from target_signal_sampling import (
    SamplingConfig,
    assign_inventory_frames,
    build_sampling_package,
    parse_frame_inventory,
    parse_station_plan,
    select_keyframes,
    main,
)


def section(
    section_id,
    *,
    camera="A",
    video_id="video-a",
    start=0,
    end=100,
):
    return {
        "section_id": section_id,
        "camera": camera,
        "video_id": video_id,
        "start_frame": start,
        "end_frame": end,
    }


def group(
    group_id,
    station_id,
    split,
    sections,
):
    return {
        "station_group_id": group_id,
        "station_id": station_id,
        "split": split,
        "sections": sections,
    }


def plan(*groups):
    return {"station_groups": list(groups)}


def target_hint(
    track_id,
    *,
    bbox=(100, 100, 200, 300),
    state="H",
    occlusion=0.0,
    exposure=0.0,
):
    return {
        "track_id": track_id,
        "bbox_xyxy": list(bbox),
        "state_hint": state,
        "occlusion_score": occlusion,
        "exposure_score": exposure,
    }


def frame(
    frame_index,
    *,
    camera="A",
    video_id="video-a",
    hint=None,
):
    frame_id = f"{camera.lower()}-{video_id}-{frame_index:06d}"
    return {
        "frame_id": frame_id,
        "image_path": f"frames/{frame_id}.jpg",
        "camera": camera,
        "video_id": video_id,
        "frame_index": frame_index,
        "timestamp_ms": frame_index * 100,
        "image_width": 1920,
        "image_height": 1080,
        "target_hint": hint,
    }


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


class StationAssignmentTests(unittest.TestCase):
    def test_assigns_a_and_b_sections_of_same_station_to_same_split(self):
        station_plan = parse_station_plan(
            plan(
                group(
                    "station-01",
                    "station-01",
                    "train",
                    [
                        section("station-01-a", camera="A", video_id="video-a"),
                        section("station-01-b", camera="B", video_id="video-b"),
                    ],
                ),
                group(
                    "station-02",
                    "station-02",
                    "val",
                    [section("station-02-a", video_id="video-c")],
                ),
            )
        )
        inventory = parse_frame_inventory(
            [
                frame(10),
                frame(10, camera="B", video_id="video-b"),
                frame(10, video_id="video-c"),
            ]
        )

        assigned = assign_inventory_frames(station_plan, inventory)

        self.assertEqual(
            [item.split.value for item in assigned],
            ["train", "train", "val"],
        )
        self.assertEqual(
            [item.station_group_id for item in assigned],
            ["station-01", "station-01", "station-02"],
        )

    def test_rejects_station_assigned_to_train_and_val(self):
        with self.assertRaisesRegex(
            AnnotationValidationError, "station.*multiple splits"
        ):
            parse_station_plan(
                plan(
                    group(
                        "station-01-part-a",
                        "station-01",
                        "train",
                        [section("section-a", video_id="video-a")],
                    ),
                    group(
                        "station-01-part-b",
                        "station-01",
                        "val",
                        [section("section-b", video_id="video-b")],
                    ),
                )
            )

    def test_rejects_continuous_section_overlap(self):
        with self.assertRaisesRegex(
            AnnotationValidationError, "continuous section overlap"
        ):
            parse_station_plan(
                plan(
                    group(
                        "station-01",
                        "station-01",
                        "train",
                        [
                            section("section-a", start=0, end=50),
                            section("section-b", start=40, end=100),
                        ],
                    )
                )
            )

    def test_rejects_frame_range_assigned_to_multiple_splits(self):
        with self.assertRaisesRegex(
            AnnotationValidationError, "frame range.*multiple splits"
        ):
            parse_station_plan(
                plan(
                    group(
                        "station-01",
                        "station-01",
                        "train",
                        [section("section-a", start=0, end=50)],
                    ),
                    group(
                        "station-02",
                        "station-02",
                        "val",
                        [section("section-b", start=40, end=100)],
                    ),
                )
            )

    def test_rejects_duplicate_section_identifier(self):
        with self.assertRaisesRegex(
            AnnotationValidationError, "duplicate section_id"
        ):
            parse_station_plan(
                plan(
                    group(
                        "station-01",
                        "station-01",
                        "train",
                        [section("same", video_id="video-a")],
                    ),
                    group(
                        "station-02",
                        "station-02",
                        "train",
                        [section("same", video_id="video-b")],
                    ),
                )
            )

    def test_rejects_unassigned_and_duplicate_frames(self):
        station_plan = parse_station_plan(
            plan(
                group(
                    "station-01",
                    "station-01",
                    "train",
                    [section("section-a", start=0, end=50)],
                )
            )
        )

        with self.assertRaisesRegex(AnnotationValidationError, "not assigned"):
            assign_inventory_frames(
                station_plan,
                parse_frame_inventory([frame(75)]),
            )

        with self.assertRaisesRegex(AnnotationValidationError, "duplicate frame_id"):
            parse_frame_inventory([frame(10), frame(10)])

        with self.assertRaisesRegex(
            AnnotationValidationError, "at least one frame"
        ):
            parse_frame_inventory([])


class KeyframeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.station_plan = parse_station_plan(
            plan(
                group(
                    "station-01",
                    "station-01",
                    "train",
                    [section("section-a", start=0, end=100)],
                )
            )
        )

    def test_generates_every_required_keyframe_reason(self):
        inventory_rows = [
            frame(0, hint=target_hint("track-1", bbox=(0, 0, 10, 10), state="U")),
            frame(10, hint=target_hint("track-1", bbox=(0, 0, 12, 12), state="U")),
            frame(20, hint=target_hint("track-1", bbox=(0, 0, 25, 25), state="U")),
            frame(30, hint=target_hint("track-1", bbox=(0, 0, 30, 30), state="L")),
            frame(
                40,
                hint=target_hint(
                    "track-1", bbox=(0, 0, 35, 35), state="L", occlusion=0.8
                ),
            ),
            frame(
                50,
                hint=target_hint(
                    "track-1", bbox=(0, 0, 100, 100), state="L", exposure=0.9
                ),
            ),
            frame(60, hint=target_hint("track-2", bbox=(0, 0, 10, 10), state="H")),
            frame(70, hint=target_hint("track-2", bbox=(0, 0, 20, 20), state="H")),
            frame(80),
            frame(90),
            frame(100),
        ]
        assigned = assign_inventory_frames(
            self.station_plan, parse_frame_inventory(inventory_rows)
        )

        selection = select_keyframes(
            assigned,
            SamplingConfig(
                max_gap_frames=30,
                distance_change_ratio=0.5,
                occlusion_threshold=0.7,
                exposure_threshold=0.7,
            ),
        )

        reasons = {
            reason
            for candidate in selection.candidates
            for reason in candidate.reasons
        }
        self.assertTrue(
            {
                "TRACK_START",
                "DISTANCE_CHANGE",
                "NEAR",
                "TRACK_END",
                "STATE_CHANGE",
                "HANDOVER",
                "OCCLUSION",
                "EXPOSURE",
                "NO_TARGET_INTERVAL",
            }.issubset(reasons)
        )
        review_reasons = {
            reason
            for item in selection.review_items
            for reason in item.reasons
        }
        self.assertTrue(
            {"STATE_CHANGE", "HANDOVER", "OCCLUSION", "EXPOSURE"}.issubset(
                review_reasons
            )
        )

    def test_creates_ordered_interpolation_segments_and_review_midpoints(self):
        inventory_rows = [
            frame(
                frame_index,
                hint=target_hint("stable-track", bbox=(0, 0, 20, 20), state="L"),
            )
            for frame_index in range(0, 101, 10)
        ]
        assigned = assign_inventory_frames(
            self.station_plan, parse_frame_inventory(inventory_rows)
        )

        selection = select_keyframes(
            assigned,
            SamplingConfig(max_gap_frames=30),
        )

        stable_candidates = [
            candidate
            for candidate in selection.candidates
            if "STABLE_INTERVAL" in candidate.reasons
        ]
        self.assertGreaterEqual(len(stable_candidates), 2)
        self.assertTrue(selection.interpolation_segments)
        for segment in selection.interpolation_segments:
            self.assertEqual(segment.track_id, "stable-track")
            self.assertLess(segment.start_frame_index, segment.end_frame_index)
            self.assertTrue(segment.intermediate_frame_ids)

        review_ids = {item.frame_id for item in selection.review_items}
        interpolation_ids = {
            segment.intermediate_frame_ids[len(segment.intermediate_frame_ids) // 2]
            for segment in selection.interpolation_segments
        }
        self.assertTrue(interpolation_ids.issubset(review_ids))


class SamplingPackageTests(unittest.TestCase):
    def test_exports_manifests_metadata_templates_and_review_checklist(self):
        station_plan = parse_station_plan(
            plan(
                group(
                    "station-01",
                    "station-01",
                    "train",
                    [
                        section("station-01-a", video_id="video-a"),
                        section(
                            "station-01-b",
                            camera="B",
                            video_id="video-b",
                        ),
                    ],
                ),
                group(
                    "station-02",
                    "station-02",
                    "val",
                    [section("station-02-a", video_id="video-c")],
                ),
            )
        )
        inventory = parse_frame_inventory(
            [
                frame(0, hint=target_hint("track-a", state="U")),
                frame(50, hint=target_hint("track-a", state="U")),
                frame(100, hint=target_hint("track-a", state="U")),
                frame(0, camera="B", video_id="video-b"),
                frame(100, camera="B", video_id="video-b"),
                frame(0, video_id="video-c"),
                frame(100, video_id="video-c"),
            ]
        )
        assigned = assign_inventory_frames(station_plan, inventory)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "package"
            summary = build_sampling_package(
                assigned,
                output,
                SamplingConfig(max_gap_frames=30),
            )

            data_manifest = read_jsonl(output / "data-manifest.jsonl")
            keyframes = read_jsonl(output / "keyframes.jsonl")
            metadata_rows = read_jsonl(output / "acquisition-metadata.jsonl")
            templates = read_jsonl(output / "manual-annotations-template.jsonl")
            interpolation = read_jsonl(output / "interpolation-segments.jsonl")
            review = read_jsonl(output / "review-checklist.jsonl")

            self.assertEqual(len(data_manifest), len(assigned))
            self.assertEqual(len(keyframes), summary["selected_frames"])
            self.assertEqual(len(metadata_rows), len(keyframes))
            self.assertEqual(len(templates), len(keyframes))
            self.assertEqual(
                {row["station_group_id"] for row in data_manifest},
                {"station-01", "station-02"},
            )
            self.assertEqual(
                {row["split"] for row in data_manifest}, {"train", "val"}
            )
            self.assertEqual(
                {row["camera"] for row in data_manifest}, {"A", "B"}
            )
            self.assertTrue(
                all(
                    {"video_id", "frame_index", "section_id"}.issubset(row)
                    for row in data_manifest
                )
            )
            self.assertTrue(all(row["targets"] is None for row in templates))
            self.assertEqual(
                [row["frame_id"] for row in metadata_rows],
                [row["frame_id"] for row in keyframes],
            )
            canonical_records = build_annotation_records(
                metadata_rows,
                [
                    {"frame_id": row["frame_id"], "targets": []}
                    for row in metadata_rows
                ],
            )
            self.assertEqual(len(canonical_records), len(metadata_rows))
            self.assertEqual(len(interpolation), summary["interpolation_segments"])
            self.assertEqual(len(review), summary["review_items"])
            self.assertTrue((output / "sampling-summary.json").is_file())

    def test_cli_validates_and_builds_sampling_package(self):
        station_plan = plan(
            group(
                "station-01",
                "station-01",
                "train",
                [section("station-01-a", video_id="video-a")],
            )
        )
        inventory_rows = [
            frame(0, hint=target_hint("track-a", state="U")),
            frame(50, hint=target_hint("track-a", state="U")),
            frame(100, hint=target_hint("track-a", state="U")),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan_path = root / "station-plan.json"
            inventory_path = root / "frame-inventory.jsonl"
            output_path = root / "package"
            write_json(plan_path, station_plan)
            write_jsonl(inventory_path, inventory_rows)

            stdout = StringIO()
            with redirect_stdout(stdout):
                validate_status = main(
                    [
                        "validate",
                        "--plan",
                        str(plan_path),
                        "--inventory",
                        str(inventory_path),
                    ]
                )
            self.assertEqual(validate_status, 0)
            self.assertGreater(json.loads(stdout.getvalue())["selected_frames"], 0)

            stdout = StringIO()
            with redirect_stdout(stdout):
                build_status = main(
                    [
                        "build",
                        "--plan",
                        str(plan_path),
                        "--inventory",
                        str(inventory_path),
                        "--output",
                        str(output_path),
                    ]
                )
            self.assertEqual(build_status, 0)
            self.assertTrue((output_path / "data-manifest.jsonl").is_file())

            invalid_plan_path = root / "invalid-plan.json"
            write_json(invalid_plan_path, {"station_groups": []})
            stderr = StringIO()
            with redirect_stderr(stderr):
                invalid_status = main(
                    [
                        "validate",
                        "--plan",
                        str(invalid_plan_path),
                        "--inventory",
                        str(inventory_path),
                    ]
                )
            self.assertEqual(invalid_status, 2)
            self.assertIn("sampling error", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
