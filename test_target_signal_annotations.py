import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from target_signal_annotations import (
    AnnotationValidationError,
    build_annotation_records,
    export_annotation_dataset,
    load_annotation_records,
    main,
)


def metadata(
    frame_id="a-trip-000001",
    *,
    camera="A",
    frame_index=1,
    timestamp_ms=100,
    section_id="station-01",
    split="train",
    width=1000,
    height=500,
):
    return {
        "frame_id": frame_id,
        "image_path": f"images/{frame_id}.jpg",
        "camera": camera,
        "video_id": f"trip-{camera.lower()}",
        "frame_index": frame_index,
        "timestamp_ms": timestamp_ms,
        "section_id": section_id,
        "split": split,
        "image_width": width,
        "image_height": height,
    }


def annotation(
    frame_id="a-trip-000001",
    *,
    configuration="ULH",
    observation="L",
    bbox=(100, 50, 300, 250),
):
    return {
        "frame_id": frame_id,
        "targets": [
            {
                "bbox_xyxy": list(bbox),
                "configuration": configuration,
                "observation": observation,
            }
        ],
    }


def no_target_annotation(frame_id):
    return {"frame_id": frame_id, "targets": []}


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


class AnnotationContractTests(unittest.TestCase):
    def test_builds_target_and_no_target_records_from_generated_metadata(self):
        metadata_rows = [
            metadata(),
            metadata(
                "b-trip-000002",
                camera="B",
                frame_index=2,
                timestamp_ms=200,
            ),
        ]
        annotation_rows = [
            annotation(),
            no_target_annotation("b-trip-000002"),
        ]

        records = build_annotation_records(metadata_rows, annotation_rows)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].source.camera.value, "A")
        self.assertEqual(records[0].target.configuration.value, "ULH")
        self.assertEqual(records[0].target.observation.value, "L")
        self.assertIsNone(records[1].target)

    def test_accepts_supported_observations_for_each_configuration(self):
        cases = [
            ("ULH", "U"),
            ("ULH", "L"),
            ("ULH", "H"),
            ("ULH", "UNLIT"),
            ("ULH", "UNREADABLE"),
            ("LH", "L"),
            ("LH", "H"),
            ("LH", "UNLIT"),
            ("LH", "UNREADABLE"),
            ("空H", "H"),
            ("空H", "UNLIT"),
            ("空H", "UNREADABLE"),
        ]

        for index, (configuration, observation) in enumerate(cases):
            with self.subTest(configuration=configuration, observation=observation):
                frame_id = f"frame-{index}"
                records = build_annotation_records(
                    [metadata(frame_id)],
                    [
                        annotation(
                            frame_id,
                            configuration=configuration,
                            observation=observation,
                        )
                    ],
                )
                self.assertEqual(records[0].target.observation.value, observation)

    def test_rejects_missing_metadata_fields(self):
        source = metadata()
        source.pop("section_id")

        with self.assertRaisesRegex(
            AnnotationValidationError, "section_id"
        ):
            build_annotation_records([source], [annotation()])

    def test_rejects_missing_manual_fields(self):
        with self.assertRaisesRegex(AnnotationValidationError, "targets"):
            build_annotation_records([metadata()], [{"frame_id": "a-trip-000001"}])

    def test_rejects_unknown_dataset_split(self):
        source = metadata()
        source["split"] = "test"

        with self.assertRaisesRegex(AnnotationValidationError, "split"):
            build_annotation_records([source], [annotation()])

    def test_rejects_multiple_current_targets(self):
        manual = annotation()
        manual["targets"].append(dict(manual["targets"][0]))

        with self.assertRaisesRegex(
            AnnotationValidationError, "at most one current target"
        ):
            build_annotation_records([metadata()], [manual])

    def test_rejects_illegal_configuration_color_pairs(self):
        cases = [("LH", "U"), ("空H", "U"), ("空H", "L")]

        for configuration, observation in cases:
            with self.subTest(configuration=configuration, observation=observation):
                with self.assertRaisesRegex(
                    AnnotationValidationError, "not valid for configuration"
                ):
                    build_annotation_records(
                        [metadata()],
                        [
                            annotation(
                                configuration=configuration,
                                observation=observation,
                            )
                        ],
                    )

    def test_rejects_unparseable_and_out_of_bounds_boxes(self):
        cases = [
            (100, 50, 100, 250),
            (-1, 50, 300, 250),
            (100, 50, 1001, 250),
            (100, 50, math.inf, 250),
            (100, 50, 300),
        ]

        for bbox in cases:
            with self.subTest(bbox=bbox):
                with self.assertRaisesRegex(AnnotationValidationError, "bbox"):
                    build_annotation_records(
                        [metadata()],
                        [annotation(bbox=bbox)],
                    )

    def test_requires_exactly_one_annotation_for_each_metadata_frame(self):
        with self.assertRaisesRegex(
            AnnotationValidationError, "missing manual annotation"
        ):
            build_annotation_records(
                [metadata(), metadata("frame-2")],
                [annotation()],
            )

    def test_rejects_annotations_without_acquisition_metadata(self):
        with self.assertRaisesRegex(
            AnnotationValidationError, "no acquisition metadata"
        ):
            build_annotation_records(
                [metadata()],
                [annotation(), no_target_annotation("frame-2")],
            )

    def test_rejects_duplicate_metadata_and_manual_annotations(self):
        with self.assertRaisesRegex(
            AnnotationValidationError, "duplicate acquisition metadata"
        ):
            build_annotation_records(
                [metadata(), metadata()],
                [annotation()],
            )

        with self.assertRaisesRegex(
            AnnotationValidationError, "duplicate manual annotation"
        ):
            build_annotation_records(
                [metadata()],
                [annotation(), annotation()],
            )


class AnnotationExportTests(unittest.TestCase):
    def setUp(self):
        self.metadata_rows = [
            metadata("frame-ulh"),
            metadata("frame-lh", camera="B"),
            metadata("frame-empty-h"),
            metadata("frame-none", camera="B"),
        ]
        self.annotation_rows = [
            annotation("frame-ulh", configuration="ULH", observation="U"),
            annotation("frame-lh", configuration="LH", observation="UNLIT"),
            annotation(
                "frame-empty-h",
                configuration="空H",
                observation="UNREADABLE",
            ),
            no_target_annotation("frame-none"),
        ]

    def test_exports_detector_labels_and_color_manifest_from_same_records(self):
        records = build_annotation_records(
            self.metadata_rows, self.annotation_rows
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = export_annotation_dataset(records, Path(temp_dir))
            output = Path(temp_dir)

            self.assertEqual(
                json.loads(
                    (output / "detector" / "classes.json").read_text(
                        encoding="utf-8"
                    )
                ),
                {
                    "0": "TARGET_ULH",
                    "1": "TARGET_LH",
                    "2": "TARGET_EMPTY_H",
                },
            )
            self.assertEqual(
                (output / "detector" / "labels" / "frame-ulh.txt").read_text(
                    encoding="utf-8"
                ),
                "0 0.200000 0.300000 0.200000 0.400000\n",
            )
            self.assertEqual(
                (output / "detector" / "labels" / "frame-lh.txt").read_text(
                    encoding="utf-8"
                ).split(" ", 1)[0],
                "1",
            )
            self.assertEqual(
                (
                    output / "detector" / "labels" / "frame-empty-h.txt"
                ).read_text(encoding="utf-8").split(" ", 1)[0],
                "2",
            )
            self.assertEqual(
                (output / "detector" / "labels" / "frame-none.txt").read_text(
                    encoding="utf-8"
                ),
                "",
            )

            color_rows = [
                json.loads(line)
                for line in (
                    output / "color" / "manifest.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [(row["frame_id"], row["target"]) for row in color_rows],
                [
                    ("frame-ulh", [1, 0, 0]),
                    ("frame-lh", [0, 0, 0]),
                ],
            )
            self.assertEqual(color_rows[0]["crop_xyxy"], [100, 50, 300, 250])
            self.assertEqual(color_rows[0]["split"], "train")
            self.assertEqual(summary["records"], 4)
            self.assertEqual(summary["detector_targets"], 3)
            self.assertEqual(summary["no_target_frames"], 1)
            self.assertEqual(summary["color_samples"], 2)
            self.assertEqual(summary["unreadable_color_samples"], 1)

    def test_loads_jsonl_and_exports_consistent_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata_path = root / "metadata.jsonl"
            annotations_path = root / "annotations.jsonl"
            write_jsonl(metadata_path, self.metadata_rows)
            write_jsonl(annotations_path, self.annotation_rows)

            records = load_annotation_records(metadata_path, annotations_path)
            summary = export_annotation_dataset(records, root / "export")

            detector_rows = [
                json.loads(line)
                for line in (
                    root / "export" / "detector" / "manifest.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [row["frame_id"] for row in detector_rows],
                [row.source.frame_id for row in records],
            )
            self.assertEqual(summary["records"], len(records))

    def test_refuses_non_empty_output_directory(self):
        records = build_annotation_records(
            self.metadata_rows, self.annotation_rows
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "export"
            output.mkdir()
            (output / "stale-label.txt").write_text("stale", encoding="utf-8")

            with self.assertRaisesRegex(
                AnnotationValidationError, "output directory must be empty"
            ):
                export_annotation_dataset(records, output)

    def test_cli_validates_and_exports_jsonl_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata_path = root / "metadata.jsonl"
            annotations_path = root / "annotations.jsonl"
            export_path = root / "export"
            write_jsonl(metadata_path, self.metadata_rows)
            write_jsonl(annotations_path, self.annotation_rows)

            stdout = StringIO()
            with redirect_stdout(stdout):
                validate_status = main(
                    [
                        "validate",
                        "--metadata",
                        str(metadata_path),
                        "--annotations",
                        str(annotations_path),
                    ]
                )
            self.assertEqual(validate_status, 0)
            self.assertEqual(json.loads(stdout.getvalue())["records"], 4)

            stdout = StringIO()
            with redirect_stdout(stdout):
                export_status = main(
                    [
                        "export",
                        "--metadata",
                        str(metadata_path),
                        "--annotations",
                        str(annotations_path),
                        "--output",
                        str(export_path),
                    ]
                )
            self.assertEqual(export_status, 0)
            self.assertEqual(json.loads(stdout.getvalue())["color_samples"], 2)
            self.assertTrue((export_path / "canonical_manifest.jsonl").is_file())

            invalid_annotations = root / "invalid-annotations.jsonl"
            write_jsonl(
                invalid_annotations,
                [
                    annotation("frame-ulh", configuration="空H", observation="U"),
                    *self.annotation_rows[1:],
                ],
            )
            stderr = StringIO()
            with redirect_stderr(stderr):
                invalid_status = main(
                    [
                        "validate",
                        "--metadata",
                        str(metadata_path),
                        "--annotations",
                        str(invalid_annotations),
                    ]
                )
            self.assertEqual(invalid_status, 2)
            self.assertIn("annotation error", stderr.getvalue())
            self.assertIn("not valid for configuration", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
