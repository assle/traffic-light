import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("视频推理.py")
SPEC = importlib.util.spec_from_file_location("video_inference", MODULE_PATH)
video_inference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(video_inference)


class FinalOutputStatisticsTests(unittest.TestCase):
    def test_surplus_color_does_not_fail_an_otherwise_valid_output(self):
        result = video_inference.cross_validate_all(
            ["下", "中"], ["红", "绿", "绿"]
        )

        self.assertFalse(result["strict_all_passed"])
        self.assertTrue(result["all_passed"])
        self.assertEqual(result["ignored_extra_colors"], 1)
        self.assertEqual(
            [pair["included_in_final"] for pair in result["pair_results"]],
            [True, True, False],
        )

    def test_missing_color_still_fails_final_output(self):
        result = video_inference.cross_validate_all(["下", "中"], ["红"])

        self.assertFalse(result["strict_all_passed"])
        self.assertFalse(result["all_passed"])
        self.assertEqual(result["ignored_extra_colors"], 0)

    def test_rule_conflict_still_fails_final_output(self):
        result = video_inference.cross_validate_all(
            ["下", "中"], ["红", "红", "绿"]
        )

        self.assertFalse(result["strict_all_passed"])
        self.assertFalse(result["all_passed"])


if __name__ == "__main__":
    unittest.main()
