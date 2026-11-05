"""Paired development comparison regression tests."""
import unittest
from compare_role_frame_models import compare_reports


def row(text, voice, expected, correct):
    return {"text": text, "voice": voice, "expected": expected,
            "exact_code": correct, "reason": None if correct else "REJECTED"}


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.base = [row(f"t{i}", "active" if i < 80 else "passive",
                         f"c{i}", i % 2 == 0) for i in range(120)]
        self.new = [row(f"t{i}", "active" if i < 80 else "passive",
                        f"c{i}", True) for i in range(120)]

    def test_paired_summary(self):
        report = compare_reports({"rows": self.base},
                                 {"rows": self.new})
        self.assertEqual(report["overall"]["total"], 120)
        self.assertEqual(report["overall"]["newly_correct"], 60)
        self.assertEqual(report["by_voice"]["passive"]["total"], 40)

    def test_different_targets_blocked(self):
        self.new[0]["expected"] = "different"
        with self.assertRaises(ValueError):
            compare_reports({"rows": self.base}, {"rows": self.new})

    def test_missing_case_blocked(self):
        self.new[0]["text"] = "other"
        with self.assertRaises(ValueError):
            compare_reports({"rows": self.base}, {"rows": self.new})

    def test_voice_mismatch_blocked(self):
        self.new[80]["voice"] = "active"
        with self.assertRaises(ValueError):
            compare_reports({"rows": self.base}, {"rows": self.new})

    def test_regressions_count(self):
        self.new[0]["exact_code"] = False
        report = compare_reports({"rows": self.base}, {"rows": self.new})
        self.assertEqual(report["overall"]["regressions"], 1)


if __name__ == "__main__":
    unittest.main()
