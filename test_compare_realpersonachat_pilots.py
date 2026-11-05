"""Pure paired comparison regression tests; no corpus/GPU needed."""
import unittest
from compare_realpersonachat_pilots import compare_reports


def row(n, response, reference="はい。", prompt=None):
    return {
        "dialogue_id": str(n), "turn_id": 2,
        "prompt": prompt or "会話:\nspeaker_0:元気？\nspeaker_1:",
        "reference": reference, "status": "GENERATED",
        "generated": response, "generated_overlap": .25,
    }


class PairedPilotTests(unittest.TestCase):
    def test_identical_dev_cases(self):
        a = {"rows": [row(1, "そうですね。"), row(2, "そうですね。")]}
        b = {"rows": [row(2, "はい。"), row(1, "お元気ですか？")]}
        result = compare_reports(a, b)
        self.assertEqual(result["cases"], 2)
        self.assertEqual(result["baseline_distinct_response_ratio"], .5)
        self.assertEqual(result["candidate_distinct_response_ratio"], 1.)

    def test_different_dev_case_rejected(self):
        with self.assertRaisesRegex(ValueError, "DIFFERENT_DEV_CASES"):
            compare_reports({"rows": [row(1, "a")]},
                            {"rows": [row(2, "b")]})

    def test_prompt_change_rejected(self):
        with self.assertRaisesRegex(ValueError, "DEV_CONTENT_MISMATCH"):
            compare_reports({"rows": [row(1, "a")]},
                            {"rows": [row(1, "a", prompt="別の会話") ]})

    def test_reference_change_rejected(self):
        with self.assertRaisesRegex(ValueError, "DEV_CONTENT_MISMATCH"):
            compare_reports({"rows": [row(1, "a")]},
                            {"rows": [row(1, "a", reference="違う回答")]})

    def test_duplicate_case_rejected(self):
        with self.assertRaisesRegex(ValueError, "DUPLICATE_CASE_KEY"):
            compare_reports({"rows": [row(1, "a"), row(1, "b")]},
                            {"rows": [row(1, "a")]})

    def test_abstention_counts(self):
        left = row(1, "a")
        left.update(status="ABSTAIN", generated=None,
                    generated_overlap=None)
        right = row(1, "はい。")
        out = compare_reports({"rows": [left]}, {"rows": [right]})
        self.assertEqual(out["baseline_coverage"], 0.)
        self.assertEqual(out["candidate_coverage"], 1.)
        self.assertEqual(out["jointly_generated_cases"], 0)
        self.assertIsNone(out["mean_overlap_delta_on_joint_cases"])


if __name__ == "__main__":
    unittest.main()
