"""Local response-generation diagnostic tests (no GPU required)."""
import unittest

from evaluate_realpersonachat_pilot import (
    calculate_report, copy_previous_baseline, overlap_ratio
)


class ResponseEvaluationTests(unittest.TestCase):
    def test_identical_overlap(self):
        self.assertEqual(overlap_ratio("こんにちは", "こんにちは"), 1.)

    def test_empty_overlap(self):
        self.assertEqual(overlap_ratio("", "こんにちは"), 0.)

    def test_copy_previous_is_last_turn(self):
        item = {"context": [{"speaker": "speaker_0", "text": "最初"},
                            {"speaker": "speaker_1", "text": "最後"}]}
        self.assertEqual(copy_previous_baseline(item), "最後")

    def test_generated_metrics(self):
        item = {
            "status": "GENERATED",
            "generated": "こんにちは",
            "reference": "こんにちは",
            "generated_overlap": 1.,
            "copy_previous_overlap": 0.,
        }
        result = calculate_report([item])
        self.assertEqual(result["exact_match_rate_on_generated"], 1.)
        self.assertEqual(result["mean_character_overlap"], 1.)
        self.assertEqual(result["copy_previous_mean_character_overlap"], 0.)

    def test_abstention_not_scored_as_generated(self):
        item = {"status": "ABSTAIN", "reason": "UNKNOWN_PROMPT_CHARACTERS",
                "generated": None, "reference": "こんにちは",
                "generated_overlap": None, "copy_previous_overlap": None}
        result = calculate_report([item])
        self.assertEqual(result["abstained"], 1)
        self.assertIsNone(result["exact_match_rate_on_generated"])
        self.assertEqual(result["abstention_reasons"],
                         {"UNKNOWN_PROMPT_CHARACTERS": 1})

    def test_detect_response_collapse(self):
        rows = [{"status": "GENERATED", "generated": "そうですね。",
                 "reference": "わかりました。",
                 "generated_overlap": 0.2,
                 "copy_previous_overlap": 0.1} for _ in range(3)]
        result = calculate_report(rows)
        self.assertEqual(result["unique_generated_count"], 1)
        self.assertAlmostEqual(result["distinct_response_ratio"], 1 / 3)
        self.assertEqual(result["most_common_generated_fraction"], 1.0)

    def test_zero_selected_rejected(self):
        with self.assertRaisesRegex(ValueError, "NO_SELECTED_ROWS"):
            calculate_report([])


if __name__ == "__main__":
    unittest.main()
