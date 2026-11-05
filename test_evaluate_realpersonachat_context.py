"""Tests for counterfactual context evaluation; no GPU needed."""
import unittest
from evaluate_realpersonachat_context import (
    choose_donor, make_counterfactual, summarize
)


def item(number, text):
    return {
        "dialogue_id": str(number), "turn_id": 4,
        "context": [
            {"speaker": "speaker_0", "text": "前の話題"},
            {"speaker": "speaker_1", "text": text},
        ],
        "next_speaker": "speaker_0", "response": "返答",
    }


class ContextCounterfactualTests(unittest.TestCase):
    def test_replaces_only_last_text(self):
        row = item(1, "今日は暑いですね")
        donor = item(2, "料理の話をします")
        altered = make_counterfactual(row, donor)
        self.assertEqual(altered["context"][-1]["text"], "料理の話をします")
        self.assertEqual(altered["context"][-1]["speaker"], "speaker_1")
        self.assertEqual(altered["context"][0], row["context"][0])
        self.assertEqual(altered["response"], "返答")
        self.assertEqual(altered["next_speaker"], "speaker_0")
        self.assertEqual(row["context"][-1]["text"], "今日は暑いですね")

    def test_reject_self_dialogue_donor(self):
        with self.assertRaisesRegex(ValueError, "DONOR_SAME_DIALOGUE"):
            make_counterfactual(item(1, "a"), item(1, "b"))

    def test_donor_selects_other_dialogue(self):
        rows = [item(1, "a"), item(1, "b"), item(2, "c")]
        self.assertEqual(choose_donor(rows, 0)["dialogue_id"], "2")

    def test_no_other_dialogue(self):
        with self.assertRaisesRegex(ValueError, "NO_OTHER_DIALOGUE"):
            choose_donor([item(1, "a"), item(1, "b")], 0)

    def test_summary_positive_fraction(self):
        rows = [
            {"status": "SCORED", "original_minus_counterfactual": 0.2},
            {"status": "SCORED", "original_minus_counterfactual": -0.1},
            {"status": "ABSTAIN", "original_minus_counterfactual": None},
        ]
        result = summarize(rows)
        self.assertEqual(result["paired_scored"], 2)
        self.assertEqual(result["abstained"], 1)
        self.assertEqual(result["positive_delta_fraction_on_scored"], 0.5)
        self.assertAlmostEqual(result["mean_delta_log_probability"], 0.05)

    def test_all_abstained(self):
        result = summarize([
            {"status": "ABSTAIN", "original_minus_counterfactual": None}
        ])
        self.assertIsNone(result["positive_delta_fraction_on_scored"])
        self.assertIsNone(result["mean_delta_log_probability"])


if __name__ == "__main__":
    unittest.main()
