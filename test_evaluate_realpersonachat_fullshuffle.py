"""Tests for full-length matched context shuffle (no GPU needed)."""
import unittest

from evaluate_realpersonachat_fullshuffle import (
    choose_matched_donor, make_full_shuffle, summarize,
)


def example(identifier, a="こんにちは。", b="元気です。", speakers=None):
    speakers = speakers or ("speaker_0", "speaker_1")
    return {
        "dialogue_id": str(identifier), "turn_id": 3,
        "context": [
            {"speaker": speakers[0], "text": a},
            {"speaker": speakers[1], "text": b},
        ],
        "next_speaker": "speaker_0",
        "response": "そうですね。",
    }


class FullContextShuffleTests(unittest.TestCase):
    def test_donor_chooses_nearest_length(self):
        source = example(1, "abc", "abcd")
        far = example(2, "長い文章です。" * 10, "別の文章です。" * 10)
        near = example(3, "xyz", "1234")
        donor = choose_matched_donor([source, far, near], source, 2)
        self.assertEqual(donor["dialogue_id"], "3")

    def test_donor_excludes_same_dialogue(self):
        source = example(1, "abc", "def")
        same_id = example(1, "xyz", "ghi")
        other = example(2, "hello", "world")
        self.assertEqual(
            choose_matched_donor([source, same_id, other],
                                 source, 2)["dialogue_id"], "2")

    def test_donor_preserves_speaker_pattern(self):
        source = example(1)
        wrong = example(2, speakers=("speaker_1", "speaker_0"))
        right = example(3, "違う話", "続き")
        donor = choose_matched_donor([source, wrong, right], source, 2)
        self.assertEqual(donor["dialogue_id"], "3")

    def test_identical_text_not_selected(self):
        source = example(1)
        copy = example(2)
        with self.assertRaisesRegex(ValueError, "NO_MATCHED_DONOR"):
            choose_matched_donor([source, copy], source, 2)

    def test_preserves_original_and_reference(self):
        src = example(1, "天気は？", "晴れです。")
        donor = example(2, "旅行の話", "温泉です。")
        shuffled = make_full_shuffle(src, donor, 2)
        self.assertEqual(len(shuffled["context"]), 2)
        self.assertEqual(shuffled["context"][0]["speaker"], "speaker_0")
        self.assertEqual(shuffled["context"][1]["speaker"], "speaker_1")
        self.assertEqual(shuffled["context"][0]["text"], "旅行の話")
        self.assertEqual(shuffled["context"][1]["text"], "温泉です。")
        self.assertEqual(shuffled["next_speaker"], src["next_speaker"])
        self.assertEqual(shuffled["response"], src["response"])
        self.assertEqual(src["context"][0]["text"], "天気は？")

    def test_same_dialogue_rejected(self):
        with self.assertRaisesRegex(ValueError, "DONOR_SAME_DIALOGUE"):
            make_full_shuffle(example(1), example(1), 2)

    def test_mismatched_context_turn_count_rejected(self):
        donor = example(2)
        donor["context"] = donor["context"][-1:]
        with self.assertRaisesRegex(ValueError, "MISMATCHED_CONTEXT_TURN_COUNT"):
            make_full_shuffle(example(1), donor, 2)

    def test_mismatched_speaker_pattern_rejected(self):
        with self.assertRaisesRegex(ValueError, "MISMATCHED_SPEAKER_SEQUENCE"):
            make_full_shuffle(example(1), example(2, speakers=(
                "speaker_1", "speaker_0")), 2)

    def test_metrics_and_abstention(self):
        rows = [
            {"status": "SCORED", "reason": None,
             "original_log_probability": -1.0,
             "shuffled_log_probability": -2.0,
             "original_minus_shuffled": 1.0,
             "context_length_difference": 2},
            {"status": "SCORED", "reason": None,
             "original_log_probability": -3.0,
             "shuffled_log_probability": -2.0,
             "original_minus_shuffled": -1.0,
             "context_length_difference": 4},
            {"status": "ABSTAIN", "reason": "UNKNOWN_PROMPT_CHARACTERS",
             "original_minus_shuffled": None},
        ]
        report = summarize(rows)
        self.assertEqual(report["selected"], 3)
        self.assertEqual(report["paired_scored"], 2)
        self.assertEqual(report["abstained"], 1)
        self.assertEqual(report["positive_delta_fraction_on_scored"], .5)
        self.assertEqual(report["mean_delta_log_probability"], 0.)
        self.assertEqual(report["mean_donor_length_difference"], 3.)

    def test_all_abstained(self):
        report = summarize([{
            "status": "ABSTAIN", "reason": "NO_MATCHED_DONOR",
            "original_minus_shuffled": None,
        }])
        self.assertIsNone(report["mean_delta_log_probability"])
        self.assertIsNone(report["positive_delta_fraction_on_scored"])


if __name__ == "__main__":
    unittest.main()
