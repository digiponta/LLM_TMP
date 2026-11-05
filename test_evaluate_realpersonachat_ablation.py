"""Tests for three-condition RealPersonaChat context ablation."""
import unittest

from evaluate_realpersonachat_ablation import make_prompts, summarize


def example():
    return {
        "dialogue_id": "42", "turn_id": 7,
        "context": [
            {"speaker": "speaker_0", "text": "今日の天気は？"},
            {"speaker": "speaker_1", "text": "晴れです。"},
        ],
        "next_speaker": "speaker_0", "response": "よかったです。",
    }


class ContextAblationTests(unittest.TestCase):
    def test_prompts_preserve_next_speaker(self):
        prompts = make_prompts(example(), 2)
        self.assertTrue(all(v.endswith("speaker_0:")
                            for v in prompts.values()))

    def test_full_last_none(self):
        prompts = make_prompts(example(), 2)
        self.assertIn("今日の天気は？", prompts["full"])
        self.assertIn("晴れです。", prompts["full"])
        self.assertNotIn("今日の天気は？", prompts["last_turn"])
        self.assertIn("晴れです。", prompts["last_turn"])
        self.assertEqual(prompts["no_context"], "会話:\nspeaker_0:")

    def test_does_not_mutate_source(self):
        original = example()
        make_prompts(original, 2)
        self.assertEqual(len(original["context"]), 2)

    def test_metrics(self):
        rows = [
            {"status": "SCORED", "reason": None, "context_turn_count": 2,
             "scores": {"full": -1.0, "last_turn": -1.2,
                        "no_context": -1.5}},
            {"status": "SCORED", "reason": None, "context_turn_count": 2,
             "scores": {"full": -2.0, "last_turn": -1.8,
                        "no_context": -1.7}},
        ]
        result = summarize(rows)
        self.assertEqual(result["all_three_scored"], 2)
        self.assertEqual(result["positive_fraction_full_vs_none"], .5)
        self.assertAlmostEqual(result["mean_delta_full_minus_none"], .1)
        self.assertEqual(result["two_turn_context_cases"], 2)

    def test_abstained_skipped_from_all_pairs(self):
        rows = [
            {"status": "ABSTAIN", "reason": "full:UNKNOWN_PROMPT_CHARACTERS",
             "context_turn_count": 2, "scores": None},
        ]
        result = summarize(rows)
        self.assertEqual(result["abstained"], 1)
        self.assertIsNone(result["mean_delta_full_minus_none"])
        self.assertEqual(result["abstention_reasons"],
                         {"full:UNKNOWN_PROMPT_CHARACTERS": 1})

    def test_single_turn_not_counted_as_two_turn(self):
        result = summarize([
            {"status": "SCORED", "reason": None, "context_turn_count": 1,
             "scores": {"full": -1.0, "last_turn": -1.0,
                        "no_context": -1.1}}
        ])
        self.assertEqual(result["two_turn_context_cases"], 0)
        self.assertEqual(result["positive_fraction_full_vs_last"], 0.0)


if __name__ == "__main__":
    unittest.main()
