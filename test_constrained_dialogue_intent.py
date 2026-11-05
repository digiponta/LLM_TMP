"""Tests for constrained intent-code scoring and paired diagnostics."""
import unittest

from constrained_dialogue_intent import (
    CANDIDATES, build_candidate_sequences, choose_best
)
from tokenizer import Tokenizer


class ConstrainedDialogueTests(unittest.TestCase):
    def setUp(self):
        self.tok = Tokenizer()
        self.tok.fit_texts(["入力:こんにちは。\nSD:" + code + "\n"
                            for code in CANDIDATES])

    def test_all_classes_enumerated(self):
        self.assertEqual(len(CANDIDATES), 8)
        self.assertEqual(len(set(CANDIDATES)), 8)

    def test_all_candidates_fit(self):
        rows, error = build_candidate_sequences(
            self.tok, "こんにちは。", context_length=96)
        self.assertIsNone(error)
        self.assertEqual(len(rows), 8)

    def test_prompt_mask_positions(self):
        rows, _ = build_candidate_sequences(
            self.tok, "こんにちは。", context_length=96)
        for code, x, y, start, count in rows:
            self.assertEqual(len(x), len(y))
            target = self.tok.decode(y[start:start+count])
            self.assertEqual(target, code + "\n")
            self.assertEqual(y[start + count - 1], self.tok.eos_id)

    def test_truncated_context_abstains(self):
        rows, error = build_candidate_sequences(
            self.tok, "こんにちは。", context_length=10)
        self.assertIsNone(rows)
        self.assertEqual(error, "CONTEXT_TOO_LONG")

    def test_unknown_input_abstains(self):
        rows, error = build_candidate_sequences(
            self.tok, "未知文字𠮟", context_length=96)
        self.assertIsNone(rows)
        self.assertEqual(error, "UNKNOWN_INPUT_TOKEN")

    def test_empty_input_abstains(self):
        rows, error = build_candidate_sequences(
            self.tok, " ", context_length=96)
        self.assertIsNone(rows)
        self.assertEqual(error, "EMPTY_INPUT")

    def test_tie_break_deterministic(self):
        scores = [
            {"code": "I(QUESTION)", "mean_log_probability": -1.},
            {"code": "I(GREETING)", "mean_log_probability": -1.},
        ]
        self.assertEqual(choose_best(scores), "I(GREETING)")

    def test_scores_select_best(self):
        scores = [
            {"code": "I(THANKS)", "mean_log_probability": -3.},
            {"code": "I(REQUEST)", "mean_log_probability": -0.2},
        ]
        self.assertEqual(choose_best(scores), "I(REQUEST)")

    def test_paired_assembly_does_not_change_model(self):
        # The scoring helper returns no altered tokenizer or checkpoint.
        before = dict(self.tok.token_to_id)
        build_candidate_sequences(self.tok, "こんにちは。")
        self.assertEqual(before, self.tok.token_to_id)


if __name__ == "__main__":
    unittest.main()
