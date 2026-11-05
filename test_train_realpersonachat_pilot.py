"""Tests for bounded local RealPersonaChat response-learning pilot."""
import json
import tempfile
import unittest
from pathlib import Path

from tokenizer import Tokenizer
from train_realpersonachat_pilot import (
    canonical_key, load_examples, prompt_text, read_sets,
)


def record(dialogue_id, turn_id, response="こんにちは。"):
    return {
        "dialogue_id": str(dialogue_id),
        "turn_id": turn_id,
        "next_speaker": "speaker_1",
        "context": [
            {"speaker": "speaker_0", "text": "おはよう。"},
            {"speaker": "speaker_1", "text": "はい。"},
            {"speaker": "speaker_0", "text": "調子はどう？"},
        ],
        "response": response,
    }


class RealPersonaChatPilotTests(unittest.TestCase):
    def test_stable_key(self):
        self.assertEqual(canonical_key(record("abc", 4)), "abc:4")

    def test_context_cap(self):
        self.assertEqual(
            prompt_text(record("a", 2), 2),
            "会話:\nspeaker_1:はい。\nspeaker_0:調子はどう？\nspeaker_1:")

    def test_target_speaker_preserved(self):
        row = record(1, 1)
        row["next_speaker"] = "speaker_0"
        self.assertTrue(prompt_text(row).endswith("speaker_0:"))

    def test_sampling_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "train.jsonl"
            rows = [record(i, 1) for i in range(20)]
            p.write_text("\n".join(json.dumps(r, ensure_ascii=False)
                                   for r in rows) + "\n", encoding="utf-8")
            self.assertEqual(
                load_examples(p, 5, 42, 2, 70),
                load_examples(p, 5, 42, 2, 70))
            self.assertEqual(len(load_examples(p, 5, 42, 2, 70)), 5)

    def test_reject_duplicate_example_key(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "train.jsonl"
            x = json.dumps(record(1, 1), ensure_ascii=False)
            p.write_text(x + "\n" + x + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "DUPLICATE_EXAMPLE"):
                load_examples(p, 10, 42, 2, 70)

    def test_max_chars_filter(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "train.jsonl"
            p.write_text(json.dumps(record(1, 1, "長" * 100),
                          ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertEqual(load_examples(p, 10, 42, 2, 70), [])

    def test_train_dev_dialogue_isolation(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            (directory / "audit.json").write_text(
                json.dumps({"status": "EXPORTED_CORPUS_HELDOUT"}),
                encoding="utf-8")
            for split in ("train", "dev"):
                (directory / (split + ".jsonl")).write_text(
                    json.dumps(record(1, 1), ensure_ascii=False) + "\n",
                    encoding="utf-8")
            class Args:
                input_dir = str(directory)
                train_limit = 10
                dev_limit = 10
                seed = 42
                context_turns = 2
                max_chars = 70
            with self.assertRaisesRegex(ValueError, "DIALOGUE_SPLIT_LEAKAGE"):
                read_sets(Args())

    def test_tokenizer_can_use_train_only(self):
        tok = Tokenizer()
        tok.fit_texts([prompt_text(record(1, 1)) + "こんにちは。"])
        self.assertIn("こ", tok.token_to_id)
        self.assertEqual(tok.encode("𠮟"), [tok.unk_id])


if __name__ == "__main__":
    unittest.main()
