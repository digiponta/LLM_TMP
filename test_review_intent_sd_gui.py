"""Headless tests for Intent SD review GUI data operations."""
import tempfile
import unittest
from pathlib import Path
from review_intent_sd_gui import (load_packet, set_review, atomic_save,
                                  load_source_text, format_source_text)

class ReviewGuiTests(unittest.TestCase):
    def test_manual_choice_preserves_ai(self):
        row = {"case_id": "c1", "ai_proxy_analysis": {"cause_assessment": "UNKNOWN"},
               "primary_cause": None, "secondary_causes": []}
        set_review(row, "TOPIC", ["CONTEXT"], "元の文脈との照合")
        self.assertEqual(row["primary_cause"], "TOPIC")
        self.assertEqual(row["ai_proxy_analysis"]["cause_assessment"], "UNKNOWN")
        self.assertEqual(row["annotation_status"], "HUMAN_REVIEW_COMPLETE")

    def test_invalid_choice(self):
        with self.assertRaises(ValueError):
            set_review({}, "INVALID", [], "reason")
        with self.assertRaises(ValueError):
            set_review({}, "TOPIC", ["TOPIC"], "reason")

    def test_requires_rationale(self):
        with self.assertRaises(ValueError):
            set_review({}, "LEXICAL", [], "")

    def test_resume_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.json"
            data = {"mismatch_cases": [{"case_id": "c1", "primary_cause": None}]}
            atomic_save(path, data)
            self.assertEqual(load_packet(path), data)

    def test_original_dialogue_display(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            atomic_save(path, {"cases": [
                {"case_id": "c1", "context_a": "元の会話A",
                 "context_b": "別の会話B", "reference_response": "参照応答",
                 "ai_proxy_review": {"label": "A"}}]})
            src = load_source_text(path)
            text = format_source_text({"case_id": "c1"}, src)
            self.assertIn("元の会話A", text)
            self.assertIn("別の会話B", text)
            self.assertIn("参照応答", text)
            self.assertNotIn("ai_proxy_review", text)
            self.assertIn("\\n", text)

    def test_missing_source_is_explicit(self):
        self.assertIn("ありません",
                      format_source_text({"case_id": "missing"}, {}))

    def test_source_text_duplicate_id_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            row = {"case_id": "c", "context_a": "A",
                   "context_b": "B", "reference_response": "R"}
            atomic_save(path, {"cases": [row, dict(row)]})
            with self.assertRaisesRegex(ValueError, "SOURCE_TEXT_DUPLICATE"):
                load_source_text(path)

    def test_missing_source_file_returns_empty_lookup(self):
        self.assertEqual(load_source_text("missing-private-context.json"), {})

    def test_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.json"
            atomic_save(path, {"mismatch_cases": [{"case_id": "c"}, {"case_id": "c"}]})
            with self.assertRaises(ValueError):
                load_packet(path)

if __name__ == "__main__":
    unittest.main()
