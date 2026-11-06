"""No-GPU regression tests for Intent SD v0.2 Topic / Context consistency."""
import unittest
from intent_sd_context_v02 import ContextSD, choose_context, topic_continuity, context_consistency
from evaluate_intent_sd_context_v02 import evaluate

def fixture():
    return {"scope": "DEV_NOT_INDEPENDENT", "reference_type": "AI_PROXY",
            "expected_case_count": 2, "cases": [
                {"case_id": "one", "reference_label": "A", "baseline_prediction": "B",
                 "context_a": {"topic_previous": "travel", "topic_current": "travel", "context_reference": "travel"},
                 "context_b": {"topic_previous": "travel", "topic_current": "music", "context_reference": "music"}},
                {"case_id": "two", "reference_label": "B", "baseline_prediction": "B",
                 "context_a": {"topic_previous": "books", "topic_current": "sport", "context_reference": "sport"},
                 "context_b": {"topic_previous": "books", "topic_current": "books", "context_reference": "books"}}]}

class ContextSDTests(unittest.TestCase):
    def test_topic_continuity(self):
        self.assertEqual(topic_continuity(ContextSD("cat", "cat")), "CONTINUOUS")
        self.assertEqual(topic_continuity(ContextSD("cat", "dog")), "SHIFTED")

    def test_no_tags_abstains(self):
        self.assertEqual(choose_context(ContextSD(None, None), ContextSD("x", "y"))["label"], "UNCLEAR")

    def test_both_continuous(self):
        self.assertEqual(choose_context(ContextSD("a", "a", "a"), ContextSD("b", "b", "b"))["label"], "BOTH")

    def test_both_shifted(self):
        self.assertEqual(choose_context(ContextSD("a", "b", "b"), ContextSD("c", "d", "d"))["label"], "NEITHER")

    def test_reference_shift(self):
        self.assertEqual(context_consistency(ContextSD("a", "a", "b")), "SHIFTED")

    def test_invalid_provenance(self):
        with self.assertRaises(ValueError):
            ContextSD("a", "b", provenance="GOLD")

    def test_unknown_fields_rejected(self):
        with self.assertRaises(ValueError):
            ContextSD.from_dict({"topic_previous": "a", "topic_current": "a", "reference_label": "A"})

    def test_paired_accuracy(self):
        r = evaluate(fixture())
        self.assertEqual(r["baseline_accuracy"], 0.5)
        self.assertEqual(r["intent_accuracy"], 1.0)
        self.assertEqual(r["intent_only_correct"], 1)

    def test_test_scope_rejected(self):
        d = fixture()
        d["scope"] = "TEST"
        with self.assertRaisesRegex(ValueError, "ONLY_DEV"):
            evaluate(d)

    def test_duplicate_rejected(self):
        d = fixture()
        d["cases"][1]["case_id"] = "one"
        with self.assertRaisesRegex(ValueError, "INVALID_CASE_IDS"):
            evaluate(d)

    def test_missing_tags_rejected(self):
        d = fixture()
        del d["cases"][0]["context_a"]
        with self.assertRaisesRegex(ValueError, "MISSING_CONTEXT_SD"):
            evaluate(d)

    def test_human_independence_required(self):
        d = fixture()
        d["cases"][0]["context_a"]["provenance"] = "HUMAN_ANNOTATED"
        with self.assertRaisesRegex(ValueError, "ANNOTATION_INDEPENDENCE"):
            evaluate(d)

    def test_deterministic(self):
        self.assertEqual(evaluate(fixture()), evaluate(fixture()))

if __name__ == "__main__":
    unittest.main()
