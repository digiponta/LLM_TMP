"""Paired v0.3 Dev evaluator safety tests."""
import copy
import unittest
from test_intent_sd_context_v03 import snapshot, intent, state
from evaluate_intent_sd_context_v03 import run

def data():
    s = state()
    shifted = intent(curr="music", utterance="u4")
    related = intent(curr="kyoto", utterance="u5")
    return {
        "scope": "DEV_NOT_INDEPENDENT", "reference_type": "AI_PROXY",
        "annotation_independent_of_reference": True, "expected_case_count": 2,
        "registry_snapshot": snapshot(),
        "cases": [
            {"case_id": "case-1", "reference_label": "A",
             "baseline_prediction": "B", "intent_a": related,
             "state_a": s, "intent_b": shifted, "state_b": s},
            {"case_id": "case-2", "reference_label": "B",
             "baseline_prediction": "A", "intent_a": shifted,
             "state_a": s, "intent_b": related, "state_b": s},
        ]
    }

class PairedDevTests(unittest.TestCase):
    def test_frozen_cases(self):
        r = run(data())
        self.assertEqual(r["case_count"], 2)
        self.assertEqual(r["v03_accuracy"], 1.0)
        self.assertEqual(r["baseline_accuracy"], 0.0)
        self.assertEqual(r["v03_only_correct"], 2)

    def test_deterministic(self):
        d = data()
        self.assertEqual(run(d), run(d))

    def test_no_test_split(self):
        d = data()
        d["scope"] = "TEST"
        with self.assertRaisesRegex(ValueError, "ONLY_DEV"):
            run(d)

    def test_no_double_id(self):
        d = data()
        d["cases"][1]["case_id"] = "case-1"
        with self.assertRaisesRegex(ValueError, "INVALID_CASE_IDS"):
            run(d)

    def test_no_hidden_input_label(self):
        d = data()
        d["cases"][0]["intent_a"]["reference_label"] = "A"
        with self.assertRaisesRegex(ValueError, "LABEL_LEAKAGE"):
            run(d)

    def test_frozen_annotation_declaration_required(self):
        d = data()
        d["annotation_independent_of_reference"] = False
        with self.assertRaisesRegex(ValueError, "ANNOTATION_INDEPENDENCE"):
            run(d)

    def test_missing_topic_coverage(self):
        d = data()
        d["cases"][0]["intent_a"]["topic_ref"] = None
        r = run(d)
        self.assertLess(r["v03_non_unclear_coverage"], 1.0)

    def test_unexpected_fields_rejected(self):
        d = data()
        d["cases"][0]["context_a"] = "private transcript"
        with self.assertRaisesRegex(ValueError, "UNEXPECTED_CASE_FIELDS"):
            run(d)

if __name__ == "__main__":
    unittest.main()
