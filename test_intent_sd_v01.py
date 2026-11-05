"""Regression tests for Intent SD Dev analysis and paired scoring (no GPU)."""
import unittest
from copy import deepcopy
from intent_sd import IntentSD
from compare_intent_sd_ablation import evaluate
from analyze_proxy_disagreements import prepare, audit
from test_compare_realpersonachat_ai_proxy import make_data

def ablation_fixture():
    return {"scope": "DEV_NOT_INDEPENDENT", "expected_case_count": 2,
            "cases": [
                {"case_id": "x", "reference_type": "AI_PROXY",
                 "reference_label": "A", "baseline_prediction": "B",
                 "intent_prediction": "A",
                 "intent_sd": {"speech_act": "QUESTION", "intent": "ASK"}},
                {"case_id": "y", "reference_type": "AI_PROXY",
                 "reference_label": "B", "baseline_prediction": "B",
                 "intent_prediction": "A",
                 "intent_sd": {"speech_act": "REQUEST", "intent": "EXPLAIN_MORE"}},
            ]}

class IntentSDTests(unittest.TestCase):
    def test_schema_roundtrip(self):
        obj = IntentSD(speech_act="REQUEST", intent="EXPLAIN_MORE",
                       target="PREVIOUS_TOPIC", context_required=True)
        self.assertEqual(IntentSD.from_dict(obj.to_dict()), obj)
        self.assertIsNone(obj.confidence)

    def test_invalid_confidence(self):
        with self.assertRaisesRegex(ValueError, "INVALID_CONFIDENCE"):
            IntentSD(speech_act="QUESTION", intent="ASK", confidence=1.5)

    def test_unknown_field(self):
        with self.assertRaisesRegex(ValueError, "UNKNOWN_INTENT_SD_FIELDS"):
            IntentSD.from_dict({"speech_act": "QUESTION", "intent": "ASK", "gold": "A"})

class ReviewTests(unittest.TestCase):
    def test_mismatch_packet_is_pending(self):
        p, a, b = make_data()
        packet = prepare(p, a, b)
        self.assertEqual(packet["mismatch_count"], 2)
        self.assertEqual(audit(packet)["pending"], 2)
        self.assertNotIn("context_a", packet["mismatch_cases"][0])

    def test_completed_annotations(self):
        p, a, b = make_data()
        packet = prepare(p, a, b)
        for case in packet["mismatch_cases"]:
            case["primary_cause"] = "TOPIC"
            case["rationale"] = "Manual diagnostic"
        self.assertEqual(audit(packet)["status"], "CAUSE_ANNOTATION_COMPLETE")

    def test_invalid_annotation(self):
        p, a, b = make_data()
        packet = prepare(p, a, b)
        packet["mismatch_cases"][0]["primary_cause"] = "TOPIC"
        with self.assertRaisesRegex(ValueError, "RATIONALE_REQUIRED"):
            audit(packet)

class AblationTests(unittest.TestCase):
    def test_paired_metrics(self):
        result = evaluate(ablation_fixture())
        self.assertEqual(result["baseline_accuracy"], .5)
        self.assertEqual(result["intent_sd_accuracy"], .5)
        self.assertEqual(result["baseline_only_correct"], 1)
        self.assertEqual(result["intent_sd_only_correct"], 1)

    def test_duplicate_ids(self):
        data = ablation_fixture()
        data["cases"][1]["case_id"] = "x"
        with self.assertRaisesRegex(ValueError, "DUPLICATE_CASE_ID"):
            evaluate(data)

    def test_sealed_test_rejected(self):
        data = ablation_fixture()
        data["scope"] = "TEST"
        with self.assertRaisesRegex(ValueError, "ONLY_DEVELOPMENT"):
            evaluate(data)

    def test_cohort_mismatch(self):
        data = ablation_fixture()
        data["expected_case_count"] = 40
        with self.assertRaisesRegex(ValueError, "COHORT_SIZE_MISMATCH"):
            evaluate(data)

    def test_no_silent_missing_sd(self):
        data = ablation_fixture()
        data["cases"][0]["intent_sd"] = None
        with self.assertRaisesRegex(ValueError, "MISSING_INTENT"):
            evaluate(data)

if __name__ == "__main__":
    unittest.main()
