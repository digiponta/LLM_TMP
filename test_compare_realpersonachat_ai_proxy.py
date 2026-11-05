"""Tests for AI-proxy-only Dev concordance (no GPU)."""
import unittest

from compare_realpersonachat_ai_proxy import concordance
from review_realpersonachat_semantics import rank_case


def make_data(labels=("A", "B", "BOTH")):
    baseline, candidate, cases = [], [], []
    for i, label in enumerate(labels):
        identity = (str(i), "2")
        cid = rank_case(identity, 42)[:16]
        a = {
            "dialogue_id": str(i), "turn_id": 2, "status": "SCORED",
            "reference": "固定応答",
            "original_prompt": "会話:\nspeaker_0:元" + str(i) + "\nspeaker_1:",
            "shuffled_prompt": "会話:\nspeaker_0:別" + str(i) + "\nspeaker_1:",
            "donor_dialogue_id": "donor" + str(i),
            "context_turn_count": 1,
            "original_minus_shuffled": (-.1 if i == 0 else .2),
        }
        b = dict(a, original_minus_shuffled=(.3 if i == 0 else -.2))
        baseline.append(a)
        candidate.append(b)
        cases.append({
            "case_id": cid, "reference_response": "固定応答",
            "context_a": a["original_prompt"],
            "context_b": a["shuffled_prompt"],
            "review": {"label": None},
            "ai_proxy_review": {"label": label, "rationale": "検証用",
                                "reviewer_type": "AI",
                                "human_verified": False},
        })
    p = {"scope": "REALPERSONACHAT_DEVELOPMENT_DEV_NOT_INDEPENDENT",
         "ai_proxy_status": "AI_PROXY_REVIEW_COMPLETE_HUMAN_REVIEW_PENDING",
         "seed": 42, "cases": cases}
    return p, {
        "status": "CORPUS_DEV_FULL_CONTEXT_SHUFFLE_NOT_INDEPENDENT",
        "rows": baseline}, {
        "status": "CORPUS_DEV_FULL_CONTEXT_SHUFFLE_NOT_INDEPENDENT",
        "rows": candidate}


class ProxyAgreementTests(unittest.TestCase):
    def test_exclusive_only(self):
        p, a, b = make_data()
        result = concordance(p, a, b)
        self.assertEqual(result["reviewed"], 3)
        self.assertEqual(result["exclusive_proxy_label_cases"], 2)
        self.assertEqual(result["baseline_concordant"], 0)
        self.assertEqual(result["candidate_concordant"], 2)
        self.assertEqual(result["candidate_minus_baseline_concordance"], 1.0)

    def test_nonexclusive_only(self):
        p, a, b = make_data(("BOTH",))
        result = concordance(p, a, b)
        self.assertIsNone(result["baseline_concordance_on_exclusive"])

    def test_changed_context_rejected(self):
        p, a, b = make_data()
        p["cases"][0]["context_a"] = "異なる文"
        with self.assertRaisesRegex(ValueError, "REVIEW_CONTEXT_MISMATCH"):
            concordance(p, a, b)

    def test_reference_changed_rejected(self):
        p, a, b = make_data()
        p["cases"][0]["reference_response"] = "異なる応答"
        with self.assertRaisesRegex(ValueError, "REFERENCE_MISMATCH"):
            concordance(p, a, b)

    def test_human_label_not_merged(self):
        p, a, b = make_data()
        p["cases"][0]["review"]["label"] = "A"
        with self.assertRaisesRegex(ValueError, "HUMAN_REVIEW_MUST_REMAIN_SEPARATE"):
            concordance(p, a, b)

    def test_fake_human_verified_rejected(self):
        p, a, b = make_data()
        p["cases"][0]["ai_proxy_review"]["human_verified"] = True
        with self.assertRaisesRegex(ValueError, "INVALID_AI_PROXY_LABEL"):
            concordance(p, a, b)

    def test_duplicate_case_id_rejected(self):
        p, a, b = make_data()
        p["cases"][1]["case_id"] = p["cases"][0]["case_id"]
        with self.assertRaisesRegex(ValueError, "DUPLICATE_PROXY_IDS"):
            concordance(p, a, b)

    def test_model_content_mismatch_rejected(self):
        p, a, b = make_data()
        b["rows"][0]["shuffled_prompt"] = "different"
        with self.assertRaisesRegex(ValueError, "MODEL_REPORT_PAIR_MISMATCH"):
            concordance(p, a, b)


if __name__ == "__main__":
    unittest.main()
