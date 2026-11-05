"""Unit tests for blinded RealPersonaChat human-review preparation."""
import unittest

from review_realpersonachat_semantics import (
    audit_reviews, build_review_pack, classify,
)


def row(n, score):
    return {
        "dialogue_id": str(n),
        "turn_id": 3,
        "status": "SCORED",
        "original_prompt": "会話:\nspeaker_0:元文脈" + str(n) + "\nspeaker_1:",
        "shuffled_prompt": "会話:\nspeaker_0:別文脈" + str(n) + "\nspeaker_1:",
        "reference": "参照応答",
        "donor_dialogue_id": "donor-" + str(n),
        "context_turn_count": 2,
        "original_minus_shuffled": score,
    }


def report(rows):
    return {"status": "CORPUS_DEV_FULL_CONTEXT_SHUFFLE_NOT_INDEPENDENT",
            "rows": rows}


class HumanReviewTests(unittest.TestCase):
    def reports(self):
        return (
            report([row(1, -.3), row(2, .4), row(3, .2),
                    row(4, -.1), row(5, -.2), row(6, .3)]),
            report([row(1, .2), row(2, -.3), row(3, .1),
                    row(4, -.3), row(5, .3), row(6, .2)]),
        )

    def test_classification(self):
        self.assertEqual(classify(-.1, .1), "RECOVERED")
        self.assertEqual(classify(.1, -.1), "REGRESSED")
        self.assertEqual(classify(.1, .2), "STABLE_OR_MIXED")

    def test_prepare_blinded(self):
        a, b = self.reports()
        pack = build_review_pack(a, b, seed=42, per_bucket=1)
        self.assertEqual(len(pack["cases"]), 3)
        self.assertEqual(pack["status"], "HUMAN_REVIEW_PENDING")
        for case in pack["cases"]:
            self.assertIsNone(case["review"]["label"])
            self.assertNotIn("original_minus_shuffled", case)
            self.assertNotIn("bucket", case)
            self.assertIn("参照応答", case["reference_response"])

    def test_determinism(self):
        a, b = self.reports()
        self.assertEqual(build_review_pack(a, b, seed=7),
                         build_review_pack(a, b, seed=7))

    def test_source_mismatch_rejected(self):
        a, b = self.reports()
        b["rows"][0]["shuffled_prompt"] = "different"
        with self.assertRaisesRegex(ValueError, "MISMATCHED_CASE_CONTENT"):
            build_review_pack(a, b)

    def test_pending_audit(self):
        a, b = self.reports()
        pack = build_review_pack(a, b, per_bucket=1)
        audit = audit_reviews(pack)
        self.assertEqual(audit["status"], "HUMAN_REVIEW_PENDING")
        self.assertEqual(audit["human_approved"], 0)
        self.assertEqual(audit["pending"], 3)

    def test_completed_audit(self):
        a, b = self.reports()
        pack = build_review_pack(a, b, per_bucket=1)
        for case in pack["cases"]:
            case["review"] = {
                "label": "BOTH", "rationale": "どちらの文脈も支持",
                "reviewer_id": "reviewer-1",
            }
        audit = audit_reviews(pack)
        self.assertEqual(audit["status"], "HUMAN_REVIEW_COMPLETE")
        self.assertEqual(audit["human_approved"], 3)

    def test_missing_rationale_invalid(self):
        a, b = self.reports()
        pack = build_review_pack(a, b, per_bucket=1)
        pack["cases"][0]["review"] = {
            "label": "A", "rationale": " ", "reviewer_id": "reviewer-1",
        }
        self.assertEqual(audit_reviews(pack)["status"], "INVALID")

    def test_invalid_label(self):
        a, b = self.reports()
        pack = build_review_pack(a, b, per_bucket=1)
        pack["cases"][0]["review"]["label"] = "AI_GUESS"
        self.assertEqual(audit_reviews(pack)["status"], "INVALID")

    def test_duplicate_case_ids_rejected(self):
        a, b = self.reports()
        pack = build_review_pack(a, b, per_bucket=2)
        pack["cases"][1]["case_id"] = pack["cases"][0]["case_id"]
        with self.assertRaisesRegex(ValueError, "DUPLICATE_REVIEW_CASE_ID"):
            audit_reviews(pack)

    def test_reject_nonpositive_allocation(self):
        a, b = self.reports()
        with self.assertRaisesRegex(ValueError, "INVALID_PER_BUCKET"):
            build_review_pack(a, b, per_bucket=0)


if __name__ == "__main__":
    unittest.main()
