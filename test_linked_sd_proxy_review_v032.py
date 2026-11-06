"""Headless tests: automatic proxy proposals separated from human approvals."""
import copy
import tempfile
import unittest
from pathlib import Path
from linked_sd_annotation_v031 import FIELDS, read_json, save_atomic
from linked_sd_proxy_review_v032 import (
    prepare_proposals, validate_proposals, decide, export_approvals,
    audit_approvals, propose_case
)

def registry():
    return {"namespace": "local", "snapshot_id": "demo-only",
            "display_names": {"CAT": "猫", "DOG": "犬", "EAT": "食べる"},
            "nodes": [
                {"sd_id": "CAT", "node_type": "CONCEPT"},
                {"sd_id": "DOG", "node_type": "CONCEPT"},
                {"sd_id": "EAT", "node_type": "PREDICATE",
                 "continuity_allowed": False}
            ]}

def source():
    return {"cases": [
        {"case_id": "c1", "context_a": "今日は猫の話です。", "context_b": "犬です。",
         "reference_response": "そうですね。",
         "proxy_label": "A", "baseline_prediction": "B",
         "gold_label": "A", "ai_proxy_analysis": {"label": "A"}},
        {"case_id": "c2", "context_a": "話題が不明です。",
         "context_b": "同じく不明です。", "reference_response": "はい。",
         "proxy_label": "B"}
    ]}

class ProxyApprovalTests(unittest.TestCase):
    def test_unambiguous_last_line(self):
        r = propose_case({"context_a": "昔の話\n猫", "context_b": "犬です"}, registry())
        self.assertEqual(r["topic_refs"]["a_current"], "CAT")
        self.assertEqual(r["topic_refs"]["a_previous"], "UNKNOWN")
        self.assertEqual(r["topic_refs"]["a_reference"], "UNKNOWN")
        self.assertEqual(r["topic_refs"]["b_current"], "DOG")

    def test_ambiguous_abstains(self):
        r = propose_case({"context_a": "猫と犬", "context_b": ""}, registry())
        self.assertEqual(r["topic_refs"]["a_current"], "UNKNOWN")
        self.assertEqual(r["reason_codes"]["a_current"], "AMBIGUOUS_MATCH")

    def test_no_segmented_previous(self):
        r = propose_case({"context_a": "猫", "context_b": "犬"}, registry())
        self.assertEqual(r["topic_refs"]["a_previous"], "UNKNOWN")

    def test_no_label_leakage(self):
        data = prepare_proposals(source(), registry())
        encoded = str(data)
        self.assertNotIn("proxy_label", encoded)
        self.assertNotIn("gold_label", encoded)
        self.assertNotIn("baseline_prediction", encoded)
        self.assertNotIn("ai_proxy_analysis", encoded)
        self.assertNotIn("今日は猫の話です", encoded)
        self.assertNotIn("参考応答", encoded)

    def test_reference_label_change_does_not_affect(self):
        a = source()
        b = copy.deepcopy(a)
        b["cases"][0]["proxy_label"] = "B"
        b["cases"][0]["gold_label"] = "NEITHER"
        self.assertEqual(prepare_proposals(a,registry()), prepare_proposals(b,registry()))

    def test_approval_export_only_approved(self):
        s = source()
        p = prepare_proposals(s,registry())
        decide(p,"c1","APPROVED")
        decide(p,"c2","REJECTED")
        out = export_approvals(p,s,registry())
        self.assertEqual(out["reviews"]["c1"]["annotation_status"],"HUMAN_REVIEW_COMPLETE")
        self.assertEqual(out["reviews"]["c2"]["annotation_status"],"PENDING")
        self.assertIn("RULE_CANDIDATE",out["reviews"]["c1"]["rationale"])
        self.assertEqual(out["reviews"]["c1"]["reviewer_id"],"LOCAL_USER_APPROVER")

    def test_no_automatic_approval(self):
        s=source()
        p=prepare_proposals(s,registry())
        out=export_approvals(p,s,registry())
        self.assertTrue(all(x["annotation_status"]=="PENDING" for x in out["reviews"].values()))

    def test_all_unknown_stays_unknown(self):
        s=source();p=prepare_proposals(s,registry())
        self.assertTrue(all(p["proposals"]["c2"]["topic_refs"][k]=="UNKNOWN" for k in FIELDS))
        decide(p,"c2","APPROVED")
        out=export_approvals(p,s,registry())
        self.assertTrue(all(v=="UNKNOWN" for v in out["reviews"]["c2"]["topic_refs"].values()))

    def test_changed_source_rejected(self):
        s=source();p=prepare_proposals(s,registry());s["cases"][0]["context_a"]="改変"
        with self.assertRaisesRegex(ValueError,"PROPOSAL_SOURCE_OR_ENGINE_CHANGED"):
            validate_proposals(p,s,registry())

    def test_changed_registry_rejected(self):
        s=source();p=prepare_proposals(s,registry());r=registry()
        r["display_names"]["CAT"]="別名"
        with self.assertRaisesRegex(ValueError,"PROPOSAL_SOURCE_OR_ENGINE_CHANGED"):
            validate_proposals(p,s,r)

    def test_invalid_decision_rejected(self):
        s=source();p=prepare_proposals(s,registry())
        with self.assertRaisesRegex(ValueError,"INVALID_APPROVAL"):
            decide(p,"c1","HUMAN_REVIEW_COMPLETE")

    def test_audit(self):
        s=source();p=prepare_proposals(s,registry())
        decide(p,"c1","APPROVED")
        decide(p,"c2","REJECTED")
        result=audit_approvals(p,s,registry())
        self.assertEqual(result["status"],"REVIEW_COMPLETE")
        self.assertEqual(result["decisions"]["APPROVED"],1)

    def test_save_resume(self):
        s=source();p=prepare_proposals(s,registry())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"proposals.json"
            save_atomic(path,p)
            self.assertEqual(validate_proposals(read_json(path),s,registry()),p)

if __name__=="__main__":
    unittest.main()
