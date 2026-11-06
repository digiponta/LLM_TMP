"""Regression tests for Intent SD v0.3 linked SD engine (no GPU)."""
import copy
import unittest
from intent_sd_context_v03 import (
    SDRegistry, resolve_topic_ref, resolve_context_reference,
    lookup_relation, classify_topic_transition, choose_context_v03,
    apply_context_transition, v02_to_v03,
)

def ref(s):
    return {"namespace": "local", "sd_id": s}

def snapshot(state="APPROVED", reverse=False, allowed=True):
    return {
        "namespace": "local",
        "snapshot_id": "unit-fixture-1",
        "nodes": [
            {"sd_id": "travel", "node_type": "TOPIC"},
            {"sd_id": "kyoto", "node_type": "ENTITY"},
            {"sd_id": "temple", "node_type": "CONCEPT"},
            {"sd_id": "music", "node_type": "TOPIC"},
            {"sd_id": "destination-of", "node_type": "PREDICATE",
             "continuity_allowed": allowed},
            {"sd_id": "rel-1", "node_type": "RELATION", "predicate_ref": "destination-of",
             "source_ref": "kyoto" if reverse else "travel",
             "target_ref": "travel" if reverse else "kyoto",
             "validation_state": state, "evidence_id": "evidence-1"},
        ],
    }

def intent(prev="travel", curr="kyoto", kind="UNKNOWN", context=None, utterance="u2", relation=None):
    return {
        "schema_version": "intent-sd/0.3",
        "utterance_id": utterance, "speech_act": "ANSWER", "intent": "ANSWER_LOCATION",
        "topic_ref": ref(curr) if curr else None,
        "previous_topic_ref": ref(prev) if prev else None,
        "context_reference": context or {"status": "NONE"},
        "topic_transition": {"kind": kind, "relation_ref": ref(relation) if relation else None},
        "provenance": "RULE_CANDIDATE",
    }

def state():
    return {"active_topic_ref": ref("travel"), "history": [
        {"utterance_id": "u1", "topic_ref": ref("travel"), "verified": True}
    ]}

class LinkedTopicTests(unittest.TestCase):
    def test_same_sd(self):
        r = classify_topic_transition(intent(curr="travel"), state(), SDRegistry(snapshot()))
        self.assertEqual(r["status"], "SAME")

    def test_related_approved_direction(self):
        r = classify_topic_transition(intent(relation="rel-1"), state(), SDRegistry(snapshot()))
        self.assertEqual(r["status"], "RELATED")
        self.assertEqual(r["evidence_path"][0]["evidence_id"], "evidence-1")

    def test_reverse_not_assumed(self):
        r = classify_topic_transition(intent(), state(), SDRegistry(snapshot(reverse=True)))
        self.assertEqual(r["status"], "SHIFTED")

    def test_pending_abstains(self):
        r = classify_topic_transition(intent(), state(), SDRegistry(snapshot(state="PENDING")))
        self.assertEqual(r["status"], "UNKNOWN")

    def test_unapproved_rejected_relation(self):
        r = classify_topic_transition(intent(relation="rel-1"), state(), SDRegistry(snapshot(state="REJECTED")))
        self.assertEqual(r["status"], "UNKNOWN")

    def test_unknown_topic_abstains(self):
        r = classify_topic_transition(intent(curr="nonexistent"), state(), SDRegistry(snapshot()))
        self.assertEqual(r["status"], "UNKNOWN")

    def test_external_namespace_not_local_identity(self):
        x = intent()
        x["topic_ref"]["namespace"] = "other-vm"
        self.assertEqual(classify_topic_transition(x, state(), SDRegistry(snapshot()))["status"], "UNKNOWN")

    def test_reference_resolution(self):
        ctx = {"status": "RESOLVED", "target_utterance_id": "u1", "topic_ref": ref("travel")}
        r = classify_topic_transition(intent(context=ctx), state(), SDRegistry(snapshot()))
        self.assertEqual(r["status"], "RELATED")

    def test_wrong_context_ref_abstains(self):
        ctx = {"status": "RESOLVED", "target_utterance_id": "u1", "topic_ref": ref("music")}
        r = classify_topic_transition(intent(context=ctx), state(), SDRegistry(snapshot()))
        self.assertEqual(r["status"], "UNKNOWN")

    def test_resume_requires_explicit_turn(self):
        s = state()
        s["history"].append({"utterance_id": "uold", "topic_ref": ref("music"), "verified": True})
        ctx = {"status": "RESOLVED", "target_utterance_id": "uold", "topic_ref": ref("music")}
        r = classify_topic_transition(intent(curr="music", context=ctx), s, SDRegistry(snapshot()))
        self.assertEqual(r["status"], "RESUMED")

    def test_no_explicit_resume(self):
        s = state()
        s["history"].append({"utterance_id": "uold", "topic_ref": ref("music"), "verified": True})
        self.assertEqual(classify_topic_transition(intent(curr="music"), s, SDRegistry(snapshot()))["status"], "SHIFTED")

    def test_update_does_not_mutate_state(self):
        s = state()
        before = copy.deepcopy(s)
        i = intent()
        d = classify_topic_transition(i, s, SDRegistry(snapshot()))
        after = apply_context_transition(s, i, d)
        self.assertEqual(s, before)
        self.assertEqual(after["active_topic_ref"], ref("kyoto"))
        self.assertEqual(len(after["history"]), 2)

    def test_unknown_does_not_update(self):
        s = state()
        self.assertEqual(apply_context_transition(s, intent(), {"status": "UNKNOWN"}), s)

    def test_graph_bound(self):
        r = lookup_relation(ref("travel"), ref("temple"), SDRegistry(snapshot()), max_nodes=1)
        self.assertEqual(r["status"], "UNKNOWN")

    def test_no_fake_relation(self):
        r = classify_topic_transition(intent(relation="fake"), state(), SDRegistry(snapshot()))
        self.assertEqual(r["status"], "UNKNOWN")

    def test_invalid_claim(self):
        r = classify_topic_transition(intent(kind="SAME"), state(), SDRegistry(snapshot()))
        self.assertEqual(r["status"], "UNKNOWN")

    def test_no_answer_leak(self):
        i = intent()
        i["reference_label"] = "A"
        with self.assertRaises(ValueError):
            classify_topic_transition(i, state(), SDRegistry(snapshot()))

    def test_duplicate_sd_rejected(self):
        x = snapshot()
        x["nodes"].append(copy.deepcopy(x["nodes"][0]))
        with self.assertRaisesRegex(ValueError, "DUPLICATE_SD_ID"):
            SDRegistry(x)

    def test_dangling_ref_rejected(self):
        x = snapshot()
        x["nodes"][-1]["target_ref"] = "not-there"
        with self.assertRaisesRegex(ValueError, "DANGLING_RELATION_REF"):
            SDRegistry(x)

    def test_frozen_repeat(self):
        reg = SDRegistry(snapshot())
        a = classify_topic_transition(intent(), state(), reg)
        b = classify_topic_transition(intent(), state(), reg)
        self.assertEqual(a, b)

    def test_choose_context(self):
        a = {"status": "RELATED"}
        b = {"status": "SHIFTED"}
        self.assertEqual(choose_context_v03(a,b)["label"], "A")
        self.assertEqual(choose_context_v03({"status":"UNKNOWN"},b)["label"], "UNCLEAR")

    def test_versioned_legacy_adapter(self):
        reg = SDRegistry(snapshot())
        mapping = {"mapping_version": "v1", "aliases": {"旅行": ref("travel")}}
        self.assertEqual(v02_to_v03("旅行",mapping,reg)["status"], "OK")
        self.assertEqual(v02_to_v03("たび",mapping,reg)["status"], "UNKNOWN")

    def test_no_graph_mutation(self):
        raw = snapshot()
        before = copy.deepcopy(raw)
        reg = SDRegistry(raw)
        reg.relation_path(("local","travel"),("local","kyoto"))
        self.assertEqual(raw, before)

if __name__ == "__main__":
    unittest.main()
