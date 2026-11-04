"""Atomicity and correctness checks for untrusted SD-candidate pipeline."""
import json
import unittest

from recursive_sd import build_sample_graph
from sd_candidate_pipeline import (
    CandidateRejected, DeterministicProposer, accept_candidate, run,
)


class CandidatePipelineTests(unittest.TestCase):
    def setUp(self):
        self.graph = build_sample_graph()
        self.proposer = DeterministicProposer(self.graph)

    def test_rule_simple(self):
        r = run("猫が魚を食べる。", self.proposer, self.graph)
        self.assertEqual(r["status"], "ACCEPTED")
        self.assertEqual(r["rendered"], r["input"])
        self.assertFalse(r["verified_fact"])

    def test_rule_nested(self):
        r = run("猫が魚を食べると、犬が鳥を見る。", self.proposer, self.graph)
        self.assertEqual(r["status"], "ACCEPTED")
        self.assertEqual(r["rendered"], r["input"])
        self.assertTrue(self.graph.validate())

    def test_id_reuse_on_second_call(self):
        text = "猫が魚を食べる、そして犬が鳥を見る。"
        a = run(text, self.proposer, self.graph)
        b = run(text, self.proposer, self.graph)
        self.assertEqual(a["sd_ref"], b["sd_ref"])
        self.assertEqual(b["node_count_before"], b["node_count_after"])

    def test_candidate_json_string(self):
        text = "猫が魚を食べる。"
        candidate = self.proposer.propose(text)
        r = accept_candidate(self.graph, text, json.dumps(candidate))
        self.assertEqual(r["status"], "ACCEPTED")

    def test_unknown_not_committed(self):
        before = self.graph.snapshot()
        r = run("猫が魚を踊る。", self.proposer, self.graph)
        self.assertEqual(r["status"], "REJECTED")
        self.assertEqual(self.graph.snapshot(), before)

    def test_surface_forgery_rejected_atomically(self):
        before = self.graph.snapshot()
        candidate = self.proposer.propose("猫が魚を食べる。")
        r = accept_candidate(self.graph, "犬が魚を食べる。", candidate)
        self.assertEqual(r["reason"], "SURFACE_MISMATCH")
        self.assertEqual(self.graph.snapshot(), before)

    def test_invented_reference(self):
        before = self.graph.snapshot()
        candidate = self.proposer.propose("猫が魚を食べる。")
        candidate["tree"]["bindings"]["subject"] = {"sd_ref": "INVENTED"}
        r = accept_candidate(self.graph, "猫が魚を食べる。", candidate)
        self.assertEqual(r["reason"], "UNKNOWN_SD_REF")
        self.assertEqual(self.graph.snapshot(), before)

    def test_predicate_in_subject_slot(self):
        candidate = self.proposer.propose("猫が魚を食べる。")
        candidate["tree"]["bindings"]["subject"] = {"sd_ref": "WATCH"}
        r = accept_candidate(self.graph, "猫が魚を食べる。", candidate)
        self.assertEqual(r["reason"], "ROLE_TYPE_MISMATCH")

    def test_extra_fields_rejected(self):
        candidate = self.proposer.propose("猫が魚を食べる。")
        candidate["tree"]["extra"] = "untrusted"
        r = accept_candidate(self.graph, "猫が魚を食べる。", candidate)
        self.assertEqual(r["reason"], "INVALID_NODE_SCHEMA")

    def test_malformed_json_rejected(self):
        r = accept_candidate(self.graph, "猫が魚を食べる。", "{broken")
        self.assertEqual(r["reason"], "INVALID_JSON")

    def test_ambiguous_atom_rejected(self):
        self.graph.add_atom("CAT-ALIAS", "猫")
        r = run("猫が魚を食べる。", self.proposer, self.graph)
        self.assertEqual(r["status"], "REJECTED")

    def test_no_claim_of_factual_truth(self):
        r = run("犬が魚を食べる。", self.proposer, self.graph)
        self.assertEqual(r["status"], "ACCEPTED")
        self.assertFalse(self.graph.nodes[r["sd_ref"]]["verified"])

    def test_depth_limit(self):
        candidate = self.proposer.propose("猫が魚を食べる。")["tree"]
        for i in range(10):
            candidate = {"template_ref": "T-AND", "bindings": {
                "first": candidate, "second": {"sd_ref": "CAT"}}}
        r = accept_candidate(self.graph, "猫が魚を食べる。", {"tree": candidate})
        self.assertEqual(r["reason"], "DEPTH_LIMIT")

    def test_unsupported_conjunction_rejected(self):
        r = run("猫が魚を食べるので、犬が鳥を見る。", self.proposer, self.graph)
        self.assertEqual(r["status"], "REJECTED")


if __name__ == "__main__":
    unittest.main()
