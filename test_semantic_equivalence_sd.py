"""Tests for semantic identity shared between active and passive clauses."""
import unittest

from semantic_equivalence_sd import SemanticNormalizer, compare_meaning
from sd_candidate_pipeline import CandidateRejected
from recursive_sd import build_sample_graph


class SemanticEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.graph = build_sample_graph()
        self.normalizer = SemanticNormalizer(self.graph)

    def test_active_passive_identity(self):
        a = self.normalizer.analyze("猫が魚を食べる。")
        b = self.normalizer.analyze("魚が猫に食べられる。")
        self.assertEqual(a["sd_ref"], b["sd_ref"])
        self.assertEqual(a["code"], "S(CAT,FISH,EAT)")
        self.assertFalse(b["surface_exact"])
        self.assertFalse(b["verified_fact"])

    def test_watch_active_passive(self):
        r = compare_meaning("犬が鳥を見る。", "鳥が犬に見られる。")
        self.assertEqual(r["status"], "COMPARED")
        self.assertTrue(r["same_meaning"])

    def test_nonidentical_roles(self):
        r = compare_meaning("犬が鳥を見る。", "犬が鳥に見られる。")
        self.assertEqual(r["status"], "COMPARED")
        self.assertFalse(r["same_meaning"])

    def test_nonidentical_predicate(self):
        r = compare_meaning("猫が魚を食べる。", "魚が猫に見られる。")
        self.assertEqual(r["status"], "COMPARED")
        self.assertFalse(r["same_meaning"])

    def test_compound_mixed_voice(self):
        a = "猫が魚を食べると、犬が鳥を見る。"
        b = "魚が猫に食べられると、鳥が犬に見られる。"
        r = compare_meaning(a, b)
        self.assertTrue(r["same_meaning"])
        self.assertEqual(r["left"]["code"], r["right"]["code"])

    def test_compound_different_connector(self):
        a = "猫が魚を食べると、犬が鳥を見る。"
        b = "魚が猫に食べられる、そして鳥が犬に見られる。"
        self.assertFalse(compare_meaning(a, b)["same_meaning"])

    def test_unknown_entity_no_mutation(self):
        before = self.graph.snapshot()
        with self.assertRaises(CandidateRejected):
            self.normalizer.analyze("宇宙が魚を食べる。")
        self.assertEqual(self.graph.snapshot(), before)

    def test_unknown_passive_no_mutation(self):
        before = self.graph.snapshot()
        with self.assertRaises(CandidateRejected):
            self.normalizer.analyze("魚が猫に踊られる。")
        self.assertEqual(self.graph.snapshot(), before)

    def test_negation_rejected(self):
        result = compare_meaning("猫が魚を食べる。", "猫が魚を食べない。")
        self.assertEqual(result["status"], "UNSUPPORTED")
        self.assertIsNone(result["same_meaning"])

    def test_question_rejected(self):
        result = compare_meaning("猫が魚を食べる。", "猫が魚を食べる？")
        self.assertEqual(result["status"], "UNSUPPORTED")

    def test_three_clauses_rejected(self):
        result = compare_meaning(
            "猫が魚を食べると、犬が鳥を見ると、猫がご飯を食べる。",
            "猫が魚を食べる。")
        self.assertEqual(result["status"], "UNSUPPORTED")

    def test_duplicate_lexical_ambiguity(self):
        self.graph.add_atom("CAT2", "猫")
        with self.assertRaises(CandidateRejected):
            self.normalizer.analyze("猫が魚を食べる。")

    def test_no_factual_promote(self):
        result = self.normalizer.analyze("ご飯が犬を食べる。")
        self.assertFalse(self.graph.nodes[result["sd_ref"]]["verified"])

    def test_multiple_parse_reuses_sd(self):
        a = self.normalizer.analyze("猫が魚を食べる。")
        before = len(self.graph.nodes)
        b = self.normalizer.analyze("魚が猫に食べられる。")
        self.assertEqual(a["sd_ref"], b["sd_ref"])
        self.assertEqual(len(self.graph.nodes), before)


if __name__ == "__main__":
    unittest.main()
