"""Regression tests for layered semantic features and safe identity."""
import unittest

from recursive_sd import build_sample_graph
from semantic_features_sd import SemanticFeatureEngine, compare_features
from sd_candidate_pipeline import CandidateRejected


class SemanticFeatureTests(unittest.TestCase):
    def setUp(self):
        self.engine = SemanticFeatureEngine()

    def test_active_passive_same_frame(self):
        r = compare_features("猫が魚を食べる。", "魚が猫に食べられる。")
        self.assertEqual(r["status"], "COMPARED")
        self.assertTrue(r["same_core"])
        self.assertTrue(r["same_frame"])

    def test_negation_same_core_different_frame(self):
        r = compare_features("猫が魚を食べる。", "猫が魚を食べない。")
        self.assertTrue(r["same_core"])
        self.assertFalse(r["same_frame"])
        self.assertEqual(r["differences"], ["polarity"])

    def test_past_same_core_different_frame(self):
        r = compare_features("猫が魚を食べる。", "猫が魚を食べた。")
        self.assertTrue(r["same_core"])
        self.assertFalse(r["same_frame"])
        self.assertEqual(r["differences"], ["tense"])

    def test_topic_same_core_different_frame(self):
        r = compare_features("猫が魚を食べる。", "猫は魚を食べる。")
        self.assertTrue(r["same_core"])
        self.assertFalse(r["same_frame"])
        self.assertEqual(r["differences"], ["topic"])

    def test_negative_passive_same_frame(self):
        r = compare_features("猫が魚を食べない。", "魚が猫に食べられない。")
        self.assertTrue(r["same_frame"])

    def test_negative_past_passive_same_frame(self):
        r = compare_features("犬が鳥を見なかった。", "鳥が犬に見られなかった。")
        self.assertTrue(r["same_frame"])

    def test_different_agent_not_same_core(self):
        r = compare_features("猫が魚を食べる。", "犬が魚を食べる。")
        self.assertFalse(r["same_core"])
        self.assertFalse(r["same_frame"])

    def test_different_predicate_not_same_core(self):
        r = compare_features("猫が魚を食べる。", "猫が魚を見る。")
        self.assertFalse(r["same_core"])

    def test_reuse_core_without_frame_collision(self):
        a = self.engine.analyze("猫が魚を食べる。")
        b = self.engine.analyze("猫が魚を食べない。")
        c = self.engine.analyze("猫が魚を食べた。")
        self.assertEqual(a["core_ref"], b["core_ref"])
        self.assertEqual(b["core_ref"], c["core_ref"])
        self.assertEqual(len({a["frame_ref"], b["frame_ref"], c["frame_ref"]}), 3)
        self.assertTrue(self.engine.graph.validate())

    def test_idempotence(self):
        a = self.engine.analyze("猫は魚を食べない。")
        before = len(self.engine.graph.nodes)
        b = self.engine.analyze("猫は魚を食べない。")
        self.assertEqual(a["frame_ref"], b["frame_ref"])
        self.assertEqual(before, len(self.engine.graph.nodes))

    def test_unverified(self):
        r = self.engine.analyze("ご飯が猫を食べた。")
        self.assertFalse(r["verified_fact"])
        self.assertFalse(self.engine.graph.nodes[r["frame_ref"]]["verified"])

    def test_unsupported_no_mutation(self):
        before = self.engine.graph.snapshot()
        with self.assertRaises(CandidateRejected):
            self.engine.analyze("猫が魚を踊った。")
        self.assertEqual(before, self.engine.graph.snapshot())

    def test_question_rejected(self):
        r = compare_features("猫が魚を食べる。", "猫が魚を食べる？")
        self.assertEqual(r["status"], "UNSUPPORTED")

    def test_unknown_concept_rejected(self):
        r = compare_features("猫が魚を食べる。", "宇宙が魚を食べる。")
        self.assertEqual(r["status"], "UNSUPPORTED")

    def test_ambiguous_lexicon_rejected(self):
        self.engine.graph.add_atom("CAT2", "猫")
        with self.assertRaises(CandidateRejected):
            self.engine.analyze("猫が魚を食べる。")

    def test_passive_not_conflated_with_role_reversal(self):
        r = compare_features("猫が魚を食べる。", "猫が魚に食べられる。")
        self.assertFalse(r["same_core"])

    def test_past_and_negation_independent(self):
        r = compare_features("猫が魚を食べない。", "猫が魚を食べなかった。")
        self.assertEqual(r["differences"], ["tense"])

    def test_clone_counter_python_future_safe(self):
        self.engine.analyze("猫が魚を食べる。")
        copy = self.engine.graph.clone()
        original = self.engine.analyze("猫が魚を食べない。")
        duplicate = SemanticFeatureEngine(copy).analyze("猫が魚を食べない。")
        self.assertEqual(original["frame_ref"], duplicate["frame_ref"])


if __name__ == "__main__":
    unittest.main()
