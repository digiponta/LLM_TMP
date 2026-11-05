"""Core+Frame compact-code tests; no model weights required."""
import unittest

from compact_frame_sd import (
    accept_frame_code, encode_frame_code, parse_frame_code, realize_frame,
)
from sd_candidate_pipeline import CandidateRejected
from semantic_features_sd import SemanticFeatureEngine
from train_frame_sd import build_frame_examples


class CompactFrameSDTests(unittest.TestCase):
    def setUp(self):
        self.engine = SemanticFeatureEngine()

    def test_short_code_roundtrip(self):
        code = "F(S(CAT,FISH,EAT),NEG,PAST,UNMARKED)"
        self.assertEqual(encode_frame_code(parse_frame_code(code)), code)

    def test_negative_past(self):
        r = accept_frame_code(self.engine, "猫が魚を食べなかった。",
                              "F(S(CAT,FISH,EAT),NEG,PAST,UNMARKED)")
        self.assertEqual(r["status"], "ACCEPTED")
        self.assertFalse(r["verified_fact"])

    def test_active_passive_share_frame(self):
        code = "F(S(CAT,FISH,EAT),POS,NONPAST,UNMARKED)"
        a = accept_frame_code(self.engine, "猫が魚を食べる。", code)
        b = accept_frame_code(self.engine, "魚が猫に食べられる。", code)
        self.assertEqual(a["frame_ref"], b["frame_ref"])

    def test_topic_does_not_merge(self):
        a = accept_frame_code(self.engine, "猫が魚を食べる。",
                              "F(S(CAT,FISH,EAT),POS,NONPAST,UNMARKED)")
        b = accept_frame_code(self.engine, "猫は魚を食べる。",
                              "F(S(CAT,FISH,EAT),POS,NONPAST,SUBJECT_TOPIC)")
        self.assertEqual(a["core_ref"], b["core_ref"])
        self.assertNotEqual(a["frame_ref"], b["frame_ref"])

    def test_wrong_polarity_rejected_without_mutation(self):
        before = self.engine.graph.snapshot()
        r = accept_frame_code(self.engine, "猫が魚を食べない。",
                              "F(S(CAT,FISH,EAT),POS,NONPAST,UNMARKED)")
        self.assertEqual(r["status"], "REJECTED")
        self.assertEqual(before, self.engine.graph.snapshot())

    def test_role_swap_rejected(self):
        r = accept_frame_code(self.engine, "猫が魚を食べる。",
                              "F(S(FISH,CAT,EAT),POS,NONPAST,UNMARKED)")
        self.assertEqual(r["reason"], "SURFACE_OR_ROLE_MISMATCH")

    def test_invalid_id_rejected(self):
        r = accept_frame_code(self.engine, "猫が魚を食べる。",
                              "F(S(CAT,INVENTED,EAT),POS,NONPAST,UNMARKED)")
        self.assertEqual(r["status"], "REJECTED")

    def test_invalid_code_rejected(self):
        with self.assertRaises(CandidateRejected):
            parse_frame_code("F(S(CAT,FISH,EAT),POS,PAST,UNMARKED);rm")

    def test_passive_topic_rejected(self):
        r = accept_frame_code(self.engine, "魚が猫に食べられる。",
                              "F(S(CAT,FISH,EAT),POS,NONPAST,SUBJECT_TOPIC)")
        self.assertEqual(r["status"], "REJECTED")

    def test_generated_split_core_disjoint(self):
        train, dev = build_frame_examples()
        self.assertFalse({tuple(x["core"]) for x in train} &
                         {tuple(x["core"]) for x in dev})
        self.assertFalse({x["text"] for x in train} &
                         {x["text"] for x in dev})
        self.assertEqual(len(train), 40 * 12)
        self.assertEqual(len(dev), 10 * 12)

    def test_generated_reference_consistency(self):
        train, dev = build_frame_examples()
        graph = self.engine.graph
        for item in train[:20] + dev[:20]:
            parsed = parse_frame_code(item["code"])
            self.assertEqual(realize_frame(parsed, graph, item["voice"]),
                             item["text"])

    def test_frame_graph_integrity(self):
        accept_frame_code(self.engine, "犬が鳥を見なかった。",
                          "F(S(DOG,BIRD,WATCH),NEG,PAST,UNMARKED)")
        self.assertTrue(self.engine.graph.validate())


if __name__ == "__main__":
    unittest.main()
