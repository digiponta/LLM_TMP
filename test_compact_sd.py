"""Compact SD codec and training-dataset regression checks."""
import unittest

from compact_sd import parse_compact, to_compact, accept_compact
from recursive_sd import build_sample_graph
from sd_candidate_pipeline import CandidateRejected
from train_compact_sd import build_examples


class CompactSDTests(unittest.TestCase):
    def setUp(self):
        self.graph = build_sample_graph()

    def test_sov_codec(self):
        code = "S(CAT,FISH,EAT)"
        self.assertEqual(to_compact(parse_compact(code)), code)

    def test_recursive_codec(self):
        code = "T(S(CAT,FISH,EAT),S(DOG,BIRD,WATCH))"
        self.assertEqual(to_compact(parse_compact(code)), code)

    def test_and_codec(self):
        code = "A(S(CAT,RICE,EAT),S(DOG,FISH,WATCH))"
        self.assertEqual(to_compact(parse_compact(code)), code)

    def test_accept_against_original(self):
        result = accept_compact(
            self.graph, "猫が魚を食べると、犬が鳥を見る。",
            "T(S(CAT,FISH,EAT),S(DOG,BIRD,WATCH))")
        self.assertEqual(result["status"], "ACCEPTED")
        self.assertFalse(result["verified_fact"])

    def test_wrong_reference_rejected_without_mutation(self):
        before = self.graph.snapshot()
        result = accept_compact(
            self.graph, "猫が魚を食べる。",
            "S(DOG,FISH,EAT)")
        self.assertEqual(result["reason"], "SURFACE_MISMATCH")
        self.assertEqual(self.graph.snapshot(), before)

    def test_unknown_id_rejected(self):
        result = accept_compact(
            self.graph, "猫が魚を食べる.",
            "S(UNKNOWN,FISH,EAT)")
        self.assertEqual(result["status"], "REJECTED")

    def test_partial_code_rejected(self):
        with self.assertRaises(CandidateRejected):
            parse_compact("T(S(CAT,FISH,EAT),")

    def test_injected_text_rejected(self):
        with self.assertRaises(CandidateRejected):
            parse_compact("S(CAT,FISH,EAT);__import__")

    def test_extra_suffix_rejected(self):
        with self.assertRaises(CandidateRejected):
            parse_compact("S(CAT,FISH,EAT)S(CAT,FISH,EAT)")

    def test_dataset_unique_surface_split(self):
        train, dev = build_examples(max_pairs=100)
        self.assertTrue(train)
        self.assertTrue(dev)
        self.assertFalse({x["text"] for x in train} & {x["text"] for x in dev})

    def test_distinct_from_json(self):
        code = "S(CAT,FISH,EAT)"
        self.assertLess(len(code), len(str(parse_compact(code))))


if __name__ == "__main__":
    unittest.main()
