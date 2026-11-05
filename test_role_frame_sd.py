"""Tests for two-head LLM targets, safe assembly, and dataset comparability."""
import unittest

from compact_frame_sd import accept_frame_code
from semantic_features_sd import SemanticFeatureEngine
from train_frame_sd import build_frame_examples
from train_role_frame_sd import (
    feature_target, make_pairs, parse_heads, role_target, target,
)


class RoleFrameTests(unittest.TestCase):
    def setUp(self):
        self.train, self.dev = build_frame_examples(seed=42)

    def test_role_target(self):
        example = {"core": ["DOG", "FISH", "WATCH"]}
        self.assertEqual(role_target(example), "R(DOG,FISH,WATCH)")

    def test_feature_target(self):
        example = {"polarity": "NEG", "tense": "PAST",
                   "topic": "UNMARKED"}
        self.assertEqual(feature_target(example), "Q(NEG,PAST,UNMARKED)")

    def test_assembly(self):
        result = parse_heads("R(DOG,FISH,WATCH)", "Q(NEG,PAST,UNMARKED)")
        self.assertEqual(result, "F(S(DOG,FISH,WATCH),NEG,PAST,UNMARKED)")

    def test_bad_role_rejected(self):
        with self.assertRaises(ValueError):
            parse_heads("R(DOG,FISH,WATCH);exit()", "Q(NEG,PAST,UNMARKED)")

    def test_bad_feature_rejected(self):
        with self.assertRaises(ValueError):
            parse_heads("R(DOG,FISH,WATCH)", "Q(POS,FUTURE,UNMARKED)")

    def test_incomplete_rejected(self):
        with self.assertRaises(ValueError):
            parse_heads("R(DOG,FISH)", "Q(POS,PAST,UNMARKED)")

    def test_targets_derived_only_from_label_not_text(self):
        item = self.train[0]
        self.assertEqual(target(item, "roles"), role_target(item))
        self.assertEqual(target(item, "features"), feature_target(item))
        self.assertRaises(ValueError, target, item, "unknown")

    def test_split_reuses_original_dev(self):
        self.assertEqual(len(self.train), 480)
        self.assertEqual(len(self.dev), 120)
        self.assertFalse({tuple(x["core"]) for x in self.train} &
                         {tuple(x["core"]) for x in self.dev})

    def test_pair_counts(self):
        for head in ("roles", "features"):
            self.assertEqual(len(make_pairs(self.train, head)), 480)
            self.assertEqual(len(make_pairs(self.dev, head)), 120)

    def test_surface_validation_blocks_swaps(self):
        e = SemanticFeatureEngine()
        before = e.graph.snapshot()
        result = accept_frame_code(e, "魚が犬に見られる。",
                                   parse_heads("R(FISH,DOG,WATCH)",
                                               "Q(POS,NONPAST,UNMARKED)"))
        self.assertEqual(result["status"], "REJECTED")
        self.assertEqual(before, e.graph.snapshot())

    def test_two_heads_correct_passive(self):
        e = SemanticFeatureEngine()
        code = parse_heads("R(DOG,FISH,WATCH)", "Q(POS,NONPAST,UNMARKED)")
        result = accept_frame_code(e, "魚が犬に見られる。", code)
        self.assertEqual(result["status"], "ACCEPTED")

    def test_dataset_pairs_have_expected_codes(self):
        for e in self.dev[:20]:
            self.assertEqual(
                parse_heads(role_target(e), feature_target(e)), e["code"]
            )


if __name__ == "__main__":
    unittest.main()
