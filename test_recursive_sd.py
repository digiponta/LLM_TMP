"""Regression tests for recursive linked SD and template-as-SD."""
import unittest

from recursive_sd import (
    RecursiveSD, build_sample_graph, compose_pair, compose_sentence,
)


class RecursiveSDTests(unittest.TestCase):
    def setUp(self):
        self.g = build_sample_graph()

    def test_template_is_sd(self):
        self.assertEqual(self.g.nodes["T-SOV"]["kind"], "template")
        self.assertEqual(self.g.nodes["T-THEN"]["kind"], "template")

    def test_nested_composition(self):
        a = compose_sentence(self.g, "CAT", "FISH", "EAT")
        b = compose_sentence(self.g, "DOG", "BIRD", "WATCH")
        c = compose_pair(self.g, a, b)
        self.assertEqual(self.g.render(c), "猫が魚を食べると、犬が鳥を見る。")
        self.assertEqual(self.g.decompose(c)["bindings"]["first"], a)
        self.assertEqual(self.g.decompose(c)["template_ref"], "T-THEN")

    def test_reference_reuse_and_roundtrip(self):
        a = compose_sentence(self.g, "CAT", "FISH", "EAT")
        b = compose_sentence(self.g, "CAT", "FISH", "EAT")
        self.assertEqual(a, b)
        desc = self.g.decompose(a)
        self.assertEqual(a, self.g.recompose(desc))
        self.assertEqual(self.g.decompose(a), desc)

    def test_unknown_is_not_fact(self):
        a = compose_sentence(self.g, "DOG", "FISH", "EAT")
        self.assertFalse(self.g.nodes[a]["verified"])

    def test_missing_reference_rejected(self):
        with self.assertRaises(ValueError):
            compose_sentence(self.g, "UNKNOWN", "FISH", "EAT")

    def test_template_not_content(self):
        with self.assertRaises(ValueError):
            compose_sentence(self.g, "T-THEN", "FISH", "EAT")

    def test_bad_roles(self):
        with self.assertRaises(ValueError):
            self.g.compose("T-SOV", {"subject": "CAT"})

    def test_cycle_guard(self):
        a = compose_sentence(self.g, "CAT", "FISH", "EAT")
        self.g.nodes[a]["refs"][0]["sd_ref"] = a
        with self.assertRaises(ValueError):
            self.g.render(a)

    def test_full_validation(self):
        a = compose_sentence(self.g, "CAT", "FISH", "EAT")
        b = compose_sentence(self.g, "DOG", "BIRD", "WATCH")
        compose_pair(self.g, a, b)
        self.assertTrue(self.g.validate())

    def test_nested_canonical_reuse(self):
        a = compose_sentence(self.g, "CAT", "RICE", "EAT")
        b = compose_sentence(self.g, "DOG", "BIRD", "WATCH")
        first = compose_pair(self.g, a, b)
        second = compose_pair(self.g, a, b)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
