"""Tests for conservative natural-language -> SD -> language roundtrips."""
import unittest

from japanese_sd_parser import JapaneseSDParser, UnsupportedSentence
from recursive_sd import build_sample_graph, compose_sentence


class JapaneseSDParserTests(unittest.TestCase):
    def setUp(self):
        self.graph = build_sample_graph()
        self.parser = JapaneseSDParser(self.graph)

    def test_simple_roundtrip(self):
        result = self.parser.analyze("猫が魚を食べる。")
        self.assertEqual(result["rendered"], result["input"])
        self.assertTrue(result["roundtrip_equal"])
        self.assertEqual(result["tree"]["template_ref"], "T-SOV")

    def test_then_roundtrip(self):
        result = self.parser.analyze("猫が魚を食べると、犬が鳥を見る。")
        self.assertTrue(result["roundtrip_equal"])
        self.assertEqual(result["tree"]["template_ref"], "T-THEN")
        self.assertEqual(result["tree"]["bindings"]["first"]["template_ref"], "T-SOV")

    def test_and_roundtrip(self):
        result = self.parser.analyze("猫が魚を食べる、そして犬が鳥を見る。")
        self.assertTrue(result["roundtrip_equal"])
        self.assertEqual(result["tree"]["template_ref"], "T-AND")

    def test_reuses_existing_sd(self):
        first = compose_sentence(self.graph, "CAT", "FISH", "EAT")
        before = len(self.graph.nodes)
        parsed = self.parser.parse("猫が魚を食べる。")
        self.assertEqual(parsed, first)
        self.assertEqual(len(self.graph.nodes), before)

    def test_idempotent_parse(self):
        text = "猫が魚を食べると、犬が鳥を見る。"
        a = self.parser.parse(text)
        before = len(self.graph.nodes)
        b = self.parser.parse(text)
        self.assertEqual(a, b)
        self.assertEqual(before, len(self.graph.nodes))

    def test_unknown_does_not_create_nodes(self):
        before = self.graph.snapshot()
        with self.assertRaises(UnsupportedSentence):
            self.parser.parse("猫が魚を踊る。")
        self.assertEqual(before, self.graph.snapshot())

    def test_malformed_second_clause_no_mutation(self):
        before = self.graph.snapshot()
        with self.assertRaises(UnsupportedSentence):
            self.parser.parse("猫が魚を食べると、犬が空を飛ぶ。")
        self.assertEqual(before, self.graph.snapshot())

    def test_ambiguous_lexeme_rejected(self):
        self.graph.add_atom("CAT-ALT", "猫")
        with self.assertRaises(UnsupportedSentence):
            self.parser.parse("猫が魚を食べる。")

    def test_unrecognized_particles_rejected(self):
        with self.assertRaises(UnsupportedSentence):
            self.parser.parse("猫は魚を食べる。")

    def test_three_clauses_rejected(self):
        with self.assertRaises(UnsupportedSentence):
            self.parser.parse("猫が魚を食べると、犬が鳥を見ると、猫がご飯を食べる。")

    def test_no_verified_factual_assertion(self):
        result = self.parser.analyze("犬が魚を食べる。")
        self.assertFalse(result["verified_fact"])
        self.assertFalse(self.graph.nodes[result["sd_ref"]]["verified"])

    def test_graph_validation(self):
        self.parser.parse("猫が魚を食べると、犬が鳥を見る。")
        self.assertTrue(self.graph.validate())


if __name__ == "__main__":
    unittest.main()
