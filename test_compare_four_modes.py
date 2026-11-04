"""Unit tests for linked-SD storage and deterministic benchmark adapters."""
import unittest

from compare_four_modes import CASES, SDMemory, evaluate


class SDComparisonTests(unittest.TestCase):
    def setUp(self):
        self.sd = SDMemory()

    def test_shared_sd_references(self):
        self.assertEqual(self.sd.nodes["FACT-1"]["template_ref"], "JA-SOV")
        self.assertEqual(self.sd.nodes["FACT-1"]["refs"][0]["sd_ref"], "CAT")
        self.assertEqual(self.sd.nodes["FACT-3"]["refs"][0]["sd_ref"], "CAT")

    def test_existing_fact_reuses_node(self):
        original = len(self.sd.nodes)
        self.assertEqual(self.sd.add_fact("DUPLICATE", ("CAT", "FISH", "EAT")), "FACT-1")
        self.assertEqual(len(self.sd.nodes), original)

    def test_unseen_composition_does_not_add_fact(self):
        original = len(self.sd.index)
        question = "主語=犬、目的語=魚、動詞=食べる。文を作って。"
        self.assertEqual(self.sd.compose_requested(question), "犬が魚を食べる。")
        self.assertEqual(len(self.sd.index), original)
        self.assertNotIn(("DOG", "FISH", "EAT"), self.sd.index)

    def test_no_unjustified_fact_inference(self):
        self.assertIsNone(self.sd.resolve_fact("犬が食べるものを教えて。"))
        self.assertIsNone(self.sd.resolve_fact("主語=犬、目的語=魚、動詞=食べる。文を作って。"))

    def test_known_fact(self):
        self.assertEqual(self.sd.resolve_fact("鳥を見るのは誰ですか？"), "犬")

    def test_development_modes(self):
        result = {m: evaluate(m, CASES, self.sd) for m in "BCD"}
        self.assertEqual(result["D"]["accuracy"], 1.0)
        self.assertEqual(result["C"]["unknown_reject"], 1.0)
        self.assertGreater(result["D"]["known_accuracy"], result["C"]["known_accuracy"])
        self.assertGreaterEqual(result["C"]["known_accuracy"], result["B"]["known_accuracy"])


if __name__ == "__main__":
    unittest.main()
