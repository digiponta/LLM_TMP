"""Dialogue intent SD and data regression tests."""
import unittest
from dialogue_intent_sd import DialogueIntentSD, INTENTS, parse_intent_code
from dialogue_intent_dataset import build_intent_examples, audit_split
from sd_candidate_pipeline import CandidateRejected


class DialogueIntentTests(unittest.TestCase):
    def setUp(self):
        self.engine = DialogueIntentSD()

    def test_labels(self):
        self.assertEqual(len(INTENTS), 8)

    def test_interning(self):
        a = self.engine.accept("こんにちは。", "I(GREETING)")
        b = self.engine.accept("おはよう。", "I(GREETING)")
        self.assertEqual(a["sd_ref"], b["sd_ref"])
        self.assertEqual(self.engine.inspect(a["sd_ref"]), "GREETING")

    def test_different_intents(self):
        a = self.engine.accept("こんにちは。", "I(GREETING)")
        b = self.engine.accept("ありがとう。", "I(THANKS)")
        self.assertNotEqual(a["sd_ref"], b["sd_ref"])

    def test_unverified(self):
        a = self.engine.accept("今日は疲れました。", "I(FEELING)")
        self.assertFalse(a["verified_fact"])
        self.assertFalse(a["semantic_alignment_verified"])

    def test_reject_bad_code_without_mutation(self):
        before = self.engine.graph.snapshot()
        r = self.engine.accept("こんにちは。", "I(UNKNOWN)")
        self.assertEqual(r["status"], "REJECTED")
        self.assertEqual(before, self.engine.graph.snapshot())

    def test_reject_malformed(self):
        r = self.engine.accept("こんにちは。", "I(QUESTION")
        self.assertEqual(r["status"], "REJECTED")

    def test_empty_reject(self):
        self.assertEqual(self.engine.accept("", "I(GREETING)")["status"],
                         "REJECTED")

    def test_parsing(self):
        self.assertEqual(parse_intent_code("I(REQUEST)"), "REQUEST")
        with self.assertRaises(CandidateRejected):
            parse_intent_code("I(WRONG)")

    def test_dev_family_split(self):
        train, dev = build_intent_examples()
        audit = audit_split(train, dev)
        self.assertEqual(audit["train_count"], 64)
        self.assertEqual(audit["dev_count"], 32)

    def test_all_labels_present(self):
        a, b = build_intent_examples()
        self.assertEqual({e["intent"] for e in a}, set(INTENTS))
        self.assertEqual({e["intent"] for e in b}, set(INTENTS))

    def test_deterministic(self):
        self.assertEqual(build_intent_examples(), build_intent_examples())

    def test_graph_validation(self):
        self.engine.accept("おはよう。", "I(GREETING)")
        self.assertTrue(self.engine.graph.validate())

    def test_wrong_label_can_be_structurally_valid(self):
        r = self.engine.accept("こんにちは。", "I(QUESTION)")
        self.assertEqual(r["status"], "ACCEPTED")
        self.assertFalse(r["semantic_alignment_verified"])


if __name__ == "__main__":
    unittest.main()
