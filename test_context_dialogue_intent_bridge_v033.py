"""Regression tests for the v0.3.3 context/intent bridge."""
import unittest
from context_dialogue_intent_bridge_v033 import ContextDialogueIntentBridge, run_dialogue


class ContextDialogueIntentBridgeTests(unittest.TestCase):
    def test_context_resolution_and_sd_acceptance(self):
        rows = run_dialogue(["スファレライトが好きです", "きれいですね", "成分は？"])
        last = rows[-1]
        self.assertEqual(last["status"], "RESOLVED")
        self.assertEqual(last["intent_sd"]["subject"], "スファレライト")
        self.assertEqual(last["intent_sd"]["source_turn"], 0)
        self.assertTrue(last["dialogue_intent_sd_ref"])

    def test_shared_intent_sd_ref(self):
        rows = run_dialogue(["スファレライトの成分は？", "化学式は？"])
        self.assertEqual(rows[0]["dialogue_intent_sd_ref"], rows[1]["dialogue_intent_sd_ref"])

    def test_no_context_does_not_create_intent(self):
        bridge = ContextDialogueIntentBridge()
        before = bridge.engine.graph.snapshot()
        result = bridge.process("成分は？")
        self.assertEqual(result["status"], "ABSTAIN")
        self.assertEqual(bridge.engine.graph.snapshot(), before)

    def test_multiple_topics_rejected(self):
        result = run_dialogue(["スファレライトとダイヤモンドの硬度は？"])[0]
        self.assertEqual(result["reason"], "MULTIPLE_TOPICS")

    def test_factual_and_semantic_checks_remain_false(self):
        result = run_dialogue(["スファレライトの成分は？"])[0]
        self.assertIs(result["verified_fact"], False)
        self.assertIs(result["semantic_alignment_verified"], False)
        self.assertIs(result["human_approved"], False)

    def test_unrelated_text_does_not_accept_intent(self):
        result = run_dialogue(["スファレライトが好きです"])[0]
        self.assertEqual(result["status"], "NO_TARGET_QUESTION")
        self.assertNotIn("dialogue_intent_sd_ref", result)


if __name__ == "__main__":
    unittest.main()
