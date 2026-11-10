import unittest
from context_intent_resolver_v033 import resolve_dialogue


class ContextIntentTests(unittest.TestCase):
    def test_followup(self):
        r = resolve_dialogue(["スファレライトが好きです", "きれいですね", "成分は？"])
        self.assertEqual(r[-1]["intent_sd"]["subject"], "スファレライト")
        self.assertEqual(r[-1]["intent_sd"]["reference"], "CONTEXT")

    def test_explicit_topic_overrides(self):
        r = resolve_dialogue(["スファレライトが好き", "ダイヤモンドの硬度は？"])
        self.assertEqual(r[-1]["intent_sd"]["subject"], "ダイヤモンド")

    def test_no_context_abstains(self):
        self.assertEqual(resolve_dialogue(["成分は？"])[0]["reason"], "NO_TOPIC_CONTEXT")

    def test_multiple_topics_abstains(self):
        self.assertEqual(resolve_dialogue(["スファレライトとダイヤモンドの硬度は？"])[0]["reason"], "MULTIPLE_TOPICS")

    def test_no_false_question(self):
        self.assertEqual(resolve_dialogue(["スファレライトの硬度が高いと思います"])[0]["status"], "NO_TARGET_QUESTION")

    def test_reference_turn(self):
        r = resolve_dialogue(["スファレライトが好き", "はい", "化学式は？"])
        self.assertEqual(r[-1]["intent_sd"]["source_turn"], 0)


if __name__ == "__main__":
    unittest.main()
