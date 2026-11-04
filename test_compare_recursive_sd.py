"""Development benchmark guard tests."""
import unittest
from types import SimpleNamespace
from recursive_sd import build_sample_graph
from compare_recursive_sd import INPUTS, predict_d, prompt, reference, run


class RecursiveBenchmarkTests(unittest.TestCase):
    def test_all_nested_development_cases(self):
        graph = build_sample_graph()
        for case in INPUTS:
            with self.subTest(case=case):
                self.assertEqual(predict_d(prompt(case), graph), reference(case))

    def test_invalid_verb_rejected(self):
        q = prompt(("猫", "魚", "踊る", "犬", "鳥", "見る", "T-AND"))
        self.assertIsNone(predict_d(q, build_sample_graph()))

    def test_question_not_fact(self):
        self.assertIsNone(
            predict_d("犬が魚を食べるのは事実ですか？", build_sample_graph()))

    def test_no_target_leak_in_prediction(self):
        graph = build_sample_graph()
        q = prompt(INPUTS[0])
        before = graph.snapshot()
        self.assertEqual(predict_d(q, graph), reference(INPUTS[0]))
        after = graph.snapshot()
        self.assertGreater(len(after), len(before))
        self.assertTrue(all(not node.get("verified", False) for node in after.values()))


if __name__ == "__main__":
    unittest.main()
