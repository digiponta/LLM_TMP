"""Tests for paired Full-Length Context Shuffle comparison; no GPU."""
import unittest

from compare_realpersonachat_fullshuffle import bootstrap_ci, compare


def record(dialogue, baseline, status="SCORED"):
    return {
        "dialogue_id": str(dialogue),
        "turn_id": 3,
        "status": status,
        "donor_dialogue_id": "999",
        "context_turn_count": 2,
        "reference": "こんにちは",
        "original_prompt": "会話:\nspeaker_0:今日は晴れです\nspeaker_1:",
        "shuffled_prompt": "会話:\nspeaker_0:夕食はカレーです\nspeaker_1:",
        "original_minus_shuffled": baseline if status == "SCORED" else None,
    }


def report(rows):
    return {"status": "CORPUS_DEV_FULL_CONTEXT_SHUFFLE_NOT_INDEPENDENT",
            "rows": rows}


class PairedFullShuffleTests(unittest.TestCase):
    def test_common_subset(self):
        a = report([record(1, .1), record(2, -.1),
                    record(3, None, status="ABSTAIN")])
        b = report([record(3, .2), record(2, .1), record(1, .3)])
        out = compare(a, b)
        self.assertEqual(out["common_selected"], 3)
        self.assertEqual(out["jointly_scored"], 2)
        self.assertEqual(out["candidate_only_scored"], 1)
        self.assertEqual(out["baseline_only_scored"], 0)
        self.assertAlmostEqual(out["candidate_minus_baseline_mean_delta"], .2)
        self.assertEqual(out["baseline_negative_to_candidate_positive"], 1)
        self.assertEqual(out["baseline_positive_to_candidate_negative"], 0)

    def test_mismatched_cases(self):
        with self.assertRaisesRegex(ValueError, "MISMATCHED_EVALUATION_CASES"):
            compare(report([record(1, .1)]), report([record(2, .2)]))

    def test_mismatched_donor(self):
        b = record(1, .2)
        b["donor_dialogue_id"] = "10"
        with self.assertRaisesRegex(ValueError, "MISMATCHED_PAIR_CONTENT"):
            compare(report([record(1, .1)]), report([b]))

    def test_mismatched_prompt(self):
        b = record(1, .2)
        b["shuffled_prompt"] = "different"
        with self.assertRaisesRegex(ValueError, "MISMATCHED_PAIR_CONTENT"):
            compare(report([record(1, .1)]), report([b]))

    def test_duplicate_case(self):
        with self.assertRaisesRegex(ValueError, "DUPLICATE_CASES"):
            compare(report([record(1, .1), record(1, .2)]),
                    report([record(1, .3)]))

    def test_all_abstained(self):
        with self.assertRaisesRegex(ValueError, "NO_JOINTLY_SCORED_CASES"):
            compare(report([record(1, None, "ABSTAIN")]),
                    report([record(1, None, "ABSTAIN")]))

    def test_bootstrap_repeatable(self):
        a = bootstrap_ci([.1, .2, -.1], seed=42)
        b = bootstrap_ci([.1, .2, -.1], seed=42)
        self.assertEqual(a, b)
        self.assertLessEqual(a[0], a[1])

    def test_bootstrap_constant(self):
        lower, upper = bootstrap_ci([.25] * 5)
        self.assertAlmostEqual(lower, .25)
        self.assertAlmostEqual(upper, .25)


if __name__ == "__main__":
    unittest.main()
