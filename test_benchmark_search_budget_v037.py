"""v0.3.7 paired search-budget sweep regression tests."""
import json
from pathlib import Path
import tempfile
import unittest

from benchmark_evidence_topology_v036 import build_graph
from benchmark_search_budget_v037 import make_cohort, run, save_new


class BudgetSweepTests(unittest.TestCase):
    def test_small_benchmark(self):
        result = run(nodes=100, relations_per_node=5, samples=12,
                     budgets=(2, 20, 100), max_hops=5)
        self.assertEqual(result["graph"]["relations"], 500)
        self.assertEqual(set(result["results"]), {"2", "20", "100"})

    def test_reachable_discovery_with_sufficient_budget(self):
        result = run(nodes=100, relations_per_node=5, samples=15,
                     budgets=(100,), max_hops=5)
        self.assertEqual(result["results"]["100"]["reachable_target"]["found"], 15)

    def test_known_reachable_is_not_false_not_found(self):
        result = run(nodes=100, relations_per_node=5, samples=20,
                     budgets=(1, 100), max_hops=5)
        counts = result["results"]["1"]["reachable_target"]["status_counts"]
        self.assertNotIn("NOT_FOUND", counts)
        self.assertEqual(sum(counts.values()), 20)

    def test_deterministic_cohort(self):
        g = build_graph("random", 100, 5, 36)
        a = make_cohort(g, 100, 10, 36)
        b = make_cohort(g, 100, 10, 36)
        self.assertEqual(a, b)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            run(nodes=100, samples=2, budgets=(100, 10))
        with self.assertRaises(ValueError):
            run(nodes=100, samples=2, budgets=(10, 10))
        with self.assertRaises(ValueError):
            run(nodes=100, samples=2, max_hops=3, target_steps=4)

    def test_existing_output_protected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "existing.json"
            path.write_text("untouched", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                save_new(path, {"status": "replacement"})
            self.assertEqual(path.read_text(encoding="utf-8"), "untouched")


if __name__ == "__main__":
    unittest.main()
