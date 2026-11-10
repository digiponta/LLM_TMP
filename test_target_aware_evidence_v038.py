import tempfile
from pathlib import Path
import unittest
from target_aware_evidence_search_v038 import search_evidence
from benchmark_target_aware_evidence_v038 import fixtures, run, save_new


class TargetAwareEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.g, self.cases = fixtures(width=8)

    def test_direct(self):
        out = search_evidence(self.g, "S", "B0", max_visits=2)
        self.assertEqual((out["status"], out["hops"]), ("FOUND", 1))
        self.assertFalse(out["verified_fact"])
        self.assertFalse(out["search"]["complete"])

    def test_budget_is_not_absence(self):
        self.assertEqual(search_evidence(self.g, "S", "T", max_visits=2)["status"], "LIMIT_REACHED")

    def test_deep_fanout_found(self):
        out = search_evidence(self.g, "S", "T", max_visits=20)
        self.assertEqual((out["status"], out["hops"]), ("FOUND", 5))

    def test_unreachable_complete(self):
        out = search_evidence(self.g, "S", "UNREACHABLE", max_visits=20)
        self.assertEqual(out["status"], "NOT_FOUND")
        self.assertTrue(out["search"]["complete"])

    def test_unapproved_edges(self):
        for target in ("P", "R"):
            self.assertEqual(search_evidence(self.g, "S", target, max_visits=20)["status"], "NOT_FOUND")

    def test_depth_limit(self):
        out = search_evidence(self.g, "S", "T", max_visits=20, max_hops=4)
        self.assertEqual(out["status"], "DEPTH_LIMIT_REACHED")
        self.assertFalse(out["search"]["complete"])

    def test_unknown(self):
        self.assertEqual(search_evidence(self.g, "UNKNOWN", "T")["status"], "UNKNOWN_NODE")

    def test_cycle_terminates(self):
        self.assertEqual(search_evidence(self.g, "S", "UNREACHABLE", max_visits=20)["status"], "NOT_FOUND")

    def test_idempotent_read_only(self):
        before = self.g.edge_count, len(self.g.nodes)
        for _ in range(3):
            search_evidence(self.g, "S", "T")
        self.assertEqual((self.g.edge_count, len(self.g.nodes)), before)

    def test_benchmark_matrix(self):
        res = run(budgets=(2, 5, 20), repeats=2)
        self.assertEqual(res["cases"]["deep_fanout"]["budgets"]["2"]["status_counts"], {"LIMIT_REACHED": 2})
        self.assertEqual(res["cases"]["deep_fanout"]["budgets"]["20"]["status_counts"], {"FOUND": 2})

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "existing.json"
            path.write_text("original", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                save_new(path, {"status": "changed"})
            self.assertEqual(path.read_text(encoding="utf-8"), "original")


if __name__ == "__main__":
    unittest.main()
