"""v0.3.6 synthetic, non-destructive regression tests."""
import tempfile
from pathlib import Path
import unittest
from benchmark_evidence_topology_v036 import evidence_fixture, evidence_benchmark, build_graph, run, save_new

class EvidenceTopologyTests(unittest.TestCase):
    def test_positive_1_to_5_hops(self):
        for hops in range(1, 6):
            graph, source, target = evidence_fixture(hops)
            result = graph.evidence_path(source, target)
            self.assertEqual(result["status"], "FOUND")
            self.assertEqual(result["hops"], hops)
            self.assertFalse(result["verified_fact"])

    def test_pending_rejected(self):
        for state in ("PENDING", "REJECTED"):
            graph, source, target = evidence_fixture(3, state)
            self.assertEqual(graph.evidence_path(source, target)["status"], "NOT_FOUND")

    def test_cycle(self):
        graph, source, target = evidence_fixture(4, "CYCLE")
        self.assertEqual(graph.evidence_path(source, target)["hops"], 4)

    def test_unknown_budget_hop(self):
        graph, source, target = evidence_fixture(3)
        self.assertEqual(graph.evidence_path(source, "MISSING")["status"], "UNKNOWN_NODE")
        self.assertEqual(graph.evidence_path(source, target, max_visits=1)["status"], "LIMIT_REACHED")
        self.assertEqual(graph.evidence_path(source, target, max_hops=2)["status"], "NOT_FOUND")

    def test_evidence_matrix(self):
        result = evidence_benchmark(3)
        self.assertEqual(len(result["cases"]), 20)
        self.assertEqual(result["cases"]["approved_5hop"]["status"], "FOUND")
        self.assertEqual(result["cases"]["rejected_5hop"]["status"], "NOT_FOUND")

    def test_topology_counts(self):
        for topology in ("random", "hierarchy", "hub"):
            graph = build_graph(topology, 25, 5, 36)
            self.assertEqual(len(graph.nodes), 25)
            self.assertEqual(graph.edge_count, 125)

    def test_small_benchmark(self):
        result = run(nodes=50, relations_per_node=5, samples=3, max_visits=60)
        self.assertEqual(set(result["topologies"]), {"random", "hierarchy", "hub"})
        self.assertEqual(result["topologies"]["hub"]["relations"], 250)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "protected.json"
            path.write_text("protected", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                save_new(path, {"status": "changed"})
            self.assertEqual(path.read_text(encoding="utf-8"), "protected")

if __name__ == "__main__":
    unittest.main()
