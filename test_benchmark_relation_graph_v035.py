import tempfile
from pathlib import Path
import unittest
from benchmark_relation_graph_v035 import run, save_new

class RelationBenchmarkTests(unittest.TestCase):
    def test_small_scale(self):
        report = run(nodes=30, relations_per_node=5, samples=5, max_hops=3, max_visits=40)
        self.assertEqual(report["correctness"], {"nodes": 30, "relations": 150})
        self.assertEqual(report["traversal"]["samples"], 5)
        self.assertFalse(report["evidence_path"]["verified_fact"])

    def test_never_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "existing.json"
            path.write_text("original", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                save_new(path, {"status": "replacement"})
            self.assertEqual(path.read_text(encoding="utf-8"), "original")

if __name__ == "__main__":
    unittest.main()
