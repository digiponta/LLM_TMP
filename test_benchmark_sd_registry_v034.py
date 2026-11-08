"""Small, non-destructive smoke tests; the 100k case is run separately."""
import json
from pathlib import Path
import tempfile
import unittest
from benchmark_sd_registry_v034 import measure, write_new_json, percentile


class BenchmarkRegistryTests(unittest.TestCase):
    def test_small_counts(self):
        data = measure(nodes=21, relations_per_node=5, queries=30,
                       traversals=20)
        self.assertEqual(data["correctness"]["registered_synthetic_nodes"], 21)
        self.assertEqual(data["correctness"]["registered_relations"], 105)
        self.assertEqual(data["correctness"]["total_sd_graph_nodes_including_templates"], 24)
        self.assertTrue(data["correctness"]["unknown_rejected"])
        self.assertEqual(data["relation_lookup"]["edges_returned"], 100)

    def test_zero_relations(self):
        data = measure(nodes=3, relations_per_node=0, queries=4, traversals=4)
        self.assertEqual(data["correctness"]["registered_relations"], 0)

    def test_percentile(self):
        self.assertEqual(percentile([1, 2, 3], 50), 2)
        self.assertEqual(percentile([1, 2], 100), 2)

    def test_exclusive_output(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "report.json"
            write_new_json(path, {"hello": "first"})
            with self.assertRaises(FileExistsError):
                write_new_json(path, {"hello": "second"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")),
                             {"hello": "first"})

    def test_repeatable_counts(self):
        a = measure(nodes=11, relations_per_node=3, queries=10, traversals=4, seed=4)
        b = measure(nodes=11, relations_per_node=3, queries=10, traversals=4, seed=4)
        self.assertEqual(a["correctness"], b["correctness"])
        self.assertEqual(a["relation_lookup"]["edges_returned"],
                         b["relation_lookup"]["edges_returned"])


if __name__ == "__main__":
    unittest.main()
