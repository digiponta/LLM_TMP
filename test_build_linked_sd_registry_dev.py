"""Ensure the v0.3.1 registry builder preserves actual RecursiveSD IDs."""
import unittest
from recursive_sd import build_sample_graph
from build_linked_sd_registry_dev import build_registry
from linked_sd_annotation_v031 import catalog

class LinkedSDRegistryTests(unittest.TestCase):
    def test_existing_ids_only(self):
        graph = build_sample_graph()
        data = build_registry(graph)
        self.assertEqual({x["sd_id"] for x in data["nodes"]},
                         {"CAT", "DOG", "FISH", "BIRD", "RICE", "EAT", "WATCH"})
        self.assertTrue({x["sd_id"] for x in data["nodes"]} <= set(graph.nodes))
    def test_no_invented_relations(self):
        data = build_registry(build_sample_graph())
        self.assertFalse(any(x["node_type"] == "RELATION" for x in data["nodes"]))
    def test_gui_catalog_only_concepts(self):
        data = build_registry(build_sample_graph())
        self.assertEqual({x["sd_id"] for x in catalog(data)},
                         {"CAT", "DOG", "FISH", "BIRD", "RICE"})
    def test_display_names(self):
        data = build_registry(build_sample_graph())
        self.assertEqual(data["display_names"]["CAT"], "猫")
    def test_deterministic_snapshot(self):
        self.assertEqual(build_registry(build_sample_graph()),
                         build_registry(build_sample_graph()))
if __name__ == "__main__":
    unittest.main()
