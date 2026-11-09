import unittest
from semantic_relation_graph_v035 import RelationGraph

class RelationGraphTests(unittest.TestCase):
    def setUp(self):
        self.g = RelationGraph(["claim", "middle", "evidence", "noise"])

    def test_approved_evidence_path(self):
        self.g.add_relation("claim", "SUPPORTED_BY", "middle", state="APPROVED")
        self.g.add_relation("middle", "SUPPORTED_BY", "evidence", state="APPROVED")
        r = self.g.evidence_path("claim", "evidence")
        self.assertEqual((r["status"], r["hops"]), ("FOUND", 2))
        self.assertFalse(r["verified_fact"])

    def test_pending_never_approved(self):
        self.g.add_relation("claim", "SUPPORTED_BY", "evidence", state="PENDING")
        self.assertEqual(self.g.evidence_path("claim", "evidence")["status"], "NOT_FOUND")
        self.assertEqual(self.g.traverse("claim", target="evidence")["status"], "FOUND")

    def test_cycles_stop(self):
        self.g.add_relation("claim", "ABOUT", "middle")
        self.g.add_relation("middle", "ABOUT", "claim")
        self.assertEqual(self.g.traverse("claim", target="noise")["status"], "NOT_FOUND")

    def test_hop_limit(self):
        self.g.add_relation("claim", "ABOUT", "middle")
        self.g.add_relation("middle", "ABOUT", "evidence")
        self.assertEqual(self.g.traverse("claim", target="evidence", max_hops=1)["status"], "NOT_FOUND")
        self.assertEqual(self.g.traverse("claim", target="evidence", max_hops=2)["hops"], 2)

    def test_visit_budget_does_not_claim_absence(self):
        self.g.add_relation("claim", "ABOUT", "middle")
        self.assertEqual(self.g.traverse("claim", target="noise", max_visits=1)["status"], "LIMIT_REACHED")

    def test_validation_no_mutation(self):
        before = self.g.edge_count
        with self.assertRaises(ValueError):
            self.g.add_relation("claim", "UNSUPPORTED", "evidence")
        with self.assertRaises(ValueError):
            self.g.add_relation("claim", "ABOUT", "missing")
        self.assertEqual(self.g.edge_count, before)

    def test_shortest_path(self):
        self.g.add_relation("claim", "ABOUT", "middle")
        self.g.add_relation("middle", "ABOUT", "evidence")
        self.g.add_relation("claim", "ABOUT", "evidence")
        self.assertEqual(self.g.traverse("claim", target="evidence")["hops"], 1)

    def test_unknown(self):
        self.assertEqual(self.g.traverse("missing")["status"], "UNKNOWN_NODE")

    def test_relation_filter(self):
        self.g.add_relation("claim", "ABOUT", "evidence")
        self.assertEqual(self.g.traverse("claim", target="evidence", kind="SUPPORTED_BY")["status"], "NOT_FOUND")

if __name__ == "__main__":
    unittest.main()
