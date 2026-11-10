"""Experimental typed relation graph, independent from RecursiveSD."""
from collections import deque
from dataclasses import dataclass

RELATIONS = frozenset(("ABOUT", "REFERS_TO", "DEPENDS_ON", "NEGATES", "SUPPORTED_BY",
                       "CONTRADICTS", "BEFORE", "AFTER", "CAUSES"))
STATES = frozenset(("PENDING", "APPROVED", "REJECTED"))

@dataclass(frozen=True)
class Relation:
    source: str
    kind: str
    target: str
    evidence_id: str | None = None
    provenance: str | None = None
    state: str = "PENDING"

    def __post_init__(self):
        if not isinstance(self.source, str) or not self.source or not isinstance(self.target, str) or not self.target:
            raise ValueError("INVALID_ENDPOINT")
        if self.kind not in RELATIONS:
            raise ValueError("UNKNOWN_RELATION")
        if self.state not in STATES:
            raise ValueError("INVALID_STATE")

    def as_dict(self):
        return {"source": self.source, "kind": self.kind, "target": self.target,
                "evidence_id": self.evidence_id, "provenance": self.provenance,
                "state": self.state}

class RelationGraph:
    """In-memory, directed multigraph; paths do not certify factual truth."""
    def __init__(self, nodes=()):
        self.nodes = set(nodes)
        self.outgoing = {}
        self.edge_count = 0

    def add_node(self, node_id):
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("INVALID_NODE_ID")
        self.nodes.add(node_id)

    def add_relation(self, source, kind, target, *, state="PENDING", evidence_id=None, provenance=None):
        edge = Relation(source, kind, target, evidence_id, provenance, state)
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("UNKNOWN_NODE")
        self.outgoing.setdefault(source, []).append(edge)
        self.edge_count += 1
        return edge

    def traverse(self, start, *, target=None, kind=None, max_hops=5, max_visits=10000, approved_only=False):
        """Bounded BFS. LIMIT_REACHED never means the target is absent."""
        if start not in self.nodes or (target is not None and target not in self.nodes):
            return {"status": "UNKNOWN_NODE", "visited": 0, "scanned_edges": 0, "path": None}
        if kind is not None and kind not in RELATIONS:
            raise ValueError("UNKNOWN_RELATION")
        if not 0 <= max_hops <= 5 or max_visits < 1:
            raise ValueError("INVALID_LIMIT")
        queue = deque([(start, ())])
        seen = {start}
        scanned = 0
        if target == start:
            return {"status": "FOUND", "visited": 1, "scanned_edges": 0, "path": [], "hops": 0}
        while queue:
            node, path = queue.popleft()
            if len(path) >= max_hops:
                continue
            for edge in self.outgoing.get(node, ()):
                scanned += 1
                if kind is not None and edge.kind != kind:
                    continue
                if approved_only and edge.state != "APPROVED":
                    continue
                if edge.target in seen:
                    continue
                if len(seen) >= max_visits:
                    return {"status": "LIMIT_REACHED", "visited": len(seen), "scanned_edges": scanned, "path": None}
                new_path = path + (edge,)
                seen.add(edge.target)
                if edge.target == target:
                    return {"status": "FOUND", "visited": len(seen), "scanned_edges": scanned,
                            "path": [item.as_dict() for item in new_path], "hops": len(new_path)}
                queue.append((edge.target, new_path))
        return {"status": "NOT_FOUND" if target is not None else "COMPLETE",
                "visited": len(seen), "scanned_edges": scanned, "path": None}

    def evidence_path(self, claim, evidence, *, max_hops=5, max_visits=10000):
        """Only APPROVED SUPPORTED_BY edges; approval is not proof of truth."""
        result = self.traverse(claim, target=evidence, kind="SUPPORTED_BY",
                               max_hops=max_hops, max_visits=max_visits, approved_only=True)
        result["verified_fact"] = False
        result["meaning"] = "approved_relation_chain_only_not_factual_truth"
        return result
