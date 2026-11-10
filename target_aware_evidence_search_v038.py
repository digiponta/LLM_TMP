"""Target-aware bounded Evidence search and conservative SKI result envelope.
Independent adapter: no RelationGraph mutation or factual truth claims.
Only APPROVED SUPPORTED_BY edges are eligible; BFS is shortest-hop first.
"""
from collections import deque
from semantic_relation_graph_v035 import RelationGraph


def search_evidence(graph: RelationGraph, source: str, target: str, *, max_hops=5, max_visits=2000):
    if not isinstance(max_hops, int) or not 0 <= max_hops <= 5 or not isinstance(max_visits, int) or max_visits < 1:
        raise ValueError("INVALID_LIMIT")
    if source not in graph.nodes or target not in graph.nodes:
        return _result("UNKNOWN_NODE", source, target, None, 0, 0, max_hops, max_visits)
    queue = deque([(source, ())])
    seen = {source}
    scanned = 0
    truncated_depth = False
    if source == target:
        return _result("FOUND", source, target, [], 1, 0, max_hops, max_visits)
    while queue:
        node, path = queue.popleft()
        for edge in graph.outgoing.get(node, ()):
            scanned += 1
            if edge.kind != "SUPPORTED_BY" or edge.state != "APPROVED":
                continue
            if edge.target in seen:
                continue
            if len(path) >= max_hops:
                truncated_depth = True
                continue
            if len(seen) >= max_visits:
                return _result("LIMIT_REACHED", source, target, None, len(seen), scanned, max_hops, max_visits)
            new_path = path + (edge,)
            seen.add(edge.target)
            if edge.target == target:
                return _result("FOUND", source, target,
                               [item.as_dict() for item in new_path],
                               len(seen), scanned, max_hops, max_visits)
            queue.append((edge.target, new_path))
    return _result("DEPTH_LIMIT_REACHED" if truncated_depth else "NOT_FOUND",
                   source, target, None, len(seen), scanned, max_hops, max_visits)


def _result(status, source, target, path, visited, scanned, max_hops, max_visits):
    return {
        "ski_payload_version": "0.3.8-experimental",
        "query_type": "EVIDENCE_PATH",
        "status": status,
        "source_sd": source,
        "target_sd": target,
        "path": path,
        "hops": len(path) if path is not None else None,
        "search": {
            "max_hops": max_hops,
            "max_visits": max_visits,
            "visited": visited,
            "scanned_edges": scanned,
            "complete": status == "NOT_FOUND",
            "termination_reason": status,
        },
        "verified_fact": False,
        "semantic_alignment_verified": False,
        "human_approved": False,
        "interpretation": "approved_relation_path_not_factual_verification",
    }
