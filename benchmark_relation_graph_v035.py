"""Synthetic, bounded 1-5 hop Relation Graph benchmark (no real data accessed)."""
import argparse
import json
from pathlib import Path
import platform
import random
import statistics
import time
import tracemalloc
from semantic_relation_graph_v035 import RelationGraph

def percentiles(values):
    values = sorted(values)
    def p(q):
        ix = (len(values) - 1) * q
        lo = int(ix)
        return round((values[lo] + (values[min(lo + 1, len(values)-1)] - values[lo]) * (ix - lo)) / 1000, 3)
    return {"samples": len(values), "p50_us": p(.5), "p95_us": p(.95), "p99_us": p(.99)}

def run(nodes=100000, relations_per_node=5, samples=100, max_hops=5, max_visits=2000, seed=35):
    if not 1 <= nodes <= 1000000 or not 0 <= relations_per_node <= 100 or not 1 <= samples <= 100000 or not 1 <= max_hops <= 5 or not 1 <= max_visits <= 1000000:
        raise ValueError("INVALID_PARAMETERS")
    rng = random.Random(seed)
    tracemalloc.start()
    start = time.perf_counter()
    graph = RelationGraph(f"N-{i:07d}" for i in range(nodes))
    node_seconds = time.perf_counter() - start
    kinds = ("ABOUT", "REFERS_TO", "DEPENDS_ON", "SUPPORTED_BY", "NEGATES")
    start = time.perf_counter()
    for i in range(nodes):
        source = f"N-{i:07d}"
        for j in range(relations_per_node):
            target = f"N-{(i + 1 + j*7919) % nodes:07d}"
            graph.add_relation(source, kinds[j % len(kinds)], target,
                               state="APPROVED" if j % 2 == 0 else "PENDING",
                               provenance="synthetic")
    relation_seconds = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    duration, statuses, visited = [], {}, []
    for _ in range(samples):
        source = f"N-{rng.randrange(nodes):07d}"
        start = time.perf_counter_ns()
        result = graph.traverse(source, max_hops=max_hops, max_visits=max_visits)
        duration.append(time.perf_counter_ns() - start)
        statuses[result["status"]] = statuses.get(result["status"], 0) + 1
        visited.append(result["visited"])
    evidence_ns = []
    evidence_hits = 0
    for _ in range(samples):
        i = rng.randrange(nodes)
        source = f"N-{i:07d}"
        target = f"N-{(i + 1 + 3*7919) % nodes:07d}"
        start = time.perf_counter_ns()
        result = graph.evidence_path(source, target, max_hops=max_hops, max_visits=max_visits)
        evidence_ns.append(time.perf_counter_ns() - start)
        evidence_hits += result["status"] == "FOUND"
    return {
        "status": "COMPLETE", "benchmark": "synthetic_relation_graph_v035",
        "scope": "benchmark-only RelationGraph; no RecursiveSD mutation",
        "parameters": {"nodes": nodes, "relations_per_node": relations_per_node, "samples": samples, "max_hops": max_hops, "max_visits": max_visits, "seed": seed},
        "correctness": {"nodes": len(graph.nodes), "relations": graph.edge_count},
        "construction": {"node_seconds": round(node_seconds, 6), "relation_seconds": round(relation_seconds, 6)},
        "traversal": {**percentiles(duration), "statuses": statuses, "mean_visited": round(statistics.mean(visited), 2),
                      "operation": "bounded BFS reachability; target unspecified"},
        "evidence_path": {**percentiles(evidence_ns), "approved_path_hits": evidence_hits,
                           "operation": "approved SUPPORTED_BY-only, target is deliberately PENDING",
                           "verified_fact": False},
        "memory": {"method": "tracemalloc Python-tracked only, not process RSS",
                   "current_mib": round(current/1048576, 3), "peak_mib": round(peak/1048576, 3)},
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }

def save_new(path, payload):
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nodes", type=int, default=100000)
    p.add_argument("--relations-per-node", type=int, default=5)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--max-hops", type=int, default=5)
    p.add_argument("--max-visits", type=int, default=2000)
    p.add_argument("--seed", type=int, default=35)
    p.add_argument("--output", default="results/relation_graph_100k_5rel_v035.json")
    a = p.parse_args()
    if Path(a.output).exists():
        p.error("OUTPUT_EXISTS_USE_NEW_FILENAME")
    result = run(a.nodes, a.relations_per_node, a.samples, a.max_hops, a.max_visits, a.seed)
    try:
        save_new(a.output, result)
    except FileExistsError:
        p.error("OUTPUT_EXISTS_USE_NEW_FILENAME")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
