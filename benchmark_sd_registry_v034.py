"""Read-only, synthetic SD Registry scale benchmark for LLM_TMP.

Existing RecursiveSD.add_atom()/nodes provide node operations.
Relations are a separate benchmark-only adjacency index, NOT a production
feature of RecursiveSD. No real dataset or model files are accessed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time
import tracemalloc

from recursive_sd import RecursiveSD


def percentile(values, percent):
    ordered = sorted(values)
    if not ordered:
        return None
    pos = (len(ordered) - 1) * percent / 100
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def latency_summary(samples_ns):
    return {
        "samples": len(samples_ns),
        "p50_us": round(percentile(samples_ns, 50) / 1000, 3),
        "p95_us": round(percentile(samples_ns, 95) / 1000, 3),
        "p99_us": round(percentile(samples_ns, 99) / 1000, 3),
    }


def measure(nodes=100000, relations_per_node=5, queries=10000, traversals=1000,
            seed=33):
    if not 1 <= nodes <= 1000000:
        raise ValueError("nodes must be in [1, 1000000]")
    if not 0 <= relations_per_node <= 100:
        raise ValueError("relations_per_node must be in [0, 100]")
    if not 1 <= queries <= 1000000 or not 1 <= traversals <= 1000000:
        raise ValueError("invalid sample counts")
    rng = random.Random(seed)
    tracemalloc.start()
    baseline_current, _ = tracemalloc.get_traced_memory()
    t0 = time.perf_counter()
    graph = RecursiveSD()
    for i in range(nodes):
        graph.add_atom(f"N-{i:07d}", f"synthetic-{i}", kind="concept")
    registration_seconds = time.perf_counter() - t0
    node_current, node_peak = tracemalloc.get_traced_memory()
    # Deterministic directed graph with exactly K outgoing edges per node.
    # Duplicates are permitted for very small N; relation multiplicity is counted.
    adjacency = {}
    t1 = time.perf_counter()
    for i in range(nodes):
        adjacency[f"N-{i:07d}"] = tuple(
            (f"REL-{j % 8}", f"N-{(i + 1 + j * 7919) % nodes:07d}")
            for j in range(relations_per_node)
        )
    relation_seconds = time.perf_counter() - t1
    build_current, build_peak = tracemalloc.get_traced_memory()
    # Tracemalloc measures Python-tracked allocations, not process RSS.
    ids = [f"N-{rng.randrange(nodes):07d}" for _ in range(queries)]
    lookup_ns = []
    for key in ids:
        t = time.perf_counter_ns()
        node = graph.nodes.get(key)
        lookup_ns.append(time.perf_counter_ns() - t)
        if node is None:
            raise AssertionError("missing registered SD ID")
    hops = [f"N-{rng.randrange(nodes):07d}" for _ in range(traversals)]
    traverse_ns = []
    traversed_edges = 0
    for key in hops:
        t = time.perf_counter_ns()
        edges = adjacency[key]
        traverse_ns.append(time.perf_counter_ns() - t)
        traversed_edges += len(edges)
        if len(edges) != relations_per_node:
            raise AssertionError("incorrect edge degree")
    unknown_key = "N-UNKNOWN"
    if graph.nodes.get(unknown_key) is not None or unknown_key in adjacency:
        raise AssertionError("unknown ID must not exist")
    edge_count = sum(map(len, adjacency.values()))
    if edge_count != nodes * relations_per_node:
        raise AssertionError("incorrect total edge count")
    end_current, end_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_nodes = len(graph.nodes)
    return {
        "status": "COMPLETE",
        "benchmark": "synthetic_sd_registry_v034",
        "backend": {
            "nodes": "RecursiveSD.add_atom / RecursiveSD.nodes",
            "relations": "benchmark-only dict[str, tuple[(role, target), ...]]",
            "disclaimer": "Relation index is not part of production RecursiveSD"
        },
        "parameters": {
            "requested_nodes": nodes,
            "relations_per_node": relations_per_node,
            "query_samples": queries,
            "relation_lookup_samples": traversals,
            "seed": seed,
        },
        "correctness": {
            "total_sd_graph_nodes_including_templates": total_nodes,
            "registered_synthetic_nodes": nodes,
            "registered_relations": edge_count,
            "unknown_rejected": True,
            "counts_valid": True,
        },
        "registration": {
            "node_seconds": round(registration_seconds, 6),
            "node_per_sec": round(nodes / max(registration_seconds, 1e-12), 2),
            "relation_seconds": round(relation_seconds, 6),
            "relation_per_sec": round(edge_count / max(relation_seconds, 1e-12), 2),
        },
        "id_lookup": latency_summary(lookup_ns),
        "relation_lookup": {
            **latency_summary(traverse_ns),
            "edges_returned": traversed_edges,
            "operation": "single-hop outgoing adjacency lookup",
        },
        "memory": {
            "unit": "MiB",
            "method": "tracemalloc Python-tracked memory (not RSS)",
            "baseline_current": round(baseline_current / 1048576, 3),
            "after_nodes_current": round(node_current / 1048576, 3),
            "after_relations_current": round(build_current / 1048576, 3),
            "end_current": round(end_current / 1048576, 3),
            "peak": round(max(node_peak, build_peak, end_peak) / 1048576, 3),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def write_new_json(path, payload):
    """Exclusive creation: never overwrite an existing user report."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, ensure_ascii=False)
        output.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=100000)
    parser.add_argument("--relations-per-node", type=int, default=5)
    parser.add_argument("--queries", type=int, default=10000)
    parser.add_argument("--traversals", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--output", default="results/sd_registry_benchmark_100k_5rel_v034.json")
    args = parser.parse_args()
    # Fail before heavy work if the report path is already occupied.
    if Path(args.output).exists():
        parser.error("output already exists; choose a NEW --output filename")
    report = measure(args.nodes, args.relations_per_node, args.queries,
                     args.traversals, args.seed)
    try:
        write_new_json(args.output, report)
    except FileExistsError:
        parser.error("output created concurrently; existing report preserved")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
