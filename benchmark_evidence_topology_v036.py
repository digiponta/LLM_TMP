"""LLM_TMP v0.3.6: Evidence Path positive/negative and graph topology benchmark.

Synthetic in-memory fixtures only. No models, human reviews or prior results read.
APPROVED paths represent approved links, not externally verified facts.
"""
import argparse
import json
import math
import platform
import random
import statistics
import time
import tracemalloc
from pathlib import Path
from semantic_relation_graph_v035 import RelationGraph

def summary(values):
    ordered = sorted(values)
    def q(percent):
        x = (len(ordered) - 1) * percent / 100
        i, j = math.floor(x), math.ceil(x)
        return round((ordered[i] + (ordered[j] - ordered[i]) * (x - i)) / 1000, 3)
    return {"samples": len(values), "p50_us": q(50), "p95_us": q(95), "p99_us": q(99)}

def evidence_fixture(hops, state="APPROVED"):
    if hops not in range(1, 6) or state not in ("APPROVED", "PENDING", "REJECTED", "CYCLE"):
        raise ValueError("INVALID_CASE")
    names = [f"SD-{i}" for i in range(hops + 1)]
    graph = RelationGraph(names)
    for i in range(hops):
        edge_state = state if state in ("PENDING", "REJECTED") and i == hops - 1 else "APPROVED"
        graph.add_relation(names[i], "SUPPORTED_BY", names[i + 1], state=edge_state,
                           evidence_id=f"EV-{i}", provenance="synthetic")
    if state == "CYCLE":
        graph.add_relation(names[-1], "SUPPORTED_BY", names[0], state="APPROVED")
    return graph, names[0], names[-1]

def evidence_benchmark(samples):
    results = {}
    for hops in range(1, 6):
        for state in ("APPROVED", "PENDING", "REJECTED", "CYCLE"):
            graph, start, end = evidence_fixture(hops, state)
            expected = "NOT_FOUND" if state in ("PENDING", "REJECTED") else "FOUND"
            durations = []
            for _ in range(samples):
                t = time.perf_counter_ns()
                result = graph.evidence_path(start, end, max_hops=5, max_visits=2000)
                durations.append(time.perf_counter_ns() - t)
                if result["status"] != expected or result["verified_fact"] is not False:
                    raise AssertionError("EVIDENCE_CASE_FAILED")
                if expected == "FOUND" and (result["hops"] != hops or
                    any(edge["kind"] != "SUPPORTED_BY" or edge["state"] != "APPROVED"
                        for edge in result["path"])):
                    raise AssertionError("UNSAFE_EVIDENCE_PATH")
            results[f"{state.lower()}_{hops}hop"] = {**summary(durations), "status": expected}
    graph, start, end = evidence_fixture(3)
    controls = {
        "unknown": graph.evidence_path(start, "MISSING")["status"],
        "visit_budget": graph.evidence_path(start, end, max_visits=1)["status"],
        "hop_limit": graph.evidence_path(start, end, max_hops=2)["status"],
    }
    if controls != {"unknown": "UNKNOWN_NODE", "visit_budget": "LIMIT_REACHED", "hop_limit": "NOT_FOUND"}:
        raise AssertionError("CONTROL_FAILED")
    return {"cases": results, "controls": controls, "verified_fact": False}

def build_graph(topology, nodes, degree, seed):
    if topology not in ("random", "hierarchy", "hub"):
        raise ValueError("INVALID_TOPOLOGY")
    graph = RelationGraph(f"N-{i:07d}" for i in range(nodes))
    rng = random.Random(seed)
    for i in range(nodes):
        src = f"N-{i:07d}"
        for j in range(degree):
            if topology == "random":
                target = rng.randrange(nodes)
            elif topology == "hierarchy":
                target = ((i * degree + j + 1) % nodes) if j % 2 == 0 else i // max(2, degree)
            else:
                target = j % max(1, math.isqrt(nodes)) if j < degree - 1 else rng.randrange(nodes)
            graph.add_relation(src, "ABOUT", f"N-{target:07d}", provenance="synthetic")
    return graph

def run(nodes=100000, relations_per_node=5, samples=100, max_hops=5,
        max_visits=2000, seed=36):
    if not 2 <= nodes <= 1000000 or not 1 <= relations_per_node <= 100 or not 1 <= samples <= 100000 or not 1 <= max_hops <= 5 or not 2 <= max_visits <= 1000000:
        raise ValueError("INVALID_PARAMETERS")
    topologies = {}
    for topology in ("random", "hierarchy", "hub"):
        tracemalloc.start()
        started = time.perf_counter()
        graph = build_graph(topology, nodes, relations_per_node, seed)
        construction = time.perf_counter() - started
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if len(graph.nodes) != nodes or graph.edge_count != nodes * relations_per_node:
            raise AssertionError("INVALID_GRAPH_COUNTS")
        rng = random.Random(seed + 101)
        latencies, visited, scanned, statuses = [], [], [], {}
        for _ in range(samples):
            source = f"N-{rng.randrange(nodes):07d}"
            t = time.perf_counter_ns()
            outcome = graph.traverse(source, max_hops=max_hops, max_visits=max_visits)
            latencies.append(time.perf_counter_ns() - t)
            visited.append(outcome["visited"])
            scanned.append(outcome["scanned_edges"])
            statuses[outcome["status"]] = statuses.get(outcome["status"], 0) + 1
        topologies[topology] = {
            "nodes": len(graph.nodes), "relations": graph.edge_count,
            "build_seconds": round(construction, 6),
            "traversal": {**summary(latencies), "statuses": statuses,
                          "mean_visited": round(statistics.mean(visited), 2),
                          "mean_scanned_edges": round(statistics.mean(scanned), 2)},
            "memory": {"current_mib": round(current / 1048576, 3),
                       "peak_mib": round(peak / 1048576, 3),
                       "method": "tracemalloc allocations, not process RSS"},
        }
        del graph
    return {
        "status": "COMPLETE", "benchmark": "evidence_topology_v036",
        "scope": "synthetic RelationGraph; no existing data modified",
        "parameters": {"nodes": nodes, "relations_per_node": relations_per_node,
                       "samples": samples, "max_hops": max_hops,
                       "max_visits": max_visits, "seed": seed},
        "evidence": evidence_benchmark(samples),
        "topologies": topologies,
        "environment": {"python": platform.python_version(), "platform": platform.platform()}
    }

def save_new(path, data):
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=100000)
    parser.add_argument("--relations-per-node", type=int, default=5)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max-hops", type=int, default=5)
    parser.add_argument("--max-visits", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--output", default="results/evidence_topology_100k_5rel_v036.json")
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("OUTPUT_EXISTS_USE_NEW_FILENAME")
    result = run(args.nodes, args.relations_per_node, args.samples,
                 args.max_hops, args.max_visits, args.seed)
    try:
        save_new(args.output, result)
    except FileExistsError:
        parser.error("OUTPUT_EXISTS_USE_NEW_FILENAME")
    print(json.dumps({"status": result["status"], "output": args.output,
                      "evidence_cases": len(result["evidence"]["cases"]),
                      "topologies": {key: value["traversal"] for key, value in result["topologies"].items()}},
                     ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
