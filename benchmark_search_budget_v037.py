"""v0.3.7: paired search-budget sweep on one synthetic Random RelationGraph.

No external data accessed. No existing report overwritten. All budget cohorts
use the SAME graph, SAME source SDs and SAME reachable target SDs.
Targets are constructed by following four edges, so reachability is known.
This is NOT an evidence/factual-truth or independent-real-data evaluation.
"""
import argparse
import json
import platform
import random
import statistics
import time
from pathlib import Path

from benchmark_evidence_topology_v036 import build_graph, summary

DEFAULT_BUDGETS = (2000, 5000, 10000, 20000)


def make_cohort(graph, nodes, samples, seed, target_steps=4):
    """Generate a deterministic matched cohort independently of search budgets."""
    if not 1 <= target_steps <= 5:
        raise ValueError("INVALID_TARGET_STEPS")
    rng = random.Random(seed + 101)
    cohort = []
    for _ in range(samples):
        source = f"N-{rng.randrange(nodes):07d}"
        target = source
        for _ in range(target_steps):
            target = graph.outgoing[target][0].target
        cohort.append((source, target))
    return cohort


def run(nodes=100000, relations_per_node=5, samples=100, max_hops=5,
        budgets=DEFAULT_BUDGETS, seed=36, target_steps=4):
    budgets = tuple(budgets)
    if not 2 <= nodes <= 1000000 or not 1 <= relations_per_node <= 100:
        raise ValueError("INVALID_GRAPH_SIZE")
    if not 1 <= samples <= 100000 or not 1 <= target_steps <= max_hops <= 5:
        raise ValueError("INVALID_SAMPLE_OR_DEPTH")
    if not budgets or any(type(b) is not int or b < 1 or b > 1000000 for b in budgets):
        raise ValueError("INVALID_BUDGET")
    if len(budgets) != len(set(budgets)) or tuple(sorted(budgets)) != budgets:
        raise ValueError("BUDGETS_MUST_BE_ASCENDING_UNIQUE")

    t = time.perf_counter()
    graph = build_graph("random", nodes, relations_per_node, seed)
    build_seconds = time.perf_counter() - t
    if len(graph.nodes) != nodes or graph.edge_count != nodes * relations_per_node:
        raise AssertionError("GRAPH_COUNT_MISMATCH")
    cohort = make_cohort(graph, nodes, samples, seed, target_steps)
    results = {}
    # Each budget is applied to the same frozen graph and exact same query pairs.
    for budget in budgets:
        reach_times = []
        discovery_times = []
        reach_statuses = {}
        discovery_statuses = {}
        reach_visits = []
        discovery_visits = []
        discovered = 0
        for source, target in cohort:
            t = time.perf_counter_ns()
            broad = graph.traverse(source, max_hops=max_hops, max_visits=budget)
            reach_times.append(time.perf_counter_ns() - t)
            reach_statuses[broad["status"]] = reach_statuses.get(broad["status"], 0) + 1
            reach_visits.append(broad["visited"])

            t = time.perf_counter_ns()
            answer = graph.traverse(source, target=target, kind="ABOUT",
                                    max_hops=max_hops, max_visits=budget)
            discovery_times.append(time.perf_counter_ns() - t)
            discovery_statuses[answer["status"]] = discovery_statuses.get(answer["status"], 0) + 1
            discovery_visits.append(answer["visited"])
            if answer["status"] == "FOUND":
                discovered += 1
                if answer["hops"] > target_steps or not answer["path"] or any(
                        edge["kind"] != "ABOUT" for edge in answer["path"]):
                    raise AssertionError("INVALID_DISCOVERED_PATH")
            elif answer["status"] != "LIMIT_REACHED":
                # Reachability within target_steps is constructed, never inferred.
                raise AssertionError("KNOWN_REACHABLE_TARGET_NOT_FOUND")
        results[str(budget)] = {
            "untargeted": {**summary(reach_times),
                           "status_counts": reach_statuses,
                           "mean_visited": round(statistics.mean(reach_visits), 2)},
            "reachable_target": {**summary(discovery_times),
                                  "status_counts": discovery_statuses,
                                  "found": discovered,
                                  "samples": samples,
                                  "discovery_rate": round(discovered / samples, 4),
                                  "mean_visited": round(statistics.mean(discovery_visits), 2)},
        }
    return {
        "status": "COMPLETE",
        "benchmark": "search_budget_sweep_v037",
        "scope": "synthetic random ABOUT graph, not Evidence Path or real-world accuracy",
        "parameters": {
            "nodes": nodes, "relations_per_node": relations_per_node,
            "samples": samples, "max_hops": max_hops,
            "budgets": list(budgets), "target_steps": target_steps, "seed": seed
        },
        "graph": {"nodes": len(graph.nodes), "relations": graph.edge_count,
                  "build_seconds": round(build_seconds, 6)},
        "cohort": {"paired": True, "source_target_pairs": samples,
                   "target_reachability": "constructively reachable in <= target_steps ABOUT hops",
                   "note": "Duplicated source/target pairs are permitted."},
        "results": results,
        "interpretation": "LIMIT_REACHED means incomplete; NOT_FOUND means not found within the depth limit.",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }


def save_new(path, report):
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2, ensure_ascii=False)
        out.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=100000)
    parser.add_argument("--relations-per-node", type=int, default=5)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max-hops", type=int, default=5)
    parser.add_argument("--target-steps", type=int, default=4)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--output", default="results/search_budget_random_100k_5rel_v037.json")
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("OUTPUT_EXISTS_USE_NEW_FILENAME")
    result = run(args.nodes, args.relations_per_node, args.samples,
                 args.max_hops, args.budgets, args.seed, args.target_steps)
    try:
        save_new(args.output, result)
    except FileExistsError:
        parser.error("OUTPUT_EXISTS_USE_NEW_FILENAME")
    print(json.dumps({"status": "COMPLETE", "output": args.output,
                      "graph": result["graph"],
                      "budgets": {key: {
                          "untargeted": value["untargeted"]["status_counts"],
                          "found": value["reachable_target"]["found"],
                          "discovery_rate": value["reachable_target"]["discovery_rate"],
                          "target_p50_us": value["reachable_target"]["p50_us"],
                          "target_p99_us": value["reachable_target"]["p99_us"]}
                          for key, value in result["results"].items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
