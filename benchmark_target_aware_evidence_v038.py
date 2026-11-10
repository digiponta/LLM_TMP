"""v0.3.8 deterministic target-aware evidence search benchmark.
Synthetic graphs only; same cases under every budget; existing reports protected.
"""
import argparse
import json
from pathlib import Path
import statistics
import time
from semantic_relation_graph_v035 import RelationGraph
from target_aware_evidence_search_v038 import search_evidence


def fixtures(width=8):
    if width < 2:
        raise ValueError("WIDTH_TOO_SMALL")
    g = RelationGraph(["S", "T", "UNREACHABLE"] +
                      [f"B{i}" for i in range(width)] + ["C", "D", "E", "P", "R"])
    for i in range(width):
        g.add_relation("S", "SUPPORTED_BY", f"B{i}", state="APPROVED", provenance="synthetic")
    g.add_relation(f"B{width-1}", "SUPPORTED_BY", "C", state="APPROVED")
    g.add_relation("C", "SUPPORTED_BY", "D", state="APPROVED")
    g.add_relation("D", "SUPPORTED_BY", "E", state="APPROVED")
    g.add_relation("E", "SUPPORTED_BY", "T", state="APPROVED")
    g.add_relation("S", "SUPPORTED_BY", "P", state="PENDING")
    g.add_relation("S", "SUPPORTED_BY", "R", state="REJECTED")
    g.add_relation("T", "SUPPORTED_BY", "S", state="APPROVED")
    return g, {
        "direct": ("S", "B0", "FOUND"),
        "deep_fanout": ("S", "T", "FOUND"),
        "unreachable": ("S", "UNREACHABLE", "NOT_FOUND"),
        "pending_only": ("S", "P", "NOT_FOUND"),
        "rejected_only": ("S", "R", "NOT_FOUND"),
    }


def run(budgets=(2, 5, 10, 20), repeats=100, width=8, max_hops=5):
    if repeats < 1 or not 0 <= max_hops <= 5:
        raise ValueError("INVALID_ARGUMENT")
    budgets = tuple(budgets)
    if not budgets or any(type(x) is not int or x < 1 for x in budgets) or tuple(sorted(set(budgets))) != budgets:
        raise ValueError("INVALID_BUDGETS")
    graph, cases = fixtures(width)
    report = {
        "status": "COMPLETE", "benchmark": "target_aware_evidence_v038",
        "scope": "synthetic approved SUPPORTED_BY paths, not factual truth",
        "parameters": {"budgets": list(budgets), "repeats": repeats, "width": width, "max_hops": max_hops},
        "cases": {},
    }
    for name, (source, target, expected) in cases.items():
        by_budget = {}
        for budget in budgets:
            times = []
            outcome = None
            for _ in range(repeats):
                t = time.perf_counter_ns()
                outcome = search_evidence(graph, source, target, max_hops=max_hops, max_visits=budget)
                times.append(time.perf_counter_ns() - t)
            if outcome["status"] == "FOUND" and expected != "FOUND":
                raise AssertionError("UNSAFE_FALSE_POSITIVE")
            if outcome["status"] == "NOT_FOUND" and expected == "FOUND":
                raise AssertionError("FALSE_ABSENCE")
            if outcome["status"] == "FOUND" and any(e["state"] != "APPROVED" for e in outcome["path"]):
                raise AssertionError("UNAPPROVED_PATH")
            by_budget[str(budget)] = {
                "status_counts": {outcome["status"]: repeats},
                "found": outcome["status"] == "FOUND",
                "visited": outcome["search"]["visited"],
                "p50_us": round(statistics.median(times) / 1000, 3),
                "max_us": round(max(times) / 1000, 3),
            }
        report["cases"][name] = {"expected_unbounded": expected, "budgets": by_budget}
    return report


def save_new(path, result):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 5, 10, 20])
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--max-hops", type=int, default=5)
    parser.add_argument("--output", default="results/target_aware_evidence_v038.json")
    a = parser.parse_args()
    if Path(a.output).exists():
        parser.error("OUTPUT_EXISTS_USE_NEW_FILENAME")
    result = run(a.budgets, a.repeats, a.width, a.max_hops)
    try:
        save_new(a.output, result)
    except FileExistsError:
        parser.error("OUTPUT_EXISTS_USE_NEW_FILENAME")
    print(json.dumps({
        "status": result["status"], "output": a.output,
        "cases": {k: {b: v["status_counts"] for b, v in case["budgets"].items()}
                  for k, case in result["cases"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
