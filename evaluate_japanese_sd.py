"""Development sweep: surface Japanese -> recursive SD -> Japanese.

Cases are generated from the same lexicon/templates as the parser. They test
implementation correctness, NOT independent semantic generalization.
"""
import argparse
import itertools
import json
from pathlib import Path

from japanese_sd_parser import JapaneseSDParser, UnsupportedSentence
from recursive_sd import build_sample_graph


def evaluate(limit=50):
    graph = build_sample_graph()
    parser = JapaneseSDParser(graph)
    subjects = ("猫", "犬")
    objects = ("魚", "鳥", "ご飯")
    predicates = ("食べる", "見る")
    basic = [f"{s}が{o}を{v}。" for s, o, v
             in itertools.product(subjects, objects, predicates)]
    cases = [{"text": x, "supported": True, "category": "simple"}
             for x in basic]
    for a, b, join in itertools.islice(
            itertools.product(basic, basic, ("と、", "、そして")), limit):
        cases.append({"text": a[:-1] + join + b,
                      "supported": True, "category": "nested"})
    for value in (
        "猫は魚を食べる。", "犬が空を飛ぶ。",
        "猫が魚を踊る。", "猫が魚を食べると、犬が空を飛ぶ。",
        "猫が魚を食べると、犬が鳥を見ると、猫がご飯を食べる。",
    ):
        cases.append({"text": value, "supported": False, "category": "unsupported"})
    rows = []
    for case in cases:
        before = len(graph.nodes)
        try:
            item = parser.analyze(case["text"])
            accepted = item["roundtrip_equal"] and graph.validate()
            same_node = parser.parse(case["text"]) == item["sd_ref"]
            accepted = accepted and same_node
            reason = None
        except UnsupportedSentence as exc:
            accepted = False
            reason = str(exc)
        correct = accepted == case["supported"]
        if not case["supported"] and len(graph.nodes) != before:
            correct = False
        rows.append({**case, "accepted": accepted, "correct": correct,
                     "reason": reason})
    supported = [x for x in rows if x["supported"]]
    unsupported = [x for x in rows if not x["supported"]]
    return {
        "status": "DEVELOPMENT_SWEEP_NOT_INDEPENDENT",
        "total": len(rows),
        "roundtrip_rate": sum(x["correct"] for x in supported) / len(supported),
        "unsupported_reject_rate":
            sum(x["correct"] for x in unsupported) / len(unsupported),
        "node_count": len(graph.nodes),
        "graph_valid": graph.validate(),
        "rows": rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nested-limit", type=int, default=50)
    p.add_argument("--output", default="results/japanese_sd_development.json")
    args = p.parse_args()
    if args.nested_limit < 0:
        p.error("--nested-limit must be >= 0")
    result = evaluate(args.nested_limit)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print({key: value for key, value in result.items() if key != "rows"})
    print("Saved:", out)


if __name__ == "__main__":
    main()
