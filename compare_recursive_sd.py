"""Development-only evaluation of recursive SD composition versus A/B/C.

NOT an independent holdout: the task grammar is fixed and visible in code.
Outputs contain per-item predictions to support later blind evaluation.
"""
import argparse
import json
import re
import time
from pathlib import Path

from compare_four_modes import LLMAdapter, SDMemory, lexical_memory, normalize
from recursive_sd import build_sample_graph, compose_pair, compose_sentence

# Syntax is deliberately explicit to isolate symbolic composition.
# No result fields are consulted by predictors.
INPUTS = [
    ("猫", "魚", "食べる", "犬", "鳥", "見る", "T-THEN"),
    ("犬", "鳥", "見る", "猫", "ご飯", "食べる", "T-THEN"),
    ("猫", "鳥", "見る", "犬", "魚", "食べる", "T-THEN"),
    ("犬", "ご飯", "食べる", "猫", "魚", "食べる", "T-THEN"),
    ("犬", "魚", "食べる", "猫", "鳥", "見る", "T-AND"),
    ("猫", "ご飯", "食べる", "犬", "鳥", "見る", "T-AND"),
    ("犬", "鳥", "見る", "犬", "魚", "食べる", "T-AND"),
    ("猫", "魚", "食べる", "猫", "鳥", "見る", "T-AND"),
]
CONNECTORS = {"T-THEN": "と、", "T-AND": "、そして"}
ENT = {"猫": "CAT", "犬": "DOG", "魚": "FISH", "鳥": "BIRD", "ご飯": "RICE"}
VERB = {"食べる": "EAT", "見る": "WATCH"}
PATTERN = re.compile(
    r"^文1\[主語=(?P<s1>[^、\]]+)、目的語=(?P<o1>[^、\]]+)、"
    r"動詞=(?P<v1>[^、\]]+)\]；"
    r"文2\[主語=(?P<s2>[^、\]]+)、目的語=(?P<o2>[^、\]]+)、"
    r"動詞=(?P<v2>[^、\]]+)\]；接続=(?P<link>T-THEN|T-AND)。"
    r"複文を作って。$"
)


def prompt(parts):
    s1, o1, v1, s2, o2, v2, link = parts
    return (f"文1[主語={s1}、目的語={o1}、動詞={v1}]；"
            f"文2[主語={s2}、目的語={o2}、動詞={v2}]；"
            f"接続={link}。複文を作って。")


def reference(parts):
    s1, o1, v1, s2, o2, v2, link = parts
    return f"{s1}が{o1}を{v1}{CONNECTORS[link]}{s2}が{o2}を{v2}。"


def predict_d(question, graph):
    match = PATTERN.fullmatch(question)
    if not match:
        return None
    v = match.groupdict()
    try:
        first = compose_sentence(graph, ENT[v["s1"]], ENT[v["o1"]], VERB[v["v1"]])
        second = compose_sentence(graph, ENT[v["s2"]], ENT[v["o2"]], VERB[v["v2"]])
        parent = compose_pair(graph, first, second, v["link"])
        parts = graph.decompose(parent)
        if graph.recompose(parts) != parent:
            raise AssertionError("Roundtrip failed")
        return graph.render(parent)
    except (KeyError, ValueError):
        return None


def run(args):
    sd = SDMemory()
    graph = build_sample_graph()
    llm = None
    if not args.structural_only:
        if not Path(args.checkpoint).is_file() or not Path(args.tokenizer).is_file():
            raise FileNotFoundError("A requires checkpoint and tokenizer")
        llm = LLMAdapter(args.checkpoint, args.tokenizer, args.device, args.max_new_tokens)

    # Distinct from the fixed original 8 examples, but not blind.
    cases = [{"id": f"nested-{i+1:02}", "question": prompt(parts),
              "expected": reference(parts), "kind": "nested_compose"}
             for i, parts in enumerate(INPUTS)]
    cases += [
        {"id": "unknown-01", "question": "犬が魚を食べるのは事実ですか？",
         "expected": None, "kind": "unknown"},
        {"id": "unknown-02", "question": "鳥がご飯を食べるのは事実ですか？",
         "expected": None, "kind": "unknown"},
        {"id": "invalid-01", "question": prompt(
            ("猫", "魚", "踊る", "犬", "鳥", "見る", "T-AND")),
         "expected": None, "kind": "invalid"},
    ]
    scores = []
    for mode in ("BCD" if args.structural_only else "ABCD"):
        rows = []
        for item in cases:
            start = time.perf_counter()
            q = item["question"]
            if mode == "A":
                answer = llm.answer(q)
            elif mode == "B":
                answer = lexical_memory(q, sd)
            elif mode == "C":
                answer = sd.resolve_fact(q)
            else:
                answer = predict_d(q, graph)
            elapsed = time.perf_counter() - start
            correct = (answer is None if item["expected"] is None else
                       normalize(answer) == normalize(item["expected"]))
            rows.append({**item, "answer": answer, "correct": correct,
                         "latency_ms": round(elapsed * 1000, 3)})
        sc = {
            "mode": mode,
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "nested_accuracy": sum(r["correct"] for r in rows if r["kind"] == "nested_compose") / len(INPUTS),
            "unknown_reject": sum(r["answer"] is None for r in rows if r["expected"] is None) / 3,
            "rows": rows,
        }
        scores.append(sc)
        print(f'{mode}: accuracy={sc["accuracy"]:.3f} nested={sc["nested_accuracy"]:.3f} unknown={sc["unknown_reject"]:.3f}')
    report = {
        "status": "DEVELOPMENT_ONLY_NOT_INDEPENDENT",
        "note": "Template-shaped prompts and known vocabulary; results demonstrate a rule-based composition implementation, not general LLM gains.",
        "case_count": len(cases), "modes": scores,
        "graph_node_count": len(graph.nodes),
        "graph_valid": graph.validate(),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", path)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--structural-only", action="store_true")
    p.add_argument("--checkpoint", default="model/model-gpu-v0.3.pt")
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--output", default="results/recursive_sd_development.json")
    run(p.parse_args())


if __name__ == "__main__":
    main()
