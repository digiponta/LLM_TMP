"""Development-only acceptance/rejection evaluation of SD candidate boundary.

Rule and malformed-fixture providers check pipeline behavior; an actual LLM
is evaluated only when explicitly requested with its trained checkpoint.
"""
import argparse
import json
from pathlib import Path

from recursive_sd import build_sample_graph
from sd_candidate_pipeline import (
    DeterministicProposer, LLMCandidateProposer, accept_candidate,
    CandidateRejected,
)

SUPPORTED = [
    "猫が魚を食べる。",
    "犬が鳥を見る。",
    "猫がご飯を食べる。",
    "犬が魚を食べる。",
    "猫が魚を食べると、犬が鳥を見る。",
    "犬が鳥を見ると、猫がご飯を食べる。",
    "猫が魚を食べる、そして犬が鳥を見る。",
    "犬が魚を食べる、そして猫が鳥を見る。",
]
UNSUPPORTED = [
    "猫は魚を食べる。",
    "犬が空を飛ぶ。",
    "猫が魚を踊る。",
    "猫が魚を食べるので、犬が鳥を見る。",
    "猫が魚を食べると、犬が鳥を見ると、猫がご飯を食べる。",
]


def evaluate(provider_kind="rule", checkpoint=None, tokenizer=None, device="auto"):
    graph = build_sample_graph()
    if provider_kind == "rule":
        proposer = DeterministicProposer(graph)
    elif provider_kind == "llm":
        if not checkpoint or not tokenizer:
            raise ValueError("checkpoint and tokenizer required")
        proposer = LLMCandidateProposer(checkpoint, tokenizer, device)
    else:
        raise ValueError(provider_kind)

    rows = []
    for supported, queries in ((True, SUPPORTED), (False, UNSUPPORTED)):
        for text in queries:
            before = graph.snapshot()
            try:
                candidate = proposer.propose(text)
            except (CandidateRejected, ValueError, TypeError) as exc:
                result = {"status": "REJECTED", "reason": type(exc).__name__}
            else:
                result = accept_candidate(graph, text, candidate)
            accepted = result["status"] == "ACCEPTED"
            unchanged_if_rejected = accepted or graph.snapshot() == before
            rows.append({
                "text": text, "supported": supported,
                "status": result["status"],
                "reason": result.get("reason"),
                "correct": (accepted == supported) and unchanged_if_rejected,
                "unchanged_if_rejected": unchanged_if_rejected,
            })
    good = [x for x in rows if x["supported"]]
    bad = [x for x in rows if not x["supported"]]
    return {
        "status": "DEVELOPMENT_ONLY_NOT_INDEPENDENT",
        "provider": provider_kind, "total": len(rows),
        "supported_accept_rate": sum(x["status"] == "ACCEPTED" for x in good) / len(good),
        "unsupported_reject_rate": sum(x["status"] == "REJECTED" for x in bad) / len(bad),
        "atomic_rejection_rate":
            sum(x["unchanged_if_rejected"] for x in rows) / len(rows),
        "total_correct": sum(x["correct"] for x in rows),
        "graph_valid": graph.validate(), "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("rule", "llm"), default="rule")
    parser.add_argument("--checkpoint", default="model/model-gpu-v0.3.pt")
    parser.add_argument("--tokenizer", default="model/tokenizer.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", default="results/sd_candidate_development.json")
    args = parser.parse_args()
    if args.provider == "llm" and (
        not Path(args.checkpoint).is_file() or not Path(args.tokenizer).is_file()
    ):
        parser.error("LLM evaluation requires trained checkpoint and tokenizer")
    result = evaluate(args.provider, args.checkpoint, args.tokenizer, args.device)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print({key: value for key, value in result.items() if key != "rows"})
    print("Saved:", out)


if __name__ == "__main__":
    main()
