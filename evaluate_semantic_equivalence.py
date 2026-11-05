"""Development fixtures for semantic equivalence (not independent holdout).

This tests whether syntactic voice alternations map to the same linked SD,
whether changed roles/predicates remain distinct, and whether unsupported
forms are not assigned a guessed semantic identity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_equivalence_sd import compare_meaning

CASES = [
    ("猫が魚を食べる。", "魚が猫に食べられる。", True),
    ("犬が鳥を見る。", "鳥が犬に見られる。", True),
    ("猫がご飯を食べる。", "ご飯が猫に食べられる。", True),
    ("魚が猫を見る。", "猫が魚に見られる。", True),
    ("猫が魚を食べると、犬が鳥を見る。",
     "魚が猫に食べられると、鳥が犬に見られる。", True),
    ("猫が魚を食べる、そして犬が鳥を見る。",
     "魚が猫に食べられる、そして鳥が犬に見られる。", True),
    ("犬が鳥を見る。", "犬が鳥に見られる。", False),
    ("猫が魚を食べる。", "猫が魚を見る。", False),
    ("猫が魚を食べる。", "犬が魚を食べる。", False),
    ("猫が魚を食べると、犬が鳥を見る。",
     "猫が魚を食べる、そして犬が鳥を見る。", False),
]
UNSUPPORTED = [
    ("猫が魚を食べる。", "猫が魚を食べない。"),
    ("猫が魚を食べる。", "猫が魚を食べる？"),
    ("猫が魚を食べる。", "魚が猫に踊られる。"),
    ("猫が魚を食べる。", "魚が猫によって食べられる。"),
    ("猫が魚を食べる。", "魚が猫に食べられた。"),
]


def evaluate():
    rows = []
    for left, right, expected in CASES:
        result = compare_meaning(left, right)
        rows.append({"left": left, "right": right, "expected": expected,
                     "actual": result["same_meaning"], "status": result["status"],
                     "correct": result["status"] == "COMPARED" and
                                result["same_meaning"] is expected})
    for left, right in UNSUPPORTED:
        result = compare_meaning(left, right)
        rows.append({"left": left, "right": right, "expected": "UNSUPPORTED",
                     "actual": result["same_meaning"], "status": result["status"],
                     "correct": result["status"] == "UNSUPPORTED" and
                                result["same_meaning"] is None})
    return {
        "status": "DEVELOPMENT_ONLY_NOT_INDEPENDENT",
        "total": len(rows),
        "positive": sum(r["expected"] is True for r in rows),
        "negative": sum(r["expected"] is False for r in rows),
        "unsupported": len(UNSUPPORTED),
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "rows": rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/semantic_equivalence_development.json")
    args = p.parse_args()
    report = evaluate()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print({k: v for k, v in report.items() if k != "rows"})
    print("Saved:", out)


if __name__ == "__main__":
    main()
