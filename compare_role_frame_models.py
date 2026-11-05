"""Compare direct Core+Frame and two-head LLMs on identical dev cases.

Read locally generated report JSONs only. This is a paired *development*
comparison, not an independent holdout or statistical proof of transfer.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def compare_reports(base, staged):
    a, b = base["rows"], staged["rows"]
    if len(a) != len(b) or len(a) != 120:
        raise ValueError("EXPECTED_IDENTICAL_120_CASES")
    left = {r["text"]: r for r in a}
    right = {r["text"]: r for r in b}
    if len(left) != len(a) or len(right) != len(b) or set(left) != set(right):
        raise ValueError("DEVELOPMENT_CASE_MISMATCH")
    for text in left:
        if left[text]["expected"] != right[text]["expected"]:
            raise ValueError("TARGET_MISMATCH")
        if left[text]["voice"] != right[text]["voice"]:
            raise ValueError("VOICE_MISMATCH")
    def summarize(items):
        total = len(items)
        base_correct = sum(left[text]["exact_code"] for text in items)
        stage_correct = sum(right[text]["exact_code"] for text in items)
        improved = sum(not left[text]["exact_code"] and
                       right[text]["exact_code"] for text in items)
        regressed = sum(left[text]["exact_code"] and
                        not right[text]["exact_code"] for text in items)
        return {
            "total": total,
            "baseline_correct": base_correct,
            "two_head_correct": stage_correct,
            "baseline_rate": base_correct / total,
            "two_head_rate": stage_correct / total,
            "delta_percentage_points": round(
                100 * (stage_correct - base_correct) / total, 3),
            "newly_correct": improved, "regressions": regressed,
        }
    pairs = list(left)
    return {
        "status": "PAIRED_DEVELOPMENT_ONLY",
        "overall": summarize(pairs),
        "by_voice": {
            voice: summarize([text for text in pairs
                              if left[text]["voice"] == voice])
            for voice in ("active", "passive")
        },
        "baseline_reasons": dict(collections.Counter(
            row.get("reason") or "ACCEPTED" for row in a)),
        "two_head_reasons": dict(collections.Counter(
            row.get("reason") or "ACCEPTED" for row in b)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="results/sd_frame_eval.json")
    p.add_argument("--two-head", default="results/sd_role_frame_eval.json")
    p.add_argument("--output", default="results/role_frame_comparison.json")
    args = p.parse_args()
    base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    new = json.loads(Path(args.two_head).read_text(encoding="utf-8"))
    report = compare_reports(base, new)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Saved:", out)


if __name__ == "__main__":
    main()
