"""Paired comparison of two private RealPersonaChat Dev-generation reports.

No access to raw corpora or Test data. Refuses mismatched prompts/references,
duplicate examples, or metrics obtained on different Dev cases. Metrics
describe generation, NOT conversational quality; no model is selected here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def case_key(row):
    return str(row["dialogue_id"]), str(row["turn_id"])


def compare_reports(baseline, candidate):
    if not isinstance(baseline.get("rows"), list) or not isinstance(candidate.get("rows"), list):
        raise ValueError("REPORT_ROWS_REQUIRED")
    left = {case_key(r): r for r in baseline["rows"]}
    right = {case_key(r): r for r in candidate["rows"]}
    if (len(left) != len(baseline["rows"]) or
            len(right) != len(candidate["rows"])):
        raise ValueError("DUPLICATE_CASE_KEY")
    if not left or set(left) != set(right):
        raise ValueError("DIFFERENT_DEV_CASES")
    rows = []
    for key in sorted(left):
        a, b = left[key], right[key]
        if a["prompt"] != b["prompt"] or a["reference"] != b["reference"]:
            raise ValueError("DEV_CONTENT_MISMATCH")
        a_valid = a["status"] == "GENERATED"
        b_valid = b["status"] == "GENERATED"
        rows.append({
            "dialogue_id": key[0], "turn_id": key[1],
            "baseline_generated": a_valid, "candidate_generated": b_valid,
            "baseline_overlap": a.get("generated_overlap") if a_valid else None,
            "candidate_overlap": b.get("generated_overlap") if b_valid else None,
            "baseline_exact": a_valid and a["generated"] == a["reference"],
            "candidate_exact": b_valid and b["generated"] == b["reference"],
            "baseline_text": a.get("generated"),
            "candidate_text": b.get("generated"),
        })
    jointly_generated = [r for r in rows if
                         r["baseline_generated"] and r["candidate_generated"]]
    n = len(rows)
    def coverage(key):
        return sum(bool(r[key]) for r in rows) / n
    def unique_ratio(key):
        values = [r[key] for r in rows if r[key] is not None]
        return len(set(values)) / len(values) if values else None
    def most_common_fraction(key):
        from collections import Counter
        values = [r[key] for r in rows if r[key] is not None]
        return Counter(values).most_common(1)[0][1] / len(values) if values else None
    mean_delta = (sum(r["candidate_overlap"] - r["baseline_overlap"]
                      for r in jointly_generated) / len(jointly_generated)
                  if jointly_generated else None)
    return {
        "status": "PAIRED_CORPUS_DEV_DIAGNOSTIC_NOT_INDEPENDENT",
        "cases": n,
        "jointly_generated_cases": len(jointly_generated),
        "baseline_coverage": coverage("baseline_generated"),
        "candidate_coverage": coverage("candidate_generated"),
        "baseline_exact_all_cases": coverage("baseline_exact"),
        "candidate_exact_all_cases": coverage("candidate_exact"),
        "mean_overlap_delta_on_joint_cases": mean_delta,
        "baseline_distinct_response_ratio": unique_ratio("baseline_text"),
        "candidate_distinct_response_ratio": unique_ratio("candidate_text"),
        "baseline_most_common_fraction": most_common_fraction("baseline_text"),
        "candidate_most_common_fraction": most_common_fraction("candidate_text"),
        "warning": "Identical Dev cases are required. Dev is not an independent "
                   "new-source holdout. Overlap and diversity do not prove relevance.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="results/realpersonachat_pilot_dev.json")
    p.add_argument("--candidate", default="results/realpersonachat_pilot_10k_dev.json")
    p.add_argument("--output", default="results/realpersonachat_paired_comparison.json")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    a = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    b = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    report = compare_reports(a, b)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise ValueError("REPORT_EXISTS_USE_OVERWRITE")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Saved:", output)


if __name__ == "__main__":
    main()
