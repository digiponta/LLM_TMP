"""Paired 10k/30k Full-Length Context Shuffle comparison (Dev only).

Checks same cases, donor, reference, both prompts and context turn count.
Computes the two model's original-vs-shuffle differences on exactly the
jointly scored cohort. Does not claim cross-tokenizer log probabilities
are directly comparable or that artificial donors are semantically wrong.
No conversation text exported in summary; Test is never read.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def key(row):
    return str(row["dialogue_id"]), str(row["turn_id"])


def indexed(report):
    if report.get("status") != "CORPUS_DEV_FULL_CONTEXT_SHUFFLE_NOT_INDEPENDENT":
        raise ValueError("INVALID_REPORT_STATUS")
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("MISSING_ROWS")
    records = {key(r): r for r in rows}
    if len(records) != len(rows):
        raise ValueError("DUPLICATE_CASES")
    return records


def bootstrap_ci(differences, seed=42, repeats=2000):
    """Percentile resampling is descriptive; examples may be correlated."""
    if not differences:
        return None
    rng = random.Random(seed)
    n = len(differences)
    means = []
    for _ in range(repeats):
        means.append(sum(differences[rng.randrange(n)]
                         for _ in range(n)) / n)
    means.sort()
    return [means[int(.025 * repeats)], means[int(.975 * repeats)]]


def compare(a, b):
    left, right = indexed(a), indexed(b)
    if set(left) != set(right):
        raise ValueError("MISMATCHED_EVALUATION_CASES")
    examined, paired = 0, []
    for identity in sorted(left):
        x, y = left[identity], right[identity]
        for field in ("reference", "original_prompt", "shuffled_prompt",
                      "donor_dialogue_id", "context_turn_count"):
            if x.get(field) != y.get(field):
                raise ValueError("MISMATCHED_PAIR_CONTENT:" + field)
        examined += 1
        if x["status"] != "SCORED" or y["status"] != "SCORED":
            continue
        paired.append((float(x["original_minus_shuffled"]),
                       float(y["original_minus_shuffled"])))
    if not paired:
        raise ValueError("NO_JOINTLY_SCORED_CASES")
    gains = [b - a for a, b in paired]
    base_win = [a > 0 for a, _ in paired]
    cand_win = [b > 0 for _, b in paired]
    return {
        "status": "PAIRED_DEV_FULLSHUFFLE_DIAGNOSTIC_NOT_INDEPENDENT",
        "common_selected": examined,
        "jointly_scored": len(paired),
        "baseline_only_scored": sum(
            left[k]["status"] == "SCORED" and right[k]["status"] != "SCORED"
            for k in left),
        "candidate_only_scored": sum(
            left[k]["status"] != "SCORED" and right[k]["status"] == "SCORED"
            for k in left),
        "baseline_positive_fraction_common": sum(base_win) / len(paired),
        "candidate_positive_fraction_common": sum(cand_win) / len(paired),
        "baseline_mean_delta_common": statistics.mean(a for a, _ in paired),
        "candidate_mean_delta_common": statistics.mean(b for _, b in paired),
        "candidate_minus_baseline_mean_delta": statistics.mean(gains),
        "bootstrap_95pct_interval_mean_delta_difference":
            bootstrap_ci(gains),
        "casewise_delta_gain_fraction": sum(g > 0 for g in gains) / len(gains),
        "baseline_negative_to_candidate_positive": sum(
            not x and y for x, y in zip(base_win, cand_win)),
        "baseline_positive_to_candidate_negative": sum(
            x and not y for x, y in zip(base_win, cand_win)),
        "warning": "This paired corpus-Dev diagnostic is not independent. "
                   "Per-model score deltas are measured under different "
                   "Train-only vocabularies; effect-size comparison is "
                   "descriptive, not a controlled same-tokenizer test. "
                   "Bootstrap CI assumes exchangeable cases and may be "
                   "too narrow if dialogues are correlated.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline",
                   default="results/realpersonachat_fullshuffle_10k_dev.json")
    p.add_argument("--candidate",
                   default="results/realpersonachat_fullshuffle_30k_dev.json")
    p.add_argument("--output",
                   default="results/realpersonachat_fullshuffle_paired.json")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    a = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    b = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    report = compare(a, b)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise ValueError("REPORT_EXISTS_USE_OVERWRITE")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Saved aggregate-only report:", output)


if __name__ == "__main__":
    main()
