"""Create and audit blinded human review cases from *existing local Dev reports*.

This is a development-review workflow, NOT an independent holdout.
The review packet contains potentially private dialogue snippets. It must
stay local and MUST NOT be committed. The preparer never inserts model scores
or predictions into the blinded review packet. All judgments start PENDING.

Labels describe which context better supports the SAME reference response:
A, B, BOTH, NEITHER, UNCLEAR. Reviewers fill exactly one label per case
and a short rationale. A/B positions are deterministic but randomized per
case; no automatic semantic labeling is done.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from compare_realpersonachat_fullshuffle import indexed, key

LABELS = {"A", "B", "BOTH", "NEITHER", "UNCLEAR"}
BUCKETS = ("RECOVERED", "REGRESSED", "STABLE_OR_MIXED")


def rank_case(identity, seed):
    key_string = str(identity[0]) + ":" + str(identity[1])
    return hashlib.sha256((str(seed) + ":" + key_string).encode()).hexdigest()


def classify(x, y):
    if x <= 0 < y:
        return "RECOVERED"
    if y <= 0 < x:
        return "REGRESSED"
    return "STABLE_OR_MIXED"


def build_review_pack(baseline, candidate, seed=42, per_bucket=15):
    if per_bucket < 1:
        raise ValueError("INVALID_PER_BUCKET")
    left, right = indexed(baseline), indexed(candidate)
    if set(left) != set(right):
        raise ValueError("MISMATCHED_CASES")
    bucketed = {bucket: [] for bucket in BUCKETS}
    for identity in left:
        a, b = left[identity], right[identity]
        for field in ("original_prompt", "shuffled_prompt", "reference",
                      "donor_dialogue_id", "context_turn_count"):
            if a.get(field) != b.get(field):
                raise ValueError("MISMATCHED_CASE_CONTENT:" + field)
        if a["status"] != "SCORED" or b["status"] != "SCORED":
            continue
        class_name = classify(a["original_minus_shuffled"],
                              b["original_minus_shuffled"])
        bucketed[class_name].append((identity, a))
    cases = []
    allocation = {}
    for bucket in BUCKETS:
        chosen = sorted(bucketed[bucket],
                        key=lambda item: rank_case(item[0], seed))[:per_bucket]
        allocation[bucket] = {"available": len(bucketed[bucket]),
                              "selected": len(chosen)}
        for identity, source in chosen:
            original_is_a = int(rank_case(identity, seed + 1), 16) % 2 == 0
            original = source["original_prompt"]
            shuffled = source["shuffled_prompt"]
            cases.append({
                "case_id": rank_case(identity, seed)[:16],
                "context_a": original if original_is_a else shuffled,
                "context_b": shuffled if original_is_a else original,
                "reference_response": source["reference"],
                "review": {
                    "label": None, "rationale": "", "reviewer_id": "",
                },
            })
    cases.sort(key=lambda case: case["case_id"])
    return {
        "status": "HUMAN_REVIEW_PENDING",
        "scope": "REALPERSONACHAT_DEVELOPMENT_DEV_NOT_INDEPENDENT",
        "protocol": "Which context better supports the fixed response?",
        "labels": {
            "A": "Only context A is clearly more suitable",
            "B": "Only context B is clearly more suitable",
            "BOTH": "Both contexts reasonably support the response",
            "NEITHER": "Neither context reasonably supports the response",
            "UNCLEAR": "Insufficient information to decide",
        },
        "blind": "No model scores, case bucket or gold side is provided.",
        "seed": seed,
        "selection": {"per_bucket": per_bucket, "available_and_selected":
                      allocation},
        "cases": cases,
        "warning": "Contains potentially private conversation excerpts. "
                   "Do not publish or commit. Labels are human judgments; "
                   "a model cannot substitute for the reviewer. "
                   "No independent Test is accessed.",
    }


def audit_reviews(pack):
    if pack.get("scope") != "REALPERSONACHAT_DEVELOPMENT_DEV_NOT_INDEPENDENT":
        raise ValueError("INVALID_REVIEW_SCOPE")
    cases = pack.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("NO_REVIEW_CASES")
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("DUPLICATE_REVIEW_CASE_ID")
    counts = Counter()
    problems = []
    for case in cases:
        cid = case["case_id"]
        review = case.get("review")
        if not isinstance(review, dict):
            problems.append(cid + ":MISSING_REVIEW")
            continue
        label = review.get("label")
        if label is None:
            counts["PENDING"] += 1
            continue
        if label not in LABELS:
            problems.append(cid + ":INVALID_LABEL")
            continue
        rationale = review.get("rationale", "")
        reviewer = review.get("reviewer_id", "")
        if not isinstance(rationale, str) or not rationale.strip():
            problems.append(cid + ":RATIONALE_REQUIRED")
            continue
        if not isinstance(reviewer, str) or not reviewer.strip():
            problems.append(cid + ":REVIEWER_REQUIRED")
            continue
        counts[label] += 1
    return {
        "status": ("INVALID" if problems else
                   "HUMAN_REVIEW_PENDING" if counts["PENDING"] else
                   "HUMAN_REVIEW_COMPLETE"),
        "total": len(cases),
        "human_approved": sum(counts[v] for v in LABELS),
        "pending": counts["PENDING"],
        "label_counts": {v: counts[v] for v in sorted(LABELS)},
        "problems": problems,
        "warning": "This is development review, NOT an independent "
                   "generalization metric. Unreviewed cases never become gold.",
    }


def write_new(path, payload, overwrite):
    dest = Path(path)
    if dest.exists() and not overwrite:
        raise ValueError("REPORT_EXISTS_USE_OVERWRITE")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return dest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "audit"))
    p.add_argument("--baseline",
                   default="results/realpersonachat_fullshuffle_10k_dev.json")
    p.add_argument("--candidate",
                   default="results/realpersonachat_fullshuffle_30k_dev.json")
    p.add_argument("--review-pack",
                   default="results/realpersonachat_semantic_review_dev.json")
    p.add_argument("--audit-output",
                   default="results/realpersonachat_semantic_review_audit.json")
    p.add_argument("--per-bucket", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.action == "prepare":
        left = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        right = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
        pack = build_review_pack(left, right, args.seed, args.per_bucket)
        destination = write_new(args.review_pack, pack, args.overwrite)
        print(json.dumps({"status": pack["status"],
                          "selected": len(pack["cases"]),
                          "allocation": pack["selection"]["available_and_selected"]},
                         ensure_ascii=False, indent=2))
        print("Private blinded review pack:", destination)
    else:
        pack = json.loads(Path(args.review_pack).read_text(encoding="utf-8"))
        report = audit_reviews(pack)
        destination = write_new(args.audit_output, report, args.overwrite)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("Audit summary:", destination)


if __name__ == "__main__":
    main()
