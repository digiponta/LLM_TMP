"""Compare frozen 10k/30k shuffle preferences with AI-proxy judgments.

A/B proxy labels are *not human ground truth*. Only exclusive A/B proxy
judgments contribute to concordance; BOTH/NEITHER/UNCLEAR are counted
separately. Review order is recovered by exact prompt/reference matching,
not inferred from model score or hidden source-side knowledge.
Results are development-only and stratified by model behavior. No Test.
The output is aggregate-only; raw private conversation text stays local.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from compare_realpersonachat_fullshuffle import indexed
from review_realpersonachat_semantics import rank_case

PROXY_LABELS = {"A", "B", "BOTH", "NEITHER", "UNCLEAR"}


def index_cases(report, seed):
    rows = indexed(report)
    lookup = {}
    for identity, row in rows.items():
        cid = rank_case(identity, seed)[:16]
        if cid in lookup:
            raise ValueError("CASE_ID_HASH_COLLISION")
        lookup[cid] = row
    return lookup


def concordance(proxy, baseline, candidate):
    if (proxy.get("scope") != "REALPERSONACHAT_DEVELOPMENT_DEV_NOT_INDEPENDENT"
            or proxy.get("ai_proxy_status")
            != "AI_PROXY_REVIEW_COMPLETE_HUMAN_REVIEW_PENDING"):
        raise ValueError("INVALID_AI_PROXY_REVIEW_STATUS")
    cases = proxy.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("MISSING_PROXY_CASES")
    ids = [c["case_id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("DUPLICATE_PROXY_IDS")
    left = index_cases(baseline, proxy["seed"])
    right = index_cases(candidate, proxy["seed"])
    counts = Counter()
    model_ok = {"baseline": 0, "candidate": 0}
    model_positive = {"baseline": 0, "candidate": 0}
    changed = Counter()
    reviewed = 0
    for case in cases:
        cid = case["case_id"]
        if cid not in left or cid not in right:
            raise ValueError("MISSING_REPORT_CASE:" + cid)
        a, b = left[cid], right[cid]
        if a["status"] != "SCORED" or b["status"] != "SCORED":
            raise ValueError("SELECTED_CASE_NOT_JOINTLY_SCORED")
        for field in ("reference", "original_prompt", "shuffled_prompt",
                      "donor_dialogue_id", "context_turn_count"):
            if a.get(field) != b.get(field):
                raise ValueError("MODEL_REPORT_PAIR_MISMATCH:" + field)
        if case["reference_response"] != a["reference"]:
            raise ValueError("REFERENCE_MISMATCH")
        a_is_original = (case["context_a"] == a["original_prompt"]
                         and case["context_b"] == a["shuffled_prompt"])
        b_is_original = (case["context_b"] == a["original_prompt"]
                         and case["context_a"] == a["shuffled_prompt"])
        if a_is_original == b_is_original:
            raise ValueError("REVIEW_CONTEXT_MISMATCH_OR_AMBIGUOUS")
        review = case.get("ai_proxy_review")
        if (not isinstance(review, dict)
                or review.get("reviewer_type") != "AI"
                or review.get("human_verified") is not False
                or review.get("label") not in PROXY_LABELS
                or not str(review.get("rationale", "")).strip()):
            raise ValueError("INVALID_AI_PROXY_LABEL")
        if case.get("review", {}).get("label") is not None:
            raise ValueError("HUMAN_REVIEW_MUST_REMAIN_SEPARATE")
        label = review["label"]
        counts[label] += 1
        reviewed += 1
        if label not in ("A", "B"):
            continue
        proxy_original = label == ("A" if a_is_original else "B")
        before = a["original_minus_shuffled"] > 0
        after = b["original_minus_shuffled"] > 0
        for name, model_original in (("baseline", before),
                                     ("candidate", after)):
            model_ok[name] += (model_original == proxy_original)
            model_positive[name] += model_original
        changed[("baseline_match" if before == proxy_original
                 else "baseline_mismatch",
                 "candidate_match" if after == proxy_original
                 else "candidate_mismatch")] += 1
    exclusive = counts["A"] + counts["B"]
    return {
        "status": "AI_PROXY_DEV_CONCORDANCE_NOT_HUMAN_VALIDATED",
        "reviewed": reviewed,
        "exclusive_proxy_label_cases": exclusive,
        "excluded_nonexclusive": reviewed - exclusive,
        "proxy_label_counts": {k: counts[k] for k in sorted(PROXY_LABELS)},
        "baseline_concordant": model_ok["baseline"],
        "candidate_concordant": model_ok["candidate"],
        "baseline_concordance_on_exclusive": (
            model_ok["baseline"] / exclusive if exclusive else None),
        "candidate_concordance_on_exclusive": (
            model_ok["candidate"] / exclusive if exclusive else None),
        "candidate_minus_baseline_concordance": (
            (model_ok["candidate"] - model_ok["baseline"]) / exclusive
            if exclusive else None),
        "baseline_match_to_candidate_mismatch": changed[
            ("baseline_match", "candidate_mismatch")],
        "baseline_mismatch_to_candidate_match": changed[
            ("baseline_mismatch", "candidate_match")],
        "proxy_preferred_original_count": None,
        "warning": "Proxy labels were assigned by an AI model and are not "
                   "human gold. This selected Dev review pack is stratified "
                   "by 10k/30k model behavior and not representative. "
                   "Concordance is descriptive only. Models may differ in "
                   "token vocabularies. Independent human review and a "
                   "pre-registered held-out evaluation are required.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--proxy",
                   default="results/realpersonachat_semantic_review_ai_proxy.json")
    p.add_argument("--baseline",
                   default="results/realpersonachat_fullshuffle_10k_dev.json")
    p.add_argument("--candidate",
                   default="results/realpersonachat_fullshuffle_30k_dev.json")
    p.add_argument("--output",
                   default="results/realpersonachat_ai_proxy_concordance.json")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    proxy = json.loads(Path(args.proxy).read_text(encoding="utf-8"))
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    result = concordance(proxy, baseline, candidate)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise ValueError("REPORT_EXISTS_USE_OVERWRITE")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Saved aggregate-only report:", output)


if __name__ == "__main__":
    main()
