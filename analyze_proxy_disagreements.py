"""Prepare a private, reviewable mismatch taxonomy for AI-proxy Dev comparisons.

No automatic error attribution: a disagreement is NOT proof the model is wrong.
The review packet contains IDs and model/proxy directional preferences only,
not raw dialogue text. Annotators can consult the existing local private pack.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from compare_realpersonachat_ai_proxy import concordance, index_cases

CATEGORIES = ("LEXICAL", "TOPIC", "SPEECH_ACT", "CONTEXT", "POLARITY", "OTHER")

def prepare(proxy, baseline, candidate):
    summary = concordance(proxy, baseline, candidate)  # strict content / ID validation
    left = index_cases(baseline, proxy["seed"])
    right = index_cases(candidate, proxy["seed"])
    rows = []
    for case in proxy["cases"]:
        label = case["ai_proxy_review"]["label"]
        if label not in ("A", "B"):
            continue
        cid = case["case_id"]
        a = left[cid]
        b = right[cid]
        original_is_a = case["context_a"] == a["original_prompt"]
        proxy_original = (label == "A") == original_is_a
        before = a["original_minus_shuffled"] > 0
        after = b["original_minus_shuffled"] > 0
        if before == proxy_original and after == proxy_original:
            continue
        rows.append({
            "case_id": cid, "proxy_label": label,
            "baseline_matches_proxy": before == proxy_original,
            "candidate_matches_proxy": after == proxy_original,
            "primary_cause": None, "secondary_causes": [],
            "annotation_status": "PENDING_HUMAN_REVIEW",
            "rationale": "",
        })
    return {
        "status": "DEV_AI_PROXY_MISMATCH_REVIEW_PENDING",
        "comparison": summary,
        "categories": list(CATEGORIES),
        "mismatch_cases": rows,
        "mismatch_count": len(rows),
        "warning": "AI proxy is not human ground truth. Causes must be assigned manually. Do not commit private case text.",
    }

def audit(packet):
    seen = set()
    counts = Counter()
    pending = 0
    for case in packet["mismatch_cases"]:
        cid = case["case_id"]
        if cid in seen:
            raise ValueError("DUPLICATE_CASE_ID")
        seen.add(cid)
        primary = case.get("primary_cause")
        secondary = case.get("secondary_causes", [])
        if primary is None:
            if secondary or case.get("rationale", "").strip():
                raise ValueError("INCOMPLETE_ANNOTATION")
            pending += 1
            continue
        if primary not in CATEGORIES or not isinstance(secondary, list):
            raise ValueError("INVALID_CAUSE")
        if len(secondary) != len(set(secondary)) or primary in secondary or any(x not in CATEGORIES for x in secondary):
            raise ValueError("INVALID_SECONDARY_CAUSES")
        if not case.get("rationale", "").strip():
            raise ValueError("RATIONALE_REQUIRED")
        counts[primary] += 1
    return {"status": "HUMAN_CAUSE_REVIEW_PENDING" if pending else "CAUSE_ANNOTATION_COMPLETE",
            "total": len(seen), "pending": pending, "primary_counts": dict(sorted(counts.items())),
            "warning": "Cause labels reflect annotation, not model correctness."}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("prepare", "audit"))
    p.add_argument("--proxy", default="results/realpersonachat_semantic_review_ai_proxy.json")
    p.add_argument("--baseline", default="results/realpersonachat_fullshuffle_10k_dev.json")
    p.add_argument("--candidate", default="results/realpersonachat_fullshuffle_30k_dev.json")
    p.add_argument("--packet", default="results/intent_sd_mismatch_review_dev.json")
    p.add_argument("--output", default="results/intent_sd_mismatch_audit_dev.json")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.mode == "prepare":
        load = lambda s: json.loads(Path(s).read_text(encoding="utf-8"))
        obj = prepare(load(args.proxy), load(args.baseline), load(args.candidate))
        target = Path(args.packet)
    else:
        obj = audit(json.loads(Path(args.packet).read_text(encoding="utf-8")))
        target = Path(args.output)
    if target.exists() and not args.overwrite:
        raise ValueError("REPORT_EXISTS_USE_OVERWRITE")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in obj.items() if k != "mismatch_cases"}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
