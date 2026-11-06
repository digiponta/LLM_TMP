"""Paired, development-only comparison using explicit Context SD v0.2.

Requires same case IDs and reference for both arms. Predictions and topic
tags must be frozen independently of labels; this scorer does not learn tags.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from intent_sd_context_v02 import ContextSD, choose_context

LABELS = frozenset(("A", "B", "BOTH", "NEITHER", "UNCLEAR"))

def evaluate(dataset):
    if dataset.get("scope") != "DEV_NOT_INDEPENDENT":
        raise ValueError("ONLY_DEV_ALLOWED")
    rows = dataset.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("NO_CASES")
    if dataset.get("expected_case_count") != len(rows):
        raise ValueError("COHORT_COUNT_MISMATCH")
    ids = [x.get("case_id") for x in rows]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("INVALID_CASE_IDS")
    reference_type = dataset.get("reference_type")
    if reference_type not in ("AI_PROXY", "HUMAN"):
        raise ValueError("INVALID_REFERENCE_TYPE")

    correct_base = correct_intent = covered = 0
    outcomes = Counter()
    prediction_counts = Counter()
    details = []
    for row in rows:
        ref, baseline = row.get("reference_label"), row.get("baseline_prediction")
        if ref not in LABELS or baseline not in LABELS:
            raise ValueError("INVALID_LABEL")
        if row.get("context_a") is None or row.get("context_b") is None:
            raise ValueError("MISSING_CONTEXT_SD")
        a, b = ContextSD.from_dict(row["context_a"]), ContextSD.from_dict(row["context_b"])
        if (a.provenance == "HUMAN_ANNOTATED" or b.provenance == "HUMAN_ANNOTATED"):
            if dataset.get("annotation_independent_of_reference") is not True:
                raise ValueError("ANNOTATION_INDEPENDENCE_UNVERIFIED")
        choice = choose_context(a, b)
        intent = choice["label"]
        bc, ic = baseline == ref, intent == ref
        correct_base += bc
        correct_intent += ic
        covered += intent != "UNCLEAR"
        outcomes[(bc, ic)] += 1
        prediction_counts[intent] += 1
        details.append({"case_id": row["case_id"], "baseline_prediction": baseline,
                        "intent_prediction": intent, "reference_label": ref,
                        "consistency_a": choice["a"], "consistency_b": choice["b"]})
    n = len(rows)
    return {
        "status": "INTENT_SD_V02_DEV_ONLY_NOT_GENERALIZATION",
        "reference_type": reference_type, "cases": n,
        "baseline_accuracy": correct_base / n,
        "intent_accuracy": correct_intent / n,
        "accuracy_difference": (correct_intent - correct_base) / n,
        "intent_coverage_non_unclear": covered / n,
        "prediction_counts": dict(sorted(prediction_counts.items())),
        "baseline_only_correct": outcomes[(True, False)],
        "intent_only_correct": outcomes[(False, True)],
        "both_correct": outcomes[(True, True)],
        "neither_correct": outcomes[(False, False)],
        "per_case": details,
        "warning": "Explicit topic tags are inputs, not NLP inference. AI proxy is not human gold. Dev results do not demonstrate independent generalization."
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="results/intent_sd_context_v02_dev.json")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    result = evaluate(json.loads(Path(args.input).read_text(encoding="utf-8")))
    dest = Path(args.output)
    if dest.exists() and not args.overwrite:
        raise ValueError("REPORT_EXISTS_USE_OVERWRITE")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k,v in result.items() if k != "per_case"}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
