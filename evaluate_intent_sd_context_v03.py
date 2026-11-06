"""CLI for frozen, paired, development-only v0.3 Linked-SD evaluation.

Source utterances and human labels remain local. Input supplies explicit SD
refs and independently frozen model choices; no NLP extraction or training.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from intent_sd_context_v03 import (
    SDRegistry, classify_topic_transition, choose_context_v03
)

LABELS = frozenset(("A", "B", "BOTH", "NEITHER", "UNCLEAR"))
DECLARED_INPUTS = frozenset(("case_id", "reference_label", "baseline_prediction",
                             "intent_a", "intent_b", "state_a", "state_b"))

def run(dataset):
    if not isinstance(dataset, dict) or dataset.get("scope") != "DEV_NOT_INDEPENDENT":
        raise ValueError("ONLY_DEV_ALLOWED")
    if dataset.get("reference_type") not in ("AI_PROXY", "HUMAN"):
        raise ValueError("INVALID_REFERENCE_TYPE")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases or type(dataset.get("expected_case_count")) is not int or len(cases) != dataset["expected_case_count"]:
        raise ValueError("COHORT_COUNT_MISMATCH")
    ids = [r.get("case_id") for r in cases]
    if any(not isinstance(cid, str) or not cid for cid in ids) or len(ids) != len(set(ids)):
        raise ValueError("INVALID_CASE_IDS")
    if dataset.get("annotation_independent_of_reference") is not True:
        raise ValueError("ANNOTATION_INDEPENDENCE_UNVERIFIED")
    registry = SDRegistry(dataset["registry_snapshot"])
    wins = Counter()
    predictions = Counter()
    details = []
    for row in cases:
        if set(row) != DECLARED_INPUTS:
            raise ValueError("UNEXPECTED_CASE_FIELDS")
        ref, baseline = row["reference_label"], row["baseline_prediction"]
        if ref not in LABELS or baseline not in LABELS:
            raise ValueError("INVALID_LABEL")
        a = classify_topic_transition(row["intent_a"], row["state_a"], registry)
        b = classify_topic_transition(row["intent_b"], row["state_b"], registry)
        result = choose_context_v03(a, b)
        predicted = result["label"]
        bc = baseline == ref
        ic = predicted == ref
        wins[(bc, ic)] += 1
        predictions[predicted] += 1
        details.append({"case_id": row["case_id"], "baseline_prediction": baseline,
                        "v03_prediction": predicted, "status_a": a["status"],
                        "status_b": b["status"]})
    n = len(cases)
    bc_count = wins[(True, True)] + wins[(True, False)]
    ic_count = wins[(True, True)] + wins[(False, True)]
    return {
        "status": "INTENT_SD_V03_DEV_ONLY_NOT_INDEPENDENT",
        "reference_type": dataset["reference_type"],
        "registry_snapshot_id": dataset["registry_snapshot"].get("snapshot_id"),
        "case_count": n, "baseline_accuracy": bc_count/n,
        "v03_accuracy": ic_count/n, "accuracy_delta": (ic_count-bc_count)/n,
        "v03_non_unclear_coverage": (n-predictions["UNCLEAR"])/n,
        "baseline_only_correct": wins[(True,False)],
        "v03_only_correct": wins[(False,True)],
        "both_correct": wins[(True,True)],
        "neither_correct": wins[(False,False)],
        "prediction_counts": dict(sorted(predictions.items())),
        "per_case": details,
        "warning": "Development-only; AI proxy is not human ground truth. Frozen topic SD references must be annotated independently of reference labels."
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="results/intent_sd_v03_dev.json")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    output = Path(a.output)
    if output.exists() and not a.overwrite:
        raise ValueError("REPORT_EXISTS_USE_OVERWRITE")
    result = run(json.loads(Path(a.input).read_text(encoding="utf-8")))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "per_case"}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
