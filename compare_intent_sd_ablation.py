"""Paired Intent SD ablation scorer (Dev only; no predictions are generated).

Input is a locally produced JSON object with scope=DEV and cases[].
Each item: case_id, reference_label, reference_type (AI_PROXY or HUMAN),
baseline_prediction, intent_prediction, intent_sd (dict or null).
Both arms share the SAME case and target. No training or Test access.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from intent_sd import IntentSD

LABELS = {"A", "B", "BOTH", "NEITHER", "UNCLEAR"}

def macro_f1(y_true, y_pred):
    classes = sorted(set(y_true) | set(y_pred))
    if not classes:
        return None
    scores = []
    for label in classes:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return sum(scores) / len(scores)

def evaluate(dataset):
    if dataset.get("scope") != "DEV_NOT_INDEPENDENT":
        raise ValueError("ONLY_DEVELOPMENT_DATA_ALLOWED")
    rows = dataset.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("EMPTY_CASES")
    if len({x["case_id"] for x in rows}) != len(rows):
        raise ValueError("DUPLICATE_CASE_ID")
    types = {x.get("reference_type") for x in rows}
    if len(types) != 1 or not types <= {"AI_PROXY", "HUMAN"}:
        raise ValueError("MIXED_OR_INVALID_REFERENCE_TYPE")
    expected = dataset.get("expected_case_count")
    if expected is not None and (not isinstance(expected, int) or expected != len(rows)):
        raise ValueError("COHORT_SIZE_MISMATCH")
    true, baseline, candidate = [], [], []
    for row in rows:
        for field in ("reference_label", "baseline_prediction", "intent_prediction"):
            if row.get(field) not in LABELS:
                raise ValueError("INVALID_LABEL:" + field)
        if row.get("intent_sd") is None:
            raise ValueError("MISSING_INTENT_SD_CANDIDATE")
        sd = IntentSD.from_dict(row["intent_sd"])
        if sd.provenance == "HUMAN_ANNOTATED" and types == {"AI_PROXY"}:
            raise ValueError("MISLEADING_HUMAN_PROVENANCE")
        true.append(row["reference_label"])
        baseline.append(row["baseline_prediction"])
        candidate.append(row["intent_prediction"])
    n = len(rows)
    bc = [a == b for a, b in zip(true, baseline)]
    cc = [a == b for a, b in zip(true, candidate)]
    return {
        "status": "PAIRED_DEV_ABLATION_NOT_INDEPENDENT",
        "reference_type": next(iter(types)),
        "cases": n,
        "baseline_accuracy": sum(bc) / n,
        "intent_sd_accuracy": sum(cc) / n,
        "accuracy_delta": (sum(cc) - sum(bc)) / n,
        "baseline_macro_f1": macro_f1(true, baseline),
        "intent_sd_macro_f1": macro_f1(true, candidate),
        "baseline_only_correct": sum(a and not b for a, b in zip(bc, cc)),
        "intent_sd_only_correct": sum(b and not a for a, b in zip(bc, cc)),
        "both_correct": sum(a and b for a, b in zip(bc, cc)),
        "neither_correct": sum(not a and not b for a, b in zip(bc, cc)),
        "reference_counts": dict(sorted(Counter(true).items())),
        "warning": "This scorer cannot establish causal benefit by itself. Use frozen arms, no label leakage, independent human labels and unseen holdout for confirmatory claims.",
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="results/intent_sd_ablation_dev.json")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    result = evaluate(json.loads(Path(a.input).read_text(encoding="utf-8")))
    path = Path(a.output)
    if path.exists() and not a.overwrite:
        raise ValueError("REPORT_EXISTS_USE_OVERWRITE")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
