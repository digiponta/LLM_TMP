"""Paired development comparison: greedy vs constrained Dialogue Intent LLM.

Uses the same frozen 32 family-disjoint samples and a locally saved epoch-35
checkpoint. The greedy baseline report is read, never recalculated or changed.
Constrained predictions are scored by the model, without label lookup.
This is development analysis, not an independent or blind holdout.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from constrained_dialogue_intent import score_constrained
from dialogue_intent_dataset import build_intent_examples, audit_split
from dialogue_intent_sd import DialogueIntentSD, INTENTS
from tokenizer import Tokenizer
from train_dialogue_intent import choose_device


def evaluate(args):
    from model import LanguageModel
    outdir = Path(args.outdir)
    split = json.loads((outdir / "split.json").read_text(encoding="utf-8"))
    train, dev = build_intent_examples(split["seed"])
    audit_split(train, dev)
    if split["train"] != train or split["dev"] != dev:
        raise ValueError("FROZEN_SPLIT_MISMATCH")
    old = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    old_rows = {r["text"]: r for r in old["rows"]}
    if len(dev) != len(old_rows) or set(old_rows) != {r["text"] for r in dev}:
        raise ValueError("BASELINE_CASE_MISMATCH")
    for item in dev:
        row = old_rows[item["text"]]
        if row["expected"] != item["code"] or row["intent"] != item["intent"]:
            raise ValueError("BASELINE_LABEL_MISMATCH")
    device = choose_device(args.device)
    tokenizer = Tokenizer.load(str(outdir / "tokenizer-dialogue-intent.json"))
    model, ckpt = LanguageModel.load_checkpoint(
        str(outdir / "model-dialogue-intent.pt"), device)
    rows = []
    engine = DialogueIntentSD()
    for item in dev:
        new = score_constrained(model, tokenizer, item["text"], device)
        prediction = new["code"]
        accepted = (engine.accept(item["text"], prediction)
                    if prediction is not None else
                    {"status": "REJECTED", "reason": new["reason"]})
        before = old_rows[item["text"]]
        rows.append({
            "text": item["text"], "intent": item["intent"],
            "expected": item["code"],
            "greedy": before["predicted"], "greedy_exact": before["exact"],
            "greedy_accepted": before["accepted"],
            "constrained": prediction,
            "constrained_exact": prediction == item["code"],
            "constrained_accepted": accepted["status"] == "ACCEPTED",
            "reason": accepted.get("reason"),
            "score_margin_uncalibrated": new.get("margin"),
        })

    def summarize(items):
        total = len(items)
        if total == 0:
            return {"count": 0}
        gc = sum(x["greedy_exact"] for x in items)
        cc = sum(x["constrained_exact"] for x in items)
        ca = sum(x["constrained_accepted"] for x in items)
        return {
            "count": total,
            "greedy_correct": gc,
            "constrained_correct": cc,
            "greedy_exact_rate": gc / total,
            "constrained_exact_rate": cc / total,
            "delta_percentage_points": round(100 * (cc - gc) / total, 3),
            "newly_correct": sum(not x["greedy_exact"] and
                                 x["constrained_exact"] for x in items),
            "regressions": sum(x["greedy_exact"] and
                               not x["constrained_exact"] for x in items),
            "constrained_accepted": ca,
            "constrained_wrong_but_accepted": sum(
                x["constrained_accepted"] and not x["constrained_exact"]
                for x in items),
        }

    result = {
        "status": "PAIRED_DEVELOPMENT_ONLY_FAMILY_DISJOINT",
        "model_epoch": ckpt.get("epoch"),
        "decision_method": "MEAN_LABEL_LOG_LIKELIHOOD_8_CLASSES",
        "evaluation_note": "Development labels are hidden at inference; "
                           "source test grammar and lexicon still overlap.",
        "overall": summarize(rows),
        "by_intent": {key: summarize([x for x in rows if x["intent"] == key])
                      for key in INTENTS},
        "wrong_label_confusion": {
            left + " -> " + right: count
            for (left, right), count in collections.Counter(
                (r["intent"], (r["constrained"] or "ABSTAIN"))
                for r in rows if not r["constrained_exact"]
            ).items()
        },
        "rows": rows,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows",)},
                     ensure_ascii=False, indent=2))
    print("Saved:", destination)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="model/dialogue-intent")
    p.add_argument("--baseline", default="results/dialogue_intent_dev.json")
    p.add_argument("--output", default="results/dialogue_intent_constrained.json")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
