"""Frozen-model Context Ablation: Full / Last Turn / No Context.

Teacher-forced mean log likelihood of the SAME Dev reference under three
input prompts, scored using the SAME tokenizer and frozen checkpoint.
Only cases for which all three conditions are scoreable are compared.
No training, Test access, automatic intent labels, or semantic accuracy claim.
Reports contain real conversation snippets: keep private and local.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from tokenizer import Tokenizer
from train_realpersonachat_pilot import get_device, prompt_text, read_sets
from evaluate_realpersonachat_context import score_reference

CONDITIONS = ("full", "last_turn", "no_context")


def make_prompts(item, context_turns):
    """Preserve next-speaker label, reference and formatting across conditions."""
    full = prompt_text(item, context_turns)
    last = dict(item, context=item["context"][-1:])
    none = dict(item, context=[])
    return {
        "full": full,
        "last_turn": prompt_text(last, context_turns),
        "no_context": prompt_text(none, context_turns),
    }


def summarize(rows):
    accepted = [r for r in rows if r["status"] == "SCORED"]
    n = len(accepted)
    def delta(a, b):
        return (statistics.mean(r["scores"][a] - r["scores"][b]
                                for r in accepted) if n else None)
    def better(a, b):
        return (sum(r["scores"][a] > r["scores"][b] for r in accepted) / n
                if n else None)
    return {
        "selected": len(rows),
        "all_three_scored": n,
        "abstained": len(rows) - n,
        "abstention_reasons": {
            k: sum(r.get("reason") == k for r in rows)
            for k in sorted({r["reason"] for r in rows if r.get("reason")})
        },
        "two_turn_context_cases": sum(
            r["context_turn_count"] >= 2 for r in accepted),
        "mean_log_probability": {
            name: (statistics.mean(r["scores"][name] for r in accepted)
                   if n else None) for name in CONDITIONS
        },
        "mean_delta_full_minus_none": delta("full", "no_context"),
        "mean_delta_full_minus_last": delta("full", "last_turn"),
        "mean_delta_last_minus_none": delta("last_turn", "no_context"),
        "positive_fraction_full_vs_none": better("full", "no_context"),
        "positive_fraction_full_vs_last": better("full", "last_turn"),
        "positive_fraction_last_vs_none": better("last_turn", "no_context"),
        "rows": rows,
    }


def evaluate(args):
    from model import LanguageModel

    directory = Path(args.outdir)
    metadata = json.loads((directory / "training.json").read_text(
        encoding="utf-8"))
    if metadata.get("status") != "CORPUS_DEV_ONLY_NOT_INDEPENDENT":
        raise ValueError("INVALID_CHECKPOINT_METADATA")
    cfg = metadata["selection"]
    if cfg["context_turns"] < 2:
        raise ValueError("MODEL_TRAINED_WITH_LESS_THAN_TWO_CONTEXT_TURNS")

    _, dev = read_sets(argparse.Namespace(
        input_dir=args.input_dir or metadata["input_dir"],
        train_limit=cfg["train_limit"], dev_limit=cfg["dev_limit"],
        context_turns=cfg["context_turns"], max_chars=cfg["max_chars"],
        seed=cfg["seed"]))
    if not 1 <= args.count <= len(dev):
        raise ValueError("INVALID_CASE_COUNT")
    device = get_device(args.device)
    tok = Tokenizer.load(str(directory / "tokenizer-chat-pilot.json"))
    model, checkpoint = LanguageModel.load_checkpoint(
        str(directory / "model-chat-pilot.pt"), device)
    if model.context_length != cfg["context"]:
        raise ValueError("CONTEXT_CONFIG_MISMATCH")

    rows = []
    for item in dev[:args.count]:
        prompts = make_prompts(item, cfg["context_turns"])
        scored = {
            name: score_reference(model, tok, prompts[name],
                                  item["response"], device)
            for name in CONDITIONS
        }
        errors = [name + ":" + scored[name][1] for name in CONDITIONS
                  if scored[name][1]]
        rows.append({
            "dialogue_id": item["dialogue_id"], "turn_id": item["turn_id"],
            "context_turn_count": len(item["context"][-cfg["context_turns"]:]),
            "reference": item["response"],
            "prompts": prompts,
            "status": "ABSTAIN" if errors else "SCORED",
            "reason": ";".join(errors) if errors else None,
            "scores": ({name: scored[name][0] for name in CONDITIONS}
                       if not errors else None),
        })
    report = summarize(rows)
    report.update({
        "status": "CORPUS_DEV_CONTEXT_ABLATION_NOT_INDEPENDENT",
        "model_epoch": checkpoint.get("epoch"),
        "method": "SAME_REFERENCE_TEACHER_FORCED_MEAN_LOG_PROBABILITY",
        "conditions": list(CONDITIONS),
        "warning": "Shortened prompts are out-of-training-distribution. "
                   "Higher likelihood is not proof of semantic understanding. "
                   "No independent Test used. Report includes private text.",
    })
    dest = Path(args.output)
    if dest.exists() and not args.overwrite:
        raise ValueError("REPORT_EXISTS_USE_OVERWRITE")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"},
                     ensure_ascii=False, indent=2))
    print("Saved private local report:", dest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="model/realpersonachat-pilot-30k")
    p.add_argument("--input-dir", default=None)
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"),
                   default="auto")
    p.add_argument("--output",
                   default="results/realpersonachat_ablation_30k_dev.json")
    p.add_argument("--overwrite", action="store_true")
    evaluate(p.parse_args())


if __name__ == "__main__":
    main()
