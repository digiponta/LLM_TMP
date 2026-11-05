"""Evaluate local RealPersonaChat multi-turn pilot generation on Dev only.

Use frozen model and frozen selected Dev examples from the training metadata.
Generates actual next utterances; records exact agreement and character
similarity against reference and compares copy-last-turn baseline. Automatic
metrics are diagnostics, NOT evidence that answers are natural or helpful.
Output contains original conversation snippets: keep report local/private.
Never touches Test JSONL, nor trains or alters checkpoint weights.
"""
from __future__ import annotations

import argparse
import difflib
import json
import statistics
from pathlib import Path

from tokenizer import Tokenizer
from train_realpersonachat_pilot import get_device, prompt_text, read_sets


def overlap_ratio(generated, reference):
    return difflib.SequenceMatcher(None, generated, reference).ratio()


def copy_previous_baseline(example):
    return example["context"][-1]["text"]


def decode_response(model, tokenizer, prompt, device, max_new_tokens=50):
    import torch
    ids = tokenizer.encode(prompt, add_bos=True)
    if tokenizer.unk_id in ids:
        return None, "UNKNOWN_PROMPT_CHARACTERS"
    if len(ids) >= model.context_length:
        return None, "PROMPT_EXCEEDS_CONTEXT"
    output = []
    budget = min(max_new_tokens, model.context_length - len(ids))
    model.eval()
    with torch.no_grad():
        for _ in range(budget):
            logits = model(torch.tensor([ids + output],
                                        device=device, dtype=torch.long))[0, -1]
            token = int(logits.argmax().item())
            if token in (tokenizer.eos_id, tokenizer.pad_id):
                break
            if tokenizer.id_to_token[token] == "\n":
                break
            output.append(token)
    return tokenizer.decode(output), None


def calculate_report(rows):
    completed = [r for r in rows if r["status"] == "GENERATED"]
    total = len(rows)
    if not total:
        raise ValueError("NO_SELECTED_ROWS")
    report = {
        "total_selected": total,
        "generated": len(completed),
        "abstained": total - len(completed),
        "abstention_reasons": {
            reason: sum(r.get("reason") == reason for r in rows)
            for reason in sorted({r.get("reason") for r in rows
                                  if r.get("reason")})
        },
        "exact_match_rate_on_generated": (
            sum(r["generated"] == r["reference"] for r in completed)
            / len(completed) if completed else None),
        "empty_generation_rate_on_generated": (
            sum(not r["generated"] for r in completed)
            / len(completed) if completed else None),
        "mean_character_overlap": (
            statistics.mean(r["generated_overlap"] for r in completed)
            if completed else None),
        "copy_previous_mean_character_overlap": (
            statistics.mean(r["copy_previous_overlap"] for r in completed)
            if completed else None),
        "mean_generated_characters": (
            statistics.mean(len(r["generated"]) for r in completed)
            if completed else None),
        "rows": rows,
    }
    return report


def evaluate(args):
    from model import LanguageModel

    outdir = Path(args.outdir)
    meta = json.loads((outdir / "training.json").read_text(encoding="utf-8"))
    if meta.get("status") != "CORPUS_DEV_ONLY_NOT_INDEPENDENT":
        raise ValueError("UNRECOGNIZED_CHECKPOINT_METADATA")
    config = meta["selection"]
    input_dir = args.input_dir or meta["input_dir"]
    # Never load Test data. Use frozen selection from original training run.
    selected = read_sets(argparse.Namespace(
        input_dir=input_dir, seed=config["seed"],
        train_limit=config["train_limit"], dev_limit=config["dev_limit"],
        context_turns=config["context_turns"],
        max_chars=config["max_chars"]))[1]
    tok = Tokenizer.load(str(outdir / "tokenizer-chat-pilot.json"))
    device = get_device(args.device)
    model, checkpoint = LanguageModel.load_checkpoint(
        str(outdir / "model-chat-pilot.pt"), device)
    if model.context_length != config["context"]:
        raise ValueError("CONTEXT_CONFIG_MISMATCH")
    if args.count < 1 or args.count > len(selected):
        raise ValueError("INVALID_EVAL_COUNT")
    rows = []
    for item in selected[:args.count]:
        prompt = prompt_text(item, config["context_turns"])
        generated, reason = decode_response(
            model, tok, prompt, device, args.max_new_tokens)
        baseline = copy_previous_baseline(item)
        result = {
            "dialogue_id": item["dialogue_id"], "turn_id": item["turn_id"],
            "prompt": prompt, "reference": item["response"],
            "generated": generated, "copy_previous": baseline,
            "status": "ABSTAIN" if reason else "GENERATED",
            "reason": reason,
            "generated_overlap": (overlap_ratio(generated, item["response"])
                                  if reason is None else None),
            "copy_previous_overlap": (overlap_ratio(baseline, item["response"])
                                      if reason is None else None),
        }
        rows.append(result)
    report = calculate_report(rows)
    report.update({
        "status": "CORPUS_DEV_QUALITATIVE_DIAGNOSTIC",
        "model_epoch": checkpoint.get("epoch"),
        "evaluation_note": "Not a blinded or independent holdout. "
                           "Character overlap is not dialogue quality.",
    })
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not args.overwrite:
        raise ValueError("REPORT_EXISTS_USE_OVERWRITE")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"},
                     ensure_ascii=False, indent=2))
    print("Local report (contains conversation text):", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default=None)
    p.add_argument("--outdir", default="model/realpersonachat-pilot")
    p.add_argument("--output", default="results/realpersonachat_pilot_dev.json")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--overwrite", action="store_true")
    evaluate(p.parse_args())


if __name__ == "__main__":
    main()
