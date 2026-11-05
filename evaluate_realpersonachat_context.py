"""Counterfactual context-sensitivity diagnostic for local dialogue pilot.

For a frozen Dev next-utterance target, compare mean token log-likelihood
under its true two-turn prompt vs a counterfactual prompt whose final
context utterance comes from a DIFFERENT Dev dialogue. Speaker roles,
turn count and target are kept fixed. The alternative is artificial and
NOT a gold-labelled "wrong" context. Report text stays local/private.
Only Dev is read; Test stays unopened. No training or changes to weights.

A positive score delta means the reference is more likely under its
original context. This is NOT a proof of dialogue understanding.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tokenizer import Tokenizer
from train_realpersonachat_pilot import get_device, prompt_text, read_sets


def make_counterfactual(row, donor):
    """Replace only last context text, preserving its speaker and next speaker."""
    if row["dialogue_id"] == donor["dialogue_id"]:
        raise ValueError("DONOR_SAME_DIALOGUE")
    changed = dict(row)
    changed["context"] = [dict(turn) for turn in row["context"]]
    changed["context"][-1]["text"] = donor["context"][-1]["text"]
    return changed


def choose_donor(rows, index):
    """Deterministic donor from another conversation; no target label used."""
    source = rows[index]
    for offset in range(1, len(rows)):
        other = rows[(index + offset) % len(rows)]
        if other["dialogue_id"] != source["dialogue_id"]:
            return other
    raise ValueError("NO_OTHER_DIALOGUE_FOR_DONOR")


def score_reference(model, tokenizer, prompt, reference, device):
    import torch
    import torch.nn.functional as F
    prefix = tokenizer.encode(prompt, add_bos=True)
    suffix = tokenizer.encode(reference + "\n", add_eos=True)
    if tokenizer.unk_id in prefix:
        return None, "UNKNOWN_PROMPT_CHARACTERS"
    if tokenizer.unk_id in suffix:
        return None, "UNKNOWN_REFERENCE_CHARACTERS"
    full = prefix + suffix
    if len(full) - 1 > model.context_length:
        return None, "CONTEXT_EXCEEDED"
    x = torch.tensor([full[:-1]], device=device, dtype=torch.long)
    target = torch.tensor(suffix, device=device, dtype=torch.long)
    start = len(prefix) - 1
    model.eval()
    with torch.no_grad():
        logits = model(x)[0, start:start + len(suffix)]
        logprobs = F.log_softmax(logits, dim=-1)
        mean = float(logprobs.gather(
            1, target[:, None]).mean().item())
    if not math.isfinite(mean):
        return None, "NONFINITE_SCORE"
    return mean, None


def summarize(rows):
    evaluated = [r for r in rows if r["status"] == "SCORED"]
    positive = sum(r["original_minus_counterfactual"] > 0 for r in evaluated)
    zero = sum(r["original_minus_counterfactual"] == 0 for r in evaluated)
    return {
        "selected": len(rows),
        "paired_scored": len(evaluated),
        "abstained": len(rows) - len(evaluated),
        "positive_delta_fraction_on_scored": (
            positive / len(evaluated) if evaluated else None),
        "tie_count": zero,
        "mean_delta_log_probability": (
            sum(r["original_minus_counterfactual"] for r in evaluated) /
            len(evaluated) if evaluated else None),
        "rows": rows,
    }


def evaluate(args):
    from model import LanguageModel

    directory = Path(args.outdir)
    meta = json.loads((directory / "training.json").read_text(encoding="utf-8"))
    if meta.get("status") != "CORPUS_DEV_ONLY_NOT_INDEPENDENT":
        raise ValueError("UNRECOGNIZED_CHECKPOINT_METADATA")
    cfg = meta["selection"]
    _, dev = read_sets(argparse.Namespace(
        input_dir=args.input_dir or meta["input_dir"],
        seed=cfg["seed"], train_limit=cfg["train_limit"],
        dev_limit=cfg["dev_limit"], context_turns=cfg["context_turns"],
        max_chars=cfg["max_chars"]))
    if not 1 <= args.count <= len(dev):
        raise ValueError("INVALID_CASE_COUNT")
    tokenizer = Tokenizer.load(str(directory / "tokenizer-chat-pilot.json"))
    device = get_device(args.device)
    model, checkpoint = LanguageModel.load_checkpoint(
        str(directory / "model-chat-pilot.pt"), device)
    if model.context_length != cfg["context"]:
        raise ValueError("CONTEXT_CONFIG_MISMATCH")
    rows = []
    for index in range(args.count):
        original = dev[index]
        donor = choose_donor(dev, index)
        alternative = make_counterfactual(original, donor)
        prompt_original = prompt_text(original, cfg["context_turns"])
        prompt_alternative = prompt_text(alternative, cfg["context_turns"])
        score_original, issue_original = score_reference(
            model, tokenizer, prompt_original, original["response"], device)
        score_alt, issue_alt = score_reference(
            model, tokenizer, prompt_alternative, original["response"], device)
        issue = issue_original or issue_alt
        rows.append({
            "dialogue_id": original["dialogue_id"],
            "turn_id": original["turn_id"],
            "donor_dialogue_id": donor["dialogue_id"],
            "reference": original["response"],
            "prompt_original": prompt_original,
            "prompt_counterfactual": prompt_alternative,
            "status": "ABSTAIN" if issue else "SCORED",
            "reason": issue,
            "original_mean_log_probability": score_original,
            "counterfactual_mean_log_probability": score_alt,
            "original_minus_counterfactual": (
                score_original - score_alt if issue is None else None),
        })
    report = summarize(rows)
    report.update({
        "status": "CORPUS_DEV_COUNTERFACTUAL_DIAGNOSTIC",
        "model_epoch": checkpoint.get("epoch"),
        "method": "SAME_REFERENCE_TEACHER_FORCED_MEAN_LOGP",
        "intervention": "REPLACE_LAST_CONTEXT_TEXT_WITH_OTHER_DEV_DIALOGUE",
        "warning": "Counterfactual context is synthetic and not necessarily "
                   "inappropriate. Positive delta is not semantic verification. "
                   "No Test accessed. Report contains conversation text.",
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
    p.add_argument("--outdir", default="model/realpersonachat-pilot-10k")
    p.add_argument("--input-dir", default=None)
    p.add_argument("--output",
                   default="results/realpersonachat_context_10k_dev.json")
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--overwrite", action="store_true")
    evaluate(p.parse_args())


if __name__ == "__main__":
    main()
