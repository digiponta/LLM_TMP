"""Full-length context shuffle diagnostic for frozen RealPersonaChat pilots.

Compare each Dev reference's teacher-forced mean token log probability
under its original context and an equal-turn-count alternative consisting
of turn TEXT from a different Dev conversation. The original conversation's
speaker slots, next-speaker marker and target are unchanged. Donor matching
is deterministic and favors similar character counts to limit length
confounding. Donor and source cannot share dialogue_id.

No Train labels, Test split, gradient, model update or semantic grading.
Artificial alternatives may still be plausible; positive deltas do NOT
prove semantic understanding. Reports contain private conversation text.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from tokenizer import Tokenizer
from train_realpersonachat_pilot import get_device, prompt_text, read_sets
from evaluate_realpersonachat_context import score_reference


def context_rows(example, count):
    return example["context"][-count:]


def choose_matched_donor(rows, source, context_turns):
    """Deterministic nearest-total-length donor with equal number of turns.

    A tie favors earlier Dev selection; only the donor's CONTEXT is used.
    """
    src = context_rows(source, context_turns)
    if not src:
        raise ValueError("SOURCE_HAS_NO_CONTEXT")
    candidates = []
    for rank, item in enumerate(rows):
        donor = context_rows(item, context_turns)
        if (item["dialogue_id"] == source["dialogue_id"]
                or len(donor) != len(src)):
            continue
        if [x["speaker"] for x in donor] != [x["speaker"] for x in src]:
            continue
        if all(a["text"] == b["text"] for a, b in zip(src, donor)):
            continue
        difference = sum(abs(len(a["text"]) - len(b["text"]))
                         for a, b in zip(src, donor))
        candidates.append((difference, rank, item))
    if not candidates:
        raise ValueError("NO_MATCHED_DONOR")
    return min(candidates, key=lambda x: (x[0], x[1]))[2]


def make_full_shuffle(source, donor, context_turns):
    src_turns = context_rows(source, context_turns)
    donor_turns = context_rows(donor, context_turns)
    if source["dialogue_id"] == donor["dialogue_id"]:
        raise ValueError("DONOR_SAME_DIALOGUE")
    if len(src_turns) != len(donor_turns):
        raise ValueError("MISMATCHED_CONTEXT_TURN_COUNT")
    if [x["speaker"] for x in src_turns] != [
            x["speaker"] for x in donor_turns]:
        raise ValueError("MISMATCHED_SPEAKER_SEQUENCE")
    if not src_turns:
        raise ValueError("SOURCE_HAS_NO_CONTEXT")
    out = dict(source)
    out["context"] = [
        {"speaker": original["speaker"], "text": alternative["text"]}
        for original, alternative in zip(src_turns, donor_turns)
    ]
    return out


def summarize(rows):
    scored = [r for r in rows if r["status"] == "SCORED"]
    n = len(scored)
    differences = [r["original_minus_shuffled"] for r in scored]
    return {
        "selected": len(rows),
        "paired_scored": n,
        "abstained": len(rows) - n,
        "abstention_reasons": {
            reason: sum(r["reason"] == reason for r in rows)
            for reason in sorted({r["reason"] for r in rows if r["reason"]})
        },
        "positive_delta_fraction_on_scored": (
            sum(v > 0 for v in differences) / n if n else None),
        "zero_delta_count": sum(v == 0 for v in differences),
        "mean_delta_log_probability": (
            statistics.mean(differences) if n else None),
        "median_delta_log_probability": (
            statistics.median(differences) if n else None),
        "mean_original_log_probability": (
            statistics.mean(r["original_log_probability"] for r in scored)
            if n else None),
        "mean_shuffled_log_probability": (
            statistics.mean(r["shuffled_log_probability"] for r in scored)
            if n else None),
        "mean_donor_length_difference": (
            statistics.mean(r["context_length_difference"] for r in scored)
            if n else None),
        "rows": rows,
    }


def evaluate(args):
    from model import LanguageModel
    directory = Path(args.outdir)
    meta = json.loads((directory / "training.json").read_text(
        encoding="utf-8"))
    if meta.get("status") != "CORPUS_DEV_ONLY_NOT_INDEPENDENT":
        raise ValueError("UNRECOGNIZED_MODEL_METADATA")
    config = meta["selection"]
    _, dev = read_sets(argparse.Namespace(
        input_dir=args.input_dir or meta["input_dir"],
        train_limit=config["train_limit"], dev_limit=config["dev_limit"],
        context_turns=config["context_turns"],
        max_chars=config["max_chars"], seed=config["seed"]))
    if not 1 <= args.count <= len(dev):
        raise ValueError("INVALID_CASE_COUNT")
    tok = Tokenizer.load(str(directory / "tokenizer-chat-pilot.json"))
    device = get_device(args.device)
    model, ckpt = LanguageModel.load_checkpoint(
        str(directory / "model-chat-pilot.pt"), device)
    if model.context_length != config["context"]:
        raise ValueError("CONTEXT_CONFIG_MISMATCH")
    rows = []
    for source in dev[:args.count]:
        try:
            donor = choose_matched_donor(
                dev, source, config["context_turns"])
            shuffled = make_full_shuffle(
                source, donor, config["context_turns"])
        except ValueError as exc:
            rows.append({
                "dialogue_id": source["dialogue_id"],
                "turn_id": source["turn_id"], "status": "ABSTAIN",
                "reason": str(exc), "original_minus_shuffled": None,
            })
            continue
        original_prompt = prompt_text(source, config["context_turns"])
        shuffled_prompt = prompt_text(shuffled, config["context_turns"])
        original_score, original_issue = score_reference(
            model, tok, original_prompt, source["response"], device)
        shuffled_score, shuffled_issue = score_reference(
            model, tok, shuffled_prompt, source["response"], device)
        issues = [tag + ":" + issue for tag, issue in (
            ("original", original_issue), ("shuffled", shuffled_issue)) if issue]
        original_turns = context_rows(source, config["context_turns"])
        donor_turns = context_rows(donor, config["context_turns"])
        length_difference = sum(
            abs(len(a["text"]) - len(b["text"]))
            for a, b in zip(original_turns, donor_turns))
        rows.append({
            "dialogue_id": source["dialogue_id"],
            "turn_id": source["turn_id"],
            "donor_dialogue_id": donor["dialogue_id"],
            "context_turn_count": len(original_turns),
            "context_length_difference": length_difference,
            "reference": source["response"],
            "original_prompt": original_prompt,
            "shuffled_prompt": shuffled_prompt,
            "status": "ABSTAIN" if issues else "SCORED",
            "reason": ";".join(issues) if issues else None,
            "original_log_probability": original_score,
            "shuffled_log_probability": shuffled_score,
            "original_minus_shuffled": (
                original_score - shuffled_score if not issues else None),
        })
    report = summarize(rows)
    report.update({
        "status": "CORPUS_DEV_FULL_CONTEXT_SHUFFLE_NOT_INDEPENDENT",
        "model_epoch": ckpt.get("epoch"),
        "method": "SAME_REFERENCE_TEACHER_FORCED_MEAN_LOGP",
        "intervention": "REPLACE_ALL_CONTEXT_TURN_TEXT_FROM_OTHER_DEV_DIALOGUE",
        "donor_selection": "MATCH_SPEAKER_SEQUENCE_AND_TURN_COUNT_NEAREST_LENGTH",
        "warning": "Artificial replacements may fit the reference. "
                   "Model sees changed text and potentially different character "
                   "statistics and lengths; this is not semantic correctness. "
                   "Dev only, no Test. Report contains private text.",
    })
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise ValueError("REPORT_EXISTS_USE_OVERWRITE")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"},
                     ensure_ascii=False, indent=2))
    print("Saved local/private report:", output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="model/realpersonachat-pilot-30k")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output",
                        default="results/realpersonachat_fullshuffle_30k_dev.json")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    parser.add_argument("--overwrite", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
