"""Preflight selected RealPersonaChat pilot examples before GPU training.

Uses only Train to fit the character tokenizer. Reports fit, unknown
characters and effective response/context lengths for both Train and Dev.
No model training and no Test access; no raw conversation text is printed.
"""
from __future__ import annotations

import argparse
import json
import statistics

from tokenizer import Tokenizer
from train_realpersonachat_pilot import prompt_text, read_sets


def audit_rows(rows, tokenizer, context_length, turns):
    counts = {"selected": len(rows), "fit": 0, "too_long": 0,
              "unknown_prompt_examples": 0,
              "unknown_target_examples": 0,
              "unknown_target_characters": 0}
    used_lengths = []
    for row in rows:
        a = tokenizer.encode(prompt_text(row, turns), add_bos=True)
        b = tokenizer.encode(row["response"] + "\n", add_eos=True)
        if tokenizer.unk_id in a:
            counts["unknown_prompt_examples"] += 1
        unknown_target = b.count(tokenizer.unk_id)
        counts["unknown_target_characters"] += unknown_target
        if unknown_target:
            counts["unknown_target_examples"] += 1
        length = len(a) + len(b) - 1
        if length > context_length:
            counts["too_long"] += 1
        else:
            counts["fit"] += 1
            used_lengths.append(length)
    counts["fit_fraction"] = counts["fit"] / len(rows) if rows else 0
    counts["mean_fit_tokens"] = (
        round(statistics.mean(used_lengths), 2) if used_lengths else None)
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="data/realpersonachat_import")
    p.add_argument("--train-limit", type=int, default=2000)
    p.add_argument("--dev-limit", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--context-turns", type=int, default=2)
    p.add_argument("--max-chars", type=int, default=70)
    p.add_argument("--context", type=int, default=256)
    args = p.parse_args()
    train, dev = read_sets(args)
    tok = Tokenizer()
    tok.fit_texts([prompt_text(row, args.context_turns)
                   + row["response"] + "\n" for row in train])
    summary = {
        "status": "PREFLIGHT_NO_TRAINING",
        "vocab_size_train_only": tok.vocab_size,
        "context_length": args.context,
        "train": audit_rows(train, tok, args.context, args.context_turns),
        "dev": audit_rows(dev, tok, args.context, args.context_turns),
        "note": "Dev UNKNOWN chars may distort token NLL. Test is untouched. "
                "No semantic quality is inferred.",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
