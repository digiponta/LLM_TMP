"""Train and evaluate a standalone dialogue-intent LLM.

This is an intentionally small development sanity check, not chatbot training.
Only the input utterance is supplied at inference. No rule-based correction
or train-label lookup is used to improve a prediction. Unknown new tokens
lead to abstention. The output is a compact I(INTENT) candidate.

A rule-based exact-match baseline is reported separately from the LLM.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from dialogue_intent_dataset import build_intent_examples, audit_split
from dialogue_intent_sd import DialogueIntentSD, INTENTS, parse_intent_code
from sd_candidate_pipeline import CandidateRejected
from tokenizer import Tokenizer
from train_compact_sd import PairDataset, decode_code, format_pair


def choose_device(name):
    import torch
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from model import LanguageModel

    train_rows, dev_rows = build_intent_examples(args.seed)
    audit = audit_split(train_rows, dev_rows)
    torch.manual_seed(args.seed)
    tok = Tokenizer()
    # Character vocabulary only. The unlabelled dev surfaces contribute
    # characters, NOT targets; all intent labels occur in train already.
    tok.fit_texts([format_pair(x["text"], x["code"]) for x in train_rows] +
                  [x["text"] for x in dev_rows])
    trainset = PairDataset(train_rows, tok, args.context)
    devset = PairDataset(dev_rows, tok, args.context)
    if len(trainset) != len(train_rows) or len(devset) != len(dev_rows):
        raise ValueError("CONTEXT_TRUNCATION")
    device = choose_device(args.device)
    model = LanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model,
                          num_layers=args.layers, hidden_dim=args.hidden_dim,
                          context_length=args.context).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                             shuffle=True, num_workers=0)
    devloader = DataLoader(devset, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)

    def run(loader, training):
        model.train(training)
        losses = []
        with torch.set_grad_enabled(training):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, tok.vocab_size), y.reshape(-1),
                    ignore_index=-100)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                losses.append(float(loss.item()))
        return sum(losses) / len(losses)

    history = []
    for epoch in range(1, args.epochs + 1):
        row = {"epoch": epoch,
               "train_loss": round(run(trainloader, True), 5),
               "dev_loss": round(run(devloader, False), 5)}
        print(row, flush=True)
        history.append(row)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(str(out / "model-dialogue-intent.pt"),
                          epoch=args.epochs, loss=history[-1]["dev_loss"])
    tok.save(str(out / "tokenizer-dialogue-intent.json"))
    (out / "split.json").write_text(
        json.dumps({"seed": args.seed, "audit": audit,
                    "train": train_rows, "dev": dev_rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "history.json").write_text(
        json.dumps({"status": "DEVELOPMENT_ONLY_FAMILY_DISJOINT",
                    "vocabulary_uses_unlabelled_dev_characters": True,
                    "history": history}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("Saved:", out)


def evaluate(args):
    from model import LanguageModel

    out = Path(args.outdir)
    split = json.loads((out / "split.json").read_text(encoding="utf-8"))
    source_train, source_dev = build_intent_examples(split["seed"])
    if split["train"] != source_train or split["dev"] != source_dev:
        raise ValueError("SPLIT_MISMATCH")
    audit_split(source_train, source_dev)
    device = choose_device(args.device)
    tok = Tokenizer.load(str(out / "tokenizer-dialogue-intent.json"))
    model, ckpt = LanguageModel.load_checkpoint(
        str(out / "model-dialogue-intent.pt"), device)
    engine = DialogueIntentSD()
    memorized = {x["text"]: x["code"] for x in source_train}
    rows = []
    for item in source_dev:
        pred, error = decode_code(model, tok, item["text"], device,
                                  max_new_tokens=args.max_new_tokens)
        if error or not pred:
            accepted = {"status": "REJECTED", "reason": error or "EMPTY"}
        else:
            accepted = engine.accept(item["text"], pred)
        rows.append({
            "text": item["text"], "expected": item["code"],
            "predicted": pred, "intent": item["intent"],
            "exact": pred == item["code"],
            "accepted": accepted["status"] == "ACCEPTED",
            "wrong_but_accepted": accepted["status"] == "ACCEPTED" and
                                  pred != item["code"],
            "reason": accepted.get("reason"),
            "baseline_exact": memorized.get(item["text"]) == item["code"],
        })
    n = len(rows)
    correct = sum(r["exact"] for r in rows)
    accepted = sum(r["accepted"] for r in rows)
    report = {
        "status": "DEVELOPMENT_ONLY_FAMILY_DISJOINT",
        "epoch": ckpt.get("epoch"), "total": n,
        "llm_exact_rate": correct / n,
        "candidate_accept_rate": accepted / n,
        "accepted_wrong_intent_count": sum(r["wrong_but_accepted"] for r in rows),
        "exact_match_baseline_rate":
            sum(r["baseline_exact"] for r in rows) / n,
        "per_intent": {
            intent: {
                "total": sum(r["intent"] == intent for r in rows),
                "correct": sum(r["intent"] == intent and r["exact"]
                               for r in rows),
            } for intent in INTENTS
        },
        "rows": rows,
    }
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print({k: v for k, v in report.items() if k != "rows"})
    print("Saved:", dest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("train", "evaluate"))
    p.add_argument("--outdir", default="model/dialogue-intent")
    p.add_argument("--output", default="results/dialogue_intent_dev.json")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--context", type=int, default=96)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=32)
    args = p.parse_args()
    if args.action == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
