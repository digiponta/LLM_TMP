"""Supervised Core+Frame SD learning with disjoint core triples in dev.

Training is deliberately separated from the existing compact-SD model.
A small causal Transformer predicts F(S(...),polarity,tense,topic).
Output-only cross-entropy and greedy evaluation; no rule-parser fallback.
The generated data shares vocabulary and grammar: NOT independent holdout.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

from compact_frame_sd import accept_frame_code, encode_frame_code, realize_frame
from semantic_features_sd import SemanticFeatureEngine
from train_compact_sd import PairDataset, decode_code, format_pair
from tokenizer import Tokenizer

ENTITY_IDS = ("CAT", "DOG", "FISH", "BIRD", "RICE")
VERB_IDS = ("EAT", "WATCH")
FEATURES = tuple(itertools.product(("POS", "NEG"), ("NONPAST", "PAST")))


def build_frame_examples(seed=42):
    """Split by *entire core triple*, then expand each into surface variations."""
    rng = random.Random(seed)
    triples = list(itertools.product(ENTITY_IDS, ENTITY_IDS, VERB_IDS))
    rng.shuffle(triples)
    cut = int(len(triples) * .8)
    train_cores, dev_cores = triples[:cut], triples[cut:]
    graph = SemanticFeatureEngine().graph

    def expand(cores):
        examples = []
        for s, o, v in cores:
            for pol, tense in FEATURES:
                for topic, voice in (("UNMARKED", "active"),
                                     ("SUBJECT_TOPIC", "active"),
                                     ("UNMARKED", "passive")):
                    parts = {"subject": s, "object": o, "verb": v,
                             "polarity": pol, "tense": tense, "topic": topic}
                    examples.append({"text": realize_frame(parts, graph, voice),
                                     "code": encode_frame_code(parts),
                                     "voice": voice, "polarity": pol,
                                     "tense": tense, "topic": topic,
                                     "core": [s, o, v]})
        rng.shuffle(examples)
        return examples

    return expand(train_cores), expand(dev_cores)


def train(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from model import LanguageModel

    torch.manual_seed(args.seed)
    train_examples, dev_examples = build_frame_examples(args.seed)
    tok = Tokenizer()
    tok.fit_texts([format_pair(e["text"], e["code"])
                   for e in train_examples + dev_examples])
    trainset = PairDataset(train_examples, tok, args.context)
    devset = PairDataset(dev_examples, tok, args.context)
    if len(trainset) != len(train_examples) or len(devset) != len(dev_examples):
        raise ValueError("Some items exceeded context; increase --context")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    model = LanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model,
                          num_layers=args.layers, hidden_dim=args.hidden_dim,
                          context_length=args.context).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                             shuffle=True, num_workers=0)
    devloader = DataLoader(devset, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)

    def loss_epoch(loader, training):
        model.train(training)
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.set_grad_enabled(training):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, tok.vocab_size),
                                       y.reshape(-1), ignore_index=-100)
                if training:
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optim.step()
            total += float(loss.item())
        return total / len(loader)

    history = []
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        tr = loss_epoch(trainloader, True)
        dev = loss_epoch(devloader, False)
        row = {"epoch": epoch, "train_loss": round(tr, 5),
               "dev_loss": round(dev, 5)}
        history.append(row)
        print(row, flush=True)
    model.save_checkpoint(str(outdir / "model-frame-sd.pt"),
                          epoch=args.epochs, loss=history[-1]["dev_loss"])
    tok.save(str(outdir / "tokenizer-frame-sd.json"))
    (outdir / "split-frame-sd.json").write_text(
        json.dumps({"seed": args.seed, "split_unit": "core_triple",
                    "train": train_examples, "dev": dev_examples},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "history-frame-sd.json").write_text(
        json.dumps({"status": "DEVELOPMENT_ONLY", "history": history,
                    "train_count": len(train_examples),
                    "dev_count": len(dev_examples),
                    "context": args.context}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("Saved:", outdir)


def evaluate(args):
    import torch
    from model import LanguageModel

    outdir = Path(args.outdir)
    saved = json.loads((outdir / "split-frame-sd.json").read_text(encoding="utf-8"))
    dev_examples = saved["dev"]
    tok = Tokenizer.load(str(outdir / "tokenizer-frame-sd.json"))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    model, checkpoint = LanguageModel.load_checkpoint(
        str(outdir / "model-frame-sd.pt"), device)
    engine = SemanticFeatureEngine()
    rows = []
    for item in dev_examples:
        generated, issue = decode_code(model, tok, item["text"], device,
                                       max_new_tokens=args.max_new_tokens)
        if issue or not generated:
            accepted = {"status": "REJECTED", "reason": issue or "EMPTY_CODE"}
        else:
            accepted = accept_frame_code(engine, item["text"], generated)
        rows.append({"text": item["text"], "expected": item["code"],
                     "generated": generated, "exact_code": generated == item["code"],
                     "status": accepted["status"],
                     "reason": accepted.get("reason"),
                     "polarity": item["polarity"], "tense": item["tense"],
                     "topic": item["topic"], "voice": item["voice"]})
    total = len(rows)
    report = {
        "status": "DEVELOPMENT_ONLY_CORE_DISJOINT",
        "total": total, "epoch": checkpoint.get("epoch"),
        "exact_code_rate": sum(r["exact_code"] for r in rows) / total,
        "sd_accept_rate": sum(r["status"] == "ACCEPTED" for r in rows) / total,
        "by_voice": {voice: {
            "count": sum(r["voice"] == voice for r in rows),
            "exact_code_rate": sum(r["exact_code"] for r in rows
                                   if r["voice"] == voice) /
                               sum(r["voice"] == voice for r in rows)}
                     for voice in ("active", "passive")},
        "rows": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print({k: v for k, v in report.items() if k != "rows"})
    print("Saved:", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("train", "evaluate"))
    p.add_argument("--outdir", default="model/sd-frame")
    p.add_argument("--output", default="results/sd_frame_eval.json")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--context", type=int, default=128)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=100)
    args = p.parse_args()
    if args.action == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
