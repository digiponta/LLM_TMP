"""Train separate LLM heads for semantic roles and frame features.

Two independently trained causal Transformers:
  roles:    R(AGENT,PATIENT,ACTION) e.g. R(DOG,FISH,WATCH)
  features: Q(POLARITY,TENSE,TOPIC) e.g. Q(NEG,PAST,UNMARKED)

The composed result is validated by the existing SD Engine. No parser-based
repair of erroneous model output and no factual approval. Train/dev use the
exact same core-disjoint seed=42 split as train_frame_sd.py. Results are
development only and use the same fixed synthetic grammar/lexicon.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from compact_frame_sd import accept_frame_code, encode_frame_code
from semantic_features_sd import SemanticFeatureEngine
from tokenizer import Tokenizer
from train_compact_sd import PairDataset, decode_code, format_pair
from train_frame_sd import build_frame_examples

ROLE_RE = re.compile(r"^R\(([A-Z][A-Z0-9_-]*),([A-Z][A-Z0-9_-]*),([A-Z][A-Z0-9_-]*)\)$")
FEATURE_RE = re.compile(r"^Q\((POS|NEG),(NONPAST|PAST),(UNMARKED|SUBJECT_TOPIC)\)$")
HEADS = ("roles", "features")


def role_target(example):
    return "R({},{},{})".format(*example["core"])


def feature_target(example):
    return "Q({},{},{})".format(
        example["polarity"], example["tense"], example["topic"]
    )


def target(example, head):
    if head == "roles":
        return role_target(example)
    if head == "features":
        return feature_target(example)
    raise ValueError("Unknown head: " + str(head))


def parse_heads(role, feature):
    a = ROLE_RE.fullmatch(role or "")
    b = FEATURE_RE.fullmatch(feature or "")
    if not a or not b:
        raise ValueError("INVALID_ROLE_OR_FEATURE_OUTPUT")
    s, o, v = a.groups()
    p, t, i = b.groups()
    parts = {"subject": s, "object": o, "verb": v,
             "polarity": p, "tense": t, "topic": i}
    return encode_frame_code(parts)


def make_pairs(examples, head):
    return [{"text": e["text"], "code": target(e, head)} for e in examples]


def device_from_arg(value):
    import torch
    return torch.device(
        "cuda" if value == "auto" and torch.cuda.is_available()
        else "cpu" if value == "auto" else value
    )


def fit_head(args, head, train_examples, dev_examples, device):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from model import LanguageModel

    train_pairs = make_pairs(train_examples, head)
    dev_pairs = make_pairs(dev_examples, head)
    tok = Tokenizer()
    # Fit on training only; any unknown dev characters are not silently added.
    tok.fit_texts([format_pair(p["text"], p["code"]) for p in train_pairs])
    trainset = PairDataset(train_pairs, tok, args.context)
    devset = PairDataset(dev_pairs, tok, args.context)
    if len(trainset) != len(train_pairs) or len(devset) != len(dev_pairs):
        raise ValueError("Context too short for " + head)
    torch.manual_seed(args.seed + (0 if head == "roles" else 1))
    model = LanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model,
                          num_layers=args.layers, hidden_dim=args.hidden_dim,
                          context_length=args.context).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                             shuffle=True, num_workers=0)
    devloader = DataLoader(devset, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)

    def epoch_loss(loader, is_train):
        model.train(is_train)
        total = 0.
        with torch.set_grad_enabled(is_train):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, tok.vocab_size), y.reshape(-1),
                    ignore_index=-100)
                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                total += loss.item()
        return total / len(loader)

    history = []
    for epoch in range(1, args.epochs + 1):
        tr = epoch_loss(trainloader, True)
        ev = epoch_loss(devloader, False)
        row = {"head": head, "epoch": epoch,
               "train_loss": round(tr, 5), "dev_loss": round(ev, 5)}
        history.append(row)
        print(row, flush=True)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(str(outdir / ("model-" + head + ".pt")),
                          epoch=args.epochs, loss=history[-1]["dev_loss"])
    tok.save(str(outdir / ("tokenizer-" + head + ".json")))
    return history


def train(args):
    device = device_from_arg(args.device)
    train_examples, dev_examples = build_frame_examples(args.seed)
    history = {head: fit_head(args, head, train_examples, dev_examples, device)
               for head in HEADS}
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "split.json").write_text(
        json.dumps({"seed": args.seed, "split_unit": "core_triple",
                    "train": train_examples, "dev": dev_examples},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "history.json").write_text(
        json.dumps({"status": "DEVELOPMENT_ONLY_CORE_DISJOINT",
                    "history": history, "device": str(device)},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", outdir)


def extract_head(head, text, models, tokenizers, device, max_new_tokens):
    result, issue = decode_code(models[head], tokenizers[head], text,
                                device, max_new_tokens=max_new_tokens)
    return result if issue is None else None, issue


def evaluate(args):
    from model import LanguageModel
    device = device_from_arg(args.device)
    directory = Path(args.outdir)
    saved = json.loads((directory / "split.json").read_text(encoding="utf-8"))
    # Guard comparison: the same 120 dev cases as the previous baseline.
    expected_train, expected_dev = build_frame_examples(saved["seed"])
    if saved["train"] != expected_train or saved["dev"] != expected_dev:
        raise ValueError("SPLIT_MISMATCH")
    models, tokenizers = {}, {}
    for head in HEADS:
        models[head], _ = LanguageModel.load_checkpoint(
            str(directory / ("model-" + head + ".pt")), device)
        tokenizers[head] = Tokenizer.load(
            str(directory / ("tokenizer-" + head + ".json")))
    rows = []
    engine = SemanticFeatureEngine()
    for example in saved["dev"]:
        role, reissue = extract_head("roles", example["text"],
                                     models, tokenizers, device,
                                     args.max_new_tokens)
        feature, feissue = extract_head("features", example["text"],
                                        models, tokenizers, device,
                                        args.max_new_tokens)
        role_ok = role == role_target(example)
        feature_ok = feature == feature_target(example)
        assembled, error = None, None
        try:
            assembled = parse_heads(role, feature)
        except ValueError as exc:
            error = str(exc)
        accepted = (accept_frame_code(engine, example["text"], assembled)
                    if assembled else {"status": "REJECTED",
                                       "reason": error or reissue or feissue})
        rows.append({
            "text": example["text"], "voice": example["voice"],
            "expected": example["code"], "role_expected": role_target(example),
            "role_generated": role, "role_exact": role_ok,
            "feature_expected": feature_target(example),
            "feature_generated": feature, "feature_exact": feature_ok,
            "generated": assembled, "exact_code": assembled == example["code"],
            "status": accepted["status"], "reason": accepted.get("reason"),
        })
    total = len(rows)
    def rate(items, field):
        return sum(bool(r[field]) for r in items) / len(items) if items else 0.
    result = {
        "status": "DEVELOPMENT_ONLY_CORE_DISJOINT",
        "architecture": "TWO_INDEPENDENT_LLM_HEADS_NO_RULE_FALLBACK",
        "total": total,
        "role_exact_rate": rate(rows, "role_exact"),
        "feature_exact_rate": rate(rows, "feature_exact"),
        "exact_code_rate": rate(rows, "exact_code"),
        "sd_accept_rate": sum(r["status"] == "ACCEPTED" for r in rows) / total,
        "by_voice": {voice: {
            "count": len(group),
            "role_exact_rate": rate(group, "role_exact"),
            "feature_exact_rate": rate(group, "feature_exact"),
            "exact_code_rate": rate(group, "exact_code"),
            "sd_accept_rate": sum(r["status"] == "ACCEPTED" for r in group)
                              / len(group),
        } for voice in HEADS_VOICES
          for group in ([r for r in rows if r["voice"] == voice],)},
        "rows": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print({k: v for k, v in result.items() if k != "rows"})
    print("Saved:", out)


HEADS_VOICES = ("active", "passive")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("train", "evaluate"))
    p.add_argument("--outdir", default="model/sd-role-frame")
    p.add_argument("--output", default="results/sd_role_frame_eval.json")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--context", type=int, default=128)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=80)
    args = p.parse_args()
    if args.action == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
