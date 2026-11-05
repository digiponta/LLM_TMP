"""Bounded multi-turn next-response pilot on local RealPersonaChat JSONL.

Uses the existing causal Transformer and a train-only character tokenizer.
No semantic/intent labels are inferred. Both dialogue participants are peers:
speaker_0 and speaker_1, not assumed user/assistant. The experiment predicts
a next utterance from preceding turns, not an autonomous chatbot.

Split files must come from import_realpersonachat.py. All outputs are local.
All dev/test records are excluded from fitting, including tokenizer fitting.
Evaluation is token NLL and perplexity (teacher-forced); not chat quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from tokenizer import Tokenizer
from model import LanguageModel


def canonical_key(row):
    return str(row["dialogue_id"]) + ":" + str(row["turn_id"])


def load_examples(path, limit, seed, context_turns, max_chars):
    """Deterministic bounded hash sample, streaming arbitrary-size JSONL."""
    if limit < 1 or context_turns < 1 or max_chars < 1:
        raise ValueError("INVALID_SELECTION_PARAMETERS")
    selected = []
    seen = set()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    # Maintain the smallest hash-ranks; bounded memory for large corpora.
    for line in path.open("r", encoding="utf-8-sig"):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("context"), list):
            raise ValueError("INVALID_CONTEXT")
        if not isinstance(row.get("response"), str):
            raise ValueError("INVALID_RESPONSE")
        if row.get("next_speaker") not in ("speaker_0", "speaker_1"):
            raise ValueError("INVALID_SPEAKER")
        key = canonical_key(row)
        if key in seen:
            raise ValueError("DUPLICATE_EXAMPLE_ID")
        seen.add(key)
        context = row["context"][-context_turns:]
        if not context or not row["response"].strip():
            continue
        if any(t.get("speaker") not in ("speaker_0", "speaker_1") or
               not isinstance(t.get("text"), str) for t in context):
            raise ValueError("INVALID_CONTEXT_TURN")
        # No mid-sentence truncation: reject oversized individual utterances.
        if len(row["response"]) > max_chars or any(
            len(t["text"]) > max_chars for t in context):
            continue
        digest = hashlib.sha256((str(seed) + ":" + key).encode()).hexdigest()
        selected.append((digest, {
            "dialogue_id": str(row["dialogue_id"]),
            "turn_id": row["turn_id"],
            "context": context,
            "next_speaker": row["next_speaker"],
            "response": row["response"],
        }))
        if len(selected) > 2 * limit:
            selected.sort(key=lambda x: x[0])
            selected = selected[:limit]
    selected.sort(key=lambda x: x[0])
    return [r for _, r in selected[:limit]]


def prompt_text(row, context_turns=2):
    turns = row["context"][-context_turns:]
    body = "".join(t["speaker"] + ":" + t["text"] + "\n" for t in turns)
    return "会話:\n" + body + row["next_speaker"] + ":"


class ResponseDataset:
    """Output-only CE; pad labels are ignored; no accidental silent drops."""
    def __init__(self, examples, tokenizer, context_length, context_turns):
        import torch
        self.items = []
        self.skipped_length = 0
        self.unknown_targets = 0
        for row in examples:
            prefix = tokenizer.encode(prompt_text(row, context_turns), add_bos=True)
            suffix = tokenizer.encode(row["response"] + "\n", add_eos=True)
            # Unknown symbols in development inputs may be mapped to UNK.
            self.unknown_targets += sum(v == tokenizer.unk_id for v in suffix)
            seq = prefix + suffix
            if len(seq) - 1 > context_length:
                self.skipped_length += 1
                continue
            x = seq[:-1]
            y = [-100] * (len(prefix) - 1) + suffix
            pad = context_length - len(x)
            self.items.append((torch.tensor(x + [tokenizer.pad_id] * pad),
                               torch.tensor(y + [-100] * pad)))
        if not self.items:
            raise ValueError("NO_EXAMPLES_FIT_CONTEXT")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def read_sets(args):
    directory = Path(args.input_dir)
    audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "EXPORTED_CORPUS_HELDOUT":
        raise ValueError("UNRECOGNIZED_DATA_AUDIT")
    train = load_examples(directory / "train.jsonl", args.train_limit,
                          args.seed, args.context_turns, args.max_chars)
    dev = load_examples(directory / "dev.jsonl", args.dev_limit,
                        args.seed, args.context_turns, args.max_chars)
    overlap = {r["dialogue_id"] for r in train} & {
        r["dialogue_id"] for r in dev}
    if overlap:
        raise ValueError("DIALOGUE_SPLIT_LEAKAGE")
    return train, dev


def get_device(name):
    import torch
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available()
                        else "cpu" if name == "auto" else name)


def run_epoch(model, loader, optimizer, device, pad_size, training):
    import torch
    import torch.nn.functional as F
    model.train(training)
    losses = 0.
    tokens = 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, pad_size), y.reshape(-1),
                ignore_index=-100, reduction="sum")
            count = int((y != -100).sum().item())
            loss = loss_sum / count
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            losses += float(loss_sum.detach().item())
            tokens += count
    return losses / tokens, tokens


def train(args):
    import torch
    from torch.utils.data import DataLoader

    train_rows, dev_rows = read_sets(args)
    torch.manual_seed(args.seed)
    tokenizer = Tokenizer()
    tokenizer.fit_texts(
        [prompt_text(r, args.context_turns) + r["response"] + "\n"
         for r in train_rows])
    train_ds = ResponseDataset(train_rows, tokenizer, args.context,
                               args.context_turns)
    dev_ds = ResponseDataset(dev_rows, tokenizer, args.context,
                             args.context_turns)
    device = get_device(args.device)
    model = LanguageModel(
        vocab_size=tokenizer.vocab_size, d_model=args.d_model,
        num_layers=args.layers, hidden_dim=args.hidden_dim,
        context_length=args.context).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        train_nll, tr_tokens = run_epoch(
            model, train_loader, optimizer, device, tokenizer.vocab_size, True)
        dev_nll, dev_tokens = run_epoch(
            model, dev_loader, None, device, tokenizer.vocab_size, False)
        record = {"epoch": epoch, "train_token_nll": round(train_nll, 5),
                  "dev_token_nll": round(dev_nll, 5),
                  "dev_ppl": round(math.exp(min(dev_nll, 60)), 4)}
        history.append(record)
        print(record, flush=True)
    model.save_checkpoint(str(out / "model-chat-pilot.pt"),
                          epoch=args.epochs, loss=history[-1]["dev_token_nll"])
    tokenizer.save(str(out / "tokenizer-chat-pilot.json"))
    meta = {
        "status": "CORPUS_DEV_ONLY_NOT_INDEPENDENT",
        "train_examples_fit": len(train_ds), "dev_examples_fit": len(dev_ds),
        "train_skipped_length": train_ds.skipped_length,
        "dev_skipped_length": dev_ds.skipped_length,
        "dev_unknown_target_chars": dev_ds.unknown_targets,
        "train_tokens_last_epoch": tr_tokens, "dev_tokens_last_epoch": dev_tokens,
        "vocab_size": tokenizer.vocab_size,
        "model_parameters": model.parameter_count,
        "input_dir": str(Path(args.input_dir).resolve()),
        "selection": {"seed": args.seed, "train_limit": args.train_limit,
                      "dev_limit": args.dev_limit, "context_turns": args.context_turns,
                      "max_chars": args.max_chars, "context": args.context},
        "history": history,
        "note": "No intent labels. Heldout by source dialogue only. "
                "Teacher-forced NLL is not evidence of coherent conversation.",
    }
    (out / "training.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", out)


def generate(args):
    import torch
    from model import LanguageModel
    directory = Path(args.outdir)
    meta = json.loads((directory / "training.json").read_text(encoding="utf-8"))
    config = meta["selection"]
    tokenizer = Tokenizer.load(str(directory / "tokenizer-chat-pilot.json"))
    device = get_device(args.device)
    model, ckpt = LanguageModel.load_checkpoint(
        str(directory / "model-chat-pilot.pt"), device)
    train_rows, dev_rows = read_sets(argparse.Namespace(
        input_dir=args.input_dir or meta["input_dir"],
        train_limit=config["train_limit"], dev_limit=config["dev_limit"],
        seed=config["seed"], context_turns=config["context_turns"],
        max_chars=config["max_chars"]))
    row = dev_rows[args.example_index]
    prefix = tokenizer.encode(
        prompt_text(row, config["context_turns"]), add_bos=True)
    if tokenizer.unk_id in prefix or len(prefix) >= model.context_length:
        print(json.dumps({"status": "ABSTAIN",
                          "reason": "UNKNOWN_OR_TOO_LONG_PROMPT"},
                         ensure_ascii=False))
        return
    predicted = []
    model.eval()
    with torch.no_grad():
        for _ in range(min(args.max_new_tokens,
                           model.context_length - len(prefix))):
            ids = prefix + predicted
            logits = model(torch.tensor([ids], device=device))[0, -1]
            next_id = int(logits.argmax().item())
            if next_id in (tokenizer.eos_id, tokenizer.pad_id):
                break
            if tokenizer.id_to_token[next_id] == "\n":
                break
            predicted.append(next_id)
    print(json.dumps({
        "status": "DEV_EXAMPLE_GENERATION_NOT_BLIND",
        "prompt": prompt_text(row, config["context_turns"]),
        "generated": tokenizer.decode(predicted),
        "reference": row["response"],
        "dialogue_id": row["dialogue_id"],
        "model_epoch": ckpt.get("epoch"),
    }, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("train", "generate"))
    p.add_argument("--input-dir", default="data/realpersonachat_import")
    p.add_argument("--outdir", default="model/realpersonachat-pilot")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--train-limit", type=int, default=2000)
    p.add_argument("--dev-limit", type=int, default=200)
    p.add_argument("--context-turns", type=int, default=2)
    p.add_argument("--max-chars", type=int, default=70)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--example-index", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=50)
    args = p.parse_args()
    if args.action == "train":
        train(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
