"""Train v0.3-style Transformer to output compact SD codes.

Train/validation splits use disjoint *surface strings* and fixed seed.
This is a development experiment, not an independent holdout. The model
receives only input text and must generate the code. The parser/SD Engine
enforces consistency; no rule-based fallback is used during evaluation.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

from compact_sd import accept_compact
from recursive_sd import build_sample_graph
from tokenizer import Tokenizer

ENT = [("CAT", "猫"), ("DOG", "犬"), ("FISH", "魚"), ("BIRD", "鳥"), ("RICE", "ご飯")]
VERB = [("EAT", "食べる"), ("WATCH", "見る")]
PAIR = [(s, o, v) for s, o, v in itertools.product(ENT, ENT, VERB)]
# All clauses are syntactic examples, never factual labels.
CLAUSES = [(s[1] + "が" + o[1] + "を" + v[1] + "。",
            "S(" + ",".join((s[0], o[0], v[0])) + ")")
           for s, o, v in PAIR]
INVALID = ["猫は魚を食べる。", "鳥が空を飛ぶ。", "犬が魚を踊る。"]


def build_examples(seed=42, max_pairs=1200):
    rng = random.Random(seed)
    examples = [{"text": surface, "code": code, "kind": "simple"}
                for surface, code in CLAUSES]
    all_pairs = list(itertools.product(range(len(CLAUSES)),
                                       range(len(CLAUSES)), ("T", "A")))
    rng.shuffle(all_pairs)
    for i, j, tag in all_pairs[:max_pairs]:
        left, lc = CLAUSES[i]
        right, rc = CLAUSES[j]
        join = "と、" if tag == "T" else "、そして"
        examples.append({"text": left[:-1] + join + right,
                         "code": f"{tag}({lc},{rc})", "kind": "nested"})
    rng.shuffle(examples)
    cut = max(1, int(len(examples) * .9))
    return examples[:cut], examples[cut:]


def format_pair(text, code):
    return "入力:" + text + "\nSD:" + code + "\n"


class PairDataset:
    """Teacher-forced sequence with loss only on SD output tokens."""
    def __init__(self, pairs, tok, context):
        import torch
        self.torch = torch
        self.items = []
        self.context = context
        for pair in pairs:
            prefix = tok.encode("入力:" + pair["text"] + "\nSD:", add_bos=True)
            target = tok.encode(pair["code"] + "\n", add_eos=True)
            if tok.unk_id in prefix or tok.unk_id in target:
                raise ValueError("Unexpected UNK token")
            # x=sequence[:-1], y=sequence[1:]; outputs only in target.
            seq = prefix + target
            if len(seq) - 1 > context:
                continue
            x = seq[:-1]
            y = seq[1:]
            labels = [-100] * (len(prefix) - 1) + target
            pad = context - len(x)
            self.items.append((x + [tok.pad_id] * pad,
                               labels + [-100] * pad))
        if not self.items:
            raise ValueError("No training examples fit context; increase --context")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        x, y = self.items[index]
        return (self.torch.tensor(x, dtype=self.torch.long),
                self.torch.tensor(y, dtype=self.torch.long))


def train(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from model import LanguageModel

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    train_pairs, eval_pairs = build_examples(args.seed, args.max_pairs)
    # Tokenizer must include both inputs and short labels. Separate artifact:
    # do not overwrite existing general-purpose model/tokenizer.
    tok = Tokenizer()
    tok.fit_texts([format_pair(p["text"], p["code"]) for p in train_pairs])
    trainset = PairDataset(train_pairs, tok, args.context)
    evalset = PairDataset(eval_pairs, tok, args.context)
    model = LanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model,
                          num_layers=args.layers, hidden_dim=args.hidden_dim,
                          context_length=args.context).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(evalset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    history = []
    for epoch in range(args.epochs):
        model.train()
        total, steps = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, tok.vocab_size),
                                   y.reshape(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total += loss.item()
            steps += 1
        model.eval()
        with torch.no_grad():
            ev, en = 0.0, 0
            for x, y in eval_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, tok.vocab_size),
                                       y.reshape(-1), ignore_index=-100)
                ev += loss.item()
                en += 1
        row = {"epoch": epoch + 1, "train_loss": round(total / steps, 5),
               "dev_loss": round(ev / max(1, en), 5)}
        history.append(row)
        print(row, flush=True)
    path = Path(args.outdir)
    path.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(str(path / "model-sd-compact.pt"),
                          epoch=args.epochs, loss=history[-1]["dev_loss"])
    tok.save(str(path / "tokenizer-sd-compact.json"))
    (path / "split-sd-compact.json").write_text(
        json.dumps({"seed": args.seed, "train": train_pairs, "dev": eval_pairs},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (path / "training-sd-compact.json").write_text(
        json.dumps({"status": "DEVELOPMENT_ONLY", "device": str(device),
                    "train_pairs": len(trainset), "dev_pairs": len(evalset),
                    "context": args.context, "history": history},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", path)


def decode_code(model, tok, text, device, max_new_tokens=100):
    """Greedy generation; stop at newline or EOS. No symbolic fallback."""
    import torch
    prefix = "入力:" + text + "\nSD:"
    ids = tok.encode(prefix, add_bos=True)
    if tok.unk_id in ids:
        return None, "UNKNOWN_INPUT_TOKEN"
    if len(ids) >= model.context_length:
        return None, "PROMPT_TOO_LONG"
    max_new_tokens = min(max_new_tokens, model.context_length - len(ids))
    model.eval()
    generated = list(ids)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = generated[-model.context_length:]
            logits = model(torch.tensor([context], dtype=torch.long, device=device))[0, -1]
            next_id = int(logits.argmax())
            if next_id in (tok.eos_id, tok.pad_id):
                break
            generated.append(next_id)
            if tok.decode([next_id], skip_special_tokens=False) == "\n":
                break
    return tok.decode(generated[len(ids):], skip_special_tokens=True).strip(), None


def evaluate(args):
    import torch
    from model import LanguageModel
    train_pairs, dev_pairs = build_examples(args.seed, args.max_pairs)
    path = Path(args.outdir)
    tok = Tokenizer.load(str(path / "tokenizer-sd-compact.json"))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    model, checkpoint = LanguageModel.load_checkpoint(
        str(path / "model-sd-compact.pt"), device)
    rows = []
    for p in dev_pairs:
        code, error = decode_code(model, tok, p["text"], device)
        graph = build_sample_graph()
        try:
            result = accept_compact(graph, p["text"], code) if code else {"status": "REJECTED", "reason": error or "EMPTY"}
        except ValueError as exc:
            result = {"status": "REJECTED", "reason": str(exc)}
        rows.append({"input": p["text"], "expected": p["code"], "generated": code,
                     "exact_code": code == p["code"],
                     "status": result["status"], "reason": result.get("reason")})
    report = {"status": "DEVELOPMENT_ONLY_NOT_INDEPENDENT",
              "model_epoch": checkpoint.get("epoch"), "total": len(rows),
              "exact_code_rate": sum(r["exact_code"] for r in rows) / len(rows),
              "sd_accept_rate": sum(r["status"] == "ACCEPTED" for r in rows) / len(rows),
              "rows": rows}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print({k: v for k, v in report.items() if k != "rows"})
    print("Saved:", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("train", "evaluate"))
    p.add_argument("--outdir", default="model/sd-compact")
    p.add_argument("--output", default="results/sd_compact_eval.json")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-pairs", type=int, default=1200)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--context", type=int, default=128)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--hidden-dim", type=int, default=256)
    args = p.parse_args()
    if args.action == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
