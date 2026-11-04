"""Four-way LLM/semantic-memory comparison; deterministic graph components.

A: raw LLM; B: lexical natural-language memory; C: linked SD lookup;
D: linked SD lookup + grammar SD composition (composition only when requested).
No benchmark target is used by a predictor.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

ENTITIES = {
    "CAT": "猫", "DOG": "犬", "FISH": "魚",
    "BIRD": "鳥", "RICE": "ご飯",
}
VERBS = {"EAT": "食べる", "WATCH": "見る"}
# Training/knowledge facts. Heldout *compositions* are NOT facts.
FACTS = [
    ("CAT", "FISH", "EAT"),
    ("DOG", "BIRD", "WATCH"),
    ("CAT", "RICE", "EAT"),
]
CASES = [
    {"id": "fact-1", "kind": "fact", "question": "猫が食べる魚は何ですか？", "expected": "魚", "triple": ["CAT", "FISH", "EAT"], "answer_role": "object"},
    {"id": "fact-2", "kind": "fact", "question": "鳥を見るのは誰ですか？", "expected": "犬", "triple": ["DOG", "BIRD", "WATCH"], "answer_role": "subject"},
    {"id": "fact-3", "kind": "fact", "question": "猫が食べるご飯は何ですか？", "expected": "ご飯", "triple": ["CAT", "RICE", "EAT"], "answer_role": "object"},
    {"id": "compose-1", "kind": "compose", "question": "主語=犬、目的語=魚、動詞=食べる。文を作って。", "expected": "犬が魚を食べる。", "triple": ["DOG", "FISH", "EAT"]},
    {"id": "compose-2", "kind": "compose", "question": "主語=猫、目的語=鳥、動詞=見る。文を作って。", "expected": "猫が鳥を見る。", "triple": ["CAT", "BIRD", "WATCH"]},
    {"id": "compose-3", "kind": "compose", "question": "主語=犬、目的語=ご飯、動詞=食べる。文を作って。", "expected": "犬がご飯を食べる。", "triple": ["DOG", "RICE", "EAT"]},
    {"id": "unknown-1", "kind": "unknown", "question": "鳥が食べるものを教えて。", "expected": None},
    {"id": "unknown-2", "kind": "unknown", "question": "犬が食べるものを教えて。", "expected": None},
]


class SDMemory:
    def __init__(self):
        self.nodes = {}
        self.index = {}
        for kind, mapping in (("concept", ENTITIES), ("predicate", VERBS)):
            for sid, text in mapping.items():
                self.nodes[sid] = {"kind": kind, "text": text, "refs": []}
        self.nodes["JA-SOV"] = {
            "kind": "template", "text": "{subject}が{object}を{verb}。",
            "roles": ["subject", "object", "verb"], "refs": [],
        }
        for number, triple in enumerate(FACTS, start=1):
            self.add_fact("FACT-" + str(number), triple)

    def add_fact(self, sid, triple):
        if tuple(triple) in self.index:
            return self.index[tuple(triple)]
        s, o, v = triple
        if any(k not in self.nodes for k in triple):
            raise ValueError("Unknown SD component")
        self.nodes[sid] = {
            "kind": "proposition", "template_ref": "JA-SOV",
            "refs": [
                {"role": "subject", "sd_ref": s},
                {"role": "object", "sd_ref": o},
                {"role": "verb", "sd_ref": v},
            ],
            "truth_state": "provided_fact",
        }
        self.index[tuple(triple)] = sid
        return sid

    def render(self, triple):
        s, o, v = triple
        template = self.nodes["JA-SOV"]["text"]
        return template.format(
            subject=self.nodes[s]["text"],
            object=self.nodes[o]["text"],
            verb=self.nodes[v]["text"],
        )

    def resolve_fact(self, question):
        # Search the question by *observed* entities and predicates only.
        # Never read the evaluation case's target triple here.
        verb_ids = [k for k, value in VERBS.items() if value in question]
        entity_ids = [k for k, value in ENTITIES.items() if value in question]
        matches = [
            triple for triple in self.index
            if triple[2] in verb_ids
            and all(e in triple[:2] for e in entity_ids)
        ]
        if len(matches) != 1:
            return None
        s, o, _ = matches[0]
        if "誰" in question:
            return self.nodes[s]["text"]
        if "何" in question or "もの" in question:
            return self.nodes[o]["text"]
        return None

    def compose_requested(self, question):
        # This generates a requested sentence, NOT a factual assertion.
        if "文を作って" not in question:
            return None
        m = re.search(r"主語=(.+?)、目的語=(.+?)、動詞=(.+?)。文を作って。", question)
        if not m:
            return None
        reverse_entities = {v: k for k, v in ENTITIES.items()}
        reverse_verbs = {v: k for k, v in VERBS.items()}
        s = reverse_entities.get(m.group(1))
        o = reverse_entities.get(m.group(2))
        v = reverse_verbs.get(m.group(3))
        if not all([s, o, v]):
            return None
        return self.render((s, o, v))


def lexical_memory(question, memory):
    # Exact whole-sentence retrieval; deliberately no grammar composition.
    # For an unknown or nonmatching query abstain.
    matches = []
    for triple in FACTS:
        s, o, v = triple
        terms = (ENTITIES[s], ENTITIES[o], VERBS[v])
        score = sum(term in question for term in terms)
        matches.append((score, triple))
    best = max(score for score, _ in matches)
    winners = [triple for score, triple in matches if score == best]
    if best < 2 or len(winners) != 1 or "文を作って" in question:
        return None
    s, o, _ = winners[0]
    if "誰" in question:
        return ENTITIES[s]
    if "何" in question or "もの" in question:
        return ENTITIES[o]
    return None


class LLMAdapter:
    def __init__(self, checkpoint, tokenizer_file, device, max_new_tokens):
        import torch
        from model import LanguageModel
        from tokenizer import Tokenizer
        self.torch = torch
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto" else device
        )
        self.tokenizer = Tokenizer.load(tokenizer_file)
        self.model, _ = LanguageModel.load_checkpoint(checkpoint, self.device)
        if self.model.vocab_size != self.tokenizer.vocab_size:
            raise ValueError("Tokenizer/model vocab mismatch")
        self.max_new_tokens = max_new_tokens

    def answer(self, question):
        ids = self.tokenizer.encode(question + "\n回答：", add_bos=True)
        if self.tokenizer.unk_id in ids:
            return None  # Do not silently score missing vocabulary as an answer.
        result = self.model.generate(
            ids, max_new_tokens=self.max_new_tokens,
            eos_id=self.tokenizer.eos_id, temperature=0,
            top_k=None, repetition_penalty=1.0,
        )
        return self.tokenizer.decode(result[len(ids):], skip_special_tokens=True).strip() or None


def normalize(text):
    return re.sub(r"[\s。．.!！?？、，]", "", text or "")


def evaluate(mode, cases, sd, llm=None):
    rows = []
    for case in cases:
        start = time.perf_counter()
        question = case["question"]
        if mode == "A":
            answer = llm.answer(question)
        elif mode == "B":
            answer = lexical_memory(question, sd)
        elif mode == "C":
            answer = sd.resolve_fact(question) if case["kind"] != "compose" else None
        elif mode == "D":
            # Routing by *request syntax*, not hidden benchmark labels.
            answer = (sd.compose_requested(question) if "文を作って" in question
                      else sd.resolve_fact(question))
        else:
            raise ValueError(mode)
        elapsed = time.perf_counter() - start
        expected = case["expected"]
        correct = (answer is None if expected is None
                   else normalize(answer) == normalize(expected))
        rows.append({
            "id": case["id"], "kind": case["kind"],
            "question": question, "expected": expected,
            "answer": answer, "correct": correct,
            "latency_ms": round(elapsed * 1000, 3),
        })
    known = [r for r in rows if r["expected"] is not None]
    unknown = [r for r in rows if r["expected"] is None]
    answered = [r for r in known if r["answer"] is not None]
    return {
        "mode": mode, "total": len(rows),
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "known_accuracy": sum(r["correct"] for r in known) / len(known),
        "coverage": len(answered) / len(known),
        "selective_accuracy": (sum(r["correct"] for r in answered) / len(answered)
                               if answered else None),
        "unknown_reject": sum(r["answer"] is None for r in unknown) / len(unknown),
        "mean_latency_ms": sum(r["latency_ms"] for r in rows) / len(rows),
        "rows": rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="model/model-gpu-v0.3.pt")
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--output", default="results/four_way_sd_comparison.json")
    p.add_argument("--structural-only", action="store_true",
                   help="Run B/C/D only; not a complete four-way comparison.")
    args = p.parse_args()
    if not args.structural_only and (
        not Path(args.checkpoint).is_file() or not Path(args.tokenizer).is_file()
    ):
        p.error("A requires trained checkpoint and tokenizer; "
                "use --structural-only for B/C/D development checks")
    sd = SDMemory()
    llm = None if args.structural_only else LLMAdapter(
        args.checkpoint, args.tokenizer, args.device, args.max_new_tokens
    )
    modes = "BCD" if args.structural_only else "ABCD"
    results = [evaluate(mode, CASES, sd, llm) for mode in modes]
    report = {
        "status": "STRUCTURAL_ONLY_NOT_FOUR_WAY" if args.structural_only
                  else "FOUR_WAY_COMPLETED",
        "note": "Illustrative, development-only fixture; not an independent holdout.",
        "facts": len(FACTS), "cases": len(CASES),
        "modes": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    for r in results:
        print(f'{r["mode"]}: accuracy={r["accuracy"]:.3f} '
              f'coverage={r["coverage"]:.3f} '
              f'unknown_reject={r["unknown_reject"]:.3f}')
    print("Saved:", output)


if __name__ == "__main__":
    main()
