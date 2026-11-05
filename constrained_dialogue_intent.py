"""Constrained classification by scoring only registered Dialogue Intent SD labels.

Scores P(code + newline + EOS | utterance) using the actual causal
Transformer. This is NOT edit-distance repair and never consults the gold
label. All eight candidate codes are compared under the same model, prompt
and tokenizer. The output code is always syntactically valid but can be
semantically wrong. Unknown-input tokens / context overflow cause abstention.

A token-average log likelihood is used to moderate class-name length bias.
Score margins are diagnostic only, not calibrated probabilities.
"""
from __future__ import annotations

import math

from dialogue_intent_sd import INTENTS

CANDIDATES = tuple("I(" + name + ")" for name in INTENTS)


def build_candidate_sequences(tokenizer, utterance, candidates=CANDIDATES,
                              context_length=96):
    if not isinstance(utterance, str) or not utterance.strip():
        return None, "EMPTY_INPUT"
    prefix = tokenizer.encode("入力:" + utterance + "\nSD:", add_bos=True)
    if tokenizer.unk_id in prefix:
        return None, "UNKNOWN_INPUT_TOKEN"
    sequences = []
    for candidate in candidates:
        suffix = tokenizer.encode(candidate + "\n", add_eos=True)
        if tokenizer.unk_id in suffix:
            return None, "UNKNOWN_OUTPUT_TOKEN"
        full = prefix + suffix
        if len(full) - 1 > context_length:
            return None, "CONTEXT_TOO_LONG"
        sequences.append((candidate, full[:-1], full[1:],
                          len(prefix) - 1, len(suffix)))
    return sequences, None


def choose_best(scored):
    """Pure policy: score entries have code and mean_log_probability fields."""
    if not scored:
        return None
    return sorted(scored, key=lambda row:
                  (-row["mean_log_probability"], row["code"]))[0]["code"]


def score_constrained(model, tokenizer, utterance, device):
    import torch
    import torch.nn.functional as F

    seqs, issue = build_candidate_sequences(tokenizer, utterance,
                        context_length=model.context_length)
    if issue:
        return {"status": "ABSTAIN", "code": None, "reason": issue,
                "scores": []}
    max_len = max(len(entry[1]) for entry in seqs)
    batch = torch.full((len(seqs), max_len), tokenizer.pad_id,
                       dtype=torch.long, device=device)
    for i, (_, x, _, _, _) in enumerate(seqs):
        batch[i, :len(x)] = torch.tensor(x, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(batch)
        log_probs = F.log_softmax(logits, dim=-1)
        scored = []
        for i, (code, x, y, start, n) in enumerate(seqs):
            target = torch.tensor(y[start:start+n], device=device)
            positions = log_probs[i, start:start+n, :]
            mean = float(positions.gather(1, target[:, None]).sum().item() / n)
            if not math.isfinite(mean):
                return {"status": "ABSTAIN", "code": None,
                        "reason": "NONFINITE_SCORE", "scores": []}
            scored.append({"code": code, "mean_log_probability": mean})
    order = sorted(scored, key=lambda row: row["mean_log_probability"],
                   reverse=True)
    return {"status": "PREDICTED", "code": choose_best(scored),
            "reason": None, "margin": (order[0]["mean_log_probability"] -
                                      order[1]["mean_log_probability"]),
            "scores": scored}
