"""Compact SD notation for supervised generation and safe graph validation.

Single sentence: S(CAT,FISH,EAT)
Two clauses:    T(S(CAT,FISH,EAT),S(DOG,BIRD,WATCH))
Conjunction:    A(S(CAT,FISH,EAT),S(DOG,BIRD,WATCH))

Only IDs from an existing SD memory may be referenced. This is *not* proof
of factual truth. All accepted expressions remain unverified.
"""
from __future__ import annotations

import re

from recursive_sd import build_sample_graph
from sd_candidate_pipeline import CandidateRejected, accept_candidate

ALIASES = {"S": ("T-SOV", ("subject", "object", "verb")),
           "T": ("T-THEN", ("first", "second")),
           "A": ("T-AND", ("first", "second"))}
TOKEN = re.compile(r"[A-Z][A-Z0-9_-]*|[(),]")
MAX_LENGTH = 160
MAX_DEPTH = 8


def parse_compact(raw):
    if not isinstance(raw, str):
        raise CandidateRejected("COMPACT_NOT_TEXT")
    text = raw.strip()
    if len(text) > MAX_LENGTH or not text:
        raise CandidateRejected("COMPACT_LENGTH")
    tokens = TOKEN.findall(text)
    if "".join(tokens) != text:
        raise CandidateRejected("COMPACT_ILLEGAL_CHAR")
    position = 0

    def consume(item):
        nonlocal position
        if position >= len(tokens) or tokens[position] != item:
            raise CandidateRejected("COMPACT_SYNTAX")
        position += 1

    def expr(depth=0):
        nonlocal position
        if depth > MAX_DEPTH or position >= len(tokens):
            raise CandidateRejected("COMPACT_DEPTH_OR_EOF")
        tag = tokens[position]
        position += 1
        if tag not in ALIASES:
            raise CandidateRejected("COMPACT_UNKNOWN_TAG")
        template, roles = ALIASES[tag]
        consume("(")
        bindings = {}
        for idx, role in enumerate(roles):
            if idx:
                consume(",")
            if tag == "S":
                if position >= len(tokens):
                    raise CandidateRejected("COMPACT_EOF")
                sd_id = tokens[position]
                position += 1
                if sd_id in ("(", ")", ","):
                    raise CandidateRejected("COMPACT_BAD_REF")
                bindings[role] = {"sd_ref": sd_id}
            else:
                bindings[role] = expr(depth + 1)
        consume(")")
        return {"template_ref": template, "bindings": bindings}

    tree = expr()
    if position != len(tokens):
        raise CandidateRejected("COMPACT_TRAILING")
    return {"tree": tree}


def to_compact(candidate):
    """Encode a canonical valid shape, without checking referenced SD IDs."""
    tree = candidate["tree"]
    inverse = {template: (tag, roles)
               for tag, (template, roles) in ALIASES.items()}

    def encode(node):
        template = node["template_ref"]
        if template not in inverse:
            raise CandidateRejected("COMPACT_UNKNOWN_TEMPLATE")
        tag, roles = inverse[template]
        refs = node["bindings"]
        if set(refs) != set(roles):
            raise CandidateRejected("COMPACT_ROLES")
        if tag == "S":
            items = [refs[r]["sd_ref"] for r in roles]
        else:
            items = [encode(refs[r]) for r in roles]
        return tag + "(" + ",".join(items) + ")"

    return encode(tree)


def accept_compact(graph, original_text, compact):
    """Reuse the existing atomic SD Engine; no rule-parser fallback."""
    return accept_candidate(graph, original_text, parse_compact(compact))


def main():
    import argparse
    import json
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="猫が魚を食べると、犬が鳥を見る。")
    p.add_argument("--compact", default="T(S(CAT,FISH,EAT),S(DOG,BIRD,WATCH))")
    args = p.parse_args()
    result = accept_compact(build_sample_graph(), args.text, args.compact)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
