"""Conservative semantic normalization for active/passive Japanese SOV clauses.

Purpose: test semantic identity rather than literal string identity.
Scope: exactly registered lexemes and the surface patterns below.
Not a general Japanese parser; no factual verification is implied.

Active:  猫が魚を食べる。        -> S(CAT,FISH,EAT)
Passive: 魚が猫に食べられる。  -> S(CAT,FISH,EAT)

The passive construction is interpreted as direct passive with an explicit
agent. In natural Japanese, context can make -られる ambiguous; this module
intentionally refuses unsupported or ambiguous forms.
"""
from __future__ import annotations

import re

from compact_sd import parse_compact, to_compact
from sd_candidate_pipeline import CandidateRejected, accept_candidate
from recursive_sd import build_sample_graph

PASSIVE_VERBS = {"食べられる": "食べる", "見られる": "見る"}
ACT = re.compile(r"^([^がをに。、]+)が([^がをに。、]+)を([^がをに。、]+)\。$")
PAS = re.compile(r"^([^がをに。、]+)が([^がをに。、]+)に([^がをに。、]+)\。$")
CONJUNCTIONS = (("と、", "T"), ("、そして", "A"))


class SemanticNormalizer:
    def __init__(self, graph):
        self.graph = graph

    def _lex(self, surface, kind):
        matches = [name for name, node in self.graph.nodes.items()
                   if node["kind"] == kind and node["text"] == surface]
        if len(matches) != 1:
            raise CandidateRejected("UNKNOWN_OR_AMBIGUOUS_" + kind.upper())
        return matches[0]

    def _simple(self, sentence):
        active = ACT.fullmatch(sentence)
        if active:
            agent, patient, verb = active.groups()
            voice = "active"
        else:
            passive = PAS.fullmatch(sentence)
            if not passive:
                raise CandidateRejected("UNSUPPORTED_GRAMMAR")
            patient, agent, verb_surface = passive.groups()
            if verb_surface not in PASSIVE_VERBS:
                raise CandidateRejected("UNSUPPORTED_PASSIVE_VERB")
            verb = PASSIVE_VERBS[verb_surface]
            voice = "passive"
        s = self._lex(agent, "concept")
        o = self._lex(patient, "concept")
        v = self._lex(verb, "predicate")
        return f"S({s},{o},{v})", voice

    def normalize(self, text):
        if not isinstance(text, str) or not text.strip():
            raise CandidateRejected("EMPTY_INPUT")
        text = text.strip()
        # Explicitly reject negation, questions, and conjunctive ambiguity.
        if "ない" in text or "ません" in text or "？" in text or "?" in text:
            raise CandidateRejected("UNSUPPORTED_POLARITY_OR_QUESTION")
        present = [(join, tag) for join, tag in CONJUNCTIONS if join in text]
        if len(present) > 1:
            raise CandidateRejected("MIXED_CONJUNCTIONS")
        if not present:
            code, voice = self._simple(text)
            return {"code": code, "voices": [voice]}
        join, tag = present[0]
        parts = text.split(join)
        if len(parts) != 2:
            raise CandidateRejected("UNSUPPORTED_CLAUSE_COUNT")
        codes, voices = [], []
        for part in parts:
            code, voice = self._simple(part if part.endswith("。") else part + "。")
            codes.append(code)
            voices.append(voice)
        return {"code": tag + "(" + ",".join(codes) + ")",
                "voices": voices}

    def analyze(self, text):
        normalized = self.normalize(text)
        # Stage-and-merge, using the canonical rendering, not the surface.
        # Original surface and voice information are retained as provenance.
        candidate = parse_compact(normalized["code"])
        root_text = self._canonical_text(candidate)
        result = accept_candidate(self.graph, root_text, candidate)
        if result["status"] != "ACCEPTED":
            return {"input": text, "status": "REJECTED", "reason": result["reason"]}
        return {"input": text, "status": "ACCEPTED",
                "sd_ref": result["sd_ref"], "code": normalized["code"],
                "voices": normalized["voices"],
                "canonical_text": root_text,
                "surface_exact": text.strip() == root_text,
                "verified_fact": False}

    def _canonical_text(self, candidate):
        # Work against a temporary graph so that analysis has one atomic commit.
        draft = self.graph.clone()

        def walk(node):
            if "sd_ref" in node:
                if node["sd_ref"] not in draft.nodes:
                    raise CandidateRejected("UNKNOWN_SD_REF")
                return node["sd_ref"]
            template = draft.nodes.get(node["template_ref"])
            if not template or template["kind"] != "template":
                raise CandidateRejected("UNKNOWN_TEMPLATE")
            refs = {key: walk(value) for key, value in node["bindings"].items()}
            return draft.compose(node["template_ref"], refs)

        root = walk(candidate["tree"])
        return draft.render(root)


def compare_meaning(a, b, graph=None):
    graph = graph or build_sample_graph()
    normalizer = SemanticNormalizer(graph)
    try:
        ra = normalizer.analyze(a)
        rb = normalizer.analyze(b)
    except CandidateRejected as exc:
        return {"status": "UNSUPPORTED", "reason": str(exc), "same_meaning": None}
    if ra["status"] != "ACCEPTED" or rb["status"] != "ACCEPTED":
        return {"status": "UNSUPPORTED", "same_meaning": None,
                "reason": ra.get("reason") or rb.get("reason")}
    return {"status": "COMPARED", "same_meaning": ra["sd_ref"] == rb["sd_ref"],
            "left": ra, "right": rb}


def main():
    import argparse
    import json
    p = argparse.ArgumentParser()
    p.add_argument("--left", default="猫が魚を食べる。")
    p.add_argument("--right", default="魚が猫に食べられる。")
    args = p.parse_args()
    print(json.dumps(compare_meaning(args.left, args.right),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
