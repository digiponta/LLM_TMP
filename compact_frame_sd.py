"""Compact SD frame codec for separately linked Core and Frame semantics.

F(S(CAT,FISH,EAT),NEG,PAST,UNMARKED) denotes:
  Core proposition: CAT/FISH/EAT
  Polarity: NEG
  Tense: PAST
  Topic: UNMARKED

Voice is a surface-level realization, not part of semantic-frame identity
for the supported direct active/passive alternation. We validate candidates
by reconstructing the supplied surface, not by substituting a rule-generated
candidate. No factual assertions are auto-approved.
"""
from __future__ import annotations
import re

from sd_candidate_pipeline import CandidateRejected
from semantic_features_sd import SemanticFeatureEngine, FORMS

PATTERN = re.compile(
    r"^F\(S\(([A-Z][A-Z0-9_-]*),([A-Z][A-Z0-9_-]*),"
    r"([A-Z][A-Z0-9_-]*)\),(POS|NEG),(NONPAST|PAST),"
    r"(UNMARKED|SUBJECT_TOPIC)\)$"
)


def parse_frame_code(code):
    if not isinstance(code, str) or len(code) > 100:
        raise CandidateRejected("INVALID_FRAME_LENGTH")
    m = PATTERN.fullmatch(code.strip())
    if not m:
        raise CandidateRejected("INVALID_FRAME_SYNTAX")
    s, o, v, pol, tense, topic = m.groups()
    return {"subject": s, "object": o, "verb": v,
            "polarity": pol, "tense": tense, "topic": topic}


def encode_frame_code(parts):
    keys = ("subject", "object", "verb", "polarity", "tense", "topic")
    if set(parts) != set(keys):
        raise CandidateRejected("INVALID_FRAME_FIELDS")
    code = "F(S({},{},{}),{},{},{})".format(*(parts[k] for k in keys))
    parse_frame_code(code)
    return code


def realize_frame(parts, graph, voice):
    """Render canonical supported surface from a candidate frame."""
    if voice not in ("active", "passive"):
        raise CandidateRejected("UNKNOWN_VOICE")
    s, o, v = (parts[key] for key in ("subject", "object", "verb"))
    for ref, kind in ((s, "concept"), (o, "concept"), (v, "predicate")):
        node = graph.nodes.get(ref)
        if not node or node["kind"] != kind:
            raise CandidateRejected("UNKNOWN_OR_ILL_TYPED_SD")
    if v not in FORMS:
        raise CandidateRejected("UNKNOWN_VERB")
    verb = FORMS[v][(parts["polarity"], parts["tense"])][
        0 if voice == "active" else 1
    ]
    subj = graph.nodes[s]["text"]
    obj = graph.nodes[o]["text"]
    if voice == "passive":
        if parts["topic"] != "UNMARKED":
            raise CandidateRejected("UNSUPPORTED_PASSIVE_TOPIC")
        return obj + "が" + subj + "に" + verb + "。"
    particle = "は" if parts["topic"] == "SUBJECT_TOPIC" else "が"
    return subj + particle + obj + "を" + verb + "。"


def accept_frame_code(engine, surface, code):
    """Atomic acceptance; a failed generation never modifies Semantic Memory."""
    try:
        parts = parse_frame_code(code)
        # Candidate-driven realization. Avoid using a rule-based source parser
        # to silently correct the LLM's candidate.
        voices = [voice for voice in ("active", "passive")
                  if _matches(parts, engine.graph, voice, surface)]
        if len(voices) != 1:
            raise CandidateRejected("SURFACE_OR_ROLE_MISMATCH")
        # Only now, after all validation, create nodes.
        graph = engine.graph
        core = graph.compose("T-SOV", {k: parts[k]
                                      for k in ("subject", "object", "verb")})
        frame = graph.compose("T-SEMANTIC-FRAME", {
            "core": core,
            "polarity": "POL-" + parts["polarity"],
            "tense": "TIME-" + parts["tense"],
            "topic": "TOP-" + ("SUBJECT" if parts["topic"] == "SUBJECT_TOPIC"
                              else "UNMARKED"),
        })
        return {"status": "ACCEPTED", "core_ref": core,
                "frame_ref": frame, "voice": voices[0],
                "verified_fact": False}
    except (CandidateRejected, ValueError, KeyError, TypeError) as exc:
        return {"status": "REJECTED", "reason": str(exc)}


def _matches(parts, graph, voice, surface):
    try:
        return realize_frame(parts, graph, voice) == surface.strip()
    except (CandidateRejected, KeyError):
        return False


def main():
    import argparse
    import json
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="猫が魚を食べなかった。")
    p.add_argument("--code", default="F(S(CAT,FISH,EAT),NEG,PAST,UNMARKED)")
    args = p.parse_args()
    print(json.dumps(accept_frame_code(SemanticFeatureEngine(), args.text,
                                       args.code), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
