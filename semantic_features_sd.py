"""Experimental SD semantic layers: event core, polarity, tense and topic.

The core proposition is canonical across simple active/passive paraphrases.
A semantic frame separately identifies polarity, tense and topicalization.
This is a narrow *rule-based* recognizer over registered vocabulary, not an
LLM result and not a general parser. Accepted expressions are NOT facts.

Examples:
  猫が魚を食べる。       core=S(CAT,FISH,EAT), POS, NONPAST, UNMARKED
  猫が魚を食べない。   same core, NEG, NONPAST, UNMARKED
  猫が魚を食べた。     same core, POS, PAST, UNMARKED
  猫は魚を食べる。     same core, POS, NONPAST, SUBJECT_TOPIC
  魚が猫に食べられる。 same frame as active, distinct voice metadata
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from recursive_sd import build_sample_graph
from sd_candidate_pipeline import CandidateRejected

FORMS = {
    "EAT": {
        ("POS", "NONPAST"): ("食べる", "食べられる"),
        ("NEG", "NONPAST"): ("食べない", "食べられない"),
        ("POS", "PAST"): ("食べた", "食べられた"),
        ("NEG", "PAST"): ("食べなかった", "食べられなかった"),
    },
    "WATCH": {
        ("POS", "NONPAST"): ("見る", "見られる"),
        ("NEG", "NONPAST"): ("見ない", "見られない"),
        ("POS", "PAST"): ("見た", "見られた"),
        ("NEG", "PAST"): ("見なかった", "見られなかった"),
    },
}
ACTIVE = re.compile(r"^([^がはをに。、？?]+)(が|は)([^がはをに。、？?]+)を([^がはをに。、？?]+)。$")
PASSIVE = re.compile(r"^([^がはをに。、？?]+)が([^がはをに。、？?]+)に([^がはをに。、？?]+)。$")


def add_feature_schema(graph):
    graph.add_atom("POL-POS", "POS", kind="feature")
    graph.add_atom("POL-NEG", "NEG", kind="feature")
    graph.add_atom("TIME-NONPAST", "NONPAST", kind="feature")
    graph.add_atom("TIME-PAST", "PAST", kind="feature")
    graph.add_atom("TOP-UNMARKED", "UNMARKED", kind="feature")
    graph.add_atom("TOP-SUBJECT", "SUBJECT_TOPIC", kind="feature")
    if "T-SEMANTIC-FRAME" not in graph.nodes:
        graph.add_template(
            "T-SEMANTIC-FRAME",
            "{core}|{polarity}|{tense}|{topic}",
            ("core", "polarity", "tense", "topic"),
        )
    return graph


class SemanticFeatureEngine:
    def __init__(self, graph=None):
        self.graph = add_feature_schema(graph if graph is not None else build_sample_graph())

    def _concept_id(self, surface):
        matches = [sid for sid, node in self.graph.nodes.items()
                   if node["kind"] == "concept" and node["text"] == surface]
        if len(matches) != 1:
            raise CandidateRejected("UNKNOWN_OR_AMBIGUOUS_CONCEPT")
        return matches[0]

    @staticmethod
    def _verb(form, voice):
        index = 0 if voice == "active" else 1
        matches = [(verb, polarity, tense)
                   for verb, forms in FORMS.items()
                   for (polarity, tense), pair in forms.items()
                   if pair[index] == form]
        if len(matches) != 1:
            raise CandidateRejected("UNKNOWN_OR_AMBIGUOUS_VERB")
        return matches[0]

    def _parse(self, text):
        if not isinstance(text, str) or not text.strip():
            raise CandidateRejected("EMPTY_INPUT")
        text = text.strip()
        if "？" in text or "?" in text:
            raise CandidateRejected("QUESTION_UNSUPPORTED")
        m = ACTIVE.fullmatch(text)
        if m:
            agent, particle, patient, verb_text = m.groups()
            voice = "active"
            topic = "SUBJECT" if particle == "は" else "UNMARKED"
        else:
            m = PASSIVE.fullmatch(text)
            if m is None:
                raise CandidateRejected("UNSUPPORTED_GRAMMAR")
            patient, agent, verb_text = m.groups()
            voice = "passive"
            topic = "UNMARKED"
        s = self._concept_id(agent)
        o = self._concept_id(patient)
        v, polarity, tense = self._verb(verb_text, voice)
        if v not in self.graph.nodes or self.graph.nodes[v]["kind"] != "predicate":
            raise CandidateRejected("UNREGISTERED_VERB")
        return {"subject": s, "object": o, "verb": v,
                "polarity": polarity, "tense": tense,
                "topic": topic, "voice": voice}

    def analyze(self, text):
        parsed = self._parse(text)  # no mutation until parse succeeds
        # Expressing a proposition is not asserting it. Its core is shared
        # even across negative and past variants; the entire frame is not.
        core = self.graph.compose("T-SOV", {key: parsed[key]
                            for key in ("subject", "object", "verb")})
        frame = self.graph.compose("T-SEMANTIC-FRAME", {
            "core": core,
            "polarity": "POL-" + parsed["polarity"],
            "tense": "TIME-" + parsed["tense"],
            "topic": "TOP-" + parsed["topic"],
        })
        return {
            "status": "ACCEPTED", "input": text, "core_ref": core,
            "frame_ref": frame,
            "core": "S({},{},{})".format(
                parsed["subject"], parsed["object"], parsed["verb"]),
            "polarity": parsed["polarity"], "tense": parsed["tense"],
            "topic": parsed["topic"], "voice": parsed["voice"],
            "verified_fact": False,
        }


def compare_features(left, right, graph=None):
    engine = SemanticFeatureEngine(graph)
    try:
        a = engine.analyze(left)
        b = engine.analyze(right)
    except CandidateRejected as exc:
        return {"status": "UNSUPPORTED", "reason": str(exc),
                "same_core": None, "same_frame": None}
    return {"status": "COMPARED",
            "same_core": a["core_ref"] == b["core_ref"],
            "same_frame": a["frame_ref"] == b["frame_ref"],
            "differences": [field for field in ("polarity", "tense", "topic")
                            if a[field] != b[field]],
            "left": a, "right": b}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--left", default="猫が魚を食べる。")
    p.add_argument("--right", default="猫が魚を食べない。")
    p.add_argument("--output", default="results/semantic_feature_comparison.json")
    args = p.parse_args()
    report = compare_features(args.left, args.right)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
