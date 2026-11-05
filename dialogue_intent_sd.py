"""Dialogue Intent SD: first-class linked SD for pragmatic utterance acts.

This module does NOT infer intent from free-form text. It accepts an
untrusted compact intent candidate and creates linked expression nodes
only after validating the candidate, registered intent ID and input type.
It is intentionally separate from Core/Frame semantic assertions.
"""
from __future__ import annotations

import argparse
import json
import re

from recursive_sd import build_sample_graph
from sd_candidate_pipeline import CandidateRejected

INTENTS = {
    "GREETING": "greeting",
    "FAREWELL": "farewell",
    "THANKS": "gratitude",
    "QUESTION": "question",
    "REQUEST": "request",
    "SELF_DISCLOSURE": "self_disclosure",
    "FEELING": "feeling",
    "ACKNOWLEDGEMENT": "acknowledgement",
}
CODE_RE = re.compile(r"^I\(([A-Z][A-Z0-9_]*)\)$")


def add_dialogue_schema(graph):
    if "T-DIALOGUE-ACT" not in graph.nodes:
        graph.add_template("T-DIALOGUE-ACT", "intent:{intent}", ("intent",))
    for intent, label in INTENTS.items():
        graph.add_atom("INTENT-" + intent, label, kind="dialogue_intent")
    return graph


def parse_intent_code(code):
    if not isinstance(code, str) or len(code) > 64:
        raise CandidateRejected("INVALID_INTENT_CODE")
    m = CODE_RE.fullmatch(code)
    if not m or m.group(1) not in INTENTS:
        raise CandidateRejected("UNKNOWN_INTENT")
    return m.group(1)


class DialogueIntentSD:
    def __init__(self, graph=None):
        self.graph = add_dialogue_schema(graph if graph is not None
                                         else build_sample_graph())

    def accept(self, utterance, code):
        """Record intent separately from surface; never claim fact verification."""
        before = len(self.graph.nodes)
        try:
            if not isinstance(utterance, str) or not utterance.strip():
                raise CandidateRejected("EMPTY_UTTERANCE")
            if len(utterance) > 512:
                raise CandidateRejected("UTTERANCE_TOO_LONG")
            intent = parse_intent_code(code)
            # Unambiguous coding of an existing INTENT SD is necessary, but
            # it does not prove that this intent matches the utterance.
            ref = self.graph.compose("T-DIALOGUE-ACT",
                                     {"intent": "INTENT-" + intent})
            return {"status": "ACCEPTED", "intent": intent, "sd_ref": ref,
                    "input": utterance, "verified_fact": False,
                    "semantic_alignment_verified": False}
        except (CandidateRejected, ValueError, TypeError) as exc:
            assert len(self.graph.nodes) == before
            return {"status": "REJECTED", "reason": str(exc)}

    def inspect(self, ref):
        node = self.graph.decompose(ref)
        if node["template_ref"] != "T-DIALOGUE-ACT":
            raise CandidateRejected("NOT_DIALOGUE_ACT")
        raw = node["bindings"]["intent"]
        if not raw.startswith("INTENT-"):
            raise CandidateRejected("INVALID_INTENT_REF")
        return raw[len("INTENT-"):]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="今日は疲れました。")
    p.add_argument("--code", default="I(SELF_DISCLOSURE)")
    args = p.parse_args()
    print(json.dumps(DialogueIntentSD().accept(args.text, args.code),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
