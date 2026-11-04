"""Conservative Japanese surface-to-SD parser for the supported grammar.

Scope: known entity and verb lexicons, exactly T-SOV, T-THEN, T-AND.
No general Japanese NLP, inference of factual truth, or learned parsing.
Failures are explicit and do not modify the graph.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from recursive_sd import build_sample_graph, compose_pair, compose_sentence


class UnsupportedSentence(ValueError):
    pass


class JapaneseSDParser:
    def __init__(self, graph):
        self.graph = graph

    def _lexicon(self, kind):
        terms = {}
        for sd_id, node in self.graph.nodes.items():
            if node["kind"] == kind:
                text = node["text"]
                if text in terms:
                    # Same surface with multiple meaning IDs is ambiguous.
                    terms[text].append(sd_id)
                else:
                    terms[text] = [sd_id]
        return terms

    def _lookup(self, surface, kind):
        ids = self._lexicon(kind).get(surface, [])
        if len(ids) != 1:
            raise UnsupportedSentence("Unknown or ambiguous " + kind + ": " + surface)
        return ids[0]

    def _simple(self, text):
        # Full match; never tokenize or guess a missing argument.
        match = re.fullmatch(r"([^がを。、と]+)が([^がを。、と]+)を(.+?)。", text)
        if match is None:
            raise UnsupportedSentence("Unsupported SOV sentence")
        subject, obj, verb = match.groups()
        s = self._lookup(subject, "concept")
        o = self._lookup(obj, "concept")
        v = self._lookup(verb, "predicate")
        return ("T-SOV", {"subject": s, "object": o, "verb": v})

    def parse(self, text):
        if not isinstance(text, str):
            raise TypeError("Expected a Japanese string")
        text = text.strip()
        if not text:
            raise UnsupportedSentence("Empty sentence")
        # First resolve every part without graph mutation.
        # This prevents partial node insertion for malformed inputs.
        if "と、" in text and "、そして" in text:
            raise UnsupportedSentence("Mixed conjunctions are not supported")
        if "と、" in text:
            parts = text.split("と、")
            template = "T-THEN"
        elif "、そして" in text:
            parts = text.split("、そして")
            template = "T-AND"
        else:
            parts = [text]
            template = None
        if template is not None and len(parts) != 2:
            raise UnsupportedSentence("Only two-clause compositions are supported")
        descriptors = []
        for part in parts:
            if not part.endswith("。"):
                part += "。"
            descriptors.append(self._simple(part))
        # All validation completed. Create/reuse interned nodes.
        ids = [self.graph.compose(t, refs) for t, refs in descriptors]
        parent = compose_pair(self.graph, ids[0], ids[1], template) if template else ids[0]
        if self.graph.render(parent) != text:
            raise UnsupportedSentence("Round-trip surface mismatch")
        return parent

    def analyze(self, text):
        parent = self.parse(text)
        def tree(node_id):
            node = self.graph.nodes[node_id]
            if node["kind"] != "expression":
                return {"sd_ref": node_id}
            return {
                "sd_ref": node_id,
                "template_ref": node["template_ref"],
                "bindings": {
                    entry["role"]: tree(entry["sd_ref"])
                    for entry in node["refs"]
                },
            }
        return {
            "input": text,
            "sd_ref": parent,
            "rendered": self.graph.render(parent),
            "roundtrip_equal": self.graph.render(parent) == text.strip(),
            "verified_fact": False,
            "tree": tree(parent),
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="猫が魚を食べると、犬が鳥を見る。")
    p.add_argument("--output", default="results/japanese_sd_roundtrip.json")
    args = p.parse_args()
    graph = build_sample_graph()
    parser = JapaneseSDParser(graph)
    try:
        result = parser.analyze(args.text)
        result["status"] = "OK"
        result["graph_valid"] = graph.validate()
        result["node_count"] = len(graph.nodes)
    except UnsupportedSentence as exc:
        result = {"status": "UNSUPPORTED", "input": args.text, "reason": str(exc)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
