"""Untrusted SD-candidate boundary: proposer -> validation -> atomic graph merge.

A candidate is a JSON tree of template IDs and *existing* SD IDs.
The validator performs no LLM inference. It admits only exactly reversible
surface representations in the currently registered grammar. Importantly,
a successful parse is NOT approval of a factual claim.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from recursive_sd import build_sample_graph

MAX_DEPTH = 8
MAX_NODES = 40


class CandidateRejected(ValueError):
    pass


def load_candidate(raw):
    """Accept JSON text or a mapping; never evaluate generated Python."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise CandidateRejected("INVALID_JSON") from exc
    if not isinstance(raw, dict):
        raise CandidateRejected("INVALID_ROOT")
    return raw


def _validate_tree(graph, node, level, counter):
    if level > MAX_DEPTH:
        raise CandidateRejected("DEPTH_LIMIT")
    counter[0] += 1
    if counter[0] > MAX_NODES:
        raise CandidateRejected("NODE_LIMIT")
    if not isinstance(node, dict):
        raise CandidateRejected("INVALID_NODE")
    if set(node) == {"sd_ref"}:
        ref = node["sd_ref"]
        if not isinstance(ref, str) or ref not in graph.nodes:
            raise CandidateRejected("UNKNOWN_SD_REF")
        if graph.nodes[ref]["kind"] not in ("concept", "predicate"):
            raise CandidateRejected("INVALID_ATOM_KIND")
        return ref
    if set(node) != {"template_ref", "bindings"}:
        raise CandidateRejected("INVALID_NODE_SCHEMA")
    template_id, bindings = node["template_ref"], node["bindings"]
    template = graph.nodes.get(template_id) if isinstance(template_id, str) else None
    if not template or template["kind"] != "template":
        raise CandidateRejected("UNKNOWN_TEMPLATE")
    if not isinstance(bindings, dict) or set(bindings) != set(template["roles"]):
        raise CandidateRejected("ROLE_MISMATCH")
    children = {role: _validate_tree(graph, bindings[role], level + 1, counter)
                for role in template["roles"]}
    if template_id == "T-SOV":
        for role, allowed in (("subject", "concept"), ("object", "concept"),
                              ("verb", "predicate")):
            if graph.nodes[children[role]]["kind"] != allowed:
                raise CandidateRejected("ROLE_TYPE_MISMATCH")
    elif template_id in ("T-THEN", "T-AND"):
        for role in ("first", "second"):
            child = graph.nodes[children[role]]
            if child["kind"] != "expression":
                raise CandidateRejected("CLAUSE_TYPE_MISMATCH")
    else:
        raise CandidateRejected("UNSUPPORTED_TEMPLATE")
    return graph.compose(template_id, children)


def accept_candidate(graph, text, raw):
    """Validate on a staged copy and merge only when every check passes."""
    before = len(graph.nodes)
    try:
        candidate = load_candidate(raw)
        if set(candidate) != {"tree"} or not isinstance(text, str) or not text.strip():
            raise CandidateRejected("INVALID_ENVELOPE")
        draft = graph.clone()
        root = _validate_tree(draft, candidate["tree"], 0, [0])
        if draft.render(root) != text.strip():
            raise CandidateRejected("SURFACE_MISMATCH")
        if not draft.validate():
            raise CandidateRejected("INVALID_GRAPH")
    except CandidateRejected as exc:
        return {"status": "REJECTED", "reason": str(exc),
                "node_count_before": before, "node_count_after": len(graph.nodes)}
    except (KeyError, ValueError, TypeError, RecursionError) as exc:
        return {"status": "REJECTED", "reason": "STRUCTURAL_ERROR",
                "node_count_before": before, "node_count_after": len(graph.nodes)}
    graph.nodes, graph.canonical, graph.ids = draft.nodes, draft.canonical, draft.ids
    return {"status": "ACCEPTED", "sd_ref": root,
            "rendered": draft.render(root), "verified_fact": False,
            "node_count_before": before, "node_count_after": len(graph.nodes)}


class DeterministicProposer:
    """A development baseline, NOT an LLM or a general Japanese parser."""
    def __init__(self, graph):
        self.graph = graph

    def propose(self, text):
        # Reject ambiguous atoms rather than guessing their meaning.
        indexes = {}
        for kind in ("concept", "predicate"):
            idx = {}
            for sd_id, node in self.graph.nodes.items():
                if node["kind"] == kind:
                    idx.setdefault(node["text"], []).append(sd_id)
            indexes[kind] = idx

        def atom(surface, kind):
            found = indexes[kind].get(surface, [])
            if len(found) != 1:
                raise CandidateRejected("UNKNOWN_OR_AMBIGUOUS_ATOM")
            return {"sd_ref": found[0]}

        def clause(sentence):
            m = re.fullmatch(r"([^がを。、と]+)が([^がを。、と]+)を(.+?)。", sentence)
            if m is None:
                raise CandidateRejected("UNSUPPORTED_GRAMMAR")
            s, o, v = m.groups()
            return {"template_ref": "T-SOV", "bindings": {
                "subject": atom(s, "concept"),
                "object": atom(o, "concept"),
                "verb": atom(v, "predicate")}}

        if "と、" in text and "、そして" in text:
            raise CandidateRejected("MIXED_CONNECTIVES")
        if "と、" in text:
            segments, connector = text.split("と、"), "T-THEN"
        elif "、そして" in text:
            segments, connector = text.split("、そして"), "T-AND"
        else:
            segments, connector = [text], None
        if len(segments) != (2 if connector else 1):
            raise CandidateRejected("UNSUPPORTED_CLAUSE_COUNT")
        trees = [clause(s if s.endswith("。") else s + "。") for s in segments]
        if connector:
            return {"tree": {"template_ref": connector, "bindings": {
                "first": trees[0], "second": trees[1]}}}
        return {"tree": trees[0]}


class LLMCandidateProposer:
    """Optional real LLM inference. No fallback from LLM to rule proposer."""
    def __init__(self, checkpoint, tokenizer, device="auto"):
        from compare_four_modes import LLMAdapter
        self.adapter = LLMAdapter(checkpoint, tokenizer, device, 180)
        self.last_diagnostic = None

    def propose(self, text):
        # Small v0.3 models are not instruction-tuned for JSON and may fail.
        prompt = (
            'Return only JSON: {"tree":{"template_ref":"T-SOV",'
            '"bindings":{"subject":{"sd_ref":"CAT"},'
            '"object":{"sd_ref":"FISH"},"verb":{"sd_ref":"EAT"}}}}. '
            "Use T-THEN/T-AND with first/second for two clauses. "
            "Known entity IDs: CAT=猫 DOG=犬 FISH=魚 BIRD=鳥 RICE=ご飯. "
            "Verb IDs: EAT=食べる WATCH=見る. "
            "No invented IDs. Input: " + text
        )
        ids = self.adapter.tokenizer.encode(prompt, add_bos=True)
        unknown_tokens = sum(token == self.adapter.tokenizer.unk_id for token in ids)
        response = self.adapter.answer(prompt)
        self.last_diagnostic = {
            "prompt_chars": len(prompt),
            "unknown_prompt_tokens": unknown_tokens,
            "response_chars": len(response) if isinstance(response, str) else 0,
            "response_preview": (response[:160] if isinstance(response, str) else None),
        }
        return load_candidate(response)


def run(text, provider, graph=None):
    graph = graph or build_sample_graph()
    try:
        proposed = provider.propose(text)
    except CandidateRejected as exc:
        return {"status": "REJECTED", "reason": str(exc),
                "input": text, "node_count": len(graph.nodes)}
    result = accept_candidate(graph, text, proposed)
    return {"input": text, "provider": type(provider).__name__, **result}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="猫が魚を食べると、犬が鳥を見る。")
    p.add_argument("--provider", choices=("rule", "llm"), default="rule")
    p.add_argument("--checkpoint", default="model/model-gpu-v0.3.pt")
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--output", default="results/sd_candidate_validation.json")
    args = p.parse_args()
    graph = build_sample_graph()
    if args.provider == "rule":
        provider = DeterministicProposer(graph)
    else:
        if not Path(args.checkpoint).is_file() or not Path(args.tokenizer).is_file():
            p.error("LLM provider requires checkpoint and tokenizer")
        provider = LLMCandidateProposer(args.checkpoint, args.tokenizer, args.device)
    result = run(args.text, provider, graph)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
