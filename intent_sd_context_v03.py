"""Intent SD v0.3 linked-topic SD runtime (read-only, deterministic).

The registry is a supplied frozen snapshot; this module never creates,
approves, or promotes Semantic Data. No natural-language extraction.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy

TOPIC_KINDS = frozenset(("TOPIC", "CONCEPT", "ENTITY"))
ACTS = frozenset(("QUESTION", "REQUEST", "ANSWER", "INFORM",
                  "ACKNOWLEDGE", "OTHER", "UNKNOWN"))
PROVENANCE = frozenset(("RULE_CANDIDATE", "MODEL_INFERRED", "HUMAN_ANNOTATED"))
TRANSITIONS = frozenset(("SAME", "RELATED", "RESUMED", "SHIFTED", "UNKNOWN"))
FORBIDDEN = frozenset(("reference_label", "proxy_label", "baseline_prediction",
                       "intent_prediction", "gold_label", "correct_answer"))

def _exact(obj, required, optional=()):
    if not isinstance(obj, dict) or not set(required) <= set(obj) or set(obj) - set(required) - set(optional):
        raise ValueError("INVALID_SCHEMA_FIELDS")

def _str(s):
    if not isinstance(s, str) or not s.strip() or s != s.strip():
        raise ValueError("INVALID_IDENTIFIER")
    return s

def sd_ref(value):
    if value is None:
        return None
    _exact(value, ("namespace", "sd_id"))
    return (_str(value["namespace"]), _str(value["sd_id"]))

def ref_dict(ref):
    return None if ref is None else {"namespace": ref[0], "sd_id": ref[1]}

class SDRegistry:
    """Read-only interface to a frozen JSON SD snapshot."""
    def __init__(self, snapshot):
        _exact(snapshot, ("namespace", "nodes"), ("snapshot_id",))
        self.namespace = _str(snapshot["namespace"])
        nodes = snapshot["nodes"]
        if not isinstance(nodes, list):
            raise ValueError("INVALID_NODES")
        self._nodes = {}
        for node in nodes:
            _exact(node, ("sd_id", "node_type"),
                   ("predicate_ref", "source_ref", "target_ref",
                    "validation_state", "evidence_id", "continuity_allowed"))
            key = _str(node["sd_id"])
            if key in self._nodes:
                raise ValueError("DUPLICATE_SD_ID")
            if node["node_type"] not in TOPIC_KINDS | {"RELATION", "PREDICATE"}:
                raise ValueError("INVALID_NODE_TYPE")
            self._nodes[key] = deepcopy(node)
        # Reject dangling relation refs and malformed relation/predicate schemas.
        for node in self._nodes.values():
            if node["node_type"] == "RELATION":
                for field in ("predicate_ref", "source_ref", "target_ref",
                              "validation_state", "evidence_id"):
                    if field not in node:
                        raise ValueError("INVALID_RELATION")
                if node["validation_state"] not in ("APPROVED", "PENDING", "REJECTED"):
                    raise ValueError("INVALID_RELATION_STATE")
                _str(node["evidence_id"])
                for field, kinds in (("predicate_ref", {"PREDICATE"}),
                                     ("source_ref", TOPIC_KINDS),
                                     ("target_ref", TOPIC_KINDS)):
                    dest = _str(node[field])
                    if dest not in self._nodes or self._nodes[dest]["node_type"] not in kinds:
                        raise ValueError("DANGLING_RELATION_REF")
            elif node["node_type"] == "PREDICATE":
                if not isinstance(node.get("continuity_allowed"), bool):
                    raise ValueError("INVALID_PREDICATE_POLICY")

    def contains_topic(self, ref):
        return (ref is not None and ref[0] == self.namespace
                and ref[1] in self._nodes
                and self._nodes[ref[1]]["node_type"] in TOPIC_KINDS)

    def validate_topic(self, ref):
        return "OK" if self.contains_topic(ref) else "UNRESOLVED"

    def relation(self, ref):
        if ref is None or ref[0] != self.namespace:
            return None
        node = self._nodes.get(ref[1])
        return node if node and node["node_type"] == "RELATION" else None

    def relation_path(self, start, end, max_hops=2, max_nodes=64):
        """Search directed, approved, evidence-backed SD edges.

        A pending edge reachable during traversal makes an unsuccessful
        search inconclusive, not SHIFTED. Bounds also produce UNKNOWN.
        """
        if not isinstance(max_hops, int) or not isinstance(max_nodes, int) or max_hops < 1 or max_nodes < 1:
            raise ValueError("INVALID_GRAPH_BOUNDS")
        if not self.contains_topic(start) or not self.contains_topic(end):
            return {"status": "UNKNOWN", "reason": "UNRESOLVED_SD_REF", "evidence_path": []}
        if start == end:
            return {"status": "FOUND", "reason": "SAME_SD", "evidence_path": []}
        edges = {}
        pending_sources = set()
        for rel in self._nodes.values():
            if rel["node_type"] != "RELATION":
                continue
            predicate = self._nodes[rel["predicate_ref"]]
            if not predicate["continuity_allowed"]:
                continue
            source = rel["source_ref"]
            if rel["validation_state"] != "APPROVED":
                if rel["validation_state"] == "PENDING":
                    pending_sources.add(source)
                continue
            edges.setdefault(source, []).append(rel)
        for group in edges.values():
            group.sort(key=lambda r: r["sd_id"])
        queue = deque([(start[1], [])])
        seen = {start[1]}
        incomplete = False
        while queue:
            node_id, path = queue.popleft()
            if node_id in pending_sources:
                incomplete = True
            if len(path) >= max_hops:
                if edges.get(node_id):
                    incomplete = True
                continue
            for rel in edges.get(node_id, []):
                nxt = rel["target_ref"]
                step = {"relation_ref": {"namespace": self.namespace, "sd_id": rel["sd_id"]},
                        "predicate_ref": {"namespace": self.namespace, "sd_id": rel["predicate_ref"]},
                        "evidence_id": rel["evidence_id"]}
                new_path = path + [step]
                if nxt == end[1]:
                    return {"status": "FOUND", "reason": "APPROVED_RELATION", "evidence_path": new_path}
                if nxt not in seen:
                    if len(seen) >= max_nodes:
                        return {"status": "UNKNOWN", "reason": "GRAPH_BOUND", "evidence_path": []}
                    seen.add(nxt)
                    queue.append((nxt, new_path))
        return {"status": "UNKNOWN" if incomplete else "NOT_FOUND",
                "reason": "INCOMPLETE_GRAPH" if incomplete else "NO_APPROVED_PATH",
                "evidence_path": []}

def resolve_topic_ref(candidate, registry):
    ref = sd_ref(candidate)
    return {"status": registry.validate_topic(ref), "topic_ref": ref_dict(ref) if registry.contains_topic(ref) else None}

def resolve_context_reference(item, history, registry):
    _exact(item, ("status",), ("target_utterance_id", "topic_ref"))
    status = item["status"]
    if status not in ("RESOLVED", "UNRESOLVED", "NONE"):
        raise ValueError("INVALID_CONTEXT_STATUS")
    if status != "RESOLVED":
        if item.get("target_utterance_id") is not None or item.get("topic_ref") is not None:
            raise ValueError("INVALID_UNRESOLVED_REFERENCE")
        return {"status": status, "topic_ref": None}
    target = _str(item.get("target_utterance_id"))
    ref = sd_ref(item.get("topic_ref"))
    matching = [h for h in history if h.get("utterance_id") == target]
    if len(matching) != 1 or not registry.contains_topic(ref):
        return {"status": "UNRESOLVED", "topic_ref": None}
    actual = sd_ref(matching[0].get("topic_ref"))
    if actual != ref:
        return {"status": "UNRESOLVED", "topic_ref": None}
    return {"status": "RESOLVED", "topic_ref": ref_dict(ref),
            "target_utterance_id": target}

def lookup_relation(previous_ref, current_ref, registry, **kwargs):
    return registry.relation_path(sd_ref(previous_ref), sd_ref(current_ref), **kwargs)

def classify_topic_transition(intent_sd, state, registry, max_hops=2, max_nodes=64):
    """State is a read-only dict with history of prior validated utterances."""
    if isinstance(intent_sd, dict) and FORBIDDEN & set(intent_sd):
        raise ValueError("LABEL_LEAKAGE")
    _exact(intent_sd, ("schema_version", "utterance_id", "speech_act",
                       "intent", "topic_ref", "previous_topic_ref",
                       "context_reference", "topic_transition", "provenance"))
    if FORBIDDEN & set(intent_sd):
        raise ValueError("LABEL_LEAKAGE")
    if intent_sd["schema_version"] != "intent-sd/0.3":
        raise ValueError("INVALID_VERSION")
    _str(intent_sd["utterance_id"])
    _str(intent_sd["intent"])
    if intent_sd["speech_act"] not in ACTS or intent_sd["provenance"] not in PROVENANCE:
        raise ValueError("INVALID_INTENT_FIELDS")
    history = state.get("history", [])
    if not isinstance(history, list):
        raise ValueError("INVALID_HISTORY")
    _exact(intent_sd["topic_transition"], ("kind",), ("relation_ref",))
    requested_kind = intent_sd["topic_transition"]["kind"]
    if requested_kind not in TRANSITIONS:
        raise ValueError("INVALID_TRANSITION")
    prev = sd_ref(intent_sd["previous_topic_ref"])
    curr = sd_ref(intent_sd["topic_ref"])
    active = sd_ref(state.get("active_topic_ref"))
    result = {"status": "UNKNOWN", "reason_codes": [], "evidence_path": [],
              "provenance": intent_sd["provenance"], "topic_ref": ref_dict(curr)}
    def decision(kind, reason, path=None):
        result.update(status=kind, reason_codes=[reason], evidence_path=path or [])
        return result
    if not registry.contains_topic(prev) or not registry.contains_topic(curr):
        return decision("UNKNOWN", "UNRESOLVED_SD_REF")
    if active is not None and prev != active:
        return decision("UNKNOWN", "PREVIOUS_TOPIC_STATE_MISMATCH")
    resolved = resolve_context_reference(intent_sd["context_reference"], history, registry)
    if resolved["status"] == "UNRESOLVED":
        return decision("UNKNOWN", "UNRESOLVED_CONTEXT_REFERENCE")
    if resolved["status"] == "RESOLVED":
        ref = sd_ref(resolved["topic_ref"])
        if ref != prev and ref != curr:
            return decision("UNKNOWN", "CONTEXT_TOPIC_MISMATCH")
    if prev == curr:
        kind, reason = "SAME", "IDENTICAL_SD_REF"
        path = []
    else:
        earlier = [h for h in history if sd_ref(h.get("topic_ref")) == curr and h.get("verified") is True]
        explicit = resolved.get("target_utterance_id") if resolved["status"] == "RESOLVED" else None
        if earlier and explicit is not None and any(h["utterance_id"] == explicit for h in earlier):
            kind, reason, path = "RESUMED", "VERIFIED_PRIOR_TURN", []
        else:
            found = registry.relation_path(prev, curr, max_hops=max_hops, max_nodes=max_nodes)
            if found["status"] == "FOUND":
                kind, reason, path = "RELATED", found["reason"], found["evidence_path"]
            elif found["status"] == "NOT_FOUND":
                kind, reason, path = "SHIFTED", found["reason"], []
            else:
                return decision("UNKNOWN", found["reason"])
    claimed_relation = sd_ref(intent_sd["topic_transition"].get("relation_ref"))
    if claimed_relation is not None:
        if kind != "RELATED" or not any(sd_ref(x["relation_ref"]) == claimed_relation for x in path):
            return decision("UNKNOWN", "CLAIMED_RELATION_NOT_IN_PATH")
    # User-proposed transition kind is not a gold label; conflicting
    # claims must be rejected instead of silently accepted.
    if requested_kind not in ("UNKNOWN", kind):
        return decision("UNKNOWN", "TRANSITION_CLAIM_CONFLICT")
    return decision(kind, reason, path)

def apply_context_transition(state, intent_sd, decision, max_history=32):
    """Returns a new transient state; never mutates SD registry or input state."""
    if decision["status"] not in ("SAME", "RELATED", "RESUMED", "SHIFTED"):
        return deepcopy(state)
    if not isinstance(max_history, int) or max_history < 1:
        raise ValueError("INVALID_HISTORY_BOUND")
    updated = deepcopy(state)
    prior = updated.get("history", [])
    if any(x["utterance_id"] == intent_sd["utterance_id"] for x in prior):
        raise ValueError("DUPLICATE_UTTERANCE_ID")
    updated["history"] = (prior + [{"utterance_id": intent_sd["utterance_id"],
                                   "topic_ref": deepcopy(intent_sd["topic_ref"]),
                                   "verified": True}])[-max_history:]
    updated["active_topic_ref"] = deepcopy(intent_sd["topic_ref"])
    return updated

def choose_context_v03(a, b):
    left, right = a["status"], b["status"]
    compatible = ("SAME", "RELATED", "RESUMED")
    if left == "UNKNOWN" or right == "UNKNOWN":
        label = "UNCLEAR"
    elif left in compatible and right == "SHIFTED":
        label = "A"
    elif right in compatible and left == "SHIFTED":
        label = "B"
    elif left in compatible and right in compatible:
        label = "BOTH"
    elif left == right == "SHIFTED":
        label = "NEITHER"
    else:
        label = "UNCLEAR"
    return {"label": label, "a": left, "b": right}

def v02_to_v03(label, mapping, registry):
    """Versioned explicit alias map. No SD node creation or synonym guess."""
    if not isinstance(mapping, dict) or mapping.get("mapping_version") is None:
        raise ValueError("UNVERSIONED_MAPPING")
    aliases = mapping.get("aliases")
    if not isinstance(aliases, dict):
        raise ValueError("INVALID_ALIASES")
    candidate = aliases.get(label) if isinstance(label, str) else None
    if not isinstance(candidate, dict):
        return {"status": "UNKNOWN", "topic_ref": None}
    result = resolve_topic_ref(candidate, registry)
    return {"status": result["status"] if result["status"] == "OK" else "UNKNOWN",
            "topic_ref": result["topic_ref"]}
