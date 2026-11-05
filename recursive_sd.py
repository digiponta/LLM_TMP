"""Recursive linked Semantic Data: all grammar templates are SD nodes.

The graph represents *expressions*, never silently promotes constructed
expressions into verified factual assertions. Only explicitly approved facts
are indexed as facts. Decomposition here is of graph nodes we created;
parsing arbitrary Japanese is outside the supported scope.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import count


class RecursiveSD:
    def __init__(self):
        self.nodes = {}
        self.canonical = {}
        self.ids = count(1)
        self.add_template("T-SOV", "{subject}が{object}を{verb}。",
                          ("subject", "object", "verb"))
        self.add_template("T-THEN", "{first}と、{second}",
                          ("first", "second"))
        self.add_template("T-AND", "{first}そして{second}",
                          ("first", "second"))

    def add_atom(self, name, text, kind="concept"):
        if name in self.nodes:
            if self.nodes[name]["text"] != text or self.nodes[name]["kind"] != kind:
                raise ValueError("Conflicting SD ID")
            return name
        self.nodes[name] = {"kind": kind, "text": text, "refs": []}
        return name

    def add_template(self, name, pattern, roles):
        if name in self.nodes:
            raise ValueError("Duplicate template ID")
        if len(set(roles)) != len(roles):
            raise ValueError("Duplicate template role")
        import string
        fields = [part for _, part, _, _ in string.Formatter().parse(pattern)
                  if part is not None]
        if set(fields) != set(roles) or len(fields) != len(roles):
            raise ValueError("Template placeholders do not match roles")
        self.nodes[name] = {
            "kind": "template", "text": pattern,
            "roles": list(roles), "refs": [],
        }
        return name

    def compose(self, template_id, bindings, *, verified=False):
        template = self.nodes.get(template_id)
        if not template or template["kind"] != "template":
            raise ValueError("Unknown template SD")
        roles = template["roles"]
        if set(bindings) != set(roles):
            raise ValueError("Missing/extra role binding")
        if any(ref not in self.nodes for ref in bindings.values()):
            raise ValueError("Unknown SD reference")
        # Guard against accidental use of a grammar node as a content element.
        if any(self.nodes[ref]["kind"] == "template" for ref in bindings.values()):
            raise ValueError("Template node cannot fill a content slot")
        key = (template_id, tuple((role, bindings[role]) for role in roles))
        if key in self.canonical:
            node_id = self.canonical[key]
            if verified and not self.nodes[node_id]["verified"]:
                raise ValueError("Approval must be handled outside compose")
            return node_id
        node_id = "SD-" + str(next(self.ids)).zfill(5)
        self.nodes[node_id] = {
            "kind": "expression", "template_ref": template_id,
            "refs": [{"role": role, "sd_ref": bindings[role]} for role in roles],
            "verified": bool(verified),
        }
        self.canonical[key] = node_id
        return node_id

    def decompose(self, node_id):
        node = self.nodes[node_id]
        if node["kind"] != "expression":
            raise ValueError("Only expression SD can be decomposed")
        return {"template_ref": node["template_ref"],
                "bindings": {ref["role"]: ref["sd_ref"] for ref in node["refs"]}}

    def recompose(self, description):
        return self.compose(description["template_ref"], description["bindings"])

    def render(self, node_id, active=None):
        active = set() if active is None else active
        if node_id in active:
            raise ValueError("Cyclic SD reference")
        node = self.nodes[node_id]
        if node["kind"] != "expression":
            return node["text"]
        active.add(node_id)
        try:
            template = self.nodes[node["template_ref"]]
            if template["kind"] != "template":
                raise ValueError("Invalid template reference")
            args = {ref["role"]: self.render(ref["sd_ref"], active)
                    for ref in node["refs"]}
            text = template["text"].format(**args)
            # Nested sentences already end with 。; conjunction expects clauses.
            if node["template_ref"] in ("T-THEN", "T-AND"):
                return text.replace("。と、", "と、").replace("。そして", "、そして")
            return text
        finally:
            active.remove(node_id)

    def validate(self):
        for node_id, node in self.nodes.items():
            if node["kind"] == "expression":
                desc = self.decompose(node_id)
                template = self.nodes.get(desc["template_ref"])
                if not template or template["kind"] != "template":
                    raise ValueError("Invalid template")
                if set(desc["bindings"]) != set(template["roles"]):
                    raise ValueError("Invalid binding")
                self.render(node_id)
        return True

    def snapshot(self):
        return deepcopy(self.nodes)

    def clone(self):
        """Copy SD graph while recreating the ID counter for Python 3.14."""
        replica = object.__new__(type(self))
        replica.nodes = deepcopy(self.nodes)
        replica.canonical = deepcopy(self.canonical)
        occupied = [int(key[3:]) for key in replica.nodes
                    if key.startswith("SD-") and key[3:].isdigit()]
        replica.ids = count(max(occupied, default=0) + 1)
        return replica


def build_sample_graph():
    graph = RecursiveSD()
    for key, text in (
        ("CAT", "猫"), ("DOG", "犬"), ("FISH", "魚"),
        ("BIRD", "鳥"), ("RICE", "ご飯"),
    ):
        graph.add_atom(key, text)
    for key, text in (("EAT", "食べる"), ("WATCH", "見る")):
        graph.add_atom(key, text, kind="predicate")
    return graph


def compose_sentence(graph, subject, obj, verb):
    return graph.compose("T-SOV",
                         {"subject": subject, "object": obj, "verb": verb})


def compose_pair(graph, first, second, connective="T-THEN"):
    return graph.compose(connective, {"first": first, "second": second})
