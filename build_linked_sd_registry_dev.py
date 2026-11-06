"""Create the v0.3.1 GUI registry from existing RecursiveSD sample atoms.

This is a SAMPLE registry: real-person conversations need separately approved
SD nodes and relations. No inferred facts, aliases or relation approvals.
"""
import argparse
import hashlib
import json
from pathlib import Path

from recursive_sd import build_sample_graph
from intent_sd_context_v03 import SDRegistry


def build_registry(graph):
    nodes = []
    names = {}
    for sd_id, node in sorted(graph.nodes.items()):
        kind = node.get("kind")
        if kind == "concept":
            nodes.append({"sd_id": sd_id, "node_type": "CONCEPT"})
            names[sd_id] = node["text"]
        elif kind == "predicate":
            nodes.append({"sd_id": sd_id, "node_type": "PREDICATE",
                          "continuity_allowed": False})
            names[sd_id] = node["text"]
        # Grammar templates and expression SDs are not approved Topic
        # relationships; do not invent Relation SDs from them.
    suffix = hashlib.sha256(json.dumps(nodes, ensure_ascii=False,
                              sort_keys=True).encode("utf-8")).hexdigest()[:16]
    snapshot = {"namespace": "local", "snapshot_id": "recursive-sample-" + suffix,
                "nodes": nodes, "display_names": names}
    SDRegistry({k: v for k, v in snapshot.items() if k != "display_names"})
    return snapshot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/linked_sd_registry_dev.json")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    dest = Path(args.output)
    if dest.exists() and not args.overwrite:
        raise ValueError("REGISTRY_EXISTS_USE_OVERWRITE")
    data = build_registry(build_sample_graph())
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print(json.dumps({"status": "VALID_SAMPLE_REGISTRY",
                      "output": str(dest), "sd_nodes": len(data["nodes"]),
                      "relations": 0, "snapshot_id": data["snapshot_id"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
