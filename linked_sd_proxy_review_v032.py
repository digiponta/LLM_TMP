"""Untrusted, deterministic topic proposals from original text; human approval only.

No label, AI-proxy preference, model scores or private transcripts enter
suggestion output. This is a conservative surface-match baseline, NOT an LLM.
"""
import re
from linked_sd_annotation_v031 import (
    FIELDS, UNKNOWN, catalog, prepare_blind_packet, validate_packet,
    digest, assign, save_atomic
)

def _mentions(text, item):
    label = item["display_name"]
    sid = item["sd_id"]
    # Display names that merely repeat IDs are not language evidence.
    if label == sid or not label.strip():
        return False
    if label.isascii() and label.isalnum():
        pattern = r"(?<![A-Za-z0-9_])" + re.escape(label) + r"(?![A-Za-z0-9_])"
        return bool(re.search(pattern, text, re.IGNORECASE))
    return label in text

def _unique_topic(text, options):
    if not text.strip():
        return UNKNOWN, "EMPTY_TEXT"
    hits = sorted({n["sd_id"] for n in options if _mentions(text, n)})
    if len(hits) == 1:
        return hits[0], "EXACT_DISPLAY_NAME_MATCH"
    return UNKNOWN, "NO_MATCH" if not hits else "AMBIGUOUS_MATCH"

def _sides(text, options):
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if not lines:
        return (UNKNOWN, "EMPTY_TEXT"), (UNKNOWN, "EMPTY_TEXT")
    # No invented turn segmentation: last line as current only when
    # original source actually contains at least 2 newline-delimited lines.
    if len(lines) == 1:
        return (UNKNOWN, "NO_TURN_BOUNDARY"), _unique_topic(lines[0], options)
    return _unique_topic("\n".join(lines[:-1]), options), _unique_topic(lines[-1], options)

def propose_case(case, snapshot):
    options = catalog(snapshot)
    result = {}
    reasons = {}
    for prefix, source_key in (("a", "context_a"), ("b", "context_b")):
        prev, current = _sides(case[source_key], options)
        result[prefix + "_previous"] = prev[0]
        result[prefix + "_current"] = current[0]
        reasons[prefix + "_previous"] = prev[1]
        reasons[prefix + "_current"] = current[1]
        # Reference resolution needs discourse structure; do not guess
        # it by scanning the fixed response (would bias A/B judgement).
        result[prefix + "_reference"] = UNKNOWN
        reasons[prefix + "_reference"] = "CONTEXT_REFERENCE_NOT_RESOLVED"
    return {"topic_refs": result, "reason_codes": reasons,
            "provenance": "RULE_CANDIDATE",
            "summary": ("既存SDの表示名との厳密一致だけを候補に採用。"
                        "話者境界が不明な項目と文脈参照先はUNKNOWN。"
                        "正誤・会話適合性を判断したものではありません。")}

def prepare_proposals(source, snapshot):
    base = prepare_blind_packet(source, snapshot)
    by_id = {c["case_id"]: c for c in source["cases"]}
    return {
        "schema_version": "linked-sd-proposals/0.3.2",
        "scope": "DEV_BLIND_PROXY_CANDIDATES",
        "registry_digest": base["registry_digest"],
        "source_digest": base["source_digest"],
        "case_ids": base["case_ids"],
        "proposals": {cid: propose_case(by_id[cid], snapshot) for cid in base["case_ids"]},
        "decisions": {cid: "PENDING" for cid in base["case_ids"]},
    }

def validate_proposals(packet, source, snapshot):
    fresh = prepare_proposals(source, snapshot)
    for key in ("schema_version", "scope", "registry_digest",
                "source_digest", "case_ids", "proposals"):
        if packet.get(key) != fresh[key]:
            raise ValueError("PROPOSAL_SOURCE_OR_ENGINE_CHANGED:" + key)
    decisions = packet.get("decisions")
    if not isinstance(decisions, dict) or set(decisions) != set(fresh["case_ids"]):
        raise ValueError("INVALID_DECISION_IDS")
    if any(x not in ("PENDING", "APPROVED", "REJECTED") for x in decisions.values()):
        raise ValueError("INVALID_DECISION_STATUS")
    return packet

def decide(packet, cid, choice):
    if cid not in packet["decisions"] or choice not in ("APPROVED", "REJECTED", "PENDING"):
        raise ValueError("INVALID_APPROVAL")
    packet["decisions"][cid] = choice

def export_approvals(packet, source, snapshot, reviewer_id="LOCAL_USER_APPROVER"):
    validate_proposals(packet, source, snapshot)
    if not isinstance(reviewer_id, str) or not reviewer_id.strip():
        raise ValueError("REVIEWER_ID_REQUIRED")
    approved = prepare_blind_packet(source, snapshot)
    allowed = {x["sd_id"] for x in catalog(snapshot)}
    for cid in approved["case_ids"]:
        if packet["decisions"][cid] == "APPROVED":
            proposal = packet["proposals"][cid]
            # Human explicitly approves a computer-produced candidate.
            # This does not turn AI/proxy labels into A/B gold.
            rationale = ("人手承認済みのRULE_CANDIDATE。"
                         + proposal["summary"]
                         + " 原因コード: "
                         + ", ".join(k + "=" + proposal["reason_codes"][k]
                                     for k in FIELDS))
            assign(approved, cid, proposal["topic_refs"], rationale,
                   reviewer_id, allowed)
    validate_packet(approved, source, snapshot)
    return approved

def audit_approvals(packet, source, snapshot):
    validate_proposals(packet, source, snapshot)
    counts = {status: list(packet["decisions"].values()).count(status)
              for status in ("PENDING", "APPROVED", "REJECTED")}
    return {"total": len(packet["case_ids"]), "decisions": counts,
            "status": "REVIEW_COMPLETE" if counts["PENDING"] == 0 else "APPROVAL_PENDING",
            "warning": "Approvals are of proxy topic suggestions, not human A/B gold."}
