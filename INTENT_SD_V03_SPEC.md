# Intent SD v0.3 — Linked SD Topic Reference (Proposed Specification)

Status: proposal only; no implementation or performance claim. Parent: v0.2.

## Decision
Topic identity is a reference to an **existing** SD node, never an ad-hoc textual category. Topic relations reference existing approved **Relation SD** nodes. Local SD IDs have meaning only within their namespace; inter-VM mapping is delegated to SKI. Unknown references never create nodes automatically.

## Proposed Intent SD schema

```json
{
  "schema_version": "intent-sd/0.3",
  "utterance_id": "U-123",
  "speech_act": "ANSWER",
  "intent": "ANSWER_LOCATION",
  "topic_ref": {"namespace": "local", "sd_id": "SD-KYOTO"},
  "previous_topic_ref": {"namespace": "local", "sd_id": "SD-TRAVEL"},
  "context_reference": {
    "status": "RESOLVED",
    "target_utterance_id": "U-122",
    "topic_ref": {"namespace": "local", "sd_id": "SD-TRAVEL"}
  },
  "topic_transition": {
    "kind": "RELATED",
    "relation_ref": {"namespace": "local", "sd_id": "SD-REL-DESTINATION"}
  },
  "provenance": "RULE_CANDIDATE"
}
```

The IDs above are illustrative, not claims about deployed SD records. Unresolved `topic_ref` is null. Context resolution status is RESOLVED, UNRESOLVED, or NONE. Transition kind is SAME, RELATED, RESUMED, SHIFTED, or UNKNOWN.

## Relationship validation
A relation SD has `predicate_ref`, `source_ref`, `target_ref`, `validation_state`, and `provenance/evidence_id`. RELATED requires an approved relation connecting the referenced topic endpoints in an allowed direction. Similar words alone are not evidence. Multi-hop graph traversal is bounded (initial proposed defaults: max 2 hops and 64 visited nodes); reject loops and incomplete traversal conservatively.

## Transition rules
- SAME: both topic refs resolve to the same SD identity.
- RESUMED: new topic matches an earlier verified SD topic with explicit turn reference.
- RELATED: verified approved relation SD connects different topic refs, and the contextual reference is compatible.
- SHIFTED: both refs exist, lookup is complete and no approved continuity relation exists. This does not mean the topics are objectively unrelated.
- UNKNOWN: missing/ambiguous SD refs, unresolved required context, unapproved relation, or incomplete graph search.

Suggested precedence: validate refs and evidence; resolve utterance references; SAME; RESUMED; RELATED; SHIFTED or UNKNOWN. Keep speech act and topic-transition classification separate. Do not promote tentative conversation context to persistent Semantic Memory.

## Planned API
`resolve_topic_ref(candidate, registry)`
`resolve_context_reference(utterance, history)`
`lookup_relation(prev_ref, current_ref, registry)`
`classify_topic_transition(intent_sd, state, registry)`
`choose_context_v03(a, b)`

Return status, reason codes, provenance and evidence path. No gold label or AI-proxy preference may enter topic resolution.

## Backwards compatibility and evaluation
Keep v0.2 unchanged. Use an explicit versioned `v02_to_v03` mapping from text labels to existing SD refs; unknown or ambiguous mappings must abstain. Freeze registry and relation approvals for comparisons.

A selectable **blind** review GUI must show Context A/B and fixed response without proxy/model preferences. Reviewers select existing SD refs, relations or UNKNOWN; record rationale, provenance and resume state. Conduct paired evaluation on identical Dev IDs. Distinguish 45 reviewed Dev cases, 40 exclusive AI-proxy A/B decisions, and 30 manually classified mismatch causes. These cohorts are not independent human gold.

Required tests: same SD with distinct surface names, relation direction and approval, resumed references, nonexistent/ambiguous SD, graph cycles and bounds, no label leakage, no silent mutation, deterministic frozen-registry behavior, and preservation of v0.2 regressions. Keep sealed Test unused until independent evaluation protocol is frozen.

## Implementation plan
1. Read-only SD registry adapter, reference validator and deterministic relation traversal.
2. Blind selection-based SD topic annotation GUI with source text.
3. Paired v0.2/v0.3 Dev evaluator and independent human-rated holdout.

This specification does not implement topic extraction or a learned semantic model.
