# Intent SD v0.3 — Execution Guide

This branch contains an **executable experimental runtime** for the
proposed Linked SD Topic Reference Model.

## Included

- `intent_sd_context_v03.py`: read-only frozen SD registry, local SDRef
  validation, approved directional Relation SD graph traversal, context
  reference resolution, SAME/RELATED/RESUMED/SHIFTED/UNKNOWN classification,
  transient state updates, deterministic A/B context decision, and explicit
  legacy v0.2 topic-label mapping.
- `demo_intent_sd_context_v03.py`: executable *synthetic* demonstration.
- `test_intent_sd_context_v03.py`: 23 no-GPU regression tests.
- `evaluate_intent_sd_context_v03.py`: development-only paired evaluator
  for actual independently prepared SD-tagged A/B cases.
- `test_evaluate_intent_sd_context_v03.py`: 8 evaluator safety tests.

The existing v0.2 implementation and tests are untouched.

## Windows commands

```powershell
cd C:\Users\inoma\git\misc\LLM_TMP
git fetch origin
git switch --track origin/feature/intent-sd-v0.3-spec

python -m unittest -v test_intent_sd_context_v03.py test_evaluate_intent_sd_context_v03.py test_intent_sd_context_v02.py
python demo_intent_sd_context_v03.py
```

If the branch already exists locally, use
`git switch feature/intent-sd-v0.3-spec` and
`git pull --ff-only origin feature/intent-sd-v0.3-spec`.

The demo contains synthetic topic IDs and synthetic evidence only; it
must not be reported as model accuracy or corpus evaluation.

## Paired real-Dev evaluation

```powershell
python evaluate_intent_sd_context_v03.py --input results/intent_sd_v03_frozen_dev.json
```

Required input fields:
```text
scope: DEV_NOT_INDEPENDENT
reference_type: AI_PROXY | HUMAN
annotation_independent_of_reference: true
expected_case_count: integer
registry_snapshot: {namespace, snapshot_id, nodes: [...]}
cases: [
 {case_id, reference_label, baseline_prediction,
  intent_a, intent_b, state_a, state_b}
]
```

`intent_a` / `intent_b` follow `INTENT_SD_V03_SPEC.md`, with the
`topic_transition.kind` set to UNKNOWN when it is not yet classified.
`state_a` and `state_b` have
`active_topic_ref` and a list `history` of
`{utterance_id, topic_ref, verified}` entries. A registry relation
node requires `predicate_ref`, `source_ref`, `target_ref`,
`validation_state`, and `evidence_id`; a predicate requires the
`continuity_allowed` boolean. The registry snapshot must be frozen
before collecting predictions.

Do not set `annotation_independent_of_reference` to true until a
real blinded annotation procedure has been performed. A flag is not
independent evidence of blinding. The current runtime does not extract
SD refs from natural language and does not create a human annotation GUI.
Build the previously agreed searchable selection-style blind SD review
GUI before interpreting corpus-level scores. Avoid raw private text in
the report or repository.

Note: graph traversal has conservative UNKNOWN handling for pending
edges and traversal budget exhaustion. SHIFTED reports absence of an
established approved path in the frozen registry, not semantic
unrelatedness in reality. Currently no SKI remote-ID resolution is
implemented: only matching local namespace IDs are accepted.

## Status / validation boundary

**Implementation committed; Windows tests still need user execution.**
No v0.3 measured improvement, human-reviewed holdout or trained topic
extractor is claimed. The independent Test dataset remains sealed.
