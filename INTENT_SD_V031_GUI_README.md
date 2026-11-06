# Intent SD v0.3.1 — Blind Linked SD Annotation GUI

Status: implemented experimental GUI, not a topic-extraction model or evidence of A/B accuracy.

## Inputs

- Private source `results/realpersonachat_semantic_review_dev.json` contains `cases[]`: `case_id`, `context_a`, `context_b`, `reference_response`. All other fields (including AI proxy and predictions) are deliberately ignored by the review screen.
- **Existing, frozen SD snapshot** under `results/linked_sd_registry_dev.json`. Required schema is compatible with `SDRegistry` v0.3: `namespace`, optional `snapshot_id`, `nodes` array with existing `sd_id` and `node_type`. Existing relation and predicate fields follow v0.3. Optional top-level `display_names` maps existing SD IDs to Japanese names (e.g. `"SD-TRAVEL": "旅行"`). No topic SDs are created automatically.

The UI offers SD ID/display-name searching and selection for Context A and B, each with previous/current/reference topic fields. `UNKNOWN` is always available. Output contains case IDs, chosen SD refs, reviewer ID, rationale, status and source/registry fingerprints; **no raw transcript, AI proxy A/B label, model prediction or reference gold label**. An output file, if present, is resumed. Registry or source changes invalidate it instead of silently mixing versions.

## Windows

```powershell
cd C:\Users\inoma\git\misc\LLM_TMP
git fetch origin
git switch --track origin/feature/intent-sd-v0.3.1

python -m unittest -v test_linked_sd_annotation_v031.py
python linked_sd_review_cli_v031.py inspect --registry results/linked_sd_registry_dev.json
python review_linked_sd_gui_v031.py --registry results/linked_sd_registry_dev.json
```

If the branch already exists: `git switch feature/intent-sd-v0.3.1` then `git pull --ff-only origin feature/intent-sd-v0.3.1`.

After annotation:

```powershell
python linked_sd_review_cli_v031.py audit --registry results/linked_sd_registry_dev.json
```

Use `--source` and `--output` to specify non-default private files. Optional `linked_sd_review_cli_v031.py init` initializes the blank review packet only when the output does not exist; launching the GUI does the initialization in memory automatically.

## Scope and safeguards

The initial 30 mismatch-cause annotations are **not** human A/B gold. Do not provide existing preferred-context judgments to the annotator. Source displays Context A, Context B and fixed response. The screenshot or screen-sharing process must also avoid showing labels outside this GUI.

This is blind annotation **of existing SD IDs**, not automated text-to-SD extraction, relation approval, context correctness adjudication or v0.3 improvement measurement. The graph evaluator consumes a separate frozen-prediction packet after topic review and independence checks; this GUI does not generate that packet automatically. The sealed Test remains unused.

## Limitations

At least one valid SD node must be supplied for meaningful selection. If no existing SD registry is available, build a read-only snapshot from the local existing SD store first; do not invent an example registry as if it were real. For every case, reviewers can assign `UNKNOWN` and explain why. The reviewer interface records **six topic references** and reasoning, not A/B correctness. Relation SD approval and graph traversal continue to be handled by the v0.3 runtime, not by this GUI.

## Bootstrap sample registry (source-backed)

The repository does not include the user's local, production SD records
under `results/` (Git ignores that directory). To run the GUI without
inventing any SD IDs, create a **sample-only** registry from the actual
`recursive_sd.build_sample_graph()` node IDs:

```powershell
python build_linked_sd_registry_dev.py
python -m unittest -v test_build_linked_sd_registry_dev.py
python linked_sd_review_cli_v031.py inspect --registry results/linked_sd_registry_dev.json
python review_linked_sd_gui_v031.py --registry results/linked_sd_registry_dev.json
```

The generator includes five existing concepts (CAT, DOG, FISH, BIRD,
RICE) and two predicates (EAT, WATCH) with no approved relation SDs.
The GUI allows choosing the five concept IDs; predicates are not Topic
choices. For real-person dialogue annotation this bootstrap file is
insufficient: replace it with a frozen registry exported from the actual
approved SD store, preserving original IDs and provenance. Do not
mistake a sample catalog for an extracted topic inventory. The builder
refuses to overwrite an existing `results/linked_sd_registry_dev.json`
unless explicitly passed `--overwrite`; replacing a registry will
invalidate existing review fingerprints and may require a fresh review.
