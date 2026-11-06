# Intent SD v0.2 — Topic / Context Consistency (experimental)

Built on `feature/intent-sd-v0.1`. This implements a **deterministic
comparator of independently supplied canonical topic IDs**, NOT a Japanese
natural-language topic extractor, trained classifier or end-to-end LLM.

## Background

The prior 30 completed **development** mismatch reviews reported primary
cause CONTEXT=17 and TOPIC=13. This describes human-selected categories in
a model-stratified sample; it is not a causal model diagnosis. Those cause
labels must **not** be copied into input topic tags or predictions.

## Context SD v0.2 fields

`ContextSD(topic_previous, topic_current, context_reference, provenance)`.
Each topic is a canonical identifier (for example, `travel`) or `null`.
`context_reference` is the topic identified by a resolved contextual
reference. It must be supplied independently of the proxy/gold preference.
The allowed provenance values are `RULE_CANDIDATE`,
`MODEL_INFERRED`, and `HUMAN_ANNOTATED`.

Topic comparison is **exact canonical-ID matching**; meaning-aware
synonym recognition is NOT implemented. A missing tag yields UNKNOWN.
Contradictions yield SHIFTED. A is preferred only if A=CONTINUOUS and
B=SHIFTED; the reverse yields B. Both continuous yields BOTH; both
shifted yields NEITHER; partial/unknown evidence yields UNCLEAR.
This policy is a deliberately narrow development baseline.

## Regression

```powershell
cd C:\Users\inoma\git\misc\LLM_TMP
git fetch origin
git switch --track origin/feature/intent-sd-v0.2
python -m unittest -v test_intent_sd_context_v02.py
```

If that branch already exists locally, switch to it and
`git pull --ff-only origin feature/intent-sd-v0.2`.

## Frozen paired evaluation

Only run after topic tags are independently collected and both branches'
predictions are frozen. An input JSON with **actual** data, not fabricated
examples, is required:

```json
{
  "scope": "DEV_NOT_INDEPENDENT",
  "reference_type": "AI_PROXY",
  "expected_case_count": 1,
  "annotation_independent_of_reference": true,
  "cases": [{
    "case_id": "example-001",
    "reference_label": "A",
    "baseline_prediction": "B",
    "context_a": {
      "topic_previous": "travel",
      "topic_current": "travel",
      "context_reference": "travel",
      "provenance": "HUMAN_ANNOTATED"
    },
    "context_b": {
      "topic_previous": "travel",
      "topic_current": "music",
      "context_reference": "music",
      "provenance": "HUMAN_ANNOTATED"
    }
  }]
}
```

The example above is **schema illustration only**, not an experiment result.
Set `annotation_independent_of_reference=true` only after an actual
blind annotation protocol; the code checks the declaration but cannot
prove independence. Keep labels concealed during annotation. Use a
selection-based topic annotation GUI for that later stage, rather than
asking the reviewer to manually edit JSON.

```powershell
python evaluate_intent_sd_context_v02.py --input results/intent_sd_context_v02_frozen_dev.json
```

The output compares baseline and intent labels on identical case IDs,
with paired correctness counts, accuracy and the abstention-aware
coverage. Reports are aggregate plus case IDs/labels, no raw transcript.
Never replace the baseline prediction with the human reference label.

Do not access the sealed Test or claim generalization from these selected
Dev cases. Next milestone: independent, blinded topic-tag annotation GUI,
then frozen A/B evaluation and an unseen human-labeled holdout.
