# Intent SD v0.1 — LLM_TMP development experiment

This branch introduces a **review and scoring scaffold**, not an intent-trained model.
Existing 10k/30k preference reports and AI-proxy labels remain unchanged.
The local AI-proxy pack includes 45 selected Dev cases in the documented
workflow; do **not** assume that every one is an exclusive A/B decision or
that the claimed "40" is verified until the actual packet is counted.

## 1. Inspect proxy disagreements

```powershell
python -m unittest -v test_compare_realpersonachat_ai_proxy.py test_intent_sd_v01.py
python analyze_proxy_disagreements.py prepare
```

Inspect `results/intent_sd_mismatch_review_dev.json`. For each mismatch
assign `primary_cause`, optional `secondary_causes`, and a non-empty
`rationale` after inspecting the source locally. Categories:
`LEXICAL`, `TOPIC`, `SPEECH_ACT`, `CONTEXT`, `POLARITY`,
`OTHER`. Mismatch means disagreement **with the AI proxy**, not
ground-truth error. The packet includes a mismatch if *either* model's
original-vs-shuffled preference differs from an exclusive proxy A/B label.

```powershell
python analyze_proxy_disagreements.py audit
```

Output paths under `results/` are local/private. Commands refuse
overwrite unless `--overwrite` is explicitly provided.

## 2. Intent SD representation

`intent_sd.py` provides a validated structure:
`speech_act`, `intent`, `topic`, `target`, `polarity`,
`context_required`, optional `confidence`, and `provenance`.
It **does not infer** any fields from a natural-language utterance.
Never inject reference labels into Intent SD extraction.

## 3. Frozen paired A/B scorer

To run the scorer, independently prepare a local JSON file containing
**actual predictions** from frozen baseline and Intent-SD-augmented
pipelines. This branch does not fabricate those predictions. Example
schema only (illustrative case, not experimental evidence):

```json
{
  "scope": "DEV_NOT_INDEPENDENT",
  "expected_case_count": 1,
  "cases": [{
    "case_id": "example-001",
    "reference_type": "AI_PROXY",
    "reference_label": "A",
    "baseline_prediction": "B",
    "intent_prediction": "A",
    "intent_sd": {
      "speech_act": "REQUEST",
      "intent": "EXPLAIN_MORE",
      "context_required": true,
      "provenance": "MODEL_INFERRED"
    }
  }]
}
```

```powershell
python compare_intent_sd_ablation.py --input results/intent_sd_pair_predictions_dev.json
```

Outputs paired accuracy, macro-F1, baseline-only wins and Intent-only wins;
it doesn't infer statistical significance. The same *case IDs, reference
labels and input contexts* must be fixed when producing both sets of
predictions. For causal evaluation freeze model weights, extraction
protocol, reference grading, and context. Prevent proxy-label leakage.

## Boundaries and next step

Do not conflate AI-proxy concordance with human-rated accuracy.
The original selected cases were stratified by model behavior and are
not an independent sample. First audit and manually analyze the mismatch
packet, then lock extraction and score identical Dev data. Independently
authored human-reviewed holdout evaluation is needed for generalization
claims. The sealed Test split must not be accessed in this phase.
