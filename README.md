# LLM_GPU

CUDA/PyTorch GPU version of the homemade LLM project.

This repository is based on the architecture and end-to-end flow proven in
`digiponta/LLM` branch `v0.3`. The original educational virtual-GPU runtime
and hand-written backward propagation are replaced by real PyTorch tensors,
CUDA kernels, and PyTorch autograd.

## Architecture

```text
Japanese corpus
    |
character tokenizer
    |
Embedding
    |
Transformer blocks
    |-- LayerNorm
    |-- single-head causal Self-Attention
    |-- residual connection
    |-- LayerNorm
    |-- FFN (Linear -> GELU -> Linear)
    |-- residual connection
    |
final LayerNorm
    |
lm_head
    |
Cross Entropy
    |
PyTorch autograd
    |
AdamW
    |
CUDA GPU
```

## Files

```text
tokenizer.py          character tokenizer compatible with the original project
dataset.py            next-token dataset / uniform sampled windows
model.py              CUDA-capable Transformer language model
train.py              GPU training loop with progress and ETA
train_corpus.py       corpus training entry point
infer.py              interactive GPU inference
check_gpu.py          CUDA/PyTorch diagnostic
prepare_wikipedia.py  Wikipedia dump -> plain text helper
requirements.txt      Python dependencies
```

## Installation

Install a CUDA-enabled PyTorch build suitable for the NVIDIA driver on the PC,
then install the project requirements.

```powershell
python -m pip install -r requirements.txt
python check_gpu.py
```

A successful setup reports:

```text
CUDA available : True
Device         : cuda
GPU            : NVIDIA ...
VRAM           : ... GiB
```

## Training data

By default the trainer looks for:

```text
data/general-ja.txt
data/data-nagato.txt
```

For convenience, if those files do not exist in this repository, it also
looks for the existing original-project files:

```text
../LLM/data/general-ja.txt
../LLM/data/data-nagato.txt
```

## Train

```powershell
python train_corpus.py
```

Default first GPU run:

```text
context length : 64
d_model        : 64
layers         : 2
FFN dimension  : 256
attention heads: 1
batch size     : 64
samples        : 20,000
epochs         : 3
learning rate  : 5e-4
```

The sample count is intentionally much larger than the CPU/virtual-GPU v0.3
smoke test because the physical GPU processes mini-batches in parallel.

Checkpoint:

```text
model/model-gpu-v0.3.pt
```

## Inference

```powershell
python infer.py
```

Generation uses temperature, top-k sampling, repetition penalty, and a bounded
context window.

## Relationship to the original v0.3

The original repository verified the complete path:

```text
corpus -> tokenizer -> model forward -> loss -> backward
       -> optimizer -> checkpoint -> reload -> inference
```

LLM_GPU preserves that path while moving tensor computation and gradient
calculation to a physical CUDA GPU.


## Four-mode SD comparison (experimental)

`compare_four_modes.py` compares four **distinct runtime strategies**:

| Mode | Runtime |
| --- | --- |
| A | Raw Transformer from `model.py` (greedy decoding; no memory) |
| B | Natural-language memory with lexical retrieval, no composition |
| C | Linked Semantic Data (SD) graph lookup, no composition |
| D | Linked SD graph lookup + grammar-template SD composition |

Every SD node (including the grammar template `JA-SOV`) is managed as an SD.
Propositions link to pre-existing concept/predicate/template SD nodes instead
of copying them. Composition is allowed **only** for an explicit request to
create a sentence; it does not assert a novel proposition as true.

The included Japanese fixture has three stored facts, three unmemorized
composition requests, and two unknown-fact requests. It is a **development
sanity check**, not a statistically meaningful independent holdout. All modes
use the same prompts and scoring, but they have different evidence access by
design. The raw LLM may not be trained on instruction-answer tasks.

### Run

From the repository root:

```powershell
python -m unittest -v test_compare_four_modes.py
python compare_four_modes.py --structural-only
```

The second command runs **B/C/D only** and saves
`results/four_way_sd_comparison.json`. It does **not** produce an A-versus-D
performance claim.

To run all four modes, first provide a **compatible trained checkpoint and
tokenizer** for the current `model.py` and `tokenizer.py`:

```powershell
python compare_four_modes.py `
  --checkpoint model/model-gpu-v0.3.pt `
  --tokenizer model/tokenizer.json `
  --device auto
```

The runner refuses to report four-mode results without these files.
The report includes per-case predictions, latency, overall exact-match
accuracy, known-answer accuracy, coverage, selective accuracy, and unknown
rejection. An unknown is correct only when the system abstains.

**Limitations:** The fixture contains template-shaped composition prompts,
lexical retrieval uses a deliberately simple matching baseline, and the
structural modes do not invoke the LLM. Consequently, any observed benefit
demonstrates the value of explicit symbolic composition on this fixture—not
an improvement in LLM understanding or independent generalization.
Future work: independently authored multi-turn and paraphrase holdouts,
balanced baseline prompting, semantic-identity collisions, negation,
nested grammar templates, and provenance-aware SKI exchange.


## Recursive linked SD composition (next experiment)

The next implementation is separated from the original eight-case benchmark.
All templates are SD nodes; sentence SDs link to existing template, entity,
predicate and child-expression SDs. `compose()` interns identical structures
to the same SD ID. `decompose()` and `recompose()` check round-trip identity;
`render()` detects reference cycles.

Run from the repository root:

```powershell
git pull origin main
python -m unittest -v test_compare_four_modes.py test_recursive_sd.py test_compare_recursive_sd.py
python compare_recursive_sd.py --structural-only
```

To include raw LLM mode A, with compatible local model/tokenizer:

```powershell
python compare_recursive_sd.py --checkpoint model/model-gpu-v0.3.pt --tokenizer model/tokenizer.json --device auto
```

The new run writes `results/recursive_sd_development.json`.
Modes B/C perform their original non-compositional retrieval, and mode D
constructs nested sentences with template SDs, under the same prompts.
The new dataset has eight two-clause composition requests plus two unknown
fact questions and one unsupported-verb request.

**Evaluation boundaries:** These are hand-authored, template-shaped
**development** examples, not a truly independent holdout. This version
demonstrates graph-level reuse and recursion, **not** free-form parsing,
learning of new grammar, factual verification, or an improvement in LLM
reasoning. No predicted values or measurements are reported in advance of
running the code. For independent validation, a reviewer should prepare
separately approved and frozen queries not accessed when tuning the parser,
templates, or SD linking policy.


## Japanese natural language -> linked SD -> natural language (experimental)

This stage adds conservative recognition for canonical Japanese SOV sentences
and two-clause conjunctions. It is the inverse of the previously implemented
template renderer **only for its supported grammar**.

New files:
- `japanese_sd_parser.py`: parses known surface expressions into recursive
  linked SD; returns an SD tree and round-trip result.
- `test_japanese_sd_parser.py`: parsing, SD sharing, ambiguity, malformed
  input, and unknown-fact safety regression tests.
- `evaluate_japanese_sd.py`: generated development cases (not an independent
  holdout), including unsupported examples.

Run:

```powershell
git pull origin main
python -m unittest -v test_compare_four_modes.py test_recursive_sd.py test_compare_recursive_sd.py test_japanese_sd_parser.py
python japanese_sd_parser.py --text "猫が魚を食べると、犬が鳥を見る。"
python evaluate_japanese_sd.py --nested-limit 50
```

Reports:
- `results/japanese_sd_roundtrip.json`
- `results/japanese_sd_development.json`

Acceptance requires exact surface round-trip using the registered grammar
templates and uniquely identified lexical SD nodes. Parsing reuses existing SD
IDs; repeated parses do not duplicate equivalent SD graphs. Unknown or
ambiguous lexical items, unsupported grammar and three-clause inputs are
rejected without partially creating nodes. All constructed propositions
remain **unverified expressions**, not asserted factual knowledge.

**Important:** At this stage, the natural-language parser is a rule-based
recognizer; the LLM is **not** used to infer SD structure. The prior four-mode
comparison and its metrics remain unchanged. The development sweep reuses the
same grammar and lexicon as the implementation and therefore must not be
presented as an independent generalization test.

Planned next: frozen human-authored paraphrase/negation/multi-clause evaluation;
separate meaning equivalence from exact textual roundtrip; optional LLM
candidate extraction with deterministic structural checks; and SKI exchange
of locally unique SD IDs with explicit provenance and cross-VM mapping.


## LLM -> untrusted SD candidate -> deterministic validation (experimental)

`sd_candidate_pipeline.py` introduces an explicit boundary between a
**proposer** and the **SD Engine**. Two proposers are supported:

- `rule`: an intentionally limited deterministic baseline for known SOV
  sentences and two-clause sentences.
- `llm`: invokes the trained local Transformer to suggest a JSON SD tree.
  The original v0.3 model is **not guaranteed to produce JSON**, so failures
  are rejected; no fallback silently replaces the LLM's answer.

The candidate schema is:

```json
{
  "tree": {
    "template_ref": "T-SOV",
    "bindings": {
      "subject": {"sd_ref": "CAT"},
      "object": {"sd_ref": "FISH"},
      "verb": {"sd_ref": "EAT"}
    }
  }
}
```

Grammar templates are SDs, while compound SDs link to existing nodes.
The validator checks known SD IDs, role/type constraints, supported
templates, maximum depth/nodes, and **exact natural-language re-rendering**.
All modifications are staged and merged only upon successful validation.
A structurally accepted sentence is an *unverified expression* and never
an automatically approved fact. The acceptance check is a surface
round-trip, not proof of semantic equivalence or correctness.

Run the new tests alongside the existing 32:

```powershell
git pull origin main
python -m unittest -v test_compare_four_modes.py test_recursive_sd.py test_compare_recursive_sd.py test_japanese_sd_parser.py test_sd_candidate_pipeline.py
python sd_candidate_pipeline.py --provider rule --text "猫が魚を食べると、犬が鳥を見る。"
python evaluate_sd_candidates.py --provider rule
```

With a compatible model and tokenizer on disk, test actual candidate
generation (this may be rejected for poor or non-JSON output):

```powershell
python sd_candidate_pipeline.py --provider llm --checkpoint model/model-gpu-v0.3.pt --tokenizer model/tokenizer.json
python evaluate_sd_candidates.py --provider llm --checkpoint model/model-gpu-v0.3.pt --tokenizer model/tokenizer.json
```

The development evaluation includes 8 supported and 5 unsupported
sentences. These are hand-curated samples, **not independent holdout**.
The rule-provider result cannot be used as evidence of LLM improvement.
The LLM-provider acceptance rate should be reported separately and only
after running it.

Planned: better structured-output extraction, independently frozen
paraphrase and negation sets, semantic-equivalence checks independent of
exact string matching, meaning-ID resolution under ambiguity, and
provenance-preserving SKI synchronization.


### Candidate failure diagnostics (post initial 0/8 LLM acceptance)

The LLM-based development run initially rejected all eight supported inputs,
while the rule-based proposer accepted all eight. The 5/5 unsupported
rejections in the LLM run **do not** demonstrate understanding, because the
LLM rejected every sample. The earlier evaluator recorded the unhelpful generic
`CandidateRejected` class name for all failures.

`evaluate_sd_candidates.py` now emits a `rejection_reasons` histogram and
the concrete exception reason. With `--provider llm`, it also records:
`unknown_prompt_tokens`, `prompt_chars`, `response_chars`, and a bounded
`response_preview` (first 160 characters). This distinguishes incompatible
tokenizer prompts from malformed model output. The recorded preview is
development-only and may contain model-generated text.

```powershell
git pull origin main
python evaluate_sd_candidates.py --provider llm `
  --checkpoint model/model-gpu-v0.3.pt `
  --tokenizer model/tokenizer.json `
  --output results/sd_candidate_llm_diagnostic.json
```

No improvement to LLM candidate-generation quality is claimed by this
diagnostic patch. A separate SD-specific supervised training/evaluation phase
will likely be necessary for the v0.3 language model.


## Compact SD supervised learning (experimental)

Use short SD output instead of verbose JSON to make structured generation
feasible for the small v0.3-style causal Transformer:

```text
猫が魚を食べる。 -> S(CAT,FISH,EAT)
猫が魚を食べると、犬が鳥を見る。
  -> T(S(CAT,FISH,EAT),S(DOG,BIRD,WATCH))
猫が魚を食べる、そして犬が鳥を見る。
  -> A(S(CAT,FISH,EAT),S(DOG,BIRD,WATCH))
```

`S` corresponds to `T-SOV`, `T` to `T-THEN`, and `A` to
`T-AND`. The short notation is parsed by `compact_sd.py` and
revalidated by the existing atomic `accept_candidate` engine. Untrusted
unknown IDs, malformed nested codes, and surface mismatches are rejected.

`train_compact_sd.py` generates syntactic input/output pairs from the
registered toy vocabulary. It trains a **separate** model and character
tokenizer, with loss masked on the prompt tokens and applied only to the
SD-output tokens. It does **not** fine-tune the old general-purpose v0.3
checkpoint; its vocabulary is not assumed compatible.

```powershell
git pull origin main
python -m unittest -v test_compact_sd.py
python compact_sd.py --text "猫が魚を食べる。" --compact "S(CAT,FISH,EAT)"
python train_compact_sd.py train --device auto --epochs 15
python train_compact_sd.py evaluate --device auto
```

Default model outputs:

- `model/sd-compact/model-sd-compact.pt`
- `model/sd-compact/tokenizer-sd-compact.json`
- `model/sd-compact/split-sd-compact.json`
- `model/sd-compact/training-sd-compact.json`
- `results/sd_compact_eval.json`

**Limitations:** No independent held-out lexicon/grammar is used. Train and
development splits use disjoint *sentences*, but share all atoms, templates,
and a generated data grammar. The measured metric is exact code generation
and SD Engine acceptance on the development split, not real-world semantic
understanding or factual accuracy. The runtime may reject unknown input
characters. No evaluation result is claimed until the model is trained and
the evaluation command runs.

The next phase is a human-authored frozen test split with new paraphrases,
unseen compositions, unsupported syntax and explicit unknown cases, followed
by testing a version trained on more natural language.


## Active/passive semantic identity experiment (development)

A conservative canonicalizer maps supported active and direct passive
sentences to **the same** recursive linked SD structure. Example:

```text
猫が魚を食べる。         -> S(CAT,FISH,EAT)
魚が猫に食べられる。     -> S(CAT,FISH,EAT)
```

Both map to the same interned SD ID, despite different original strings.
This is an experimental step toward semantic identity across paraphrases,
not general natural-language understanding or a claim that the statements
are factually true. The voice and original surface are preserved in the
analysis result; all expressions retain `verified_fact: false`.

The current scope covers the existing finite lexicon, direct passives
with explicit agent marked by `に`, and two supported conjunctions.
The rule normalizer does **not use an LLM**. It conservatively rejects
negation, questions, three-clause sentences, unknown words, and additional
passive constructions. Natural passive morphology may be ambiguous
depending on context, which is outside this experimental subset.

Files:
- `semantic_equivalence_sd.py`: voice normalization, SD interning,
  canonical graph identity, and pairwise comparison.
- `test_semantic_equivalence_sd.py`: positive/negative/unknown regression
  tests.
- `evaluate_semantic_equivalence.py`: small, non-independent development
  comparison set (six positive, four negative, five unsupported pairs).

Run:

```powershell
git pull origin main
python -m unittest -v test_semantic_equivalence_sd.py
python semantic_equivalence_sd.py --left "猫が魚を食べる。" --right "魚が猫に食べられる。"
python evaluate_semantic_equivalence.py
```

Development output: `results/semantic_equivalence_development.json`.

Next, combine this canonicalizer with an LLM proposer and evaluate frozen,
human-authored paraphrases with independent semantic labels. The earlier
94.4% compact-SD score remains a separate **development** baseline; it
cannot be extrapolated to active/passive normalization.


## Layered semantic features (experimental)

`semantic_features_sd.py` introduces two independent identities: a shared core proposition and a full semantic frame. Supported direct active/passive pairs share the core. Polarity (positive/negative), tense (nonpast/past), and subject-topic marking are stored as separate SD feature nodes and therefore distinguish frames even when they share a core. This is a finite-rule prototype rather than LLM-based recognition, general Japanese semantics, or factual verification.

Run:

```powershell
 git pull origin main
 python -m unittest -v test_semantic_features_sd.py
 python semantic_features_sd.py --left "猫が魚を食べる。" --right "猫が魚を食べない。"
```

Expected: `same_core: true`, `same_frame: false`, `differences: ["polarity"]`. The new regression module defines 18 test methods; their results must be checked locally. The SD graph clone now reconstructs its ID counter rather than deep-copying `itertools.count`, avoiding the Python 3.14 deprecation path in the candidate validator and equivalence normalizer. The pre-existing compact SD generator and its 94.4% development baseline are unchanged. Independent holdout, ambiguity, question scope, and actual LLM feature extraction remain future work.


## Core + Frame compact SD supervised experiment

This experiment extends the earlier compact code with *semantic frame features*
without changing the already-trained SD-compact checkpoint.

```text
猫が魚を食べなかった。
  -> F(S(CAT,FISH,EAT),NEG,PAST,UNMARKED)
猫は魚を食べる。
  -> F(S(CAT,FISH,EAT),POS,NONPAST,SUBJECT_TOPIC)
魚が猫に食べられる。
  -> F(S(CAT,FISH,EAT),POS,NONPAST,UNMARKED)
```

The `S(...)` part is the canonical Core Proposition. The outer `F(...)`
is a separate Semantic Frame specifying polarity, tense and topic. Direct
active/passive voice is retained as surface provenance rather than a separate
frame identity. A candidate generated by the LLM is checked by deterministic
surface realization *from the candidate*, existing ID/type validation and
canonical graph linking. Rejected candidates do not mutate Semantic Memory.
Passing the validator is not approval of the factual statement.

Files:
- `compact_frame_sd.py`: codec, constrained candidate validation and
  Core/Frame graph interning.
- `train_frame_sd.py`: supervised training of a **new** model with output-only
  loss, separate tokenizer/checkpoint, frozen split and greedy inference.
- `test_compact_frame_sd.py`: twelve regression tests.

Run on Windows:

```powershell
git pull origin main
python -m unittest -v test_compact_frame_sd.py
python compact_frame_sd.py --text "猫が魚を食べなかった。" --code "F(S(CAT,FISH,EAT),NEG,PAST,UNMARKED)"
python train_frame_sd.py train --device auto --epochs 20
python train_frame_sd.py evaluate --device auto
```

Default artifacts:
- `model/sd-frame/model-frame-sd.pt`
- `model/sd-frame/tokenizer-frame-sd.json`
- `model/sd-frame/split-frame-sd.json`
- `model/sd-frame/history-frame-sd.json`
- `results/sd_frame_eval.json`

For a more demanding **development** split, entire Core triples
(subject, object, verb) are assigned to either train (40) or dev (10).
Each triple produces 12 surfaces: positive/negative × nonpast/past ×
active-given/active-topic/passive. This yields 480 training and 120
development examples, with no Core triple shared between the splits.
Vocabulary, inflection rules and grammar remain shared, so this is **not**
independent natural-language generalization or a blind holdout.

Report `exact_code_rate`, `sd_accept_rate` and breakdown by active/passive.
The existing 94.4% compact-SD development result is a historical reference,
not a controlled apples-to-apples comparison because tasks/splits differ.


## Two-head role-first Core/Frame learning (experimental)

The direct frame LLM baseline on 120 core-disjoint **development** examples:
70.833% overall (85/120), 92.5% active (74/80), 27.5% passive (11/40).
Most rejected candidates swap the agent/patient on Japanese passives.

The two-head experiment trains **two independent small Transformers** on
exactly the same training and development examples and split as
`train_frame_sd.py` (same seed 42, 480 train / 120 dev):
- **roles head:** `R(DOG,FISH,WATCH)` identifies semantic agent,
  patient, and action irrespective of surface voice.
- **features head:** `Q(NEG,PAST,UNMARKED)` identifies polarity,
  nonpast/past and subject-topic attributes.
- **assembler:** validates both outputs and constructs
  `F(S(DOG,FISH,WATCH),NEG,PAST,UNMARKED)`.
- **SD Engine:** validates the *predicted candidate* against the original
  surface and rejects mismatches without mutating Semantic Memory.

Neither head receives ground-truth labels, voice flags, or rule-parser
outputs at evaluation time; both receive **only the input sentence**. No
symbolic fallback corrects prediction failures. Evaluation reports separate
role, feature, full-code and accepted-frame accuracy, plus active/passive
breakdowns.

```powershell
git pull origin main
python -m unittest -v test_role_frame_sd.py test_compare_role_frame_models.py
python train_role_frame_sd.py train --device auto --epochs 20
python train_role_frame_sd.py evaluate --device auto
python compare_role_frame_models.py --baseline results/sd_frame_eval.json --two-head results/sd_role_frame_eval.json
```

Artifacts live separately under `model/sd-role-frame/`:
`model-roles.pt`, `model-features.pt`, tokenizers for each, `split.json`,
and `history.json`. Reports go to `results/sd_role_frame_eval.json`
and `results/role_frame_comparison.json`. The comparison script refuses
differing rows, targets or voice labels and reports newly-correct and
regressed cases on the **paired** 120-item development set.

These synthetic samples share lexicon, inflections and grammar. The split
is core-triple-disjoint, not independent natural-language generalization.
No score is claimed until the user trains and evaluates the new heads.


## Dialogue Intent SD: first step toward ordinary conversation (experimental)

Adds a third, **independently trained** dialogue-intent model alongside the
existing Core-role and Semantic-Frame models. It does NOT yet implement a
dialogue-context engine, a conversation-response generator or open-ended chat.

Intent labels: `GREETING`, `FAREWELL`, `THANKS`, `QUESTION`,
`REQUEST`, `SELF_DISCLOSURE`, `FEELING`, `ACKNOWLEDGEMENT`.
Each is a registered SD atom, wrapped by the `T-DIALOGUE-ACT` template
as `I(INTENT)`. SD graph interning shares an intent-type node across
utterances. **This is a dialogue-act category, not a unique representation
of the full utterance; it does not preserve who said what or the original
content within that node.** Neither a structurally accepted intent SD nor
its label is validated against utterance meaning by `accept()` alone.
Factual assertions remain unverified.

`dialogue_intent_dataset.py` includes 8 pragmatic classes × 6 example
families × 2 variants = 96 utterances. Split by whole example family:
64 training and 32 development utterances, all classes in each split.
The dataset is deliberately small, hand-authored and **not an independent
holdout**. The character tokenizer includes *unlabelled development input
characters* for fixed coverage; no development intent labels are exposed
to the model during training, but this is not a strict unseen-vocabulary
protocol. The evaluation reports exact intent-code rate, syntactic
candidate acceptance, and importantly **accepted wrong-intent count**.
Unlike Core+Frame SD surface-validation, the intent validator cannot tell
that an in-vocabulary but incorrect label is semantically wrong.

```powershell
git pull origin main
python -m unittest -v test_dialogue_intent_sd.py
python dialogue_intent_sd.py --text "今日は疲れました。" --code "I(FEELING)"
python dialogue_intent_dataset.py
python train_dialogue_intent.py train --device auto --epochs 35
python train_dialogue_intent.py evaluate --device auto
```

New files: `dialogue_intent_sd.py`, `dialogue_intent_dataset.py`,
`train_dialogue_intent.py` and `test_dialogue_intent_sd.py`.
Model/tokenizer/split/history are written under
`model/dialogue-intent/`; the report is
`results/dialogue_intent_dev.json`. Existing Role/Frame checkpoints and
historical scores are unaffected.

Next: independently authored intent paraphrases, multi-intent utterances,
unknown/abstain cases, actual dialogue context, response-action selection,
and grounding of conversational content with provenance and speaker roles.


## Constrained Dialogue Intent decoding (no retraining)

The previous 35-epoch Intent LLM scored 17/32 exact on the development
split (53.125%), with 26/32 structurally accepted and 9/32 **valid but
incorrect intent labels**. These are user-observed results, not claims
about general conversational understanding.

`constrained_dialogue_intent.py` evaluates exactly the eight registered
`I(INTENT)` codes under the **same saved Transformer**. Rather than
generating arbitrary characters greedily, it teacher-forces each complete
candidate code (including newline and EOS), scores the **mean token log
likelihood**, and chooses the highest-scoring valid candidate. No gold
label or heuristic parser is used to repair its answer. Unknown input
characters and overlong prompts lead to abstention; otherwise a
known label is selected even for unsupported dialogue. The score gap is
*uncalibrated* and must not be interpreted as confidence.

`evaluate_constrained_dialogue_intent.py` reads the earlier fixed
`results/dialogue_intent_dev.json` for a strict per-case comparison and
uses the existing checkpoint/tokenizer/split from
`model/dialogue-intent/`. It rejects mismatched test items or labels.
The resulting report records gains, regressions, per-Intent results,
wrong-label confusions and valid-but-wrong candidate counts. It does
**not** overwrite or silently modify the original baseline report.

```powershell
git pull origin main
python -m unittest -v test_constrained_dialogue_intent.py
python evaluate_constrained_dialogue_intent.py --device auto
```

Output: `results/dialogue_intent_constrained.json`.

This is only a paired **development** diagnostic with a single fixed
checkpoint; it is not an independent holdout. Since the set of possible
outputs is constrained to known labels, improvements to syntax acceptance
are not proof of semantic classification. Training changes, hierarchical
intent recognition, calibrated abstention and multi-turn context remain
future work.


## Local RealPersonaChat dialogue JSON importer (13,583 files observed locally)

The RealPersonaChat corpus, as checked on Windows, contains file-per-dialogue
UTF-8 JSON with fields `dialogue_id`, `interlocutors`, `utterances`,
`evaluations`; utterances include `utterance_id`, `interlocutor_id`,
`text` and `timestamp`. Use `-Encoding UTF8` with Windows PowerShell 5.1
when reading Japanese JSON.

`import_realpersonachat.py` reads this source format **without altering
the originals**. It validates speakers, unique dialogue/utterance IDs and
nonempty text, sorts utterances by their explicit ID, and constructs
multi-turn next-utterance examples containing previous speaker/text pairs,
the next speaker, and the next utterance. Both conversation participants
are treated symmetrically as `speaker_0`/`speaker_1` based on the source
`interlocutors` order; **AA/AB are not assumed to be user/assistant**.
Evaluations are ignored, not copied as quality labels. Intent labels are
`null`, because the corpus does not provide ground-truth intent classes.

All next-turn samples from a source dialogue share the same deterministic
content-hash split, 90/5/5 train/dev/test. Identical full-dialogue content
is deduplicated. This protects against conversation-level leakage but does
**not** establish an independent evaluation: paraphrases or overlapping
dialogues may still cross splits. Split proportions are approximate due
to hash assignment and the corpus may contain sensitive or copyrighted
material; check original dataset usage terms and privacy before training.

**Audit only (default, no text output files):**

```powershell
cd C:\Users\inoma\git\misc\LLM_TMP
git pull origin main
python -m unittest -v test_import_realpersonachat.py
$source = "C:\Users\inoma\git\misc\LLM_CHA\data-src\real-persona-chat\real_persona_chat\dialogues"
python import_realpersonachat.py --source $source --limit-files 100
python import_realpersonachat.py --source $source
```

After checking usage rights and personal-data handling, export only on
the local machine, first with a 100-file smoke test into a separate fresh
directory:

```powershell
python import_realpersonachat.py --source $source --limit-files 100 --write --output data/realpersonachat_smoke
python import_realpersonachat.py --source $source --write --output data/realpersonachat_import
```

Exports: `train.jsonl`, `dev.jsonl`, `test.jsonl`, `audit.json`.
The importer refuses overwriting an existing export. These paths are
gitignored. **Do not commit source dialogue content to GitHub.**

This is a **data-preparation step only**, not trained dialogue generation
or auto-labelled Intent learning. Next steps: license/privacy audit,
conversation-aware tokenizer, model context budgeting, response-training
pipeline, independent evaluation, and separate human-reviewed Intent
annotations. The existing 32-item Intent evaluation must not be promoted
to an independent final benchmark.


## Multi-turn RealPersonaChat response-generation pilot

Observed local audit (2026-11-05): 13,583 JSON files; 13,577 unique
valid conversations, 4 duplicate-content files excluded, and 2
duplicate-dialogue-ID files excluded. The exported corpus currently has
353,929 Train, 20,383 Dev, 20,549 Test **next-utterance examples**.
The 14 importer unit tests passed on Windows. This is observational
data from the operator's run, not a fresh independent evaluation.

`train_realpersonachat_pilot.py` trains a **separate**, tiny causal
Transformer on a bounded sample, using the local JSONL exporter output.
No file is pushed to GitHub, and all model/data artifacts are gitignored.
It preserves `speaker_0` / `speaker_1` rather than making assumptions
about assistant/user roles. Input format is preceding turns plus an
explicit next-speaker ID; target is the next utterance. Its loss is
supervised *only on generated response tokens*, not on the input prompt.

Tokenizer is fitted using Train examples **only**. Dev unseen characters
are handled as UNK and counted; this is a limitation. The model has
character-level tokens and a fixed context length (default 256): examples
that do not fit are dropped, with counts reported. Data is sampled
deterministically (SHA256 keyed by dialogue ID + turn ID) from the
pre-existing corpus-disjoint split, with default caps Train 2,000 /
Dev 200; **Test stays entirely unused**. Token NLL/PPL on Dev is a
diagnostic, *not* proof of coherent or helpful conversation.

```powershell
cd C:\Users\inoma\git\misc\LLM_TMP
git pull origin main
python -m unittest -v test_train_realpersonachat_pilot.py
python train_realpersonachat_pilot.py train --device auto --epochs 3 --train-limit 2000 --dev-limit 200
python train_realpersonachat_pilot.py generate --device auto --example-index 0
```

Training artifacts: `model/realpersonachat-pilot/` (checkpoint,
tokenizer, training.json). The `generate` command uses an existing
**development** example with its reference response shown for qualitative
comparison; it does not evaluate an independent blind holdout. No
persona labels, intent ground truth, or proof of factual correctness is
created. Review licensing and privacy restrictions before using actual
conversations for machine-learning training, even when files stay local.
Further work: privacy filtering, character-vocab optimization, better
context windows, response planning, reviewed Intent labels and
independent new-source testing.


## Response-pilot diagnostics: preflight + generated-response Dev audit

The importer is complete, but the user's own GPU pilot training results
have **not** yet been observed. New diagnostics allow the pilot to be
assessed before and after GPU training.

**Step 1: data-fit preflight, no GPU or response-model training**

```powershell
git pull origin main
python -m unittest -v test_train_realpersonachat_pilot.py test_evaluate_realpersonachat_pilot.py
python audit_realpersonachat_pilot.py --input-dir data/realpersonachat_import --train-limit 2000 --dev-limit 200 --context 256
```

The preflight computes the Train-only character vocabulary size, effective
Train and Dev sample counts, context-length exclusions, and Dev UNK
character counts. It reads Train + Dev only, never Test, and prints no
conversation snippets. If many examples are rejected, increase context
(e.g. 384) or reduce max chars/context turns **before** training, while
keeping a separate experiment checkpoint for fair comparison.

**Step 2: pilot training (if not already run)**

```powershell
python train_realpersonachat_pilot.py train --device auto --epochs 3 --train-limit 2000 --dev-limit 200
```

**Step 3: small Dev-generation audit using that checkpoint**

```powershell
python evaluate_realpersonachat_pilot.py --device auto --count 20
```

Local private report: `results/realpersonachat_pilot_dev.json`, with
prompt, actual generated response, reference response, and a
`copy_previous` baseline per case. The report also includes generation
coverage, exact matching, character overlap and empty-response rate. It
refuses overwriting an existing report unless `--overwrite` is passed.
The report contains conversation snippets and should **not** be published
without review. Similarity and exact-match metrics are only diagnostics:
alternative responses may be valid, so also inspect individual examples.
The input selection is the same local corpus Dev subset used for training
loss monitoring, NOT a blinded evaluation. No Test data is touched.

The two new scripts are `audit_realpersonachat_pilot.py` and
`evaluate_realpersonachat_pilot.py`; their supporting regression tests
are in `test_evaluate_realpersonachat_pilot.py`.


## Follow-up after first RealPersonaChat GPU pilot (2026-11-05)

Observed operator results: 2,000 Train / 200 Dev selected, context 256,
3 epochs. Train NLL 4.29513 -> 3.18845; Dev NLL 3.56798 -> 3.32128,
PPL 35.445 -> 27.6959. All examples fit context. Train-only
vocabulary contained 1,821 characters; Dev had 28/200 prompts with
unknown characters and 8/200 targets containing 11 unknown characters.

Dev generation audit on 20 cases: 18 generated, 2 abstained
(UNKNOWN_PROMPT_CHARACTERS), 0 exact matches; 17/18 generated texts
were the generic phrase "そうですね。" and the remaining one was
"そうですね！いですね！". Mean character overlap was ~0.2568
versus ~0.2375 for a copy-last-turn baseline. These results show
**severe response collapse** and do not establish contextual reasoning.

Updates in this revision:
- Fix the regression test: it incorrectly expected `お` in a tokenizer
  fitted on example text containing `こ` but no `お`.
- Use a context manager to close local JSONL files even when invalid
  examples raise exceptions (removes ResourceWarning).
- Add `unique_generated_count`, `distinct_response_ratio`,
  `most_common_generated_count` and
  `most_common_generated_fraction` to the Dev generation report.
- Test response-collapse metrics without requiring GPU.

```powershell
git pull origin main
python -m unittest -v test_train_realpersonachat_pilot.py test_evaluate_realpersonachat_pilot.py
python evaluate_realpersonachat_pilot.py --device auto --count 20 --overwrite
```

The updated evaluation does **not** require retraining and uses the same
checkpoint and Dev sample. It overwrites the existing local report only
when explicitly requested. Next compare larger supervised training
samples and alternative decoding against the same frozen Dev cases,
without using the Test split for model selection. Do not interpret
generic acknowledgements as high-quality dialogue or automate persona
memory ingestion from this pilot.


## Scale experiment: 2k vs 10k supervised conversation examples

**Hypothesis:** Increasing training examples from 2,000 to 10,000 while
keeping model architecture, seed, generation algorithm and the 200-item
Dev subset fixed may reduce generic-response collapse. This is a
development experiment, not a conclusion or guarantee. Previous 2k run:
Dev Token NLL 3.32128 (epoch 3); 18/20 Dev responses generated, 17/18
were "そうですね。", with a 1,821-symbol Train-only character vocabulary.
The previous 14-test suite surfaced a *test fixture* failure (expected a
character absent from its training sample) and file-close warnings;
both were corrected in commit 9dffa4c and preceding updates.

**Preflight for scaled run** (Train only for tokenizer fit):

```powershell
git pull origin main
python -m unittest -v test_train_realpersonachat_pilot.py test_evaluate_realpersonachat_pilot.py test_compare_realpersonachat_pilots.py
python audit_realpersonachat_pilot.py --train-limit 10000 --dev-limit 200 --context 256
```

**New checkpoint directory** to avoid overwriting the existing 2k
checkpoint and training record:

```powershell
python train_realpersonachat_pilot.py train --device auto --epochs 3 --train-limit 10000 --dev-limit 200 --outdir model/realpersonachat-pilot-10k
```

**Matched Dev-generation audit** (same first 20 Dev examples, max 50
tokens, greedy decoding; these reports contain text and must remain
private/local):

```powershell
python evaluate_realpersonachat_pilot.py --device auto --count 20 --outdir model/realpersonachat-pilot-10k --output results/realpersonachat_pilot_10k_dev.json
python compare_realpersonachat_pilots.py --baseline results/realpersonachat_pilot_dev.json --candidate results/realpersonachat_pilot_10k_dev.json
```

New `compare_realpersonachat_pilots.py` requires identical
`dialogue_id`/`turn_id`, identical prompt and reference for each
case, and rejects duplicates. It reports per-case cohort coverage,
exact-match, mean character-overlap delta on jointly-generated cases,
distinct response ratio and most-common response fraction. Unlike a
naive comparison of generated-only averages, it uses exactly the same
Dev cases and exposes changes in abstention. Scores do not measure
meaning, appropriateness or safety. `test_compare_realpersonachat_pilots.py`
tests this strict pairing. **Test JSONL remains unused**. Do not merge
any generated conversation text into permanent Semantic Memory without
source verification and human review.

After this comparison, decide whether to add more data, length-controlled
decoding and vocab fallback, or invest in a dialogue-act planner. Avoid
claiming that 10k sampling necessarily improves dialogue coherence.


## Next controlled experiment: 30k training and context counterfactuals

User-observed 10k run (3 epochs, 200 fixed Dev): 21 tests passed,
Dev NLL 2.91524, Dev PPL 18.4533, 20/20 generated, 14 unique
outputs, most-common response 25%, mean generated length 11.5.
Compared with 2k on the same 20 Dev examples, coverage rose from
90% to 100%, and response variety improved. **NLL/PPL are not perfectly
comparable across 2k and 10k checkpoints**, as their Train-only character
vocabularies differ (1,821 vs 2,420). Increased text diversity does not
by itself demonstrate context comprehension.

Use a fresh 30k checkpoint directory with the **same model architecture,
seed, 200 Dev examples, 2 preceding turns, 256 context, three epochs and
greedy generation**. Keep the 10k checkpoint unchanged.

```powershell
cd C:\Users\inoma\git\misc\LLM_TMP
git pull origin main
python -m unittest -v test_evaluate_realpersonachat_context.py test_compare_realpersonachat_pilots.py
python audit_realpersonachat_pilot.py --train-limit 30000 --dev-limit 200 --context 256
python train_realpersonachat_pilot.py train --device auto --epochs 3 --train-limit 30000 --dev-limit 200 --outdir model/realpersonachat-pilot-30k
python evaluate_realpersonachat_pilot.py --device auto --count 20 --outdir model/realpersonachat-pilot-30k --output results/realpersonachat_pilot_30k_dev.json
python compare_realpersonachat_pilots.py --baseline results/realpersonachat_pilot_10k_dev.json --candidate results/realpersonachat_pilot_30k_dev.json --output results/realpersonachat_10k_30k_comparison.json
```

**Counterfactual sensitivity, not semantic grading.** The new
`evaluate_realpersonachat_context.py` computes mean teacher-forced
log probability of each frozen Dev reference response under the original
prompt and an artificial prompt. The only intervention is replacing
the *last context turn's text* with a turn from a different Dev dialogue;
speaker identifiers, earlier turn(s), next speaker and response target
are preserved. It reports the fraction of scored cases where original
log probability is greater than counterfactual log probability, mean
log-probability delta and skip count (unknown characters, context
overflow, etc.). Because donor context might still be compatible, these
statistics are not accuracy or factual verification. This Dev
counterfactual is for *development only*, and must not substitute for a
separately authored, human-judged independent evaluation.

```powershell
python evaluate_realpersonachat_context.py --device auto --count 20 --outdir model/realpersonachat-pilot-10k --output results/realpersonachat_context_10k_dev.json
python evaluate_realpersonachat_context.py --device auto --count 20 --outdir model/realpersonachat-pilot-30k --output results/realpersonachat_context_30k_dev.json
```

The context reports include original/donor prompts and reference
responses; keep them **local and private**. Only Train and Dev split
files are read. Test remains sealed for a future final evaluation.
The diagnostic outputs are not compared across checkpoints as
log-probabilities directly; each model's original-vs-alternative
difference is measured on its own tokenizer/vocabulary.


## Context Ablation: frozen 30k model / Full vs Last Turn vs No Context

After the user's 30k run, Dev NLL decreased to 2.57207 with
20/20 generated, 14 unique responses and a 15% most-common-response
fraction. However, the previous last-turn-shuffle counterfactual gave
10/20 positive differences, mean +0.017187; this was insufficient
to establish context comprehension. It is not evidence that the
model necessarily disregards context.

The new `evaluate_realpersonachat_ablation.py` uses exactly the
same saved checkpoint/tokenizer and selected *Dev* examples to score
a frozen reference response (teacher-forced mean log probability)
under three prompt conditions:

1. `full`: the original two previous turns and the next-speaker marker.
2. `last_turn`: only the most recent previous turn, same marker.
3. `no_context`: conversation header and next-speaker marker only.

Each reference is identical under all three conditions. It scores
all three conditions per example, excludes a case from paired
comparisons if **any** condition has unknown prompt/reference
characters or exceeds the context, and reports all-three coverage,
mean log probability, mean within-case differences, sign fractions,
and count of cases with two original turns. This controls for
different response lengths/vocabularies *within a single model*.
The models were trained using Full Context, so shorter inputs are
out-of-training-distribution: this diagnostic cannot independently
prove semantic understanding or quantify a causal training benefit.
All prompts/references in the output must remain local and private.
No Test file is touched.

Run the regression tests first:

```powershell
git pull origin main
python -m unittest -v test_evaluate_realpersonachat_ablation.py test_evaluate_realpersonachat_context.py
```

Run both saved checkpoints with 200 fixed Dev items:

```powershell
python evaluate_realpersonachat_ablation.py --device auto --count 200 --outdir model/realpersonachat-pilot-10k --output results/realpersonachat_ablation_10k_dev.json
python evaluate_realpersonachat_ablation.py --device auto --count 200 --outdir model/realpersonachat-pilot-30k --output results/realpersonachat_ablation_30k_dev.json
```

If either report already exists, explicitly pass `--overwrite`
only when replacement is intended. Next: inspect cases where Full
Context loses to No Context, then design targeted semantic,
human-judged conversation-pair checks. Keep final independent Test
sealed until the protocol is frozen.


## Full-Length Context Shuffle: 10k vs 30k frozen Dev diagnostic

User-observed Context Ablation using 200 Dev cases: 10k had
194/200 all-three-condition scores, with Full-minus-Last +0.06413
and Full-minus-No-Context +0.97885. The 30k checkpoint had
198/200, Full-minus-Last +0.30168 and Full-minus-No-Context
+1.86572. These are *within-model* score differences and do not
prove semantic understanding. No-Context is out of the training
prompt distribution.

`evaluate_realpersonachat_fullshuffle.py` now evaluates a
*same-length-structure* alternative: both context turns are replaced
with text from a different Dev dialogue, while keeping the same
number of turns, speaker sequence and reference response. A
deterministic donor selector favors similar text lengths, rejects the
same conversation and rejects no-op identical text. This comparison
does not remove context, reducing (but not eliminating) prompt-format
confounding. It reuses saved checkpoints and Train-only tokenizers,
does not update model weights and does not access Test.

```powershell
git pull origin main
python -m unittest -v test_evaluate_realpersonachat_fullshuffle.py test_evaluate_realpersonachat_ablation.py
python evaluate_realpersonachat_fullshuffle.py --device auto --count 200 --outdir model/realpersonachat-pilot-10k --output results/realpersonachat_fullshuffle_10k_dev.json
python evaluate_realpersonachat_fullshuffle.py --device auto --count 200 --outdir model/realpersonachat-pilot-30k --output results/realpersonachat_fullshuffle_30k_dev.json
```

Key metrics: `paired_scored`, `positive_delta_fraction_on_scored`,
`mean_delta_log_probability`, `median_delta_log_probability`,
`mean_donor_length_difference`, and `abstention_reasons`.
Positive delta means the original prompt assigned a higher
teacher-forced mean token log-probability to the original reference.
It is not a semantic correctness or generation-quality score.
Artificially swapped text may remain sensible; donor lengths and
wording can still affect likelihood. Reports contain real conversation
text and must remain private. Output is not overwritten unless
`--overwrite` is explicitly used.

Next: use a separate manually reviewed set of compatible/incompatible
dialogue-context pairs, compare the same **fixed cohort** across
checkpoints, and only use the independent Test after the final
evaluation protocol is locked.


## Strict paired common-cohort comparison after Full-Length Shuffle (2026-11-05)

User-observed run: 16 related tests passed. Among 200 Dev cases, 10k
scored 188, with 63.83% positive original-minus-shuffle and
mean delta +0.02948; 30k scored 196, 71.43% positive and mean
delta +0.04911. These model-level results use DIFFERENT scored
cohorts, so cannot by themselves quantify a matched-case improvement.

`compare_realpersonachat_fullshuffle.py` reads ONLY the two existing
private local evaluation reports. It verifies that their 200 case keys,
donor dialogue identities, original and shuffled prompts, reference
responses and turn counts are identical; then excludes all cases in
which either model abstained. It produces a JSON summary with NO source
utterance text, including joint cohort size, both sign fractions and
mean deltas, direction-change counts and an exploratory seeded
bootstrap interval on the within-case difference of score deltas.
Because the models have different Train-only token vocabularies,
the score differences are still not perfectly comparable; the
bootstrap confidence interval also assumes exchangeable cases and
is descriptive, not a confirmatory population inference.

```powershell
git pull origin main
python -m unittest -v test_compare_realpersonachat_fullshuffle.py
python compare_realpersonachat_fullshuffle.py --baseline results/realpersonachat_fullshuffle_10k_dev.json --candidate results/realpersonachat_fullshuffle_30k_dev.json --output results/realpersonachat_fullshuffle_paired.json
```

Do not reuse these Dev results as independent model validation.
Next prioritize **human-judged topical consistency** for true-vs-
shuffled dialogues (with private text retained locally), controlled
speaker/length and tokenizer normalization, and independent holdout
after the evaluation protocol is fixed.


## Human-reviewed Semantic Context Evaluation: blinded Dev review

Prior user-run paired Full-Length Shuffle evaluation on 188 common Dev
cases: original-context win fraction 10k 0.6383 vs 30k 0.7074,
mean within-model log-likelihood delta +0.02948 vs +0.04682.
39 cases switched unfavorable-to-favorable, while 26 switched
favorable-to-unfavorable. These are model likelihood preferences,
not human-rated contextual suitability.

`review_realpersonachat_semantics.py` creates a *local/private* blinded
review pack from the TWO original scored Dev shuffle JSON reports.
It selects up to 15 examples from each of three diagnostic strata:
RECOVERED, REGRESSED and STABLE_OR_MIXED. Strata are sampled
deterministically by hashed case ID; they are **not a random
representative Dev sample**. Reviewers see only context A, context B,
the same reference answer, and empty review fields. The source's
original/shuffled order is pseudorandomized per case. No model outputs,
likelihood scores or gold-side labels appear in the packet.
The report's original case IDs are pseudonymized, although its
conversation text is **not anonymized**.

```powershell
git pull origin main
python -m unittest -v test_review_realpersonachat_semantics.py
python review_realpersonachat_semantics.py prepare --baseline results/realpersonachat_fullshuffle_10k_dev.json --candidate results/realpersonachat_fullshuffle_30k_dev.json --per-bucket 15 --review-pack results/realpersonachat_semantic_review_dev.json
```

Open `results/realpersonachat_semantic_review_dev.json` locally and
fill each case's `review` object by hand:

- `label`: `"A"` if A alone is better suited to the target response,
  `"B"` if B alone, `"BOTH"` if both, `"NEITHER"` if neither,
  or `"UNCLEAR"` if the context does not suffice.
- `rationale`: short, non-empty explanation of why the answer fits.
- `reviewer_id`: a non-empty pseudonymous identifier.
- Leave `label: null` for any item not yet reviewed; do not
  invent judgments from the model's metrics.

```powershell
python review_realpersonachat_semantics.py audit --review-pack results/realpersonachat_semantic_review_dev.json --audit-output results/realpersonachat_semantic_review_audit.json
```

The audit produces HUMAN_REVIEW_PENDING, INVALID or
HUMAN_REVIEW_COMPLETE and counts labels. It does **not**
automatically claim human-review completion or semantic accuracy.
Because selected cases are stratified by model behavior, the completed
set is qualitative failure analysis, not an unbiased population
accuracy estimate. Keep all text-bearing files out of Git and
external publication; `results/` is ignored by Git.
Test remains sealed. A separate unbiased, independently curated
evaluation should follow the review-protocol freeze.
