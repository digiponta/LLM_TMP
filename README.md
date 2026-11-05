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
