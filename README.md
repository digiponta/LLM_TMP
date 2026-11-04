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
