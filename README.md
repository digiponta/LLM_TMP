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
