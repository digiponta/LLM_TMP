# Intent SD v0.3.2 — Candidate generation + approve-only GUI

This replaces six manual SD topic entries per case with **untrusted
computer-proposed SD topic references**, and lets the user review the
original source and press **Approve / Reject / Defer**. Nothing is
automatically approved.

## Quick start on Windows

```powershell
cd C:\Users\inoma\git\misc\LLM_TMP
git fetch origin
git switch --track origin/feature/intent-sd-v0.3.2-proxy-approval

python -m unittest -v test_linked_sd_proxy_review_v032.py test_linked_sd_annotation_v031.py
python linked_sd_proxy_cli_v032.py prepare
python review_linked_sd_approval_gui_v032.py
python linked_sd_proxy_cli_v032.py audit
```

For an existing local branch, use `git switch feature/intent-sd-v0.3.2-proxy-approval`
and `git pull --ff-only origin feature/intent-sd-v0.3.2-proxy-approval`.

All commands accept local source and registry paths via `--source`
and `--registry`. Defaults are
`results/realpersonachat_semantic_review_dev.json` and
`results/linked_sd_registry_dev.json` respectively.

The GUI shows Context A/B and fixed reference response, followed by six
automatically selected Topic SD references and *reason codes*. You need
not type the individual topic fields or rationale.

## Source and model behavior

- Reads only the case ID, Context A/B and fixed response for UI; the
  candidate model **does not inspect the fixed response for predictions**.
- Candidates use exact displayed SD labels from the current frozen
  registry (Japanese substring match, bounded English word match).
- When a context has newline-separated lines, the earlier lines and
  final line are separately inspected for previous/current candidates.
  This is only a heuristic; newlines are not guaranteed turn boundaries.
- When no unambiguous existing SD is identified, returns `UNKNOWN`.
  Context-reference resolution is always `UNKNOWN` in this baseline.
- AI proxy A/B labels, model scores and existing human judgments are
  never inputs to the proposal model; the proposal output also excludes
  raw conversation text.

The sample registry produced by `build_linked_sd_registry_dev.py`
contains only a handful of concepts such as 猫/犬. It cannot provide
meaningful automatic Topic matching for general conversations. Use an
existing source-backed, frozen real SD registry for broad coverage.
No external API/model inference is performed by the current proposal engine.
It is a conservative rule-based代理入力, not advanced LLM understanding.

## Output separation

- `results/linked_sd_proxy_proposals_v032_dev.json`: frozen proposals,
  reason codes and human decision states `PENDING/APPROVED/REJECTED`.
- `results/linked_sd_proxy_approved_v032_dev.json`: only explicitly
  approved cases are marked `HUMAN_REVIEW_COMPLETE`. Rejected and
  deferred cases remain `PENDING`; none become human A/B gold.

Both are local private outputs under ignored `results/`.
Resume validates the source text fingerprint, registry fingerprint and
candidate engine's deterministic output. Changed inputs are rejected,
not silently merged. The CLI `prepare` does not overwrite existing proposals.
The approval GUI saves after each decision and on closing.

```powershell
python linked_sd_proxy_cli_v032.py audit
python linked_sd_proxy_cli_v032.py export
```

**Approval meaning:** the human approves the displayed SD-topic candidate
as suitable for subsequent testing; it is *not* an independent judgment
of which context A/B is correct, not validation of model scores, and not
approval for persistent SD graph facts.

**Known limitation:** a reviewer who only presses Approve/Reject cannot
fix a wrong individual field. Reject makes the case remain excluded
from human-approved topic inputs; to improve coverage, expand the
source-backed SD registry or upgrade the proposal engine. Never silently
overwrite a rejection by generating a different model prediction.
