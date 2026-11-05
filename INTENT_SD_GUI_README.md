# Selection-based human review (Windows)

This GUI is an **optional local/manual** review interface for Intent SD v0.1.
It uses built-in Python Tkinter (no pip dependencies). It does not assign
a human label automatically and it never treats AI proxy as human gold.

## Run

```powershell
python -m unittest -v test_review_intent_sd_gui.py
python analyze_proxy_disagreements.py prepare  # 初回のみ。既存ファイルがあれば省略
python review_intent_sd_gui.py
```

Or double-click `START_INTENT_SD_REVIEW.bat` after preparing the mismatch JSON.

For an alternative local packet (including the AI proxy review addendum):

```powershell
python review_intent_sd_gui.py --input results/intent_sd_mismatch_ai_proxy_review.json --output results/intent_sd_mismatch_human_review_dev.json
```

The UI has a single-choice primary cause, multi-choice secondary causes,
required rationale, previous/next navigation, defer, and safe local save.
It automatically resumes the existing output file (when present) and
never modifies the original input.

**Review limitation:** The default mismatch packet excludes raw conversation
text; where missing, do not infer linguistic error causes from the agreement
pattern alone. Keep the completed JSON and any context-bearing review pack
private under `results/` (gitignored). To validate reviewed categories:

```powershell
python analyze_proxy_disagreements.py audit --packet results/intent_sd_mismatch_human_review_dev.json
```

## Display original dialogue text (updated)

The GUI now looks up each reviewed `case_id` in the **local**
`results/realpersonachat_semantic_review_ai_proxy.json` file.
It displays **Context A**, **Context B**, and the fixed **reference
response** in the scrolling text area, in addition to the existing
agreement indicators. The text is loaded for display only; it is never
copied into the human annotation output.

Default command:

```powershell
python review_intent_sd_gui.py
```

If your source review pack has another path, specify it:

```powershell
python review_intent_sd_gui.py --source-text results/realpersonachat_semantic_review_dev.json
```

The source JSON must contain `cases` with matching `case_id`,
`context_a`, `context_b`, `reference_response`.
If this file is missing or a case cannot be matched, the GUI
shows a clear warning instead of making up conversation text.
Keep the source review file local; do not commit it.

You do **not** need to regenerate the mismatch review packet, so
existing classifications and pending reviews remain available.
