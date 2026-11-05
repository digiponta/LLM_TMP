# Selection-based human review (Windows)

This GUI is an **optional local/manual** review interface for Intent SD v0.1.
It uses built-in Python Tkinter (no pip dependencies). It does not assign
a human label automatically and it never treats AI proxy as human gold.

## Run

```powershell
python -m unittest -v test_review_intent_sd_gui.py
python analyze_proxy_disagreements.py prepare
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
