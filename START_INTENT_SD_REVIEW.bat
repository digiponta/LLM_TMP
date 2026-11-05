@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -3 review_intent_sd_gui.py --input "results/intent_sd_mismatch_review_dev.json" --output "results/intent_sd_mismatch_human_review_dev.json"
if errorlevel 1 pause
