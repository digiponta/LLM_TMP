"""Offline Tkinter human-cause review UI for LLM_TMP Intent SD v0.1.

Only an explicit click saves a human annotation. AI-proxy fields stay separate.
Use private local JSON; never commit case text or completed review packets.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

CAUSES = ("LEXICAL", "TOPIC", "SPEECH_ACT", "CONTEXT", "POLARITY", "OTHER")
LABELS = {
    "LEXICAL": "語彙・言い換え依存", "TOPIC": "話題・対象の誤認",
    "SPEECH_ACT": "質問・依頼・報告の誤認", "CONTEXT": "文脈参照の失敗",
    "POLARITY": "否定・肯定の誤認", "OTHER": "その他・判断不能",
}

def load_packet(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    rows = data.get("mismatch_cases")
    if not isinstance(rows, list):
        raise ValueError("MISSING_MISMATCH_CASES")
    ids = [x.get("case_id") for x in rows]
    if len(ids) != len(set(ids)) or not all(ids):
        raise ValueError("DUPLICATE_OR_MISSING_CASE_ID")
    return data

def set_review(case, primary, secondary, rationale):
    if primary not in CAUSES:
        raise ValueError("主原因を選んでください")
    if (len(secondary) != len(set(secondary)) or primary in secondary
            or any(x not in CAUSES for x in secondary)):
        raise ValueError("副原因の選択が不正です")
    if not rationale.strip():
        raise ValueError("判断理由を入力してください")
    case.update(primary_cause=primary, secondary_causes=list(secondary),
                rationale=rationale.strip(), annotation_status="HUMAN_REVIEW_COMPLETE")

def atomic_save(path, data):
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".intent_review_", dir=str(dest.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

class ReviewWindow(tk.Tk):
    def __init__(self, source, output):
        super().__init__()
        self.packet = load_packet(source)
        self.output = Path(output)
        self.position = 0
        self.title("LLM_TMP — Intent SD 人手レビュー")
        self.geometry("940x730")
        self.primary = tk.StringVar()
        self.secondary = {k: tk.BooleanVar(value=False) for k in CAUSES}
        self.head = ttk.Label(self, font=("", 13, "bold"))
        self.head.pack(anchor="w", padx=16, pady=12)
        self.details = tk.Text(self, height=12, wrap="word", state="disabled")
        self.details.pack(fill="both", expand=True, padx=16, pady=6)
        ttk.Label(self, text="主原因：1つ選択").pack(anchor="w", padx=16)
        main = ttk.Frame(self)
        main.pack(fill="x", padx=16)
        for i, key in enumerate(CAUSES):
            ttk.Radiobutton(main, text=f"{key} — {LABELS[key]}", value=key,
                            variable=self.primary).grid(row=i//2, column=i%2, sticky="w", padx=10)
        ttk.Label(self, text="副原因：複数選択可能").pack(anchor="w", padx=16, pady=(10, 0))
        extras = ttk.Frame(self)
        extras.pack(fill="x", padx=16)
        for i, key in enumerate(CAUSES):
            ttk.Checkbutton(extras, text=key, variable=self.secondary[key]).grid(row=0, column=i, padx=5)
        ttk.Label(self, text="判断理由（必須。本文が無ければ保留を推奨）").pack(anchor="w", padx=16, pady=(10, 0))
        self.reason = tk.Text(self, height=4, wrap="word")
        self.reason.pack(fill="x", padx=16, pady=5)
        nav = ttk.Frame(self)
        nav.pack(fill="x", padx=16, pady=10)
        ttk.Button(nav, text="前へ", command=lambda: self.move(-1)).pack(side="left", padx=5)
        ttk.Button(nav, text="保存して次へ", command=self.save_next).pack(side="left", padx=5)
        ttk.Button(nav, text="保留して次へ", command=lambda: self.move(1)).pack(side="left", padx=5)
        ttk.Button(nav, text="ファイルに保存", command=self.save).pack(side="right", padx=5)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.show()

    def current(self):
        return self.packet["mismatch_cases"][self.position]

    def show(self):
        case = self.current()
        total = len(self.packet["mismatch_cases"])
        done = sum(bool(x.get("primary_cause")) for x in self.packet["mismatch_cases"])
        self.head.config(text=f"{self.position+1}/{total}  |  分類済 {done}  |  ID {case['case_id']}")
        lines = [f"代理ラベル: {case.get('proxy_label')}",
                 f"10k一致: {case.get('baseline_matches_proxy')} / 30k一致: {case.get('candidate_matches_proxy')}"]
        if case.get("ai_proxy_analysis"):
            lines.append("AI代理分析: " + json.dumps(case["ai_proxy_analysis"], ensure_ascii=False, indent=2))
        for key in ("context_a", "context_b", "reference_response"):
            if key in case:
                lines.append(f"{key}: {case[key]}")
        if not any(k in case for k in ("context_a", "context_b", "reference_response")):
            lines.append("元の会話本文は含まれていません。原因を断定せず必要なら保留してください。")
        self.details.config(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", "\n\n".join(lines))
        self.details.config(state="disabled")
        self.primary.set(case.get("primary_cause") or "")
        for k, v in self.secondary.items():
            v.set(k in case.get("secondary_causes", []))
        self.reason.delete("1.0", "end")
        self.reason.insert("1.0", case.get("rationale", ""))

    def move(self, step):
        nxt = self.position + step
        if 0 <= nxt < len(self.packet["mismatch_cases"]):
            self.position = nxt
            self.show()

    def save(self):
        atomic_save(self.output, self.packet)
        messagebox.showinfo("保存", f"保存しました: {self.output}")

    def save_next(self):
        try:
            set_review(self.current(), self.primary.get(),
                       [k for k,v in self.secondary.items() if v.get()],
                       self.reason.get("1.0", "end").strip())
        except ValueError as e:
            messagebox.showwarning("入力確認", str(e))
            return
        atomic_save(self.output, self.packet)
        self.move(1)

    def close(self):
        if messagebox.askyesno("終了", "レビュー状態を保存して終了しますか？"):
            atomic_save(self.output, self.packet)
        self.destroy()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/intent_sd_mismatch_review_dev.json")
    p.add_argument("--output", default="results/intent_sd_mismatch_human_review_dev.json")
    a = p.parse_args()
    source = a.output if Path(a.output).exists() else a.input
    if not load_packet(source)["mismatch_cases"]:
        raise ValueError("NO_REVIEW_CASES")
    ReviewWindow(source, a.output).mainloop()

if __name__ == "__main__":
    main()
