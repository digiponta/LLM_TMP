"""GUI: review automated SD topic proposals, click approve/reject only."""
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

from linked_sd_annotation_v031 import read_json, save_atomic, validate_packet, catalog, UNKNOWN
from linked_sd_proxy_review_v032 import (
    prepare_proposals, validate_proposals, decide, export_approvals, audit_approvals
)

LABELS = {
    "a_previous": "Context A / 前のTopic",
    "a_current": "Context A / 現在のTopic",
    "a_reference": "Context A / 文脈参照",
    "b_previous": "Context B / 前のTopic",
    "b_current": "Context B / 現在のTopic",
    "b_reference": "Context B / 文脈参照",
}

class ApprovalWindow(tk.Tk):
    def __init__(self, source_path, registry_path, proposals_path, approved_path, reviewer_id):
        super().__init__()
        self.source = read_json(source_path)
        self.snapshot = read_json(registry_path)
        self.proposals_path = Path(proposals_path)
        self.approved_path = Path(approved_path)
        self.reviewer_id = reviewer_id
        self.packet = (read_json(self.proposals_path) if self.proposals_path.exists()
                       else prepare_proposals(self.source, self.snapshot))
        validate_proposals(self.packet, self.source, self.snapshot)
        if self.approved_path.exists():
            # Never replace an unrelated or older human review file.
            validate_packet(read_json(self.approved_path), self.source, self.snapshot)
        self.originals = {c["case_id"]: c for c in self.source["cases"]}
        self.labels = {x["sd_id"]: x["display_name"] for x in catalog(self.snapshot)}
        self.pos = 0
        self.title("LLM_TMP — SD候補の確認・承認 v0.3.2")
        self.geometry("1060x860")
        self.head = ttk.Label(self, font=("", 13, "bold"))
        self.head.pack(anchor="w", padx=14, pady=8)
        ttk.Label(self, text="自動生成されたSD候補です。元会話と照合し、承認／却下／保留を選んでください。").pack(anchor="w", padx=14)
        self.context = tk.Text(self, height=17, wrap="word", state="disabled")
        self.context.pack(fill="both", expand=True, padx=14, pady=8)
        ttk.Label(self, text="自動提案：既存SDとの文字列一致（精度未検証）", font=("", 11, "bold")).pack(anchor="w", padx=14)
        self.suggestion = tk.Text(self, height=13, wrap="word", state="disabled")
        self.suggestion.pack(fill="both", expand=True, padx=14, pady=6)
        nav = ttk.Frame(self)
        nav.pack(fill="x", padx=14, pady=11)
        ttk.Button(nav, text="前へ", command=lambda: self.move(-1)).pack(side="left", padx=5)
        ttk.Button(nav, text="承認して次へ", command=lambda: self.record("APPROVED")).pack(side="left", padx=5)
        ttk.Button(nav, text="却下して次へ", command=lambda: self.record("REJECTED")).pack(side="left", padx=5)
        ttk.Button(nav, text="保留して次へ", command=lambda: self.record("PENDING")).pack(side="left", padx=5)
        ttk.Button(nav, text="保存", command=self.save).pack(side="right", padx=5)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.show()

    @staticmethod
    def put(widget, content):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    def show(self):
        cid = self.packet["case_ids"][self.pos]
        c = self.originals[cid]
        p = self.packet["proposals"][cid]
        stats = audit_approvals(self.packet, self.source, self.snapshot)
        counts = stats["decisions"]
        self.head.configure(text=(
            f'{self.pos + 1}/{stats["total"]} | 承認 {counts["APPROVED"]}'
            f' / 却下 {counts["REJECTED"]} / 保留 {counts["PENDING"]}'
            f' | ID {cid} | 現在 {self.packet["decisions"][cid]}'))
        self.put(self.context,
                 "【Context A】\n" + c["context_a"] +
                 "\n\n【Context B】\n" + c["context_b"] +
                 "\n\n【参照応答】\n" + c["reference_response"])
        lines = []
        for key, label in LABELS.items():
            val = p["topic_refs"][key]
            display = val if val == UNKNOWN else f'{val}（{self.labels.get(val, "?")}）'
            lines.append(f'{label}: {display}')
            lines.append(f'  根拠: {p["reason_codes"][key]}')
        lines.append("\n" + p["summary"])
        self.put(self.suggestion, "\n".join(lines))

    def move(self, delta):
        nxt = self.pos + delta
        if 0 <= nxt < len(self.packet["case_ids"]):
            self.pos = nxt
            self.show()

    def persist(self):
        validate_proposals(self.packet, self.source, self.snapshot)
        output = export_approvals(self.packet, self.source, self.snapshot, self.reviewer_id)
        save_atomic(self.proposals_path, self.packet)
        save_atomic(self.approved_path, output)

    def record(self, choice):
        cid = self.packet["case_ids"][self.pos]
        candidate = self.packet["proposals"][cid]
        if choice == "APPROVED" and all(v == UNKNOWN for v in candidate["topic_refs"].values()):
            if not messagebox.askyesno("全項目UNKNOWN",
                   "候補はすべてUNKNOWNです。未解決として承認しますか？"):
                return
        old = self.packet["decisions"][cid]
        decide(self.packet, cid, choice)
        try:
            self.persist()
        except Exception as exc:
            decide(self.packet, cid, old)
            messagebox.showerror("保存に失敗", str(exc))
            return
        self.move(1)
        self.show()

    def save(self):
        try:
            self.persist()
        except Exception as exc:
            messagebox.showerror("保存に失敗", str(exc))
            return
        messagebox.showinfo("保存", str(self.proposals_path))

    def close(self):
        try:
            self.persist()
        except Exception as exc:
            if not messagebox.askyesno("保存失敗", f"{exc}\n保存せず終了しますか？"):
                return
        self.destroy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="results/realpersonachat_semantic_review_dev.json")
    parser.add_argument("--registry", default="results/linked_sd_registry_dev.json")
    parser.add_argument("--proposals", default="results/linked_sd_proxy_proposals_v032_dev.json")
    parser.add_argument("--approved", default="results/linked_sd_proxy_approved_v032_dev.json")
    parser.add_argument("--reviewer-id", default="LOCAL_USER_APPROVER")
    args = parser.parse_args()
    ApprovalWindow(args.source, args.registry, args.proposals,
                   args.approved, args.reviewer_id).mainloop()

if __name__ == "__main__":
    main()
