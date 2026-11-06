"""Selection-based blind Tkinter GUI for existing SD topic annotations."""
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from linked_sd_annotation_v031 import (UNKNOWN,FIELDS,read_json,catalog,
    prepare_blind_packet,validate_packet,assign,save_atomic,audit)

LABELS={"a_previous":"A: 直前の話題","a_current":"A: 現在の話題",
        "a_reference":"A: 文脈参照先","b_previous":"B: 直前の話題",
        "b_current":"B: 現在の話題","b_reference":"B: 文脈参照先"}

class LinkedSDReviewApp(tk.Tk):
    def __init__(self, source, registry, output):
        super().__init__()
        self.source=read_json(source)
        self.snapshot=read_json(registry)
        self.options=catalog(self.snapshot)
        self.packet=(read_json(output) if Path(output).exists()
                     else prepare_blind_packet(self.source,self.snapshot))
        validate_packet(self.packet,self.source,self.snapshot)
        self.output=Path(output)
        self.by_id={c["case_id"]:c for c in self.source["cases"]}
        self.allowed={r["sd_id"] for r in self.options}
        self.pos=0
        self.title("LLM_TMP — Blind Linked SD Topic Review v0.3.1")
        self.geometry("1050x840")
        self.header=ttk.Label(self,font=("",13,"bold"))
        self.header.pack(anchor="w",padx=12,pady=7)
        ttk.Label(self,text="AI代理ラベルとモデル予測は非表示。既存SDを選択、判定不能ならUNKNOWN。").pack(anchor="w",padx=12)
        self.text=tk.Text(self,height=15,wrap="word",state="disabled")
        self.text.pack(fill="both",expand=True,padx=12,pady=8)
        toolbar=ttk.Frame(self);toolbar.pack(fill="x",padx=12)
        ttk.Label(toolbar,text="SD検索:").pack(side="left")
        self.search=tk.StringVar()
        ttk.Entry(toolbar,textvariable=self.search,width=36).pack(side="left",padx=6)
        self.search.trace_add("write",lambda *_:self.refilter())
        form=ttk.Frame(self);form.pack(fill="x",padx=12,pady=8)
        self.selected={key:tk.StringVar(value=UNKNOWN) for key in FIELDS}
        self.combos={}
        for i,key in enumerate(FIELDS):
            ttk.Label(form,text=LABELS[key]).grid(row=i,column=0,sticky="w",padx=4,pady=4)
            cb=ttk.Combobox(form,textvariable=self.selected[key],state="readonly",width=55)
            cb.grid(row=i,column=1,sticky="we",padx=4,pady=4)
            self.combos[key]=cb
        form.columnconfigure(1,weight=1)
        ttk.Label(self,text="Reviewer ID（必須）").pack(anchor="w",padx=12)
        self.reviewer=tk.StringVar()
        ttk.Entry(self,textvariable=self.reviewer).pack(fill="x",padx=12)
        ttk.Label(self,text="理由（必須）").pack(anchor="w",padx=12,pady=(6,0))
        self.reason=tk.Text(self,height=3,wrap="word")
        self.reason.pack(fill="x",padx=12,pady=4)
        nav=ttk.Frame(self);nav.pack(fill="x",padx=12,pady=9)
        ttk.Button(nav,text="前へ",command=lambda:self.move(-1)).pack(side="left",padx=4)
        ttk.Button(nav,text="分類して次へ",command=self.save_next).pack(side="left",padx=4)
        ttk.Button(nav,text="保留して次へ",command=lambda:self.move(1)).pack(side="left",padx=4)
        ttk.Button(nav,text="保存",command=self.save).pack(side="right",padx=4)
        self.protocol("WM_DELETE_WINDOW",self.close)
        self.refilter()
        self.show()

    def desc(self,row):
        return row["sd_id"]+" | "+row["display_name"]+" ("+row["node_type"]+")"

    def refilter(self):
        q=self.search.get().strip().casefold()
        choices=[UNKNOWN]+[self.desc(n) for n in self.options if q in (n["sd_id"]+" "+n["display_name"]).casefold()]
        for key,cb in self.combos.items():
            chosen=self.selected[key].get()
            cb["values"]=choices+([chosen] if chosen not in choices else [])

    def choice(self,value):
        if value==UNKNOWN:return UNKNOWN
        id_=value.split(" | ",1)[0]
        matches=[r for r in self.options if r["sd_id"]==id_]
        if len(matches)!=1 or value!=self.desc(matches[0]):
            raise ValueError("INVALID_SD_CHOICE")
        return id_

    def show(self):
        cid=self.packet["case_ids"][self.pos]
        item=self.by_id[cid]
        review=self.packet["reviews"][cid]
        done=sum(v["annotation_status"]=="HUMAN_REVIEW_COMPLETE" for v in self.packet["reviews"].values())
        self.header.configure(text=f'{self.pos+1}/{len(self.packet["case_ids"])} | 完了 {done} | ID {cid}')
        lines=["【Context A】",item["context_a"],"【Context B】",item["context_b"],
               "【参照応答】",item["reference_response"]]
        self.text.config(state="normal");self.text.delete("1.0","end")
        self.text.insert("1.0","\n\n".join(lines));self.text.config(state="disabled")
        labels={r["sd_id"]:self.desc(r) for r in self.options}
        for k in FIELDS:
            val=review["topic_refs"][k]
            self.selected[k].set(labels[val] if val!=UNKNOWN else UNKNOWN)
        self.reviewer.set(review["reviewer_id"])
        self.reason.delete("1.0","end");self.reason.insert("1.0",review["rationale"])
        self.refilter()

    def move(self,n):
        nxt=self.pos+n
        if 0<=nxt<len(self.packet["case_ids"]):
            self.pos=nxt;self.show()

    def save_next(self):
        try:
            cid=self.packet["case_ids"][self.pos]
            refs={key:self.choice(self.selected[key].get()) for key in FIELDS}
            assign(self.packet,cid,refs,self.reason.get("1.0","end").strip(),
                   self.reviewer.get(),self.allowed)
            save_atomic(self.output,self.packet)
        except (ValueError,KeyError) as e:
            messagebox.showwarning("確認",str(e));return
        self.move(1)
        self.show()

    def save(self):
        save_atomic(self.output,self.packet)
        r=audit(self.packet,self.source,self.snapshot)
        messagebox.showinfo("保存",f'保存済み: {self.output}\n{r["complete"]}/{r["total"]} 完了')

    def close(self):
        if messagebox.askyesno("終了","レビュー状態を保存して終了しますか？"):
            save_atomic(self.output,self.packet)
        self.destroy()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",default="results/realpersonachat_semantic_review_dev.json")
    p.add_argument("--registry",required=True)
    p.add_argument("--output",default="results/linked_sd_annotations_v031_dev.json")
    a=p.parse_args()
    LinkedSDReviewApp(a.source,a.registry,a.output).mainloop()

if __name__=="__main__":
    main()
