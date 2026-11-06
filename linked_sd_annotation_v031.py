"""Blind selection of existing SD topic IDs; never import AI proxy or gold labels."""
import json, hashlib, os, tempfile
from pathlib import Path
from intent_sd_context_v03 import SDRegistry

FIELDS=("a_previous","a_current","a_reference","b_previous","b_current","b_reference")
UNKNOWN="UNKNOWN"

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def digest(x):
    return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def catalog(snapshot):
    SDRegistry({k:v for k,v in snapshot.items() if k!="display_names"})
    names=snapshot.get("display_names",{})
    if not isinstance(names,dict): raise ValueError("INVALID_DISPLAY_NAMES")
    out=[]
    for n in snapshot["nodes"]:
        if n["node_type"] in ("TOPIC","CONCEPT","ENTITY"):
            label=names.get(n["sd_id"],n["sd_id"])
            if not isinstance(label,str) or not label.strip(): raise ValueError("INVALID_DISPLAY_NAME")
            out.append({"sd_id":n["sd_id"],"display_name":label,"node_type":n["node_type"]})
    return sorted(out,key=lambda n:n["sd_id"])

def prepare_blind_packet(source,snapshot):
    cases=source.get("cases")
    if not isinstance(cases,list) or not cases: raise ValueError("SOURCE_CASES_REQUIRED")
    ids=[]
    for c in cases:
        cid=c.get("case_id")
        if not isinstance(cid,str) or not cid.strip() or cid in ids: raise ValueError("INVALID_OR_DUPLICATE_CASE_ID")
        if not all(isinstance(c.get(k),str) for k in ("context_a","context_b","reference_response")):
            raise ValueError("SOURCE_TEXT_REQUIRED")
        ids.append(cid)
    catalog(snapshot)
    text=[{k:c[k] for k in ("case_id","context_a","context_b","reference_response")} for c in cases]
    return {"schema_version":"linked-sd-annotation/0.3.1","scope":"DEV_BLIND_SD_TOPIC_ANNOTATION",
            "registry_digest":digest(snapshot),"source_digest":digest(text),"case_ids":ids,
            "reviews":{cid:{"annotation_status":"PENDING",
                             "topic_refs":{k:UNKNOWN for k in FIELDS},"rationale":"","reviewer_id":""} for cid in ids}}

def validate_packet(packet,source,snapshot):
    fresh=prepare_blind_packet(source,snapshot)
    for key in ("schema_version","scope","registry_digest","source_digest","case_ids"):
        if packet.get(key)!=fresh[key]: raise ValueError("REVIEW_SOURCE_OR_REGISTRY_CHANGED:"+key)
    if not isinstance(packet.get("reviews"),dict) or set(packet["reviews"])!=set(fresh["case_ids"]):
        raise ValueError("INVALID_REVIEW_IDS")
    allowed={r["sd_id"] for r in catalog(snapshot)}
    for cid in packet["case_ids"]:
        r=packet["reviews"][cid]
        if set(r)!=set(("annotation_status","topic_refs","rationale","reviewer_id")) or set(r["topic_refs"])!=set(FIELDS):
            raise ValueError("INVALID_REVIEW_FIELDS")
        if any(v!=UNKNOWN and v not in allowed for v in r["topic_refs"].values()):
            raise ValueError("UNKNOWN_SD_SELECTION")
        if r["annotation_status"] not in ("PENDING","HUMAN_REVIEW_COMPLETE"):
            raise ValueError("INVALID_REVIEW_STATUS")
        if r["annotation_status"]=="HUMAN_REVIEW_COMPLETE" and (not r["rationale"].strip() or not r["reviewer_id"].strip()):
            raise ValueError("INCOMPLETE_HUMAN_REVIEW")
    return packet

def assign(packet,cid,refs,rationale,reviewer_id,allowed):
    if cid not in packet["reviews"] or set(refs)!=set(FIELDS): raise ValueError("INVALID_ANNOTATION_CASE")
    if any(v!=UNKNOWN and v not in allowed for v in refs.values()): raise ValueError("INVALID_SD_CHOICE")
    if not isinstance(rationale,str) or not rationale.strip(): raise ValueError("RATIONALE_REQUIRED")
    if not isinstance(reviewer_id,str) or not reviewer_id.strip(): raise ValueError("REVIEWER_REQUIRED")
    packet["reviews"][cid]={"annotation_status":"HUMAN_REVIEW_COMPLETE","topic_refs":dict(refs),
                             "rationale":rationale.strip(),"reviewer_id":reviewer_id.strip()}
    return packet

def audit(packet,source,snapshot):
    validate_packet(packet,source,snapshot)
    complete=[r for r in packet["reviews"].values() if r["annotation_status"]=="HUMAN_REVIEW_COMPLETE"]
    return {"status":"COMPLETE" if len(complete)==len(packet["case_ids"]) else "PENDING",
            "total":len(packet["case_ids"]),"complete":len(complete),
            "pending":len(packet["case_ids"])-len(complete),
            "unknown_fields":sum(sum(v==UNKNOWN for v in r["topic_refs"].values()) for r in complete),
            "warning":"Topic refs are human annotations, not human A/B gold labels."}

def save_atomic(path,packet):
    dest=Path(path);dest.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=".linked_sd_review_",dir=str(dest.parent))
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:
            json.dump(packet,f,ensure_ascii=False,indent=2);f.write("\n")
        os.replace(tmp,dest)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
