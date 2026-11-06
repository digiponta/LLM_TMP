"""Inspect/init/audit blind Linked SD topic annotations."""
import argparse,json
from pathlib import Path
from linked_sd_annotation_v031 import (read_json,catalog,prepare_blind_packet,
                                       validate_packet,audit,save_atomic)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("inspect","init","audit"))
    p.add_argument("--source",default="results/realpersonachat_semantic_review_dev.json")
    p.add_argument("--registry",required=True)
    p.add_argument("--output",default="results/linked_sd_annotations_v031_dev.json")
    a=p.parse_args()
    source,snapshot=read_json(a.source),read_json(a.registry)
    if a.action=="inspect":
        packet=prepare_blind_packet(source,snapshot)
        result={"status":"VALID","cases":len(packet["case_ids"]),
                "sd_nodes":len(catalog(snapshot)),
                "registry_digest":packet["registry_digest"]}
    elif a.action=="init":
        if Path(a.output).exists():raise ValueError("REPORT_EXISTS_REFUSE_OVERWRITE")
        packet=prepare_blind_packet(source,snapshot)
        save_atomic(a.output,packet)
        result={"status":"PENDING","cases":len(packet["case_ids"]),"output":a.output}
    else:
        packet=read_json(a.output)
        validate_packet(packet,source,snapshot)
        result=audit(packet,source,snapshot)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
