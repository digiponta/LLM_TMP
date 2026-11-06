"""Prepare/audit auto-generated Linked SD candidates without opening the GUI."""
import argparse
import json
from pathlib import Path
from linked_sd_annotation_v031 import read_json, save_atomic
from linked_sd_proxy_review_v032 import prepare_proposals, validate_proposals, audit_approvals, export_approvals

def main():
    p=argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare","audit","export"))
    p.add_argument("--source",default="results/realpersonachat_semantic_review_dev.json")
    p.add_argument("--registry",default="results/linked_sd_registry_dev.json")
    p.add_argument("--proposals",default="results/linked_sd_proxy_proposals_v032_dev.json")
    p.add_argument("--approved",default="results/linked_sd_proxy_approved_v032_dev.json")
    p.add_argument("--reviewer-id",default="LOCAL_USER_APPROVER")
    a=p.parse_args()
    src,reg=read_json(a.source),read_json(a.registry)
    path=Path(a.proposals)
    if a.action=="prepare":
        if path.exists():
            existing=read_json(path)
            validate_proposals(existing,src,reg)
            print(json.dumps({"status":"EXISTS_VALIDATED_NO_OVERWRITE",
                              "audit":audit_approvals(existing,src,reg)},ensure_ascii=False,indent=2))
            return
        packet=prepare_proposals(src,reg)
        save_atomic(path,packet)
        print(json.dumps({"status":"CANDIDATES_PREPARED_UNAPPROVED",
                          "audit":audit_approvals(packet,src,reg)},ensure_ascii=False,indent=2))
    else:
        packet=read_json(path)
        validate_proposals(packet,src,reg)
        if a.action=="audit":
            print(json.dumps(audit_approvals(packet,src,reg),ensure_ascii=False,indent=2))
        else:
            out=Path(a.approved)
            if out.exists():
                from linked_sd_annotation_v031 import validate_packet
                validate_packet(read_json(out),src,reg)
            exported=export_approvals(packet,src,reg,a.reviewer_id)
            save_atomic(out,exported)
            print(json.dumps({"status":"APPROVALS_EXPORTED",
                              "approved_count":sum(v["annotation_status"]=="HUMAN_REVIEW_COMPLETE"
                                                   for v in exported["reviews"].values())},
                             ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()
