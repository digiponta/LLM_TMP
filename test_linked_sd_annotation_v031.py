"""Headless v0.3.1 blind SD annotation regression tests."""
import copy,tempfile,unittest
from pathlib import Path
from linked_sd_annotation_v031 import (FIELDS,UNKNOWN,catalog,prepare_blind_packet,
                                       validate_packet,assign,audit,save_atomic,read_json)
def reg():
    return {"namespace":"local","snapshot_id":"test","display_names":{"T1":"旅行","T2":"京都"},
            "nodes":[{"sd_id":"T1","node_type":"TOPIC"},{"sd_id":"T2","node_type":"ENTITY"}]}
def source():
    return {"cases":[{"case_id":"c1","context_a":"旅行の話","context_b":"京都の話",
                      "reference_response":"どちらへ？","proxy_label":"A",
                      "model_prediction":"B","gold_label":"A","raw_score":0.9}]}
class BlindReviewTests(unittest.TestCase):
    def test_blind_no_leak(self):
        s=str(prepare_blind_packet(source(),reg()))
        for term in ("旅行の話","京都の話","どちらへ","proxy_label","gold_label","raw_score","model_prediction"):
            self.assertNotIn(term,s)
    def test_catalog_names(self):
        self.assertEqual(catalog(reg())[0]["display_name"],"旅行")
    def test_default_unknown(self):
        p=prepare_blind_packet(source(),reg())
        self.assertTrue(all(v==UNKNOWN for v in p["reviews"]["c1"]["topic_refs"].values()))
    def test_assign(self):
        p=prepare_blind_packet(source(),reg())
        refs={k:UNKNOWN for k in FIELDS};refs["a_previous"]="T1"
        assign(p,"c1",refs,"本文の内容","reviewer",{"T1","T2"})
        self.assertEqual(audit(p,source(),reg())["complete"],1)
    def test_invalid_choice(self):
        with self.assertRaisesRegex(ValueError,"INVALID_SD_CHOICE"):
            assign(prepare_blind_packet(source(),reg()),"c1",{k:"FAKE" for k in FIELDS},"理由","reviewer",{"T1","T2"})
    def test_requires_reason(self):
        with self.assertRaisesRegex(ValueError,"RATIONALE"):
            assign(prepare_blind_packet(source(),reg()),"c1",{k:UNKNOWN for k in FIELDS},"","reviewer",{"T1","T2"})
    def test_requires_reviewer(self):
        with self.assertRaisesRegex(ValueError,"REVIEWER"):
            assign(prepare_blind_packet(source(),reg()),"c1",{k:UNKNOWN for k in FIELDS},"理由","",{"T1","T2"})
    def test_source_changed(self):
        p=prepare_blind_packet(source(),reg());s=source();s["cases"][0]["context_a"]="改変"
        with self.assertRaisesRegex(ValueError,"SOURCE_OR_REGISTRY_CHANGED"):
            validate_packet(p,s,reg())
    def test_registry_changed(self):
        p=prepare_blind_packet(source(),reg());r=reg()
        r["nodes"].append({"sd_id":"T3","node_type":"TOPIC"})
        with self.assertRaisesRegex(ValueError,"SOURCE_OR_REGISTRY_CHANGED"):
            validate_packet(p,source(),r)
    def test_duplicate_id(self):
        s=source();s["cases"].append(copy.deepcopy(s["cases"][0]))
        with self.assertRaisesRegex(ValueError,"DUPLICATE"):
            prepare_blind_packet(s,reg())
    def test_pending(self):
        self.assertEqual(audit(prepare_blind_packet(source(),reg()),source(),reg())["status"],"PENDING")
    def test_save_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/"review.json";p=prepare_blind_packet(source(),reg())
            save_atomic(f,p)
            self.assertEqual(validate_packet(read_json(f),source(),reg()),p)
    def test_reject_forged_review(self):
        p=prepare_blind_packet(source(),reg());p["reviews"]["c1"]["topic_refs"]["a_current"]="fake"
        with self.assertRaisesRegex(ValueError,"UNKNOWN_SD_SELECTION"):
            validate_packet(p,source(),reg())
if __name__=="__main__":unittest.main()
