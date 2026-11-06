"""Small synthetic v0.3 demonstration (no corpus and no gold labels)."""
import json
from intent_sd_context_v03 import SDRegistry, classify_topic_transition, choose_context_v03

def ref(value):
    return {"namespace": "local", "sd_id": value}

def run():
    snapshot = {"namespace": "local", "snapshot_id": "synthetic-v03",
                "nodes": [
                    {"sd_id": "TRAVEL", "node_type": "TOPIC"},
                    {"sd_id": "KYOTO", "node_type": "ENTITY"},
                    {"sd_id": "MUSIC", "node_type": "TOPIC"},
                    {"sd_id": "DESTINATION_OF", "node_type": "PREDICATE",
                     "continuity_allowed": True},
                    {"sd_id": "R1", "node_type": "RELATION",
                     "predicate_ref": "DESTINATION_OF",
                     "source_ref": "TRAVEL", "target_ref": "KYOTO",
                     "validation_state": "APPROVED",
                     "evidence_id": "SYNTHETIC_EXAMPLE"}]}
    registry = SDRegistry(snapshot)
    history = {"active_topic_ref": ref("TRAVEL"),
               "history": [{"utterance_id": "U1", "topic_ref": ref("TRAVEL"),
                            "verified": True}]}
    def example(current, relation=None):
        return {"schema_version": "intent-sd/0.3",
                "utterance_id": "U2", "speech_act": "ANSWER",
                "intent": "ANSWER_LOCATION",
                "topic_ref": ref(current), "previous_topic_ref": ref("TRAVEL"),
                "context_reference": {"status": "RESOLVED",
                                      "target_utterance_id": "U1",
                                      "topic_ref": ref("TRAVEL")},
                "topic_transition": {"kind": "UNKNOWN",
                                     "relation_ref": ref(relation) if relation else None},
                "provenance": "RULE_CANDIDATE"}
    a = classify_topic_transition(example("KYOTO", "R1"), history, registry)
    b = classify_topic_transition(example("MUSIC"), history, registry)
    print(json.dumps({"example": "SYNTHETIC_NOT_EVALUATION",
                      "context_a": a, "context_b": b,
                      "choice": choose_context_v03(a, b)},
                     ensure_ascii=False, indent=2))

if __name__ == "__main__":
    run()
