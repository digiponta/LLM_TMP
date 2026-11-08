"""LLM_TMP v0.3.3: conservative context resolver -> DialogueIntentSD bridge.

No model training, factual verification, or automatic approval is performed.
The existing DialogueIntentSD contract is left unchanged.
"""
import json
from context_intent_resolver_v033 import TopicContext, resolve
from dialogue_intent_sd import DialogueIntentSD


class ContextDialogueIntentBridge:
    def __init__(self, engine=None):
        self.engine = engine if engine is not None else DialogueIntentSD()
        self.context = TopicContext()
        self.turn = 0

    def process(self, utterance):
        current_turn = self.turn
        result = resolve(utterance, self.context, current_turn)
        self.context = result["next_context"]
        self.turn += 1

        if result["status"] != "RESOLVED":
            response = {"status": result["status"], "turn": current_turn}
            if "reason" in result:
                response["reason"] = result["reason"]
            return response

        proposal = result["intent_sd"]
        # Validates only the recognized dialogue act; this does not attest
        # that the subject/attribute match the user's intended meaning.
        accepted = self.engine.accept(utterance, "I(" + proposal["act"] + ")")
        if accepted["status"] != "ACCEPTED":
            return {
                "status": "REJECTED",
                "turn": current_turn,
                "reason": accepted.get("reason", "INTENT_REJECTED"),
            }
        return {
            "status": "RESOLVED",
            "turn": current_turn,
            "intent_sd": proposal,
            "dialogue_intent_sd_ref": accepted["sd_ref"],
            "verified_fact": False,
            "semantic_alignment_verified": False,
            "human_approved": False,
        }


def run_dialogue(utterances):
    bridge = ContextDialogueIntentBridge()
    return [bridge.process(utterance) for utterance in utterances]


if __name__ == "__main__":
    example = ["スファレライトが好きです", "きれいな宝石ですね", "成分は？"]
    print(json.dumps(run_dialogue(example), ensure_ascii=False, indent=2))
