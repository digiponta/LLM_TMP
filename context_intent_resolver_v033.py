"""Conservative, deterministic context resolver for LLM_TMP v0.3.3.

Development baseline only. It does not attest factual truth or run an LLM.
Only explicit known topic mentions are remembered; no guessing on ambiguity.
"""
from dataclasses import dataclass
import re

TOPICS = ("スファレライト", "ダイヤモンド", "量子力学")
ATTRIBUTE_PATTERNS = {
    "化学組成": (r"成分", r"化学組成", r"化学式", r"何でできて"),
    "硬度": (r"硬度", r"硬さ"),
}

@dataclass(frozen=True)
class TopicContext:
    topic: str | None = None
    source_turn: int | None = None


def resolve(text: str, context: TopicContext, turn: int):
    if not isinstance(text, str) or not text.strip():
        return {"status": "ABSTAIN", "reason": "EMPTY_INPUT", "next_context": context}
    mentioned = [topic for topic in TOPICS if topic in text]
    if len(mentioned) > 1:
        return {"status": "ABSTAIN", "reason": "MULTIPLE_TOPICS", "next_context": context}
    explicit = mentioned[0] if mentioned else None
    next_context = TopicContext(explicit, turn) if explicit else context
    attributes = [name for name, patterns in ATTRIBUTE_PATTERNS.items()
                  if any(re.search(pattern, text) for pattern in patterns)]
    if len(attributes) > 1:
        return {"status": "ABSTAIN", "reason": "MULTIPLE_ATTRIBUTES", "next_context": next_context}
    question = bool(re.search(r"[？?]|(?:教えて|知りたい|とは|ですか|なの)", text))
    if not question or not attributes:
        return {"status": "NO_TARGET_QUESTION", "next_context": next_context}
    topic = explicit or context.topic
    if not topic:
        return {"status": "ABSTAIN", "reason": "NO_TOPIC_CONTEXT", "next_context": next_context}
    return {
        "status": "RESOLVED",
        "intent_sd": {
            "act": "QUESTION", "subject": topic, "attribute": attributes[0],
            "reference": "EXPLICIT" if explicit else "CONTEXT",
            "source_turn": turn if explicit else context.source_turn,
            "semantic_alignment_verified": False,
        },
        "next_context": next_context,
    }


def resolve_dialogue(utterances):
    context = TopicContext()
    results = []
    for turn, text in enumerate(utterances):
        result = resolve(text, context, turn)
        context = result["next_context"]
        results.append({k: v for k, v in result.items() if k != "next_context"})
    return results


if __name__ == "__main__":
    import json
    example = ["スファレライトが好きです", "きれいな宝石ですね", "成分は？"]
    print(json.dumps(resolve_dialogue(example), ensure_ascii=False, indent=2))
