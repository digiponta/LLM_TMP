"""Intent SD v0.2: explicit topic/context consistency, no text inference.

Topic IDs are independently supplied canonical tags, NOT automatically
extracted from speech. Missing information yields UNKNOWN rather than guesses.
"""
from dataclasses import dataclass, asdict
from typing import Optional

PROVENANCE = frozenset(("HUMAN_ANNOTATED", "MODEL_INFERRED", "RULE_CANDIDATE"))

def _tag(value):
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("INVALID_TOPIC_TAG")
    return value.strip()

@dataclass(frozen=True)
class ContextSD:
    topic_previous: Optional[str]
    topic_current: Optional[str]
    context_reference: Optional[str] = None
    provenance: str = "RULE_CANDIDATE"

    def __post_init__(self):
        for key in ("topic_previous", "topic_current", "context_reference"):
            _tag(getattr(self, key))
        if self.provenance not in PROVENANCE:
            raise ValueError("INVALID_PROVENANCE")

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) - set(cls.__dataclass_fields__):
            raise ValueError("INVALID_CONTEXT_SD")
        return cls(**data)

    def to_dict(self):
        return asdict(self)

def topic_continuity(sd):
    before, after = _tag(sd.topic_previous), _tag(sd.topic_current)
    if before is None or after is None:
        return "UNKNOWN"
    return "CONTINUOUS" if before == after else "SHIFTED"

def context_consistency(sd):
    before, ref = _tag(sd.topic_previous), _tag(sd.context_reference)
    if before is None or ref is None:
        return "UNKNOWN"
    return "CONTINUOUS" if before == ref else "SHIFTED"

def aggregate_consistency(sd):
    signals = (topic_continuity(sd), context_consistency(sd))
    if "SHIFTED" in signals:
        return "SHIFTED"
    if "UNKNOWN" in signals:
        return "UNKNOWN"
    return "CONTINUOUS"

def choose_context(a, b):
    left, right = aggregate_consistency(a), aggregate_consistency(b)
    if left == "CONTINUOUS" and right == "SHIFTED":
        label = "A"
    elif right == "CONTINUOUS" and left == "SHIFTED":
        label = "B"
    elif left == right == "CONTINUOUS":
        label = "BOTH"
    elif left == right == "SHIFTED":
        label = "NEITHER"
    else:
        label = "UNCLEAR"
    return {"label": label, "a": left, "b": right,
            "a_topic": topic_continuity(a), "b_topic": topic_continuity(b),
            "a_reference": context_consistency(a),
            "b_reference": context_consistency(b)}
