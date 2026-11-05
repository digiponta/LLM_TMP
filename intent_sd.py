"""Intent SD v0.1: validated conversational-intent representation.

This schema is a representation contract, NOT an intent extractor and
NOT a learned classifier. Unknowns remain unknown; no fixed confidence
is manufactured. Provenance and topic context stay explicit.
"""
from dataclasses import dataclass, asdict
from typing import Optional

SPEECH_ACTS = frozenset(("QUESTION", "REQUEST", "INFORM", "ACKNOWLEDGE", "OTHER", "UNKNOWN"))
POLARITIES = frozenset(("POSITIVE", "NEGATIVE", "MIXED", "UNKNOWN"))

@dataclass(frozen=True)
class IntentSD:
    speech_act: str
    intent: str
    topic: Optional[str] = None
    target: Optional[str] = None
    polarity: str = "UNKNOWN"
    context_required: bool = False
    confidence: Optional[float] = None
    provenance: str = "UNVERIFIED_CANDIDATE"

    def __post_init__(self):
        if self.speech_act not in SPEECH_ACTS:
            raise ValueError("INVALID_SPEECH_ACT")
        if not isinstance(self.intent, str) or not self.intent.strip():
            raise ValueError("INVALID_INTENT")
        if self.polarity not in POLARITIES:
            raise ValueError("INVALID_POLARITY")
        if not isinstance(self.context_required, bool):
            raise ValueError("INVALID_CONTEXT_REQUIRED")
        if self.confidence is not None and (
            isinstance(self.confidence, bool) or
            not isinstance(self.confidence, (float, int)) or
            not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("INVALID_CONFIDENCE")
        if self.provenance not in ("UNVERIFIED_CANDIDATE", "HUMAN_ANNOTATED", "MODEL_INFERRED"):
            raise ValueError("INVALID_PROVENANCE")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, item):
        if not isinstance(item, dict):
            raise ValueError("INVALID_INTENT_SD")
        allowed = set(cls.__dataclass_fields__)
        if set(item) - allowed:
            raise ValueError("UNKNOWN_INTENT_SD_FIELDS")
        return cls(**item)
