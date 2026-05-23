"""Conservative emotion selection for BoBe responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


EmotionName = Literal["happy", "thinking", "curious", "surprised", "sad", "neutral"]


@dataclass(frozen=True)
class EmotionDecision:
    """Emotion metadata for a spoken response."""

    emotion: EmotionName
    reason: str

    @property
    def should_play_emotion(self) -> bool:
        """Return whether this decision should trigger a robot emotion move."""
        return self.emotion != "neutral"


_KEYWORDS: tuple[tuple[EmotionName, tuple[str, ...]], ...] = (
    ("sad", ("sorry", "unfortunately", "sad", "failed", "error", "can't", "cannot")),
    ("surprised", ("wow", "surprising", "unexpected", "amazing", "astonishing")),
    ("happy", ("great", "glad", "happy", "excellent", "nice", "awesome", "wonderful")),
    ("curious", ("curious", "wonder", "explore", "investigate")),
    ("thinking", ("think", "consider", "maybe", "let me check", "i can check")),
)


def classify_emotion(text: str) -> EmotionDecision:
    """Classify response text into a conservative emotion decision."""
    normalized = text.strip().lower()
    if not normalized:
        return EmotionDecision("neutral", "empty")

    for emotion, keywords in _KEYWORDS:
        for keyword in keywords:
            if keyword in normalized:
                return EmotionDecision(emotion, f"matched:{keyword}")

    return EmotionDecision("neutral", "no_clear_signal")
