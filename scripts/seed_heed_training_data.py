#!/usr/bin/env python3
"""Seed minimal positive/negative clips so heed train can run with TTS augmentation."""

from __future__ import annotations

from pathlib import Path

from heed.audio import save_wav
from heed.tts import synthesize_phrase

PROJECT = Path(__file__).resolve().parents[1] / "wake_training" / "hey_jarvis"
POSITIVE_PHRASE = "hey jarvis"
NEGATIVE_PHRASES = (
    "hey there",
    "what time is it",
    "good morning",
    "turn it off",
)


def main() -> None:
    pos_dir = PROJECT / "positive"
    neg_dir = PROJECT / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    for idx, clip in enumerate(synthesize_phrase(POSITIVE_PHRASE, 8, seed=42)):
        save_wav(pos_dir / f"seed_pos_{idx:03d}.wav", clip)

    neg_idx = 0
    for phrase_idx, phrase in enumerate(NEGATIVE_PHRASES):
        for clip in synthesize_phrase(phrase, 2, seed=100 + phrase_idx):
            save_wav(neg_dir / f"seed_neg_{neg_idx:03d}.wav", clip)
            neg_idx += 1

    print(f"wrote {len(list(pos_dir.glob('*.wav')))} positives and {len(list(neg_dir.glob('*.wav')))} negatives")


if __name__ == "__main__":
    main()
