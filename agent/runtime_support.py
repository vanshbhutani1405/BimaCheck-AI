"""Provider-independent runtime helpers used by the future LiveKit adapter."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Callable

from agent.workflow import normalize_tts_text


def audio_cache_key(text: str, *, voice: str, model: str, language: str) -> str:
    """Stable cache key includes normalized text and every synthesis option."""
    material = "\0".join((normalize_tts_text(text), voice, model, language))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class AudioCache:
    """Small in-memory cache; only caller-approved audio is inserted."""
    def __init__(self):
        self._items: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self._items.get(key)

    def put_approved(self, key: str, audio: bytes, *, listener_verdict: str) -> None:
        if listener_verdict != "ok":
            raise ValueError("only listener-approved audio may enter the cache")
        self._items[key] = bytes(audio)


@dataclass(frozen=True)
class TurnTiming:
    transcript_final_ms: float | None
    response_first_audio_ms: float
    cached_playback: bool


class TurnTimer:
    """Monotonic-clock helper; caller supplies speech-end and audio-start marks."""
    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self._clock = clock
        self._speech_end = clock()
        self._transcript = None

    def transcript_final(self) -> None:
        self._transcript = self._clock()

    def first_audio(self, *, cached: bool) -> TurnTiming:
        now = self._clock()
        stt_ms = None if self._transcript is None else (self._transcript - self._speech_end) * 1000
        return TurnTiming(stt_ms, (now - self._speech_end) * 1000, cached)
