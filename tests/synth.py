"""Synthetic test recordings: voice-like signals with the defects the cleaner is meant to fix."""

import io
import wave

import numpy as np

RATE = 16000


def voice(seconds: float, seed: int = 0) -> np.ndarray:
    """Voice-like signal: a gliding harmonic tone in ~4 Hz syllables, with short hissy consonants."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    pitch = 140 + 30 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(pitch) / RATE
    vowels = sum(np.sin(k * phase) / k for k in range(1, 12))
    syllables = np.clip(np.sin(2 * np.pi * 4 * t), 0, None) ** 2  # smooth onsets: no steps, no clicks
    hiss = np.diff(rng.standard_normal(t.size + 1))  # first difference: mostly high frequencies
    consonants = hiss * np.clip(np.sin(2 * np.pi * 4 * t + 2.5) - 0.7, 0, None) / 0.3
    signal = 0.25 * vowels * syllables + 0.06 * consonants
    return signal / np.abs(signal).max() * 0.5


def build(parts, noise: float = 0.0, hum_hz: int | None = None, hum: float = 0.0,
          clicks_per_second: float = 0.0, gain: float = 1.0, seed: int = 0) -> np.ndarray:
    """(seconds, kind) parts with kind "voice" or "pause", plus optional defects over the whole thing."""
    rng = np.random.default_rng(seed)
    audio = np.concatenate([voice(s, seed + i) if kind == "voice" else np.zeros(int(s * RATE))
                            for i, (s, kind) in enumerate(parts)]) * gain
    t = np.arange(audio.size) / RATE
    if noise:
        audio = audio + noise * rng.standard_normal(audio.size)
    if hum_hz:
        audio = audio + hum * sum(np.sin(2 * np.pi * hum_hz * h * t) / h for h in (1, 2, 3))
    if clicks_per_second:
        for at in rng.integers(0, audio.size - 4, int(clicks_per_second * audio.size / RATE)):
            audio[at:at + 2] += rng.choice([-0.6, 0.6])
    return audio


def wav_bytes(audio: np.ndarray, rate: int = RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()
