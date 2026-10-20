"""The measurements the agents decide from: each defect is detected, and clean audio raises nothing."""

import shutil

import pytest

from app.analysis import analyze

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def test_clean_recording(wav):
    a = analyze(wav(noise=0.0005))
    assert a.snr_db > 40 and a.clicks_per_minute == 0 and a.hum_hz is None and a.clipped_ratio == 0
    assert -20 < a.loudness_lufs < -12
    assert a.long_pauses == 1 and 1.8 <= a.long_pause_seconds <= 2.2
    assert a.issues == ["1 pauses longer than 1 s (2 s in total)"]
    assert abs(a.duration_seconds - 14) < 0.1


def test_noise(wav):
    a = analyze(wav(noise=0.03))
    assert a.snr_db < 25
    assert any("background noise" in issue for issue in a.issues)


@pytest.mark.parametrize("mains", [50, 60])
def test_hum(wav, mains):
    a = analyze(wav(noise=0.0005, hum_hz=mains, hum=0.02))
    assert a.hum_hz == mains and a.hum_db > 15
    assert any(f"{mains} Hz mains hum" in issue for issue in a.issues)


def test_clicks(wav):
    a = analyze(wav(noise=0.0005, clicks_per_second=1))
    assert a.clicks_per_minute > 10
    assert any("clicks" in issue for issue in a.issues)


def test_voice_edges_are_not_clicks(wav):
    # Pitch pulses and consonant onsets are steep too, but they repeat or come in bursts.
    assert analyze(wav(noise=0.0005, gain=1.5)).clicks_per_minute == 0


def test_clipping_and_loudness(wav):
    loud = analyze(wav(noise=0.0005, gain=3))
    assert loud.clipped_ratio > 0.01 and loud.peak_db > -0.1
    assert any("clipping" in issue for issue in loud.issues) and any("too loud" in i for i in loud.issues)
    quiet = analyze(wav(noise=0.0002, gain=0.05))
    assert quiet.loudness_lufs < -35 and any("too quiet" in issue for issue in quiet.issues)


def test_excerpt(wav):
    src = wav([(4, "voice"), (4, "pause"), (4, "voice")], noise=0.001)
    a = analyze(src, start_seconds=3, limit_seconds=5)
    assert abs(a.analyzed_seconds - 5) < 0.06 and abs(a.duration_seconds - 5) < 0.06
    assert a.long_pauses == 1
