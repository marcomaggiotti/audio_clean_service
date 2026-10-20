"""Decoding and the cleaning filters (ported from voice_embedding_service, plus the new repair steps)."""

import shutil
import subprocess

import numpy as np
import pytest

from app import audio
from app.analysis import analyze
from app.audio import InvalidAudioError, decode_audio, detect_format, process_audio, silence_threshold_db, spectrogram_png
from tests.synth import RATE, wav_bytes

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def tone(seconds=1.0, rate=16000, channels=1, freq=440.0) -> bytes:
    import io
    import wave

    t = np.arange(int(seconds * rate)) / rate
    samples = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    if channels > 1:
        samples = np.repeat(samples[:, None], channels, axis=1).ravel()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


def rms_db(x: np.ndarray) -> float:
    x = x.astype(np.float32) / (32768.0 if x.dtype == np.int16 else 1.0)
    return float(20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-12))


# ------------------------------------------------------------------ decoding

def test_detect_format():
    assert detect_format(tone()) == "wav"
    assert detect_format(b"ID3\x04\x00") == "mp3"
    assert detect_format(b"\xff\xfb\x90\x00") == "mp3"
    assert detect_format(b"OggS\x00\x02" + b"\x00" * 10) == "ogg"
    assert detect_format(b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 8) == "mp4"
    assert detect_format(b"fLaC\x00\x00\x00\x22") == "flac"
    with pytest.raises(InvalidAudioError):
        detect_format(b"%PDF-1.7")


@pytest.mark.parametrize("use_ffmpeg", [True, False])
def test_decode_wav_resamples_to_16k_mono(monkeypatch, use_ffmpeg):
    if use_ffmpeg and not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    if not use_ffmpeg:
        monkeypatch.setattr(audio.shutil, "which", lambda _: None)
    samples = decode_audio(tone(seconds=2.0, rate=44100, channels=2))
    assert samples.dtype == np.float32
    assert abs(samples.shape[0] - 32000) <= 2
    assert 0.25 < np.abs(samples).max() <= 0.31


def test_mp3_without_ffmpeg_is_rejected(monkeypatch):
    monkeypatch.setattr(audio.shutil, "which", lambda _: None)
    with pytest.raises(InvalidAudioError, match="ffmpeg"):
        decode_audio(b"ID3\x04\x00" + b"\x00" * 100)


@needs_ffmpeg
@pytest.mark.parametrize("codec,ext", [("libmp3lame", "mp3"), ("libopus", "ogg"), ("aac", "m4a"), ("flac", "flac")])
def test_decode_compressed_formats(tmp_path, codec, ext):
    path = tmp_path / f"tone.{ext}"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1.5",
                    "-c:a", codec, str(path)], check=True)
    assert abs(decode_audio(path.read_bytes()).shape[0] / 16000 - 1.5) < 0.1


# ------------------------------------------------------------------ cleaning filters

@needs_ffmpeg
def test_silence_removal_cuts_long_gaps_only(wav, tmp_path):
    src = wav([(2, "voice"), (4, "pause"), (2, "voice"), (0.5, "pause"), (2, "voice")], noise=0.001)
    out = tmp_path / "out.flac"
    notes = process_audio(src, out, remove_silence=True, min_silence_seconds=1.0)
    seconds = decode_audio(out).shape[0] / RATE
    assert 7.2 < seconds < 7.8, seconds  # the 4 s gap shrinks to 1 s; the 0.5 s gap stays
    assert "pauses longer than 1s shortened to 1s" in notes[0]


@needs_ffmpeg
def test_silence_removal_skips_when_speech_and_noise_are_alike(wav, tmp_path):
    src = wav([(3, "pause")], noise=0.05)  # just noise: nothing to tell apart
    assert silence_threshold_db(src) is None
    notes = process_audio(src, tmp_path / "o.flac", remove_silence=True)
    assert "skipped" in notes[0] and abs(decode_audio(tmp_path / "o.flac").shape[0] / RATE - 3) < 0.05


@needs_ffmpeg
def test_loudness_normalization_lifts_quiet_audio(wav, tmp_path):
    src = wav(gain=0.02, noise=0.0001)
    out = tmp_path / "loud.flac"
    notes = process_audio(src, out, loudnorm=True)
    assert rms_db(decode_audio(out, as_int16=True)) - rms_db(decode_audio(src, as_int16=True)) > 25
    assert "→ -16 LUFS" in notes[0]
    assert np.abs(decode_audio(out)).max() <= 0.86  # limiter ceiling (-1.5 dBFS)
    assert abs(analyze(out).loudness_lufs + 16) < 1.5


@needs_ffmpeg
def test_denoise_reduces_steady_noise_and_strength_matters(wav, tmp_path):
    src = wav(noise=0.02)
    gap = slice(int(6.5 * RATE), int(7.5 * RATE))  # inside the pause: noise only
    before = rms_db(decode_audio(src)[gap])
    after = {}
    for db in (6, 20):
        out = tmp_path / f"clean{db}.flac"
        process_audio(src, out, denoise=True, denoise_db=db)
        after[db] = rms_db(decode_audio(out)[gap])
    assert after[6] < before - 3
    assert after[20] < after[6] - 5


@needs_ffmpeg
def test_dehum_removes_mains_hum(wav, tmp_path):
    src = wav(noise=0.0005, hum_hz=60, hum=0.03)
    assert analyze(src).hum_hz == 60
    out = tmp_path / "dehum.flac"
    notes = process_audio(src, out, dehum_hz=60)
    assert "60 Hz mains hum" in notes[0]
    assert analyze(out).hum_hz is None


@needs_ffmpeg
def test_declick_removes_clicks(wav, tmp_path):
    src = wav(noise=0.0005, clicks_per_second=1)
    out = tmp_path / "declick.flac"
    process_audio(src, out, declick=True)
    assert analyze(out).clicks_per_minute < analyze(src).clicks_per_minute / 2


@needs_ffmpeg
def test_declip_leaves_headroom_for_rebuilt_peaks(wav, tmp_path):
    src = wav(noise=0.0005, gain=3)
    assert analyze(src).clipped_ratio > 0.01
    out = tmp_path / "declip.flac"
    process_audio(src, out, declip=True)
    assert analyze(out).clipped_ratio < 0.0005


@needs_ffmpeg
def test_highpass_removes_rumble(tmp_path):
    from tests.synth import build

    t = np.arange(14 * RATE) / RATE
    rumbly = build([(14, "voice")], noise=0.0005) + 0.2 * np.sin(2 * np.pi * 35 * t)
    src = tmp_path / "rumble.wav"
    src.write_bytes(wav_bytes(rumbly))
    out = tmp_path / "hp.flac"
    process_audio(src, out, highpass_hz=80)
    assert analyze(out).rumble_ratio_db < analyze(src).rumble_ratio_db - 10


@needs_ffmpeg
def test_excerpt_processing(wav, tmp_path):
    src = wav([(4, "voice"), (4, "pause"), (4, "voice")], noise=0.001)
    pause, speech = tmp_path / "pause.flac", tmp_path / "speech.flac"
    process_audio(src, pause, denoise=True, start_seconds=4.5, limit_seconds=3)
    process_audio(src, speech, denoise=True, start_seconds=8.5, limit_seconds=3)
    assert abs(decode_audio(pause).shape[0] / RATE - 3) < 0.05
    assert rms_db(decode_audio(pause)) < -55 < -30 < rms_db(decode_audio(speech))


@needs_ffmpeg
def test_spectrogram_puts_frequencies_in_the_right_rows(tmp_path):
    from PIL import Image

    src = tmp_path / "tone.wav"
    src.write_bytes(tone(seconds=4, freq=1000))  # 1 kHz = 1/4 of 0-4 kHz
    meta = spectrogram_png(src, tmp_path / "s.png", width=40, height=100)
    assert meta == {"duration": 4.0, "max_hz": 4000}
    image = np.asarray(Image.open(tmp_path / "s.png").convert("L"), dtype=np.float32)
    assert image.shape[0] == 100
    assert abs(int(image.mean(axis=1).argmax()) - 75) <= 2  # 1 kHz: 3/4 of the way down
