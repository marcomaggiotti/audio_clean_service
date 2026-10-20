"""Measure a recording's problems, so the agent can decide how to clean it.

One streaming pass over the audio decoded at 16 kHz (memory stays at one chunk plus a few floats per
50 ms frame), plus an EBU R128 loudness pass in ffmpeg. Everything is measured, nothing is guessed
from the file type: a phone voice note and a studio podcast get different answers for the same
reason a human engineer would give them different treatment.
"""

from pathlib import Path

import numpy as np

from app.audio import SAMPLE_RATE, AudioProcessingError, measure_loudness, pcm_chunks, probe_duration, threshold_from_levels
from app.schemas import AudioAnalysis

FRAME = SAMPLE_RATE // 20  # 50 ms
WELCH_FFT = 16384  # ~1 Hz bins: mains hum harmonics are narrow and stationary
SPECTRUM_SECONDS = 180.0  # long-term spectrum from the first 3 minutes is plenty
CLIP_LEVEL = 32440  # ~0.99 of full scale
CLICK_RATIO = 4.0  # a click's jump is this many times the biggest other jump within ±20 ms
CLICK_SPAN = 320  # ±20 ms: always holds another pitch period of a voice (F0 >= 50 Hz)
CLICK_MIN_JUMP = 0.02  # and is at least this big (fraction of full scale), so hiss can't qualify
HUM_MIN_DB = 10.0  # a harmonic this far above its neighbourhood counts as hum
LONG_PAUSE_FRAMES = 20  # 1 s

_BANDS = np.fft.rfftfreq(FRAME, 1 / SAMPLE_RATE)
_LOW_BAND = (_BANDS >= 300) & (_BANDS < 2000)
_HIGH_BAND = (_BANDS >= 2000) & (_BANDS < 8000)
_FRAME_WINDOW = np.hanning(FRAME).astype(np.float32)


def analyze(path: Path, limit_seconds: float | None = None, start_seconds: float | None = None) -> AudioAnalysis:
    """Measure `path` (optionally an excerpt). Raises AudioProcessingError if there is nothing to measure."""
    filters = f"atrim=start={start_seconds:.3f},asetpts=PTS-STARTPTS" if start_seconds else None
    levels, low, high, clicks = [], [], [], 0
    spectrum, spectra, welch_window = np.zeros(WELCH_FFT // 2 + 1), 0, np.hanning(WELCH_FFT)
    pending = np.empty(0, dtype=np.float32)
    total, clipped, peak = 0, 0, 0
    for chunk in pcm_chunks(path, SAMPLE_RATE, filters, limit_seconds, chunk_samples=FRAME * 512):
        total += chunk.shape[0]
        magnitude = np.abs(chunk.astype(np.int32))
        clipped += int((magnitude >= CLIP_LEVEL).sum())
        peak = max(peak, int(magnitude.max(initial=0)))
        audio = chunk.astype(np.float32) / 32768.0

        usable = audio.shape[0] // FRAME * FRAME
        if usable:
            frames = audio[:usable].reshape(-1, FRAME)
            levels.append(20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9))
            power = np.abs(np.fft.rfft(frames * _FRAME_WINDOW, axis=1)) ** 2
            low.append(power[:, _LOW_BAND].sum(axis=1))
            high.append(power[:, _HIGH_BAND].sum(axis=1))
            clicks += _count_clicks(frames)

        if spectra * WELCH_FFT < SPECTRUM_SECONDS * SAMPLE_RATE:
            pending = np.concatenate([pending, audio])
            while pending.shape[0] >= WELCH_FFT:
                spectrum += np.abs(np.fft.rfft(pending[:WELCH_FFT] * welch_window)) ** 2
                spectra += 1
                pending = pending[WELCH_FFT:]
    if not levels:
        raise AudioProcessingError("Audio too short to analyze")

    levels, low, high = np.concatenate(levels), np.concatenate(low), np.concatenate(high)
    analyzed = total / SAMPLE_RATE
    floor, speech = float(np.percentile(levels, 10)), float(np.percentile(levels, 90))
    threshold = threshold_from_levels(levels)
    quiet = levels < threshold if threshold is not None else np.zeros(levels.shape, dtype=bool)
    # Consonant energy: the speech frames' 2-8 kHz vs 300 Hz-2 kHz power, each after subtracting the
    # background's own (quietest frames), so removing hiss doesn't read as losing consonants.
    talking = levels >= (threshold if threshold is not None else np.median(levels))
    background = levels <= np.percentile(levels, 15)
    speech_high = max(high[talking].mean() - high[background].mean(), 1e-9)
    speech_low = max(low[talking].mean() - low[background].mean(), 1e-9)
    brightness = 10 * np.log10(speech_high / speech_low)
    pauses = _runs(quiet)
    long_pauses = [n for n in pauses if n >= LONG_PAUSE_FRAMES]

    hum_hz, hum_db, rumble = None, 0.0, -60.0
    if spectra:
        hum_hz, hum_db = _hum(spectrum)
        freqs = np.fft.rfftfreq(WELCH_FFT, 1 / SAMPLE_RATE)
        sub = spectrum[(freqs >= 20) & (freqs < 80)].sum()
        voice = spectrum[(freqs >= 80) & (freqs < 4000)].sum()
        rumble = float(10 * np.log10((sub + 1e-12) / (voice + 1e-12)))

    duration = probe_duration(path) if not (start_seconds or limit_seconds) else None
    result = AudioAnalysis(
        duration_seconds=round(duration or analyzed, 2),
        analyzed_seconds=round(analyzed, 2),
        noise_floor_db=round(floor, 1),
        speech_level_db=round(speech, 1),
        snr_db=round(speech - floor, 1),
        loudness_lufs=_round(measure_loudness(path, filters, limit_seconds)),
        peak_db=round(20 * np.log10(peak / 32768 + 1e-9), 1),
        clipped_ratio=round(clipped / max(total, 1), 5),
        clicks_per_minute=round(clicks / max(analyzed / 60, 1 / 60), 1),
        hum_hz=hum_hz,
        hum_db=round(hum_db, 1),
        rumble_ratio_db=round(rumble, 1),
        speech_brightness_db=round(float(brightness), 1),
        silence_ratio=round(float(quiet.mean()), 3),
        long_pauses=len(long_pauses),
        long_pause_seconds=round(sum(long_pauses) * FRAME / SAMPLE_RATE, 1),
    )
    result.issues = describe_issues(result)
    return result


def describe_issues(a: AudioAnalysis) -> list[str]:
    issues = []
    if a.snr_db < 15:
        issues.append(f"loud background noise (speech only {a.snr_db:.0f} dB above it)")
    elif a.snr_db < 30:
        issues.append(f"audible background noise (speech {a.snr_db:.0f} dB above it)")
    if a.clipped_ratio >= 0.0005:
        issues.append(f"clipping ({a.clipped_ratio:.2%} of samples at full scale)")
    if a.clicks_per_minute >= 3:
        issues.append(f"clicks or pops ({a.clicks_per_minute:.0f} per minute)")
    if a.hum_hz:
        issues.append(f"{a.hum_hz} Hz mains hum ({a.hum_db:.0f} dB over its surroundings)")
    if a.rumble_ratio_db > -15:
        issues.append(f"low-frequency rumble ({a.rumble_ratio_db:.0f} dB below 80 Hz)")
    if a.loudness_lufs is None:
        issues.append("no audible signal")
    elif a.loudness_lufs < -24:
        issues.append(f"too quiet ({a.loudness_lufs:.0f} LUFS)")
    elif a.loudness_lufs > -12:
        issues.append(f"too loud ({a.loudness_lufs:.0f} LUFS)")
    if a.long_pauses:
        issues.append(f"{a.long_pauses} pauses longer than 1 s ({a.long_pause_seconds:.0f} s in total)")
    return issues


_OFFSETS = np.arange(FRAME - 1)


def _count_clicks(frames: np.ndarray) -> int:
    """Frames whose steepest sample-to-sample jump is isolated: nothing within ±20 ms comes close.

    Voices are full of steep edges too (consonant onsets, and a glottal pulse every pitch period),
    but those repeat or come in bursts; a click is one or two samples standing alone. So the steepest
    jump is compared with the biggest other jump around it, not with the frame's average.
    """
    slope = np.abs(np.diff(frames, axis=1))
    steepest = slope.argmax(axis=1)
    distance = np.abs(_OFFSETS[None, :] - steepest[:, None])
    others = np.where((distance > 2) & (distance <= CLICK_SPAN), slope, 0).max(axis=1) + 1e-9
    biggest = slope[np.arange(slope.shape[0]), steepest]
    return int(((biggest / others > CLICK_RATIO) & (biggest > CLICK_MIN_JUMP)).sum())


def _hum(spectrum: np.ndarray) -> tuple[int | None, float]:
    """Mains frequency (50 or 60 Hz) whose harmonics stand out most, with that prominence in dB."""
    freqs = np.fft.rfftfreq(WELCH_FFT, 1 / SAMPLE_RATE)
    best_hz, best_db = None, 0.0
    for base in (50, 60):
        prominences = []
        for harmonic in range(1, 5):
            f = base * harmonic
            peak = spectrum[np.abs(freqs - f) <= 2].max()
            around = spectrum[(np.abs(freqs - f) >= 5) & (np.abs(freqs - f) <= 15)]
            prominences.append(10 * np.log10((peak + 1e-12) / (np.median(around) + 1e-12)))
        # Two strongest harmonics: one lucky peak (a sung note, a tone) isn't a hum.
        prominence = float(np.mean(sorted(prominences)[-2:]))
        if prominence > best_db:
            best_hz, best_db = base, prominence
    return (best_hz, best_db) if best_db >= HUM_MIN_DB else (None, best_db)


def _runs(mask: np.ndarray) -> list[int]:
    """Lengths of the runs of True in `mask`."""
    if not mask.any():
        return []
    padded = np.concatenate([[0], mask.astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(padded))
    return (edges[1::2] - edges[::2]).tolist()


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 1)
