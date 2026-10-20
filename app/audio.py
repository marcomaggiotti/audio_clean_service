"""Audio validation, decoding and the cleaning filters.

Ported from voice_embedding_service (app/audio.py): format detection, decoding to 16 kHz mono, the
processed-copy filters (loudness, noise reduction, silence removal) and the spectrogram renderer.
The filter chain is extended with de-clipping, de-clicking, hum removal and a separate high-pass, and
every step's strength is a parameter, so the cleaning agent can choose and tune them per recording.
"""

import io
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000


class InvalidAudioError(ValueError):
    """Raised when an upload is not decodable audio in a supported format."""


# Container formats accepted, with the media type used when serving the original file back.
MEDIA_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",    # Telegram voice notes (Opus)
    "mp4": "audio/mp4",    # M4A / AAC, Telegram video notes
    "flac": "audio/flac",
}


def detect_format(data: bytes) -> str:
    """Return the container format from the file header, or raise InvalidAudioError."""
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:4] == b"OggS":
        return "ogg"
    if data[:4] == b"fLaC":
        return "flac"
    if len(data) >= 8 and data[4:8] == b"ftyp":
        return "mp4"
    if data[:3] == b"ID3" or (len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return "mp3"
    raise InvalidAudioError("Unsupported audio format: expected WAV, MP3, OGG/Opus, M4A or FLAC")


def decode_audio(source: bytes | Path, max_seconds: float | None = None, as_int16: bool = False) -> np.ndarray:
    """Decode an audio file (any format in MEDIA_TYPES) into mono float32 at 16 kHz in [-1, 1].

    ffmpeg reads from a file, not a pipe: MP4/M4A written by phones (and Telegram video notes) keep
    their index after the media data, which ffmpeg can only reach on a seekable input.
    With max_seconds, decoding stops shortly after that point, so memory stays bounded no matter
    how long a small, low-bitrate file is; callers detect "too long" from the returned length.
    as_int16 returns the raw 16-bit samples (half the memory).
    """
    if isinstance(source, Path):
        with source.open("rb") as f:
            fmt = detect_format(f.read(16))
    else:
        fmt = detect_format(source)
    limit = max_seconds + 1 if max_seconds is not None else None
    if shutil.which("ffmpeg"):
        if isinstance(source, Path):
            audio = _decode_ffmpeg(source, limit, as_int16)
        else:
            with tempfile.NamedTemporaryFile(suffix=f".{fmt}") as tmp:
                tmp.write(source)
                tmp.flush()
                audio = _decode_ffmpeg(Path(tmp.name), limit, as_int16)
    elif fmt == "wav":
        audio = _decode_wav(source.read_bytes() if isinstance(source, Path) else source)
        if limit is not None:
            audio = audio[: int(limit * SAMPLE_RATE)]
        if as_int16:
            audio = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    else:
        raise InvalidAudioError(f"{fmt.upper()} decoding requires ffmpeg to be installed")
    if audio.size == 0:
        raise InvalidAudioError("Audio file contains no samples")
    return audio


def duration_seconds(audio: np.ndarray) -> float:
    return audio.shape[0] / SAMPLE_RATE


def _decode_ffmpeg(path: Path, limit_seconds: float | None = None, as_int16: bool = False) -> np.ndarray:
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path)]
    if limit_seconds is not None:
        cmd += ["-t", f"{limit_seconds:.3f}"]
    cmd += ["-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip().splitlines()
        raise InvalidAudioError(f"Could not decode audio: {detail[-1] if detail else 'ffmpeg failed'}")
    if as_int16:
        return np.frombuffer(proc.stdout, dtype=np.int16)
    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32)
    audio *= 1 / 32768.0  # in place: avoids a second full-length float copy
    return audio


def _decode_wav(data: bytes) -> np.ndarray:
    """Pure-Python fallback for PCM WAV when ffmpeg is unavailable."""
    try:
        with wave.open(io.BytesIO(data)) as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as e:
        raise InvalidAudioError(f"Could not decode WAV: {e}") from e

    if width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    elif width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise InvalidAudioError(f"Unsupported WAV sample width: {width * 8} bit")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE and audio.size:
        target_len = int(round(audio.size * SAMPLE_RATE / rate))
        audio = np.interp(
            np.linspace(0, audio.size - 1, target_len), np.arange(audio.size), audio
        ).astype(np.float32)
    return audio


def probe_duration(src: Path) -> float | None:
    """Container duration from ffprobe, without decoding; None if unknown."""
    if not shutil.which("ffprobe"):
        return None
    proc = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)],
                          capture_output=True, text=True, check=False)
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


# ---------------------------------------------------------------- cleaning filters
# Each step runs in ffmpeg as a stream, so memory stays at ffmpeg's ~55 MB whatever the length. EBU
# R128 `loudnorm` is not used: it resamples to 192 kHz internally (~105 MB and ~30 s for 30 minutes);
# loudness is instead measured in a first pass and applied as a gain with a limiter.

TARGET_LUFS = -16.0  # podcast loudness target
LIMIT = 0.84  # limiter ceiling, about -1.5 dBFS
DENOISE_DB = 10  # mild: steady noise drops ~10 dB; stronger settings start eating consonants
MIN_SPEECH_GAP_DB = 6.0  # below this speech/noise contrast, silences can't be told from speech
DECLIP_HEADROOM_DB = 6
HUM_HARMONICS = 4  # notches at the mains frequency and its first harmonics


class AudioProcessingError(RuntimeError):
    pass


def _ffmpeg(args: list[str]) -> str:
    proc = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-y", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        lines = proc.stderr.strip().splitlines()
        raise AudioProcessingError(f"ffmpeg failed: {lines[-1] if lines else proc.returncode}")
    return proc.stderr


def pcm_chunks(src: Path, rate: int, filters: str | None = None, limit_seconds: float | None = None,
               chunk_samples: int = 1 << 16):
    """Yield the decoded audio as int16 arrays of `chunk_samples` (the last may be shorter).

    Read from ffmpeg piece by piece, so memory stays at one chunk whatever the recording's length.
    """
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(src)]
    if limit_seconds:
        cmd += ["-t", f"{limit_seconds:.3f}"]
    if filters:
        cmd += ["-af", filters]
    cmd += ["-f", "s16le", "-ac", "1", "-ar", str(rate), "pipe:1"]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        pending = b""
        while data := proc.stdout.read(chunk_samples * 2 - len(pending)):
            pending += data
            if len(pending) == chunk_samples * 2:
                yield np.frombuffer(pending, dtype=np.int16)
                pending = b""
        if len(pending) >= 2:
            yield np.frombuffer(pending[: len(pending) // 2 * 2], dtype=np.int16)
        error = proc.stderr.read().decode(errors="replace").strip()
    if proc.returncode != 0:
        raise AudioProcessingError(f"Could not decode audio: {error.splitlines()[-1] if error else 'ffmpeg failed'}")


def _levels_db(path: Path, filters: str | None = None, limit_seconds: float | None = None) -> np.ndarray:
    """RMS level (dBFS) of every 50 ms of the audio, after `filters`."""
    frame = SAMPLE_RATE // 20
    levels = []
    for chunk in pcm_chunks(path, SAMPLE_RATE, filters, limit_seconds, chunk_samples=frame * 1024):
        usable = chunk.shape[0] // frame * frame
        if usable:
            block = chunk[:usable].astype(np.float32).reshape(-1, frame) / 32768.0
            levels.append(20 * np.log10(np.sqrt((block ** 2).mean(axis=1)) + 1e-9))
    return np.concatenate(levels) if levels else np.empty(0, dtype=np.float32)


def noise_floor_db(path: Path, limit_seconds: float | None = None) -> float | None:
    """The recording's background level: the 10th percentile of its 50 ms levels."""
    levels = _levels_db(path, None, limit_seconds)
    return float(np.percentile(levels, 10)) if levels.size else None


def threshold_from_levels(levels: np.ndarray) -> float | None:
    """Silence threshold (dBFS) from 50 ms levels; see silence_threshold_db."""
    if not levels.size:
        return None
    floor, speech = np.percentile(levels, 10), np.percentile(levels, 90)
    if speech - floor < MIN_SPEECH_GAP_DB:
        return None
    return float(floor + max(3.0, (speech - floor) / 3))


def silence_threshold_db(path: Path, filters: str | None = None, limit_seconds: float | None = None) -> float | None:
    """Level (dBFS) below which this recording is silence, from its own noise floor and speech level.

    A fixed threshold fails both ways: on a quiet recording it eats soft speech, on a noisy one it
    never fires. The 10th and 90th percentiles of 50 ms levels estimate the noise floor and the
    speech level (measured after `filters`, e.g. noise reduction); the threshold sits a third of the
    way up, at least 3 dB over the floor. None if speech isn't clearly louder than the background.
    """
    return threshold_from_levels(_levels_db(path, filters, limit_seconds))


def measure_loudness(src: Path, filters: str | None = None, limit_seconds: float | None = None) -> float | None:
    """Integrated loudness (EBU R128, LUFS) after `filters`; None when there is no audible signal."""
    head = ["-i", str(src)] + (["-t", f"{limit_seconds:.3f}"] if limit_seconds else [])
    chain = ",".join(([filters] if filters else []) + ["ebur128=framelog=quiet"])
    measured = _ffmpeg(["-loglevel", "info", *head, "-af", chain, "-f", "null", "-"])
    found = re.findall(r"I:\s+(-?[\d.]+|-inf) LUFS", measured)
    loudness = float(found[-1]) if found and found[-1] != "-inf" else None
    return None if loudness is None or loudness < -70 else loudness


def process_audio(src: Path, dst: Path, *, loudnorm: bool = False, denoise: bool = False,
                  remove_silence: bool = False, min_silence_seconds: float = 1.0,
                  declip: bool = False, declick: bool = False, dehum_hz: int | None = None,
                  highpass_hz: float | None = None, denoise_db: float = DENOISE_DB,
                  target_lufs: float = TARGET_LUFS, limit_seconds: float | None = None,
                  start_seconds: float | None = None) -> list[str]:
    """Write a processed copy of `src` to `dst` (16 kHz mono 16-bit FLAC). Returns notes on what was done.

    Order: repairs first (clipping, clicks: they need the untouched waveform), then hum and rumble,
    then broadband noise reduction, then silence removal (so cleaned-up gaps count as silence), then
    loudness (so the gain is measured on what will actually be heard). `start_seconds` and
    `limit_seconds` select an excerpt, which is how the agent trials a strategy on a preview.
    """
    notes, chain = [], []
    head = (["-ss", f"{start_seconds:.3f}"] if start_seconds else []) + ["-i", str(src)]
    head += ["-t", f"{limit_seconds:.3f}"] if limit_seconds else []
    if declip:
        # Rebuilt peaks rise above full scale; without headroom the 16-bit output would clip them again.
        chain += ["adeclip", f"volume={-DECLIP_HEADROOM_DB}dB"]
        notes.append(f"clipped peaks reconstructed (level lowered {DECLIP_HEADROOM_DB} dB to fit them)")
    if declick:
        chain.append("adeclick")
        notes.append("clicks and pops removed")
    if dehum_hz:
        # Notches (Q 10: 6 Hz wide at 60 Hz) at the mains frequency and its harmonics. Narrower ones
        # (Q 30) miss mains that drift by a hertz or two; voices lose about 1 dB at Q 10.
        chain += [f"bandreject=f={dehum_hz * h}:t=q:w=10" for h in range(1, HUM_HARMONICS + 1)]
        notes.append(f"{dehum_hz} Hz mains hum notched out (and {HUM_HARMONICS - 1} harmonics)")
    if highpass_hz:
        chain.append(f"highpass=f={highpass_hz:g}")
        notes.append(f"rumble below {highpass_hz:g} Hz removed (high-pass)")
    if denoise:
        # afftdn needs the noise level up front; its own tracking (tn) barely reacts on short or
        # loud-noise recordings, so give it the floor measured after the steps before it.
        floor = _excerpt_floor(src, ",".join(chain) or None, start_seconds, limit_seconds)
        floor = -50.0 if floor is None else min(-20.0, max(-80.0, floor))
        chain.append(f"afftdn=nr={denoise_db:g}:nf={floor:.0f}")
        notes.append(f"noise reduction: steady noise at about {floor:.0f} dBFS cut by {denoise_db:g} dB")
    if remove_silence:
        levels = _excerpt_levels(src, ",".join(chain) or None, start_seconds, limit_seconds)
        threshold = threshold_from_levels(levels)
        if threshold is None:
            notes.append("silence removal skipped: speech and background are too close in level to separate")
        else:
            # Leading silence is cut to 0.2 s; every later pause longer than min_silence_seconds is
            # shortened to min_silence_seconds, so sentences still breathe and words don't run together.
            chain.append(
                f"silenceremove=start_periods=1:start_threshold={threshold:.1f}dB:start_silence=0.2"
                f":stop_periods=-1:stop_duration={min_silence_seconds:g}:stop_threshold={threshold:.1f}dB")
            notes.append(f"pauses longer than {min_silence_seconds:g}s shortened to {min_silence_seconds:g}s "
                         f"(silence = below {threshold:.0f} dBFS)")
    if loudnorm:
        loudness = _excerpt_loudness(src, ",".join(chain) or None, start_seconds, limit_seconds)
        if loudness is None:
            notes.append("loudness normalization skipped: no audible signal")
        else:
            gain = max(-20.0, min(40.0, target_lufs - loudness))
            chain += [f"volume={gain:.1f}dB", f"alimiter=limit={LIMIT}:level=false"]
            notes.append(f"loudness {loudness:.1f} → {target_lufs:.0f} LUFS ({gain:+.1f} dB, peaks limited)")
    tmp = dst.with_name(dst.name + ".tmp.flac")
    try:
        _ffmpeg(["-loglevel", "error", *head, *(["-af", ",".join(chain)] if chain else []),
                 "-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16", "-c:a", "flac", str(tmp)])
        if tmp.stat().st_size < 100:
            raise AudioProcessingError("Processing left no audio")
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
    return notes


def _excerpt_filters(filters: str | None, start_seconds: float | None) -> str | None:
    # Measurements read the file with pcm_chunks/measure_loudness, which have no input seek; an
    # atrim at the front of the chain selects the same excerpt the processing pass will see.
    if not start_seconds:
        return filters
    return ",".join([f"atrim=start={start_seconds:.3f}", "asetpts=PTS-STARTPTS"] + ([filters] if filters else []))


def _excerpt_levels(src, filters, start_seconds, limit_seconds) -> np.ndarray:
    # `-t` is an output option in pcm_chunks, like in the processing pass: it limits what comes out of
    # the filters, so after the atrim it is the excerpt's own length.
    return _levels_db(src, _excerpt_filters(filters, start_seconds), limit_seconds)


def _excerpt_floor(src, filters, start_seconds, limit_seconds) -> float | None:
    levels = _excerpt_levels(src, filters, start_seconds, limit_seconds)
    return float(np.percentile(levels, 10)) if levels.size else None


def _excerpt_loudness(src, filters, start_seconds, limit_seconds) -> float | None:
    return measure_loudness(src, _excerpt_filters(filters, start_seconds), limit_seconds)


# ---------------------------------------------------------------- spectrogram
SPECTROGRAM_RATE = 8000  # speech lives below 4 kHz; halves the work and the decoded size
SPECTROGRAM_FFT = 512
SPECTROGRAM_RANGE_DB = 80.0
SPECTROGRAM_MIN_RANGE_DB = 40.0
# Dark blue -> purple -> orange -> pale yellow (perceptually ordered, readable in light and dark UIs).
_COLORMAP_STOPS = np.array([[0, 4, 10, 30], [0.3, 70, 20, 110], [0.55, 180, 50, 90], [0.8, 245, 140, 40],
                            [1.0, 252, 250, 190]], dtype=np.float32)


def spectrogram_png(src: Path, dst: Path, width: int = 1000, height: int = 200,
                    limit_seconds: float | None = None) -> dict:
    """Render a spectrogram (time left to right, 0-4 kHz bottom to top) to `dst` as PNG.

    The audio is decoded at 8 kHz and read in chunks; every `hop` samples one column of 4 averaged
    FFTs is kept (about 2x the image width), then columns are merged down to `width`. Memory stays
    at a few MB whatever the length. Returns {"duration", "max_hz"} for drawing the axes.
    """
    from PIL import Image

    estimate = probe_duration(src) or 600.0
    if limit_seconds:
        estimate = min(estimate, limit_seconds)
    hop = max(SPECTROGRAM_FFT, int(estimate * SPECTROGRAM_RATE) // (2 * width))
    window = np.hanning(SPECTROGRAM_FFT).astype(np.float32)
    offsets = np.linspace(0, hop - SPECTROGRAM_FFT, 4).astype(int)
    columns, total, pending = [], 0, np.empty(0, dtype=np.int16)
    for chunk in pcm_chunks(src, SPECTROGRAM_RATE, None, limit_seconds, chunk_samples=max(hop, 1 << 15)):
        total += chunk.shape[0]
        pending = np.concatenate([pending, chunk])
        while pending.shape[0] >= hop:
            frames = np.stack([pending[o:o + SPECTROGRAM_FFT] for o in offsets]).astype(np.float32) / 32768.0
            columns.append((np.abs(np.fft.rfft(frames * window, axis=1)) ** 2).mean(axis=0))
            pending = pending[hop:]
    if not columns:
        raise AudioProcessingError("Audio too short for a spectrogram")
    power = np.stack(columns, axis=1)
    groups = np.array_split(np.arange(power.shape[1]), min(width, power.shape[1]))
    power = np.stack([power[:, g].mean(axis=1) for g in groups], axis=1)
    # Colours span this recording's own range (2nd to 99.7th percentile, at least 40 dB), so quiet
    # recordings aren't black and noise doesn't wash everything out.
    db = 10 * np.log10(power + 1e-12)
    low, top = np.percentile(db, 2), np.percentile(db, 99.7)
    low = max(min(low, top - SPECTROGRAM_MIN_RANGE_DB), top - SPECTROGRAM_RANGE_DB)
    level = np.clip((db - low) / (top - low), 0, 1) ** 1.3  # slight gamma: background darker, speech pops
    rows = np.linspace(db.shape[0] - 1, 0, height).round().astype(int)  # low frequencies at the bottom
    stops = _COLORMAP_STOPS
    rgb = np.stack([np.interp(level[rows], stops[:, 0], stops[:, c]) for c in (1, 2, 3)], axis=-1)
    Image.fromarray(rgb.astype(np.uint8), "RGB").save(dst, optimize=True)
    return {"duration": round(total / SPECTROGRAM_RATE, 2), "max_hz": SPECTROGRAM_RATE // 2}
