"""API and storage models."""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, model_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Goal(str, Enum):
    """What the cleaned audio is for; the agent weighs its options differently for each."""

    transcription = "transcription"  # speech recognition: intelligibility first, pauses can go
    podcast = "podcast"  # listening: broadcast loudness, quiet background, natural pauses
    archival = "archival"  # preservation: fix defects only, change as little as possible


class CleaningPlan(BaseModel):
    """One cleaning strategy: which filters run and how strongly. Every step is optional."""

    declip: bool = Field(False, description="Reconstruct clipped (flattened) peaks")
    declick: bool = Field(False, description="Remove clicks, pops and other short impulses")
    dehum_hz: int | None = Field(None, description="Notch out mains hum at 50 or 60 Hz and its harmonics")
    highpass_hz: float | None = Field(None, ge=20, le=200, description="Remove rumble below this frequency")
    denoise: bool = Field(False, description="FFT noise reduction of steady background noise")
    denoise_db: float = Field(10.0, ge=3, le=30, description="How much steady noise is lowered, in dB")
    remove_silence: bool = Field(False, description="Shorten long pauses")
    min_silence_seconds: float = Field(1.0, ge=0.3, le=10, description="Pauses longer than this are cut to it")
    loudnorm: bool = Field(False, description="Normalize loudness (EBU R128) with a peak limiter")
    target_lufs: float = Field(-16.0, ge=-30, le=-10, description="Loudness target in LUFS")

    @model_validator(mode="after")
    def _check(self):
        if self.dehum_hz not in (None, 50, 60):
            raise ValueError("dehum_hz must be 50, 60 or null")
        return self

    def is_empty(self) -> bool:
        return not (self.declip or self.declick or self.dehum_hz or self.highpass_hz or self.denoise
                    or self.remove_silence or self.loudnorm)

    def labels(self) -> list[str]:
        labels = []
        if self.declip:
            labels.append("declipped")
        if self.declick:
            labels.append("declicked")
        if self.dehum_hz:
            labels.append(f"{self.dehum_hz} Hz hum removed")
        if self.highpass_hz:
            labels.append(f"high-pass {self.highpass_hz:g} Hz")
        if self.denoise:
            labels.append(f"denoised {self.denoise_db:g} dB")
        if self.remove_silence:
            labels.append(f"pauses ≤ {self.min_silence_seconds:g}s")
        if self.loudnorm:
            labels.append(f"{self.target_lufs:g} LUFS")
        return labels or ["unchanged"]


class AudioAnalysis(BaseModel):
    """Measurements the agent decides from. Levels are dBFS unless noted."""

    duration_seconds: float
    analyzed_seconds: float = Field(description="Length actually measured (long files are capped)")
    noise_floor_db: float = Field(description="Background level: 10th percentile of 50 ms levels")
    speech_level_db: float = Field(description="Foreground level: 90th percentile of 50 ms levels")
    snr_db: float = Field(description="speech_level_db - noise_floor_db")
    loudness_lufs: float | None = Field(None, description="Integrated loudness (EBU R128); null if inaudible")
    peak_db: float
    clipped_ratio: float = Field(description="Share of samples at or near full scale")
    clicks_per_minute: float = Field(description="Short impulses that stand out from their surroundings")
    hum_hz: int | None = Field(None, description="Mains hum frequency, if detected")
    hum_db: float = Field(0.0, description="How far the strongest hum harmonic rises over its neighbourhood")
    rumble_ratio_db: float = Field(description="Energy below 80 Hz relative to 80 Hz-4 kHz (higher = more rumble)")
    speech_brightness_db: float = Field(description="2-8 kHz vs 300 Hz-2 kHz energy in loud frames (consonants)")
    silence_ratio: float = Field(description="Share of time below the silence threshold")
    long_pauses: int = Field(description="Pauses longer than 1 s")
    long_pause_seconds: float = Field(description="Total length of those pauses")
    issues: list[str] = Field(default_factory=list, description="Problems spotted, in plain words")


class Trial(BaseModel):
    """A strategy tried on a preview excerpt, with what it measured afterwards."""

    plan: CleaningPlan
    score: float = Field(description="Estimated quality of the result (higher is better)")
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    after: AudioAnalysis
    notes: list[str] = Field(default_factory=list)


class CleaningDecision(BaseModel):
    agent: str = Field(description="Which agent decided: heuristic or claude (with the model)")
    goal: Goal
    plan: CleaningPlan
    reasoning: list[str] = Field(default_factory=list)
    baseline_score: float | None = Field(None, description="Score of the untouched preview")
    trials: list[Trial] = Field(default_factory=list)
    preview: dict[str, float] = Field(default_factory=dict, description="Excerpt the trials ran on (start, seconds)")


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class Job(BaseModel):
    id: str
    title: str
    filename: str
    content_type: str
    size_bytes: int
    goal: Goal = Goal.transcription
    agent: str = Field("auto", description="Agent requested: auto, heuristic or claude")
    requested_plan: CleaningPlan | None = Field(None, description="Explicit plan; skips the agent")
    status: JobStatus = JobStatus.queued
    stage: str | None = Field(None, description="analyzing, deciding, cleaning or measuring while processing")
    error: str | None = None
    attempts: int = 0
    analysis: AudioAnalysis | None = None
    decision: CleaningDecision | None = None
    notes: list[str] = Field(default_factory=list, description="What the cleaning did")
    result: AudioAnalysis | None = Field(None, description="Analysis of the cleaned audio")
    metrics: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class JobSummary(BaseModel):
    id: str
    title: str
    status: JobStatus
    stage: str | None
    goal: Goal
    agent: str | None = Field(None, description="Agent that decided, once it has")
    plan: list[str] | None = Field(None, description="Labels of the chosen plan")
    created_at: datetime

    @classmethod
    def of(cls, job: Job) -> "JobSummary":
        return cls(id=job.id, title=job.title, status=job.status, stage=job.stage, goal=job.goal,
                   agent=job.decision.agent if job.decision else None,
                   plan=job.decision.plan.labels() if job.decision else None, created_at=job.created_at)


class JobCreated(BaseModel):
    id: str
    status: JobStatus
    status_url: str


class HealthResponse(BaseModel):
    status: str
    ffmpeg: bool
    default_agent: str
    claude_available: bool
    claude_model: str
    queue_depth: int
    active_jobs: int
