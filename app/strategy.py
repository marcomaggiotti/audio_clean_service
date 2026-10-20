"""Trying cleaning strategies on a preview excerpt and scoring the result.

Both agents (rule-based and Claude) work the same way: propose a plan, run it on a short excerpt of
the recording, measure what came out, compare. The score is the shared yardstick: it rewards a
quieter background and a correct level, and penalizes anything the cleaning breaks (lost consonants,
remaining clicks, clipping, hum) according to the goal.
"""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from app.analysis import analyze
from app.audio import process_audio
from app.schemas import AudioAnalysis, CleaningPlan, Goal, Trial

# Goal-specific weights: how much each property of the result matters.
GOAL_WEIGHTS = {
    Goal.transcription: {"loudness": 0.4, "distortion": 3.0, "pauses": 0.3, "min_pause": 0.5, "intervention": 0.0},
    Goal.podcast: {"loudness": 0.8, "distortion": 2.0, "pauses": 0.15, "min_pause": 0.8, "intervention": 0.0},
    Goal.archival: {"loudness": 0.0, "distortion": 4.0, "pauses": 0.0, "min_pause": 0.0, "intervention": 1.0},
}
TARGET_LUFS = {Goal.transcription: -16.0, Goal.podcast: -16.0, Goal.archival: None}
SNR_CAP = 45.0  # beyond this the background is inaudible; more reduction earns nothing

SCORING_RULES = """\
score = noise + loudness + clipping + clicks + hum + rumble + distortion + pauses + intervention
- noise: the preview's speech-to-background ratio in dB, capped at 45
- loudness: -w x |LUFS - target| (target -16 LUFS; weight 0.4 transcription, 0.8 podcast, 0 archival)
- clipping: -1000 x share of samples at full scale
- clicks: -0.2 x clicks per minute
- hum: -(hum prominence - 6 dB) when above 6 dB
- rumble: -0.5 x (rumble ratio + 20 dB) when above -20 dB
- distortion: -w x (consonant-band loss beyond 2 dB) (w 3 transcription, 2 podcast, 4 archival)
- pauses: -w x seconds of pause per minute left after shortening (w 0.3 transcription, 0.15 podcast),
  and -10 x (minimum pause length below 0.5 s transcription / 0.8 s podcast) for unnatural pacing;
  archival: -10 for removing any silence
- intervention (archival only): -1 per step used, and -10 for normalizing loudness"""


@dataclass
class Preview:
    """The excerpt every strategy is tried on, with its untouched measurements."""

    source: Path
    goal: Goal
    start: float
    seconds: float
    before: AudioAnalysis
    workdir: Path
    trials: list[Trial] = field(default_factory=list)

    @property
    def baseline_score(self) -> float:
        return score(self.before, self.before, CleaningPlan(), self.goal)[0]

    def try_plan(self, plan: CleaningPlan) -> Trial:
        """Run `plan` on the excerpt and score the result. Silence removal is not run (it would skew
        the noise measurements by deleting quiet frames); its effect is predicted from the pauses
        measured in the cleaned excerpt."""
        out = self.workdir / f"trial{len(self.trials)}.flac"
        measured = plan.model_copy(update={"remove_silence": False})
        notes = process_audio(self.source, out, **measured.model_dump(), start_seconds=self.start,
                              limit_seconds=self.seconds)
        after = analyze(out)
        out.unlink(missing_ok=True)
        total, breakdown = score(self.before, after, plan, self.goal)
        trial = Trial(plan=plan, score=total, score_breakdown=breakdown, after=after, notes=notes)
        self.trials.append(trial)
        return trial

    def untouched(self) -> Trial:
        """Leaving the audio alone is always an option; nothing needs to run to score it."""
        total, breakdown = score(self.before, self.before, CleaningPlan(), self.goal)
        return Trial(plan=CleaningPlan(), score=total, score_breakdown=breakdown, after=self.before)

    def best(self) -> Trial:
        return max([self.untouched(), *self.trials], key=lambda t: t.score)


def choose_preview(duration: float, seconds: float) -> tuple[float, float]:
    """(start, length) of the excerpt: skip the first fifth of long files (intros, jingles, silence)."""
    if duration <= seconds * 2:
        return 0.0, min(seconds, duration)
    return round(min(duration * 0.2, 120.0), 2), seconds


def open_preview(source: Path, goal: Goal, analysis: AudioAnalysis, seconds: float, workdir: Path) -> Preview:
    start, length = choose_preview(analysis.duration_seconds, seconds)
    before = analyze(source, limit_seconds=length, start_seconds=start)
    return Preview(source=source, goal=goal, start=start, seconds=length, before=before, workdir=workdir)


def score(before: AudioAnalysis, after: AudioAnalysis, plan: CleaningPlan, goal: Goal) -> tuple[float, dict[str, float]]:
    w = GOAL_WEIGHTS[goal]
    parts = {"noise": min(after.snr_db, SNR_CAP)}
    target = TARGET_LUFS[goal]
    if target is not None:
        parts["loudness"] = -w["loudness"] * abs((after.loudness_lufs if after.loudness_lufs is not None else -70) - target)
    parts["clipping"] = -1000 * after.clipped_ratio
    parts["clicks"] = -0.2 * after.clicks_per_minute
    parts["hum"] = -max(0.0, after.hum_db - 6) if after.hum_hz else 0.0
    parts["rumble"] = -0.5 * max(0.0, after.rumble_ratio_db + 20)
    parts["distortion"] = -w["distortion"] * max(0.0, before.speech_brightness_db - after.speech_brightness_db - 2)

    minutes = max(after.analyzed_seconds / 60, 0.25)
    if goal is Goal.archival:
        parts["pauses"] = -10.0 if plan.remove_silence else 0.0
        steps = len([label for label in plan.labels() if label != "unchanged"])
        parts["intervention"] = -w["intervention"] * steps - (10.0 if plan.loudnorm else 0.0)
    else:
        left = after.long_pause_seconds
        if plan.remove_silence:
            left = min(left, after.long_pauses * plan.min_silence_seconds)
        parts["pauses"] = -w["pauses"] * left / minutes
        if plan.remove_silence:
            parts["pauses"] -= 10 * max(0.0, w["min_pause"] - plan.min_silence_seconds)
    parts = {k: round(v, 2) for k, v in parts.items()}
    return round(sum(parts.values()), 2), parts


def scratch_dir(parent: Path) -> tempfile.TemporaryDirectory:
    parent.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="trials-", dir=parent)
