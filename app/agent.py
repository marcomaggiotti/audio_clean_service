"""Agents that decide the cleaning strategy for a recording.

- HeuristicAgent: rules turn each measured problem into a repair, then a sweep of noise-reduction
  strengths is tried on a preview excerpt and the best-scoring one wins. Offline and deterministic.
- ClaudeAgent: Claude gets the measurements, the goal and the scoring rules, and works with two
  tools: `try_plan` runs a candidate plan on the preview and reports what it measured, and
  `submit_plan` hands in the final choice with its reasoning. It can test combinations no rule
  anticipated (e.g. a gentler denoise plus a higher high-pass), and explains its trade-offs.

Either way the result is a CleaningDecision: the plan, why, and every trial with its score.
"""

import json
import logging
import os
from typing import Protocol

from pydantic import ValidationError

from app.schemas import AudioAnalysis, CleaningDecision, CleaningPlan, Goal
from app.strategy import SCORING_RULES, Preview

logger = logging.getLogger(__name__)


class AgentError(RuntimeError):
    """The agent could not reach a decision (API error, refusal, no plan submitted)."""


class CleaningAgent(Protocol):
    name: str

    def decide(self, analysis: AudioAnalysis, preview: Preview) -> CleaningDecision: ...


# ---------------------------------------------------------------- rule-based agent

DENOISE_SWEEP = {Goal.transcription: [6, 10, 16], Goal.podcast: [6, 10, 16, 24], Goal.archival: [4, 8]}


def rule_based_plan(a: AudioAnalysis, goal: Goal) -> tuple[CleaningPlan, list[str]]:
    """Repairs and level/pause handling that follow directly from the measurements (no denoise yet)."""
    plan, why = CleaningPlan(), []
    if a.clipped_ratio >= 0.0005:
        plan.declip = True
        why.append(f"{a.clipped_ratio:.2%} of samples are clipped: reconstruct the flattened peaks first")
    if a.clicks_per_minute >= 3:
        plan.declick = True
        why.append(f"{a.clicks_per_minute:.0f} clicks per minute: remove impulses before other filters smear them")
    if a.hum_hz:
        plan.dehum_hz = a.hum_hz
        why.append(f"{a.hum_hz} Hz hum stands {a.hum_db:.0f} dB over its surroundings: notch it and its harmonics")
    rumble_limit = -15 if goal is Goal.archival else -25
    if a.rumble_ratio_db > rumble_limit:
        plan.highpass_hz = 80
        why.append(f"energy below 80 Hz is {a.rumble_ratio_db:.0f} dB relative to the voice band: high-pass at 80 Hz")
    if goal is Goal.podcast:
        plan.loudnorm = True
        why.append("podcast: normalize to -16 LUFS so it plays at the same level as other shows")
    elif goal is Goal.transcription and (a.loudness_lufs is None or abs(a.loudness_lufs + 16) > 4):
        plan.loudnorm = True
        level = "inaudible" if a.loudness_lufs is None else f"{a.loudness_lufs:.0f} LUFS"
        why.append(f"transcription: level is {level}, bring it to -16 LUFS")
    if goal is Goal.transcription and a.long_pause_seconds >= 3:
        plan.remove_silence, plan.min_silence_seconds = True, 0.6
        why.append(f"{a.long_pause_seconds:.0f} s of long pauses: shorten them to 0.6 s, speech recognition doesn't need them")
    elif goal is Goal.podcast and a.long_pauses and a.long_pause_seconds / a.long_pauses >= 2:
        plan.remove_silence, plan.min_silence_seconds = True, 1.0
        why.append(f"pauses average {a.long_pause_seconds / a.long_pauses:.1f} s: tighten them to 1 s, keeping a natural pace")
    if goal is Goal.archival:
        why.append("archival: only fix defects, keep timing and level as recorded")
    return plan, why


class HeuristicAgent:
    name = "heuristic"

    def decide(self, analysis: AudioAnalysis, preview: Preview) -> CleaningDecision:
        base, reasoning = rule_based_plan(analysis, preview.goal)
        candidates = [base]
        if preview.before.snr_db < 40:
            candidates += [base.model_copy(update={"denoise": True, "denoise_db": float(db)})
                           for db in DENOISE_SWEEP[preview.goal]]
            reasoning.append(f"background is {preview.before.snr_db:.0f} dB below speech: "
                             f"compare no noise reduction with {', '.join(map(str, DENOISE_SWEEP[preview.goal]))} dB")
        for plan in candidates:
            if not plan.is_empty():
                preview.try_plan(plan)
        best = preview.best()
        tried = "; ".join(f"{', '.join(t.plan.labels())}: {t.score:.1f}" for t in preview.trials)
        reasoning.append(f"scores on the preview (untouched {preview.baseline_score:.1f}): {tried}")
        if best.plan.is_empty():
            reasoning.append("nothing scored better than the untouched audio: leave it as it is")
        else:
            reasoning.append(f"chose the best: {', '.join(best.plan.labels())}")
        return CleaningDecision(agent=self.name, goal=preview.goal, plan=best.plan, reasoning=reasoning,
                                baseline_score=preview.baseline_score, trials=preview.trials,
                                preview={"start": preview.start, "seconds": preview.seconds})


# ---------------------------------------------------------------- Claude agent

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "declip": {"type": "boolean", "description": "Reconstruct clipped peaks (ffmpeg adeclip)"},
        "declick": {"type": "boolean", "description": "Remove clicks and pops (ffmpeg adeclick)"},
        "dehum_hz": {"type": "integer", "description": "Mains hum to notch out with 3 harmonics: 50, 60, or 0 for none"},
        "highpass_hz": {"type": "number", "description": "High-pass cutoff against rumble, 20-200 Hz, or 0 for none"},
        "denoise": {"type": "boolean", "description": "FFT noise reduction of steady background noise (ffmpeg afftdn)"},
        "denoise_db": {"type": "number", "description": "Noise reduction strength, 3-30 dB. Above ~16 dB consonants start to suffer"},
        "remove_silence": {"type": "boolean", "description": "Shorten pauses longer than min_silence_seconds"},
        "min_silence_seconds": {"type": "number", "description": "Pauses are cut down to this length, 0.3-10 s"},
        "loudnorm": {"type": "boolean", "description": "Normalize loudness to target_lufs with a -1.5 dBFS limiter"},
        "target_lufs": {"type": "number", "description": "Loudness target, -30 to -10 LUFS"},
    },
    "required": ["declip", "declick", "dehum_hz", "highpass_hz", "denoise", "denoise_db", "remove_silence",
                 "min_silence_seconds", "loudnorm", "target_lufs"],
    "additionalProperties": False,
}

TOOLS = [
    {
        "name": "try_plan",
        "description": "Run a cleaning plan on the preview excerpt and measure the result. Returns the score "
                       "(higher is better), its breakdown, the measurements after cleaning and what each filter "
                       "did. Use it to compare candidate plans before submitting.",
        "input_schema": PLAN_SCHEMA,
        "strict": True,
    },
    {
        "name": "submit_plan",
        "description": "Submit the final cleaning plan for the whole recording, with your reasoning. Call this "
                       "exactly once, when you are done experimenting.",
        "input_schema": {
            "type": "object",
            "properties": {
                "plan": PLAN_SCHEMA,
                "reasoning": {"type": "array", "items": {"type": "string"},
                              "description": "Short points: what the recording needs, what you tried, why this plan"},
            },
            "required": ["plan", "reasoning"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

SYSTEM_PROMPT = """\
You are an audio restoration engineer choosing how to clean one spoken-word recording. You are given \
its measurements and what the cleaned audio is for (the goal). Experiment with try_plan on a preview \
excerpt, compare the scores and measurements, then call submit_plan with the plan to apply to the \
whole recording.

Filters run in this fixed order: declip, declick, dehum, high-pass, noise reduction, silence removal, \
loudness. Use only what the recording needs: every filter can also damage speech. Goals:
- transcription: maximum intelligibility for speech recognition. Avoid strong noise reduction, which \
removes consonants; long pauses can be shortened.
- podcast: pleasant listening at -16 LUFS, quiet background, natural pacing.
- archival: fix real defects (clipping, clicks, hum) and change as little else as possible.

The preview score is computed like this (it can't hear everything, so use judgement as well, and \
explain any choice that departs from the highest score):
{scoring}

Silence removal is not run during trials; its effect on the pause terms is predicted. You can try at \
most {max_trials} plans. Be efficient: start from the rule-based suggestion, vary what matters, then submit."""


def _plan_from_tool(raw: dict) -> CleaningPlan:
    data = dict(raw)
    data["dehum_hz"] = data.get("dehum_hz") or None
    data["highpass_hz"] = data.get("highpass_hz") or None
    return CleaningPlan(**data)


def _plan_to_tool(plan: CleaningPlan) -> dict:
    data = plan.model_dump()
    data["dehum_hz"] = data["dehum_hz"] or 0
    data["highpass_hz"] = data["highpass_hz"] or 0
    return data


def claude_credentials_configured() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


class ClaudeAgent:
    """Tool-use loop over the Messages API; `client` is an anthropic.Anthropic (tests pass a fake)."""

    def __init__(self, client=None, model: str = "claude-opus-5-5", effort: str = "medium",
                 max_trials: int = 6, max_turns: int = 12):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.max_trials = max_trials
        self.max_turns = max_turns

    @property
    def name(self) -> str:
        return f"claude ({self.model})"

    def decide(self, analysis: AudioAnalysis, preview: Preview) -> CleaningDecision:
        import anthropic

        suggestion, rules = rule_based_plan(analysis, preview.goal)
        brief = {
            "goal": preview.goal.value,
            "recording": analysis.model_dump(),
            "preview": {"start_seconds": preview.start, "seconds": preview.seconds,
                        "measurements": preview.before.model_dump(), "untouched_score": preview.baseline_score},
            "rule_based_suggestion": {"plan": _plan_to_tool(suggestion), "because": rules},
        }
        system = SYSTEM_PROMPT.format(scoring=SCORING_RULES, max_trials=self.max_trials)
        messages = [{"role": "user", "content": "Decide how to clean this recording.\n\n" + json.dumps(brief, indent=1)}]

        for _ in range(self.max_turns):
            try:
                response = self.client.beta.messages.create(
                    model=self.model,
                    max_tokens=16000,
                    system=system,
                    tools=TOOLS,
                    messages=messages,
                    thinking={"type": "adaptive"},
                    output_config={"effort": self.effort},
                    # On a safety decline, the API re-runs the request on Anthropic's recommended model.
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                )
            except anthropic.APIStatusError as e:
                raise AgentError(f"Claude API error {e.status_code}: {e.message}") from e
            except anthropic.APIConnectionError as e:
                raise AgentError(f"Could not reach the Claude API: {e}") from e

            if response.stop_reason == "refusal":
                raise AgentError("Claude declined the request")
            if response.stop_reason == "max_tokens":
                raise AgentError("Claude's answer was cut off (max_tokens)")
            messages.append({"role": "assistant", "content": response.content})

            calls = [b for b in response.content if b.type == "tool_use"]
            if not calls:
                messages.append({"role": "user", "content": "Call submit_plan with your final plan."})
                continue
            results = []
            for call in calls:
                if call.name == "submit_plan":
                    try:
                        plan = _plan_from_tool(call.input["plan"])
                    except (ValidationError, KeyError, TypeError) as e:
                        results.append(self._error(call.id, f"Invalid plan: {e}"))
                        continue
                    return CleaningDecision(
                        agent=self.name, goal=preview.goal, plan=plan,
                        reasoning=[str(r) for r in call.input.get("reasoning", [])],
                        baseline_score=preview.baseline_score, trials=preview.trials,
                        preview={"start": preview.start, "seconds": preview.seconds})
                results.append(self._try(call, preview))
            messages.append({"role": "user", "content": results})
        raise AgentError(f"No plan submitted after {self.max_turns} turns")

    def _try(self, call, preview: Preview) -> dict:
        if call.name != "try_plan":
            return self._error(call.id, f"Unknown tool {call.name}")
        if len(preview.trials) >= self.max_trials:
            return self._error(call.id, f"Trial limit ({self.max_trials}) reached: submit a plan now")
        try:
            plan = _plan_from_tool(call.input)
        except (ValidationError, TypeError) as e:
            return self._error(call.id, f"Invalid plan: {e}")
        trial = preview.try_plan(plan)
        after = trial.after.model_dump(exclude={"duration_seconds", "analyzed_seconds"})
        content = {"score": trial.score, "score_breakdown": trial.score_breakdown, "after": after,
                   "filters": trial.notes, "trials_left": self.max_trials - len(preview.trials)}
        return {"type": "tool_result", "tool_use_id": call.id, "content": json.dumps(content)}

    @staticmethod
    def _error(tool_use_id: str, message: str) -> dict:
        return {"type": "tool_result", "tool_use_id": tool_use_id, "content": message, "is_error": True}
