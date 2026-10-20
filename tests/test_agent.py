"""The agents' decisions: the rule-based one per defect and goal, the Claude one's tool-use loop."""

import shutil

import pytest

from app.agent import AgentError, ClaudeAgent, HeuristicAgent, rule_based_plan
from app.analysis import analyze
from app.schemas import CleaningPlan, Goal
from app.strategy import open_preview, score
from tests.conftest import FakeClaude, tool_plan

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def decide(agent, path, goal, tmp_path):
    analysis = analyze(path)
    return agent.decide(analysis, open_preview(path, goal, analysis, 10, tmp_path))


# ------------------------------------------------------------------ rule-based agent

def test_clean_audio_is_left_alone_for_transcription(wav, tmp_path):
    decision = decide(HeuristicAgent(), wav(noise=0.0005), Goal.transcription, tmp_path)
    assert decision.plan.is_empty()
    assert "leave it as it is" in decision.reasoning[-1]


def test_noise_gets_the_best_scoring_reduction(wav, tmp_path):
    decision = decide(HeuristicAgent(), wav(noise=0.03), Goal.transcription, tmp_path)
    assert decision.plan.denoise
    tried = {t.plan.denoise_db for t in decision.trials if t.plan.denoise}
    assert tried == {6.0, 10.0, 16.0}
    best = max(decision.trials, key=lambda t: t.score)
    assert decision.plan == best.plan and best.score > decision.baseline_score + 5


@pytest.mark.parametrize("defect,expected", [
    ({"hum_hz": 50, "hum": 0.02}, {"dehum_hz": 50}),
    ({"clicks_per_second": 1}, {"declick": True}),
    ({"gain": 3}, {"declip": True}),
])
def test_each_defect_gets_its_repair(wav, tmp_path, defect, expected):
    decision = decide(HeuristicAgent(), wav(noise=0.0005, **defect), Goal.archival, tmp_path)
    for field, value in expected.items():
        assert getattr(decision.plan, field) == value, decision.reasoning
    assert not decision.plan.loudnorm and not decision.plan.remove_silence  # archival keeps level and timing


def test_goals_change_the_plan(wav, tmp_path):
    path = wav([(4, "voice"), (3, "pause"), (4, "voice"), (3, "pause"), (4, "voice")], noise=0.0005, gain=0.1)
    analysis = analyze(path)
    podcast, _ = rule_based_plan(analysis, Goal.podcast)
    transcription, _ = rule_based_plan(analysis, Goal.transcription)
    archival, _ = rule_based_plan(analysis, Goal.archival)
    assert podcast.loudnorm and podcast.remove_silence and podcast.min_silence_seconds == 1.0
    assert transcription.loudnorm and transcription.remove_silence and transcription.min_silence_seconds == 0.6
    assert archival.is_empty()


def test_score_penalizes_lost_consonants(wav, tmp_path):
    analysis = analyze(wav(noise=0.0005))
    duller = analysis.model_copy(update={"speech_brightness_db": analysis.speech_brightness_db - 6})
    plan = CleaningPlan(denoise=True)
    assert score(analysis, duller, plan, Goal.transcription)[1]["distortion"] == pytest.approx(-12)
    assert score(analysis, analysis, plan, Goal.transcription)[1]["distortion"] == 0


# ------------------------------------------------------------------ Claude agent

def test_claude_experiments_then_submits(wav, tmp_path):
    fake = FakeClaude([
        [("text", "Noisy; trying two strengths."),
         ("tool", "try_plan", tool_plan(denoise=True, denoise_db=8)),
         ("tool", "try_plan", tool_plan(denoise=True, denoise_db=14, highpass_hz=80))],
        [("tool", "submit_plan", {"plan": tool_plan(denoise=True, denoise_db=14, highpass_hz=80),
                                  "reasoning": ["background 18 dB down", "14 dB scored best"]})],
    ])
    agent = ClaudeAgent(client=fake, model="claude-opus-5-5", effort="medium")
    decision = decide(agent, wav(noise=0.03), Goal.transcription, tmp_path)

    assert decision.agent == "claude (claude-opus-5-5)"
    assert decision.plan == CleaningPlan(denoise=True, denoise_db=14, highpass_hz=80)
    assert decision.reasoning == ["background 18 dB down", "14 dB scored best"]
    assert [t.plan.denoise_db for t in decision.trials] == [8, 14]

    first, second = fake.requests
    assert first["model"] == "claude-opus-5-5" and first["thinking"] == {"type": "adaptive"}
    assert first["output_config"] == {"effort": "medium"}
    assert first["fallbacks"] == "default" and first["betas"] == ["server-side-fallback-2026-07-01"]
    assert all(tool["strict"] for tool in first["tools"])
    assert '"goal": "transcription"' in first["messages"][0]["content"]
    # Both trial results go back in one user message, after the assistant turn (kept whole).
    results = second["messages"][-1]["content"]
    assert [r["type"] for r in results] == ["tool_result", "tool_result"]
    assert '"score"' in results[0]["content"] and "is_error" not in results[0]
    assert second["messages"][-2]["role"] == "assistant"


def test_claude_gets_errors_for_bad_plans_and_the_trial_limit(wav, tmp_path):
    fake = FakeClaude([
        [("tool", "try_plan", tool_plan(dehum_hz=55))],
        [("tool", "try_plan", tool_plan(denoise=True))],
        [("tool", "try_plan", tool_plan(denoise=True, denoise_db=12))],
        [("text", "Done.")],
        [("tool", "submit_plan", {"plan": tool_plan(denoise=True), "reasoning": ["ok"]})],
    ])
    decision = decide(ClaudeAgent(client=fake, max_trials=1), wav(noise=0.03), Goal.podcast, tmp_path)
    assert decision.plan.denoise
    bad, ok, over = (fake.requests[i]["messages"][-1]["content"][0] for i in (1, 2, 3))
    assert bad["is_error"] and "dehum_hz" in bad["content"]
    assert "is_error" not in ok
    assert over["is_error"] and "Trial limit" in over["content"]
    assert fake.requests[4]["messages"][-1]["content"] == "Call submit_plan with your final plan."


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_claude_failures_raise(wav, tmp_path, stop):
    with pytest.raises(AgentError):
        decide(ClaudeAgent(client=FakeClaude([stop])), wav(noise=0.03), Goal.podcast, tmp_path)


def test_claude_must_submit_within_the_turn_limit(wav, tmp_path):
    fake = FakeClaude([[("text", "thinking aloud")]] * 3)
    with pytest.raises(AgentError, match="No plan submitted"):
        decide(ClaudeAgent(client=fake, max_turns=3), wav(noise=0.03), Goal.podcast, tmp_path)
