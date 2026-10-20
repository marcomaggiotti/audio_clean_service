"""The HTTP API: upload, job lifecycle, downloads, auth and validation."""

import json
import shutil

import pytest

from app.schemas import Job, JobStatus
from tests.conftest import tool_plan, wait_for
from tests.synth import build, wav_bytes

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")

NOISY = wav_bytes(build([(6, "voice"), (2, "pause"), (6, "voice")], noise=0.03))


def upload(client, data=NOISY, headers=None, **form):
    return client.post("/jobs", files={"file": ("episode.wav", data, "audio/wav")}, data=form, headers=headers)


def test_agent_cleans_an_upload(client):
    res = upload(client, goal="podcast")
    assert res.status_code == 202
    job = wait_for(client, res.json()["id"])
    assert job["status"] == "completed", job

    assert job["decision"]["agent"] == "heuristic" and job["decision"]["goal"] == "podcast"
    assert job["decision"]["plan"]["denoise"] and job["decision"]["plan"]["loudnorm"]
    assert job["decision"]["trials"] and job["decision"]["reasoning"]
    assert job["result"]["snr_db"] > job["analysis"]["snr_db"] + 5
    assert abs(job["result"]["loudness_lufs"] + 16) < 2
    assert any("noise reduction" in n for n in job["notes"])
    assert set(job["metrics"]) == {"analyze_seconds", "decide_seconds", "clean_seconds"}

    cleaned = client.get(f"/jobs/{job['id']}/audio")
    assert cleaned.headers["content-type"] == "audio/flac" and cleaned.content[:4] == b"fLaC"
    assert client.get(f"/jobs/{job['id']}/audio?version=original").content == NOISY
    for version in ("original", "cleaned"):
        png = client.get(f"/jobs/{job['id']}/spectrogram?version={version}")
        assert png.status_code == 200 and png.content[:4] == b"\x89PNG"

    listed = client.get("/jobs").json()
    assert listed[0]["id"] == job["id"] and listed[0]["agent"] == "heuristic" and listed[0]["plan"]


def test_explicit_plan_skips_the_agent(client):
    res = upload(client, plan=json.dumps({"highpass_hz": 100, "loudnorm": True, "target_lufs": -20}))
    job = wait_for(client, res.json()["id"])
    assert job["decision"]["agent"] == "user" and job["decision"]["trials"] == []
    assert abs(job["result"]["loudness_lufs"] + 20) < 2


def test_claude_agent_through_the_api(client, claude):
    claude.script = [
        [("tool", "try_plan", tool_plan(denoise=True, denoise_db=12))],
        [("tool", "submit_plan", {"plan": tool_plan(denoise=True, denoise_db=12), "reasoning": ["noisy"]})],
    ]
    job = wait_for(client, upload(client, agent="claude").json()["id"])
    assert job["status"] == "completed", job
    assert job["decision"]["agent"] == "claude (claude-opus-5-5)"
    assert job["decision"]["plan"]["denoise_db"] == 12 and job["decision"]["reasoning"] == ["noisy"]


def test_rule_based_agent_takes_over_when_claude_fails(client, claude):
    claude.script = ["refusal"]
    job = wait_for(client, upload(client, agent="claude").json()["id"])
    assert job["status"] == "completed" and job["decision"]["agent"] == "heuristic"
    assert "could not decide (Claude declined the request)" in job["decision"]["reasoning"][0]


def test_auto_uses_claude_when_available(settings, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import Components, create_app

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    settings.cleaning_agent = "auto"
    with TestClient(create_app(settings, Components())) as c:
        health = c.get("/health").json()
        assert health["default_agent"] == "heuristic" and not health["claude_available"]
        assert upload(c, agent="claude").status_code == 422  # no credentials
    settings.cleaning_agent = "claude"  # forced, but there is no key: still works, rule-based
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")  # blank, as a skipped Render prompt may leave it
    with TestClient(create_app(settings, Components())) as c:
        assert c.get("/health").json()["default_agent"] == "heuristic"
        job = wait_for(c, upload(c).json()["id"])
        assert job["status"] == "completed" and job["decision"]["agent"] == "heuristic"
    settings.cleaning_agent = "auto"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with TestClient(create_app(settings, Components())) as c:
        assert c.get("/health").json()["default_agent"] == "claude"


def test_analyze_endpoint(client):
    res = client.post("/analyze", files={"file": ("x.wav", NOISY, "audio/wav")})
    assert res.status_code == 200
    assert res.json()["snr_db"] < 25 and any("background noise" in i for i in res.json()["issues"])


def test_validation(client, settings):
    assert upload(client, data=b"%PDF-1.7 not audio").status_code == 415
    assert upload(client, plan="{not json").status_code == 422
    assert upload(client, plan=json.dumps({"dehum_hz": 55})).status_code == 422
    assert upload(client, goal="karaoke").status_code == 422
    too_big = wav_bytes(build([(200, "voice")]))
    assert upload(client, data=too_big).status_code == 413  # 200 s > max_audio_seconds (120)
    assert client.get("/jobs/nope").status_code == 404
    assert client.get("/strategies").json()["goals"]["archival"]["intervention"] == 1.0
    assert list((client.app.state.pipeline.data_dir / "jobs").glob("*/original")) == []  # nothing left behind


def test_cleaned_audio_waits_for_the_job(client):
    pipeline = client.app.state.pipeline
    pipeline.repository.save(Job(id="q" * 32, title="Queued", filename="x.wav", content_type="audio/wav",
                                 size_bytes=0))
    assert client.get(f"/jobs/{'q' * 32}/audio").status_code == 409
    assert client.get(f"/jobs/{'q' * 32}/spectrogram").status_code == 409


def test_retry_and_delete(client):
    pipeline = client.app.state.pipeline
    job_id = upload(client).json()["id"]
    wait_for(client, job_id)
    assert client.post(f"/jobs/{job_id}/retry").status_code == 409  # only failed jobs

    original = pipeline.original_path(job_id).read_bytes()
    pipeline.original_path(job_id).unlink()  # e.g. the disk was wiped
    job = pipeline.repository.get(job_id)
    job.status = JobStatus.queued
    pipeline.repository.save(job)
    pipeline.enqueue(job_id)
    failed = wait_for(client, job_id)
    assert failed["status"] == "failed" and "no longer available" in failed["error"]

    pipeline.original_path(job_id).write_bytes(original)
    assert client.post(f"/jobs/{job_id}/retry").status_code == 202
    assert wait_for(client, job_id)["status"] == "completed"

    assert client.delete(f"/jobs/{job_id}").status_code == 204
    assert client.get(f"/jobs/{job_id}").status_code == 404
    assert not pipeline.job_dir(job_id).exists()


def test_api_token(client, settings):
    settings.api_token = "s3cret"
    assert upload(client).status_code == 401
    assert client.get("/jobs").status_code == 401
    assert client.post("/analyze", files={"file": ("x.wav", NOISY, "audio/wav")}).status_code == 401
    assert client.get("/health").status_code == 200
    res = upload(client, headers={"Authorization": "Bearer s3cret"})
    assert res.status_code == 202
    assert wait_for(client, res.json()["id"], headers={"X-API-Key": "s3cret"})["status"] == "completed"


def test_interrupted_jobs_resume_on_restart(settings, claude):
    from fastapi.testclient import TestClient

    from app.main import Components, create_app

    with TestClient(create_app(settings, Components(claude_client=claude))) as c:
        job_id = upload(c).json()["id"]
        wait_for(c, job_id)
        pipeline = c.app.state.pipeline
        job = pipeline.repository.get(job_id)
        job.status, job.attempts = JobStatus.processing, 1  # as if a deploy killed it mid-way
        pipeline.repository.save(job)
        stuck = job.model_copy(update={"id": "s" * 32, "attempts": 2})
        pipeline.repository.save(stuck)
    with TestClient(create_app(settings, Components(claude_client=claude))) as c:
        assert wait_for(c, job_id)["status"] == "completed"
        assert c.get(f"/jobs/{'s' * 32}").json()["status"] == "failed"


def test_healthz_is_dependency_free(client, monkeypatch):
    assert client.get("/healthz").json() == {"status": "ok"}
    monkeypatch.setattr("app.main.shutil.which", lambda _: None)
    assert client.get("/healthz").status_code == 503


def test_render_blueprint_matches_the_settings():
    """Every variable render.yaml sets is one the service reads, and the health check path exists."""
    from pathlib import Path

    import yaml

    from app.config import Settings

    root = Path(__file__).parent.parent
    service = yaml.safe_load((root / "render.yaml").read_text())["services"][0]
    assert service["healthCheckPath"] == "/healthz" and service["disk"]["mountPath"] == "/data"
    read = {name.upper() for name in Settings.model_fields} | {"ANTHROPIC_API_KEY"}
    assert {var["key"] for var in service["envVars"]} <= read
