import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import Components, create_app
from tests.synth import build, wav_bytes

VOICE = [(6, "voice"), (2, "pause"), (6, "voice")]


class FakeClaude:
    """Stands in for anthropic.Anthropic: replays scripted responses and records every request.

    Each script entry is a list of ("text", str) / ("tool", name, input) blocks, or a stop reason
    string ("refusal", "max_tokens") for a response without content.
    """

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        entry = self.script.pop(0)
        if isinstance(entry, str):
            return SimpleNamespace(stop_reason=entry, content=[])
        content = []
        for i, block in enumerate(entry):
            if block[0] == "text":
                content.append(SimpleNamespace(type="text", text=block[1]))
            else:
                content.append(SimpleNamespace(type="tool_use", id=f"toolu_{len(self.requests)}_{i}",
                                               name=block[1], input=block[2]))
        has_tools = any(b.type == "tool_use" for b in content)
        return SimpleNamespace(stop_reason="tool_use" if has_tools else "end_turn", content=content)


def tool_plan(**changes) -> dict:
    """A complete plan in the Claude tool's format (0 = off for dehum_hz / highpass_hz)."""
    plan = {"declip": False, "declick": False, "dehum_hz": 0, "highpass_hz": 0, "denoise": False,
            "denoise_db": 10, "remove_silence": False, "min_silence_seconds": 1.0, "loudnorm": False,
            "target_lufs": -16}
    plan.update(changes)
    return plan


@pytest.fixture
def wav(tmp_path):
    """Write a synthetic recording (see tests/synth.build) and return its path."""
    def write(parts=VOICE, **kw) -> Path:
        path = tmp_path / f"in{len(list(tmp_path.glob('in*.wav')))}.wav"
        path.write_bytes(wav_bytes(build(parts, **kw)))
        return path
    return write


@pytest.fixture
def settings(tmp_path):
    return Settings(
        data_dir=str(tmp_path / "data"),
        max_upload_mb=5,
        max_audio_seconds=120,
        api_token=None,  # independent of an API_TOKEN exported in the developer's shell
        cleaning_agent="heuristic",  # never call the real API from tests
        preview_seconds=10,
    )


@pytest.fixture
def claude():
    return FakeClaude([])


@pytest.fixture
def client(settings, claude):
    with TestClient(create_app(settings, Components(claude_client=claude))) as c:
        yield c


def wait_for(client, job_id, timeout=60.0, headers=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/jobs/{job_id}", headers=headers).json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {body}")
