"""Runtime configuration, read from environment variables."""

import os
from functools import lru_cache

from pydantic import BaseModel


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


class Settings(BaseModel):
    # HTTP
    host: str = "0.0.0.0"
    port: int = int(_env("PORT", "8000"))

    # Shared secret for every /jobs and /analyze route. Empty = no auth.
    api_token: str | None = _env("API_TOKEN")

    # Storage: uploads, cleaned audio, spectrograms and the job list.
    data_dir: str = _env("DATA_DIR", "./data")

    # Uploads
    max_upload_mb: int = int(_env("MAX_UPLOAD_MB", "100"))
    max_audio_seconds: int = int(_env("MAX_AUDIO_SECONDS", "7200"))

    # Background worker
    worker_concurrency: int = int(_env("WORKER_CONCURRENCY", "1"))
    max_queue_size: int = int(_env("MAX_QUEUE_SIZE", "20"))

    # Agent that chooses the cleaning strategy: auto (Claude when credentials are set, otherwise the
    # rule-based agent), heuristic, or claude.
    cleaning_agent: str = (_env("CLEANING_AGENT", "auto") or "auto").lower()
    claude_model: str = _env("CLAUDE_MODEL", "claude-opus-5-5")
    claude_effort: str = _env("CLAUDE_EFFORT", "medium")  # low | medium | high | xhigh | max
    agent_max_trials: int = int(_env("AGENT_MAX_TRIALS", "6"))
    # Length of the excerpt every candidate strategy is tried on.
    preview_seconds: float = float(_env("PREVIEW_SECONDS", "45"))
    # Long recordings are measured on their first part only (the analysis is a full decode).
    analysis_max_seconds: float = float(_env("ANALYSIS_MAX_SECONDS", "900"))

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
