"""Audio Clean Service API: upload a recording, an agent decides how to clean it, download the result."""

import asyncio
import hmac
import json
import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from pydantic import ValidationError

from app.agent import ClaudeAgent, CleaningAgent, HeuristicAgent, claude_credentials_configured
from app.analysis import analyze
from app.audio import MEDIA_TYPES, AudioProcessingError, InvalidAudioError, detect_format, probe_duration
from app.config import Settings, get_settings
from app.jobs import CleaningPipeline, JobNotFound, JobRepository, QueueFullError
from app.schemas import (
    AudioAnalysis,
    CleaningPlan,
    Goal,
    HealthResponse,
    Job,
    JobCreated,
    JobStatus,
    JobSummary,
)
from app.strategy import GOAL_WEIGHTS, SCORING_RULES

logger = logging.getLogger(__name__)

UPLOAD_CHUNK = 1024 * 1024
PROTECTED_PREFIXES = ("/jobs", "/analyze")


class AgentChoice(str, Enum):
    auto = "auto"
    heuristic = "heuristic"
    claude = "claude"


class Version(str, Enum):
    original = "original"
    cleaned = "cleaned"


@dataclass
class Components:
    """Injectable dependencies; tests pass fakes, production builds them from Settings."""

    claude_client: object | None = None  # anthropic.Anthropic; tests pass a scripted fake


def _token_ok(request: Request, expected: str) -> bool:
    """Accept `X-API-Key: <token>` or `Authorization: Bearer <token>` (scheme case-insensitive)."""
    scheme, _, credentials = request.headers.get("authorization", "").partition(" ")
    bearer = credentials.strip() if scheme.lower() == "bearer" else ""
    supplied = request.headers.get("x-api-key", "").strip() or bearer
    return hmac.compare_digest(supplied.encode(), expected.encode())


class Agents:
    """Resolves a requested agent name to an agent; the Claude one is built once, on first use."""

    def __init__(self, settings: Settings, claude_client=None):
        self.settings = settings
        self.claude_client = claude_client
        self._claude: ClaudeAgent | None = None

    @property
    def claude_available(self) -> bool:
        return self.claude_client is not None or claude_credentials_configured()

    def resolve(self, requested: str) -> str:
        name = self.settings.cleaning_agent if requested == "auto" else requested
        if name == "auto":
            name = "claude" if self.claude_available else "heuristic"
        return name

    def __call__(self, requested: str) -> CleaningAgent:
        if self.resolve(requested) != "claude":
            return HeuristicAgent()
        if self._claude is None:
            s = self.settings
            self._claude = ClaudeAgent(client=self.claude_client, model=s.claude_model, effort=s.claude_effort,
                                       max_trials=s.agent_max_trials)
        return self._claude


def create_app(settings: Settings | None = None, components: Components | None = None) -> FastAPI:
    settings = settings or get_settings()
    components = components or Components()
    data_dir = Path(settings.data_dir)
    agents = Agents(settings, components.claude_client)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not shutil.which("ffmpeg"):
            logger.error("ffmpeg is not installed: every job will fail")
        if settings.cleaning_agent not in ("auto", "heuristic", "claude"):
            raise RuntimeError(f"CLEANING_AGENT must be auto, heuristic or claude, not {settings.cleaning_agent!r}")
        pipeline = CleaningPipeline(
            JobRepository(data_dir / "jobs.json"), data_dir, agents,
            preview_seconds=settings.preview_seconds, analysis_max_seconds=settings.analysis_max_seconds,
            max_audio_seconds=settings.max_audio_seconds, concurrency=settings.worker_concurrency,
            max_queue_size=settings.max_queue_size)
        await pipeline.start()
        app.state.pipeline = pipeline
        logger.info("Ready: default agent %s", agents.resolve("auto"))
        yield
        await pipeline.stop()

    app = FastAPI(
        title="Audio Clean Service",
        version="1.0.0",
        description="Upload a recording; an agent measures it, tries cleaning strategies on a preview and "
                    "applies the best one (noise reduction, hum, clicks, clipping, rumble, pauses, loudness).",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def require_token(request: Request, call_next):
        if settings.api_token and request.url.path.startswith(PROTECTED_PREFIXES):
            if not _token_ok(request, settings.api_token):
                return JSONResponse({"detail": "Missing or invalid API token"}, status_code=401,
                                    headers={"WWW-Authenticate": "Bearer"})
        return await call_next(request)

    def pipeline_of(request: Request) -> CleaningPipeline:
        return request.app.state.pipeline

    def get_job(pipeline: CleaningPipeline, job_id: str) -> Job:
        try:
            return pipeline.repository.get(job_id)
        except JobNotFound:
            raise HTTPException(404, "Job not found")

    async def save_upload(file: UploadFile, dst: Path) -> tuple[str, int]:
        """Stream the upload to `dst` within the size limit; returns (format, size). Cleans up on error."""
        dst.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        try:
            with dst.open("wb") as out:
                while chunk := await file.read(UPLOAD_CHUNK):
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise HTTPException(413, f"File larger than {settings.max_upload_mb} MB")
                    out.write(chunk)
            with dst.open("rb") as f:
                try:
                    fmt = detect_format(f.read(16))
                except InvalidAudioError as e:
                    raise HTTPException(415, str(e))
            duration = await asyncio.to_thread(probe_duration, dst)
            if duration is not None and duration > settings.max_audio_seconds + 1:
                raise HTTPException(413, f"Audio longer than {settings.max_audio_seconds} s ({duration:.0f} s)")
            return fmt, size
        except BaseException:
            dst.unlink(missing_ok=True)
            raise

    @app.get("/", include_in_schema=False)
    async def index():
        return RedirectResponse("/docs")

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request):
        pipeline = pipeline_of(request)
        has_ffmpeg = shutil.which("ffmpeg") is not None
        return HealthResponse(status="ok" if has_ffmpeg else "degraded", ffmpeg=has_ffmpeg,
                              default_agent=agents.resolve("auto"), claude_available=agents.claude_available,
                              claude_model=settings.claude_model, queue_depth=pipeline.queue.qsize(),
                              active_jobs=pipeline.active)

    @app.get("/healthz", include_in_schema=False)
    async def healthz():
        """Liveness for the platform health check: 503 if ffmpeg is missing. Never calls the Claude API."""
        if not shutil.which("ffmpeg"):
            return JSONResponse({"status": "unhealthy", "detail": "ffmpeg is not installed"}, status_code=503)
        return {"status": "ok"}

    @app.get("/strategies")
    async def strategies():
        """What the agents can do and how they compare strategies."""
        return {
            "goals": {g.value: GOAL_WEIGHTS[g] for g in Goal},
            "agents": {"heuristic": "rules per measured problem, plus a sweep of noise-reduction strengths",
                       "claude": f"{settings.claude_model} experimenting with plans through tool use",
                       "auto": f"currently {agents.resolve('auto')}"},
            "plan": CleaningPlan.model_json_schema(),
            "scoring": SCORING_RULES.splitlines(),
        }

    @app.post("/analyze", response_model=AudioAnalysis)
    async def analyze_upload(request: Request, file: UploadFile = File(...)):
        """Measure a recording's problems without cleaning it."""
        tmp = data_dir / "tmp" / uuid.uuid4().hex
        await save_upload(file, tmp)
        try:
            return await asyncio.to_thread(analyze, tmp, settings.analysis_max_seconds)
        except (AudioProcessingError, InvalidAudioError) as e:
            raise HTTPException(422, str(e))
        finally:
            tmp.unlink(missing_ok=True)

    @app.post("/jobs", response_model=JobCreated, status_code=202)
    async def create_job(
        request: Request,
        file: UploadFile = File(...),
        goal: Goal = Form(Goal.transcription),
        agent: AgentChoice = Form(AgentChoice.auto),
        plan: str | None = Form(None, description="Optional CleaningPlan as JSON: skips the agent"),
        title: str | None = Form(None),
    ):
        """Upload audio to clean. Unless a plan is given, the agent decides how; poll the status URL."""
        requested_plan = None
        if plan:
            try:
                requested_plan = CleaningPlan.model_validate(json.loads(plan))
            except (json.JSONDecodeError, ValidationError) as e:
                raise HTTPException(422, f"Invalid plan: {e}")
        if agent is AgentChoice.claude and not agents.claude_available:
            raise HTTPException(422, "The Claude agent needs ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN)")

        pipeline = pipeline_of(request)
        job_id = uuid.uuid4().hex
        fmt, size = await save_upload(file, pipeline.original_path(job_id))
        job = Job(id=job_id, title=title or Path(file.filename or "recording").stem, filename=file.filename or "upload",
                  content_type=MEDIA_TYPES[fmt], size_bytes=size, goal=goal, agent=agent.value,
                  requested_plan=requested_plan)
        try:
            pipeline.register(job)
        except QueueFullError as e:
            raise HTTPException(503, str(e), headers={"Retry-After": "30"})
        return JobCreated(id=job.id, status=job.status, status_url=f"/jobs/{job.id}")

    @app.get("/jobs", response_model=list[JobSummary])
    async def list_jobs(request: Request, limit: int = Query(50, ge=1, le=500)):
        return [JobSummary.of(j) for j in pipeline_of(request).repository.list(limit)]

    @app.get("/jobs/{job_id}", response_model=Job)
    async def job_status(request: Request, job_id: str):
        """Status, measurements before and after, the agent's decision with its trials, and notes."""
        return get_job(pipeline_of(request), job_id)

    @app.get("/jobs/{job_id}/audio")
    async def job_audio(request: Request, job_id: str, version: Version = Version.cleaned):
        pipeline = pipeline_of(request)
        job = get_job(pipeline, job_id)
        if version is Version.original:
            return FileResponse(pipeline.original_path(job_id), media_type=job.content_type, filename=job.filename)
        if job.status != JobStatus.completed:
            raise HTTPException(409, f"Job is {job.status.value}; the cleaned audio isn't ready")
        return FileResponse(pipeline.cleaned_path(job_id), media_type="audio/flac",
                            filename=f"{Path(job.filename).stem}.cleaned.flac")

    @app.get("/jobs/{job_id}/spectrogram")
    async def job_spectrogram(request: Request, job_id: str, version: Version = Version.cleaned):
        """PNG spectrogram (time left to right, 0-4 kHz bottom to top), rendered on first request."""
        pipeline = pipeline_of(request)
        job = get_job(pipeline, job_id)
        if version is Version.cleaned and job.status != JobStatus.completed:
            raise HTTPException(409, f"Job is {job.status.value}; the cleaned audio isn't ready")
        try:
            path = await asyncio.to_thread(pipeline.make_spectrogram, job_id, version.value)
        except AudioProcessingError as e:
            raise HTTPException(422, str(e))
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})

    @app.post("/jobs/{job_id}/retry", response_model=JobCreated, status_code=202)
    async def retry_job(request: Request, job_id: str):
        pipeline = pipeline_of(request)
        job = get_job(pipeline, job_id)
        if job.status != JobStatus.failed:
            raise HTTPException(409, f"Only failed jobs can be retried (this one is {job.status.value})")
        job.status, job.error, job.attempts = JobStatus.queued, None, 0
        pipeline.repository.save(job)
        try:
            pipeline.enqueue(job.id)
        except QueueFullError as e:
            raise HTTPException(503, str(e), headers={"Retry-After": "30"})
        return JobCreated(id=job.id, status=job.status, status_url=f"/jobs/{job.id}")

    @app.delete("/jobs/{job_id}", status_code=204)
    async def delete_job(request: Request, job_id: str):
        pipeline = pipeline_of(request)
        try:
            pipeline.repository.delete(job_id)
        except JobNotFound:
            raise HTTPException(404, "Job not found")
        pipeline.remove_files(job_id)
        return Response(status_code=204)

    return app
