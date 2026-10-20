"""Job storage and the background cleaning pipeline: analyze -> decide -> clean -> measure."""

import asyncio
import json
import logging
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path

from app.agent import AgentError, CleaningAgent, HeuristicAgent
from app.analysis import analyze
from app.audio import process_audio, spectrogram_png
from app.schemas import CleaningDecision, Job, JobStatus, utcnow
from app.strategy import open_preview, scratch_dir

logger = logging.getLogger(__name__)

# A job found mid-processing at boot was interrupted (deploy, crash, OOM kill). Re-run it once more,
# but if it was interrupted again, fail it instead of crash-looping on every boot.
MAX_ATTEMPTS = 2


class JobNotFound(KeyError):
    pass


class QueueFullError(RuntimeError):
    pass


class JobRepository:
    """Jobs in one JSON file, written atomically. Enough for a single-instance service."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        if path.exists():
            for raw in json.loads(path.read_text()):
                job = Job.model_validate(raw)
                self._jobs[job.id] = job

    def save(self, job: Job, only_if_exists: bool = False) -> None:
        with self._lock:
            if only_if_exists and job.id not in self._jobs:
                return  # deleted meanwhile; don't bring it back
            job.updated_at = utcnow()
            self._jobs[job.id] = job.model_copy(deep=True)
            self._flush()

    def get(self, job_id: str) -> Job:
        with self._lock:
            if job_id not in self._jobs:
                raise JobNotFound(job_id)
            return self._jobs[job_id].model_copy(deep=True)

    def list(self, limit: int = 100) -> list[Job]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return [j.model_copy(deep=True) for j in jobs[:limit]]

    def delete(self, job_id: str) -> None:
        with self._lock:
            if self._jobs.pop(job_id, None) is None:
                raise JobNotFound(job_id)
            self._flush()

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps([j.model_dump(mode="json") for j in self._jobs.values()]))
        tmp.replace(self.path)


class CleaningPipeline:
    def __init__(self, repository: JobRepository, data_dir: Path, agent_for: Callable[[str], CleaningAgent],
                 preview_seconds: float = 45, analysis_max_seconds: float = 900,
                 max_audio_seconds: float = 7200, concurrency: int = 1, max_queue_size: int = 20):
        self.repository = repository
        self.data_dir = data_dir
        self.agent_for = agent_for  # requested agent name -> agent
        self.preview_seconds = preview_seconds
        self.analysis_max_seconds = analysis_max_seconds
        self.max_audio_seconds = max_audio_seconds
        self.concurrency = concurrency
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max_queue_size)
        self._workers: list[asyncio.Task] = []
        self.active = 0

    # ------------------------------------------------------------ files
    def job_dir(self, job_id: str) -> Path:
        return self.data_dir / "jobs" / job_id

    def original_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "original"

    def cleaned_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "cleaned.flac"

    def spectrogram_path(self, job_id: str, version: str) -> Path:
        return self.job_dir(job_id) / f"{version}.spectrogram.png"

    def make_spectrogram(self, job_id: str, version: str) -> Path:
        path = self.spectrogram_path(job_id, version)
        if not path.exists():
            src = self.original_path(job_id) if version == "original" else self.cleaned_path(job_id)
            spectrogram_png(src, path, limit_seconds=self.max_audio_seconds)
        return path

    def remove_files(self, job_id: str) -> None:
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)

    # ------------------------------------------------------------ queue
    async def start(self) -> None:
        self._workers = [asyncio.create_task(self._worker()) for _ in range(self.concurrency)]
        for job in self.repository.list(limit=1000):
            if job.status == JobStatus.processing and job.attempts >= MAX_ATTEMPTS:
                job.status, job.stage = JobStatus.failed, None
                job.error = f"Processing was interrupted {job.attempts} times; not retried automatically"
                self.repository.save(job)
                continue
            if job.status in (JobStatus.queued, JobStatus.processing):
                job.status, job.stage = JobStatus.queued, None
                self.repository.save(job)
                try:
                    self.enqueue(job.id)
                except QueueFullError:
                    logger.warning("Queue full on startup; %s stays queued", job.id)

    async def stop(self) -> None:
        for task in self._workers:
            task.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)

    def register(self, job: Job) -> None:
        """Persist a new job whose upload is already at original_path(id), and queue it."""
        self.repository.save(job)
        try:
            self.enqueue(job.id)
        except QueueFullError:
            self.repository.delete(job.id)
            self.remove_files(job.id)
            raise

    def enqueue(self, job_id: str) -> None:
        try:
            self.queue.put_nowait(job_id)
        except asyncio.QueueFull as e:
            raise QueueFullError("Cleaning queue is full, try again later") from e

    async def _worker(self) -> None:
        while True:
            job_id = await self.queue.get()
            try:
                self.active += 1
                try:
                    await asyncio.to_thread(self.process, job_id)
                finally:
                    self.active -= 1
            except Exception:  # never let one job kill the worker
                logger.exception("Job %s crashed the worker", job_id)
            finally:
                self.queue.task_done()

    # ------------------------------------------------------------ processing
    def process(self, job_id: str) -> None:
        try:
            job = self.repository.get(job_id)
        except JobNotFound:
            return  # deleted while queued
        job.status, job.error, job.attempts = JobStatus.processing, None, job.attempts + 1
        metrics: dict[str, float] = {}
        try:
            src = self.original_path(job_id)
            if not src.exists():
                raise RuntimeError("The uploaded audio is no longer available")

            job = self._stage(job, "analyzing")
            started = time.perf_counter()
            job.analysis = analyze(src, limit_seconds=self.analysis_max_seconds)
            metrics["analyze_seconds"] = time.perf_counter() - started

            job = self._stage(job, "deciding")
            started = time.perf_counter()
            job.decision = self.decide(job)
            metrics["decide_seconds"] = time.perf_counter() - started

            job = self._stage(job, "cleaning")
            started = time.perf_counter()
            dst = self.cleaned_path(job_id)
            self.spectrogram_path(job_id, "cleaned").unlink(missing_ok=True)
            job.notes = process_audio(src, dst, **job.decision.plan.model_dump(),
                                      limit_seconds=self.max_audio_seconds)
            metrics["clean_seconds"] = time.perf_counter() - started

            job = self._stage(job, "measuring")
            job.result = analyze(dst, limit_seconds=self.analysis_max_seconds)
            if job.decision.plan.remove_silence:
                job.notes.append("after shortening pauses there are fewer quiet moments to measure the background "
                                 "on, so the cleaned audio's noise floor and SNR read worse than they sound; "
                                 "the decision's trials (run without silence removal) compare noise fairly")
            job.status, job.stage = JobStatus.completed, None
        except Exception as e:  # noqa: BLE001 - every failure is reported on the job
            logger.exception("Job %s failed", job_id)
            job.status, job.stage, job.error = JobStatus.failed, None, str(e) or e.__class__.__name__
        job.metrics = {k: round(v, 2) for k, v in metrics.items()}
        self._save_unless_deleted(job)

    def decide(self, job: Job) -> CleaningDecision:
        """The requested plan if there is one, else the agent's decision (rule-based if Claude fails)."""
        if job.requested_plan is not None:
            return CleaningDecision(agent="user", goal=job.goal, plan=job.requested_plan,
                                    reasoning=["plan given with the request"])
        with scratch_dir(self.job_dir(job.id)) as tmp:
            agent = self.agent_for(job.agent)
            preview = open_preview(self.original_path(job.id), job.goal, job.analysis, self.preview_seconds, Path(tmp))
            try:
                return agent.decide(job.analysis, preview)
            except AgentError as e:
                if isinstance(agent, HeuristicAgent):
                    raise
                logger.warning("Agent %s failed on %s (%s); using the rule-based agent", agent.name, job.id, e)
                preview.trials.clear()
                decision = HeuristicAgent().decide(job.analysis, preview)
                decision.reasoning.insert(0, f"{agent.name} could not decide ({e}); the rule-based agent did")
                return decision

    def _stage(self, job: Job, stage: str) -> Job:
        job.stage = stage
        self._save_unless_deleted(job)
        return job

    def _save_unless_deleted(self, job: Job) -> None:
        self.repository.save(job, only_if_exists=True)
