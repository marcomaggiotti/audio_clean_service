# Audio Clean Service

Upload a spoken-word recording. An agent measures what is wrong with it, tries cleaning strategies
on a short preview, picks the best one for your goal, and applies it to the whole file.

```
 POST /jobs ─► queue ─► analyze ──────────► decide (agent) ──────────────► clean ─────► measure again
 202 + job id           noise floor, SNR,   propose plans, run each on a   ffmpeg chain  GET /jobs/{id}
                        loudness, clipping, 45 s preview, score the        on the whole  GET /jobs/{id}/audio
                        clicks, hum,        result, keep the best          file          GET /jobs/{id}/spectrogram
                        rumble, pauses
```

The cleaning filters, decoding and spectrograms come from
[voice_embedding_service](https://github.com/marcomaggiotti/voice_embedding_service) (`app/audio.py`,
its "processed copies" feature): loudness normalization, light noise reduction and silence removal.
This service extends them with de-clipping, de-clicking, mains hum removal and a separate high-pass,
makes each step's strength a parameter, and adds the analysis and the agents that choose them.

## The cleaning steps

Filters run in this order, all in ffmpeg as a stream (about 55 MB of RAM whatever the length):

| Step | Plan field | What it does |
|---|---|---|
| De-clip | `declip` | Rebuilds flattened peaks (`adeclip`), then lowers the level 6 dB so the rebuilt peaks fit |
| De-click | `declick` | Removes clicks and pops (`adeclick`) before other filters smear them |
| De-hum | `dehum_hz` (50/60) | Q 10 notches at the mains frequency and 3 harmonics |
| High-pass | `highpass_hz` | Removes rumble below the cutoff |
| Noise reduction | `denoise`, `denoise_db` | `afftdn` lowering steady noise by `denoise_db`, using the floor measured after the steps before it |
| Shorten pauses | `remove_silence`, `min_silence_seconds` | Pauses longer than the minimum are cut to it; "silence" comes from the recording's own noise floor and speech level, and the step is skipped when the two are too close to separate |
| Loudness | `loudnorm`, `target_lufs` | EBU R128 loudness measured after everything else, then a gain and a −1.5 dBFS limiter (not ffmpeg's `loudnorm`, which resamples to 192 kHz and needs ~105 MB) |

Output is 16 kHz mono FLAC. The original upload is kept and never changed.

## How the strategy is decided

**1. Analysis** (`app/analysis.py`): one streaming pass at 16 kHz, plus an EBU R128 loudness pass.

| Measurement | How |
|---|---|
| Noise floor / speech level / SNR | 10th and 90th percentile of 50 ms levels |
| Loudness, peak, clipping | EBU R128 integrated loudness; share of samples at ≥ 0.99 full scale |
| Clicks per minute | Frames whose steepest sample-to-sample jump is 4× bigger than any other jump within ±20 ms. Voices have steep edges too (a glottal pulse every pitch period, consonant onsets), but those repeat or come in bursts; a click stands alone |
| Mains hum | Long-term spectrum with ~1 Hz bins: 50 and 60 Hz harmonics against their ±5–15 Hz neighbourhood (the two strongest harmonics must stand out, so one sung note isn't "hum") |
| Rumble | Energy at 20–80 Hz relative to 80 Hz–4 kHz |
| Speech brightness | 2–8 kHz vs 300 Hz–2 kHz energy in speech frames, after subtracting the background's own. This is how lost consonants (over-denoising) are detected |
| Pauses | Runs longer than 1 s below the recording's own silence threshold |

**2. Trials on a preview** (`app/strategy.py`): each candidate plan runs on a 45 s excerpt (from 20 %
into long files, to skip intros) and the result is analyzed again and scored:

```
score = noise + loudness + clipping + clicks + hum + rumble + distortion + pauses + intervention
```

The score rewards a quieter background (SNR, capped at 45 dB) and the right level, and penalizes what
the cleaning leaves or breaks: clipping, clicks, hum, rumble, and **lost consonants** (brightness drop
beyond 2 dB). The weights depend on the goal. "Leave it alone" is always a candidate, and ties go to
the gentler plan. `GET /strategies` returns the exact rules.

| Goal | Priorities |
|---|---|
| `transcription` (default) | Intelligibility. Consonant loss is penalized hard; long pauses can go (down to 0.6 s) |
| `podcast` | −16 LUFS, quiet background, pauses tightened to 1 s but not below 0.8 s |
| `archival` | Fix real defects only: every step costs a point, normalizing or cutting silence costs 10 |

**3. The agents** (`app/agent.py`):

- **`heuristic`**: turns each measured defect into a repair (declip, declick, dehum, high-pass), sets
  level and pause handling from the goal, then sweeps noise-reduction strengths (6/10/16 dB for
  transcription, up to 24 dB for podcasts, 4/8 dB for archival) and keeps the best-scoring plan.
  Offline and deterministic.
- **`claude`**: Claude (`claude-opus-5-5` by default) gets the measurements, the goal, the scoring
  rules and the rule-based suggestion, and works with two tools: `try_plan` runs a plan on the
  preview and returns its score and measurements, and `submit_plan` hands in the final plan with its
  reasoning. It can try combinations no rule anticipates (a gentler denoise with a higher high-pass,
  a different loudness target) and explains its trade-offs. At most `AGENT_MAX_TRIALS` trials. If
  the API fails, declines, or no plan is submitted, the rule-based agent decides instead and the
  decision says so.
- **`auto`** (default): Claude when `ANTHROPIC_API_KEY` (or `ANTHROPIC_AUTH_TOKEN`) is set, otherwise
  the rule-based agent.

Every job keeps the full decision: the plan, the reasoning, the untouched preview's score, and each
trial with its score breakdown and after-measurements.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/jobs` | Multipart: `file` (WAV, MP3, OGG/Opus, M4A, FLAC), optional `goal` (`transcription`, `podcast`, `archival`), `agent` (`auto`, `heuristic`, `claude`), `plan` (a JSON plan: skips the agent), `title`. Returns `202` with the job id |
| `GET` | `/jobs` | Recent jobs with status, agent and chosen plan |
| `GET` | `/jobs/{id}` | Status and stage, analysis before and after, the decision with all trials, filter notes, timings |
| `GET` | `/jobs/{id}/audio?version=cleaned\|original` | Cleaned FLAC (`409` until completed) or the original upload |
| `GET` | `/jobs/{id}/spectrogram?version=cleaned\|original` | PNG spectrogram, 0–4 kHz, rendered on first request |
| `POST` | `/jobs/{id}/retry` | Re-queue a failed job |
| `DELETE` | `/jobs/{id}` | Remove the job and its files |
| `POST` | `/analyze` | Multipart `file`: measurements and detected issues only, no cleaning |
| `GET` | `/strategies` | Goals, agents, the plan schema and the scoring rules |
| `GET` | `/health` | ffmpeg present, default agent, whether Claude is configured, queue depth |

When `API_TOKEN` is set, `/jobs` and `/analyze` require `X-API-Key: <token>` or
`Authorization: Bearer <token>`.

Example:

```bash
curl -F file=@episode.mp3 -F goal=podcast localhost:8000/jobs
# {"id": "3f2c…", "status": "queued", "status_url": "/jobs/3f2c…"}
curl localhost:8000/jobs/3f2c…            # decision.reasoning explains the plan
curl -o clean.flac localhost:8000/jobs/3f2c…/audio

# Skip the agent with an explicit plan:
curl -F file=@call.ogg -F 'plan={"dehum_hz": 50, "denoise": true, "denoise_db": 8}' localhost:8000/jobs
```

## Run locally

```bash
pip install -r requirements-dev.txt
python main.py                                  # API docs at http://localhost:8000/docs
ANTHROPIC_API_KEY=sk-ant-... python main.py     # with the Claude agent
pytest
```

`ffmpeg` and `ffprobe` must be on the PATH. Tests use synthetic voice-like recordings
(`tests/synth.py`) with each defect added, and a scripted fake in place of the Claude API, so they
need no network or API key.

Docker:

```bash
docker build -t audio-clean .
docker run -p 8000:8000 -v $PWD/data:/data -e ANTHROPIC_API_KEY audio-clean
```

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `API_TOKEN` | – | Requires a token for `/jobs` and `/analyze` |
| `DATA_DIR` | `./data` | Uploads, cleaned audio, spectrograms, `jobs.json` |
| `MAX_UPLOAD_MB` / `MAX_AUDIO_SECONDS` | `100` / `7200` | Upload limits |
| `WORKER_CONCURRENCY` / `MAX_QUEUE_SIZE` | `1` / `20` | Parallel jobs; uploads get `503` when the queue is full |
| `CLEANING_AGENT` | `auto` | `auto`, `heuristic` or `claude`: the default for jobs that don't choose |
| `ANTHROPIC_API_KEY` | – | Enables the Claude agent |
| `CLAUDE_MODEL` | `claude-opus-5-5` | Model for the Claude agent |
| `CLAUDE_EFFORT` | `medium` | `low` … `max`: how hard Claude thinks per turn |
| `AGENT_MAX_TRIALS` | `6` | Plans Claude may try on the preview |
| `PREVIEW_SECONDS` | `45` | Length of the excerpt strategies are tried on |
| `ANALYSIS_MAX_SECONDS` | `900` | Long recordings are measured on their first 15 minutes |

## Known limitations

- The score is a proxy: it can't hear musical noise or a voice that sounds "underwater". The
  consonant-band check catches the most common damage (over-denoising), and the archival goal
  keeps interventions minimal. Listen to the result for anything that matters.
- After pauses are shortened, the cleaned file has fewer quiet moments to measure the background on,
  so its after-measurements report a higher noise floor than it sounds; the trials (run without
  silence removal) are the fair comparison, and the job notes say so.
- Jobs and their queue are in-process with a JSON job list, so one instance owns its jobs. A job
  interrupted by a restart re-runs once on the next boot.
- Clicks are counted conservatively (one per 50 ms frame, only isolated jumps), so heavy crackle
  reads lower than it is.
