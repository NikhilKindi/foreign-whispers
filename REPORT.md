# Foreign Whispers — Project Report

---

## Demo Video

> **Dubbed output video:** [Google Drive link](https://drive.google.com/YOUR_VIDEO_LINK_HERE)
>
> Upload your dubbed output video to Google Drive, set sharing to "Anyone with the link", and replace the link above.

---

## 1. Pipeline Overview

Foreign Whispers is an automated video dubbing pipeline: YouTube English video in, Spanish-dubbed video out. Six stages, four Docker containers, all open-source.

```
YouTube URL → Download → Transcribe → Diarize → Translate → TTS → Stitch → Dubbed MP4
```

| Container | Port | Role |
|-----------|------|------|
| `foreign-whispers-stt` (speaches) | 8000 | Whisper GPU inference |
| `foreign-whispers-tts` (Chatterbox) | 8020 | TTS + voice cloning (GPU) |
| `foreign-whispers-api` (FastAPI) | 8080 | Pipeline orchestrator (CPU) |
| `foreign-whispers-frontend` (Next.js) | 8501 | Dubbing Studio UI |

```bash
docker compose --profile nvidia up -d    # GPU mode
docker compose --profile cpu up -d       # CPU-only fallback
```

---

## 2. Task Completion — Full Checklist

Below is every task from the [documentation](https://aegean.ai/aiml-common/projects/nlp/foreign-whispers), with status and the files that were edited or created.

### 2.1 Download Integration Notebook

[Documentation page](https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/download_integration/download_integration)

No coding tasks — exploration and run-through only.

| Task | Status | Notes |
|------|--------|-------|
| Run `fw.download()` via SDK | Done | |
| Inspect downloaded artifacts | Done | |
| Visualize caption timeline | Done | |

### 2.2 Transcription Integration Notebook

[Documentation page](https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/transcription_integration/transcription_integration)

No coding tasks — exploration and run-through only.

| Task | Status | Notes |
|------|--------|-------|
| Transcribe using YouTube captions | Done | |
| Force Whisper STT | Done | |
| Compare YouTube vs Whisper segments | Done | |
| Inspect JSON structure | Done | |

### 2.3 Translation Integration Notebook

[Documentation page](https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/translation_integration/translation_integration)

| Task | Status | Files |
|------|--------|-------|
| Run translation via API | Done | |
| Analyze translation length expansion | Done | |
| Implement `get_shorter_translations()` | Done | `foreign_whispers/reranking.py` |

The docs said this was a "stub that currently returns an empty list." It now implements 4 shortening strategies: rule-based Spanish phrase contraction, MarianMT independent translation, argos re-translation from simplified source, and smart truncation at sentence/clause/word boundaries. Also implemented `pick_optimal_translation()` which selects the longest candidate that fits the duration budget.

### 2.4 Diarization Integration Notebook

[Documentation page](https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/diarization_integration/diarization_integration)

#### Task 1: `assign_speakers` Merge Function

| Step | Status | File |
|------|--------|------|
| 1.1 Read existing diarization code | Done | |
| 1.2 Write TDD tests | Done | Tests provided by docs |
| 1.3 Implement `assign_speakers()` | Done | `foreign_whispers/diarization.py` |
| 1.4 Re-run tests (4/4 pass) | Done | |
| 1.5 Commit | Done | |

For each segment, finds the diarization interval with the greatest temporal overlap and assigns that speaker. Does not mutate input. Defaults to `SPEAKER_00` when diarization is empty.

#### Task 2: Diarize API Endpoint

| Step | Status | File |
|------|--------|------|
| 2.1 Add `diarizations_dir` to Settings | Done | `api/src/core/config.py` |
| 2.2 Create response schema | Done | `api/src/schemas/diarize.py` (created) |
| 2.3 Create router with endpoint | Done | `api/src/routers/diarize.py` (created) |
| 2.4 Register router in main.py | Done | `api/src/main.py` |
| 2.5 Implement the endpoint (replace YOUR CODE HERE) | Done | `api/src/routers/diarize.py` |
| 2.6 Rebuild and test | Done | |
| 2.7 Commit | Done | |

Endpoint extracts audio via ffmpeg, runs pyannote diarization, extracts per-speaker voice clips, caches result to disk, returns `DiarizeResponse`.

#### Task 3: Merge Speaker Labels Into Transcription

| Step | Status | File |
|------|--------|------|
| 3.1 Add merge step to diarize endpoint | Done | `api/src/routers/diarize.py` (lines 83-88) |
| 3.2 Verify the merge | Done | |
| 3.3 Commit | Done | |

After diarization completes, the endpoint loads the transcription JSON, calls `assign_speakers()`, and writes back the labeled segments.

#### Task 4: Frontend Pipeline Integration

| Step | Status | File |
|------|--------|------|
| 4.1 Add `diarizeVideo()` API client function | Done | `frontend/src/lib/api.ts` |
| 4.2 Add `DiarizeResponse` type + `"diarize"` to `PipelineStage` | Done | `frontend/src/lib/types.ts` |
| 4.3 Wire diarize into pipeline hook | Done | `frontend/src/hooks/use-pipeline.ts` |
| 4.4 Add diarize row to pipeline table | Done | `frontend/src/components/pipeline-table.tsx` |
| 4.4 Add diarize status message | Done | `frontend/src/components/pipeline-status-bar.tsx` |
| 4.5 Build and test | Done | |
| 4.6 Commit | Done | |

#### Task 5: Per-Speaker TTS Voice Selection

| Step | Status | File |
|------|--------|------|
| 5.1 Design voice assignment strategy | Done | Round-robin from extracted clips via diarization |
| 5.2 Implement and test | Done | `api/src/routers/tts.py` (voice map logic, lines 63-96) |

The TTS endpoint reads the diarization cache, builds a speaker → voice WAV mapping, and passes it per-segment to the TTS engine. Falls back to `resolve_speaker_wav()` when extracted clips are missing.

### 2.5 Alignment Integration Notebook

[Documentation page](https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/alignment_integration/alignment_integration)

#### Task 1: Improve TTS Duration Prediction

| Step | Status | File |
|------|--------|------|
| Measure baseline heuristic error | Done | |
| Replace `_estimate_duration()` | Done | `foreign_whispers/alignment.py` (lines 36-55) |

Replaced the naive chars/15 heuristic with a syllable-rate model: counts vowel-cluster syllables, applies 5.2 syll/s rate calibrated for Chatterbox Spanish, adds punctuation pause penalties and inter-word micro-pauses.

#### Task 2: Duration-Aware Translation Re-ranking

| Step | Status | File |
|------|--------|------|
| Identify over-budget segments | Done | |
| Implement `get_shorter_translations()` | Done | `foreign_whispers/reranking.py` (lines 264-342) |
| Implement `pick_optimal_translation()` | Done | `foreign_whispers/reranking.py` (lines 345-389) |

Same as Translation notebook task — 4 shortening strategies with budget-aware candidate selection.

#### Task 3: Beat the Greedy Optimizer

| Step | Status | File |
|------|--------|------|
| Record greedy baseline metrics | Done | |
| Implement `global_align_dp()` | Done | `foreign_whispers/alignment.py` (lines 317-448) |
| Compare against greedy | Done | |

Implemented a gap-splitting optimizer that evaluates 6 allocation splits per inter-segment gap. Minimizes a penalty function combining stretch severity and downstream starvation.

**Limitation:** This is a pairwise adjacent-segment heuristic, not the full `dp[i][b]` formulation described in the docstring. A true DP would require discretizing the continuous gap budget, which was not completed.

#### Task 4: Build a Dubbing Quality Scorecard

| Step | Status | File |
|------|--------|------|
| Implement `dubbing_scorecard()` | Done | `foreign_whispers/evaluation.py` (lines 125-192) |
| Implement `_compute_intelligibility()` | Done | `foreign_whispers/evaluation.py` (lines 58-85) |
| Implement `_compute_semantic_fidelity()` | Done | `foreign_whispers/evaluation.py` (lines 88-122) |

Five dimensions (timing accuracy, stretch quality, intelligibility, semantic fidelity, naturalness) each normalized to [0,1] with a weighted overall score.

**Limitation:** `_compute_intelligibility()` uses a stretch-based proxy when Whisper is unavailable. When Whisper IS available, it returns a hardcoded 0.8 instead of doing a real STT round-trip.

### 2.6 TTS Integration Notebook

[Documentation page](https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/tts_integration/tts_integration)

#### Task 1: Understand the Chatterbox Client

| Step | Status | Notes |
|------|--------|-------|
| Read Chatterbox client code | Done | Read-only exploration |
| Explore available reference voices | Done | |

#### Task 2: Voice Resolution Function (TDD)

| Step | Status | File |
|------|--------|------|
| 2.1 Write TDD tests | Done | Tests provided by docs |
| 2.2 Implement `resolve_speaker_wav()` | Done | `foreign_whispers/voice_resolution.py` (created) |
| 2.3 Re-run tests (5/5 pass) | Done | |
| 2.4 Commit | Done | |

Resolution order: speaker-specific WAV → language default → global fallback. Returns a relative path for the Chatterbox container.

#### Task 3: Add `speaker_wav` Parameter to TTS API

| Step | Status | File |
|------|--------|------|
| 3.1 Add `speakers_dir` to Settings | Done | `api/src/core/config.py` |
| 3.2 Add `speaker_wav` query parameter to endpoint | Done | `api/src/routers/tts.py` (line 32) |
| 3.3 Pass `speaker_wav` through service layer | Done | `api/src/services/tts_service.py` |
| 3.4 Test manually | Done | |
| 3.5 Commit | Done | |

#### Task 4: Per-Speaker Voice Assignment

| Step | Status | File |
|------|--------|------|
| 4.1 Build speaker-to-voice mapping | Done | `api/src/routers/tts.py` (lines 63-86) |
| 4.2 Modify TTS to accept per-segment speaker hints | Done | `api/src/routers/tts.py`, `api/src/services/tts_service.py` |
| 4.3 Test with multi-speaker video | Done | |
| 4.4 Commit | Done | |

Reads the diarization voice map, builds `{SPEAKER_00: "es/SPEAKER_00.wav", ...}`, passes to TTS engine which switches voice per segment.

### 2.7 Stitch Integration Notebook

[Documentation page](https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/stitch_integration/stitch_integration)

No coding tasks — verification and run-through only.

| Task | Status | Notes |
|------|--------|-------|
| Run stitch via API | Done | |
| Inspect output artifacts | Done | |
| View generated VTT captions | Done | Rolling two-line format |

### 2.8 End-to-End Pipeline Notebook

[Documentation page](https://aegean.ai/aiml-common/projects/nlp/foreign-whispers/pipeline_end_to_end/pipeline_end_to_end)

| Task | Status | Notes |
|------|--------|-------|
| P1 — Download | Done | |
| P2 — Transcribe | Done | |
| P3 — Translate | Done | |
| P4 — TTS (with alignment) | Done | |
| P5 — Stitch | Done | |
| Show pipeline artifacts | Done | |

---

## 3. Problems Faced and Root Causes

### 3.1 Voice Cloning: Whole Video Had One Voice

**Root cause: Leading space in HF token broke diarization silently.**

The `.env` file had `HF_TOKEN= hf_RBg...` (space before the token). Docker passed this to the container. pyannote.audio rejected the whitespace-prefixed token but `diarize_audio()` only checked `if not hf_token` — a string with a space is truthy, so it passed the guard but failed silently in the pyannote pipeline.

**Result chain:**
1. `diarize_audio()` returned `[]` — no speakers detected
2. `extract_speaker_clips()` returned `{}` — no per-speaker WAVs
3. TTS fell back to `es/default.wav` for ALL segments → single voice for entire video

**Evidence:** All diarization cache files contained `{"segments": [], "voice_map": {}}`.

**Fixes applied:**
- Removed leading space from `.env`
- Added `.strip()` defense in `api/src/services/alignment_service.py` (line 23)
- Frontend never passed `speaker_wav` to the API → fixed in `frontend/src/lib/api.ts` and `frontend/src/hooks/use-pipeline.ts`
- Diarization step was skipped when only voice cloning was enabled → fixed condition in `frontend/src/hooks/use-pipeline.ts` (line 197)

### 3.2 Alignment Not Up to the Mark

**Root causes (multiple):**

1. **Duration prediction is heuristic.** The syllable-rate model is calibrated for average Chatterbox output, but actual TTS duration varies ±30% per segment.
2. **No VAD silence regions.** `global_align` accepts `silence_regions` from VAD, but the TTS engine always passes `[]`. Without silence data, `gap_shift` actions never fire — the aligner can only accept or stretch, never borrow from adjacent silence.
3. **`global_align_dp` is not true DP.** Pairwise gap splitting, not globally optimal.
4. **Translation re-ranking limited.** MarianMT requires `transformers` (heavy), argos re-translation often produces similar-length output. Smart truncation works but loses meaning.

### 3.3 TTS Issues on CPU

**Root cause: No local NVIDIA GPU.**

Chatterbox TTS requires CUDA. On CPU, the engine falls back to Coqui TTS which does NOT support voice cloning, produces lower quality speech, and uses a single fixed voice. The fallback logic is in `api/src/services/tts_engine.py` → `_make_tts_engine()`.

### 3.4 YouTube Bot Detection on Cloud GPU

**Root cause: YouTube blocks datacenter IP ranges.**

When running on Lightning.ai (cloud GPU), `yt-dlp` received `ERROR: Sign in to confirm you're not a bot`. YouTube blocks datacenter IPs regardless of cookies — detects the IP mismatch between cookie origin and request source.

**Fixes applied:**
- Added `cookiefile` support in `api/src/services/download_engine.py`
- Uncommented `cookies.txt` volume mount in `docker-compose.yml`
- Added registry-first video resolution in `api/src/routers/download.py` to skip YouTube API calls for registered videos
- **Primary workaround:** `scp` pipeline data from local machine to cloud to bypass YouTube download entirely

---

## 4. Complete File Edit Summary

### Files Created

| File | Purpose |
|------|---------|
| `foreign_whispers/voice_resolution.py` | Voice WAV resolution with 3-level fallback |
| `api/src/schemas/diarize.py` | Pydantic response model for diarize endpoint |
| `api/src/routers/diarize.py` | `POST /api/diarize/{video_id}` endpoint |

### Library Files Edited

| File | What was changed |
|------|-----------------|
| `foreign_whispers/diarization.py` | Added `assign_speakers()`, `_pick_best_segments()`, `extract_speaker_clips()` |
| `foreign_whispers/alignment.py` | Replaced `_estimate_duration()` with syllable-rate model, added `global_align_dp()` |
| `foreign_whispers/reranking.py` | Implemented `get_shorter_translations()` with 4 strategies, `pick_optimal_translation()` |
| `foreign_whispers/evaluation.py` | Implemented `dubbing_scorecard()`, `_compute_intelligibility()`, `_compute_semantic_fidelity()` |

### API Files Edited

| File | What was changed |
|------|-----------------|
| `api/src/core/config.py` | Added `diarizations_dir`, `speakers_dir` properties |
| `api/src/main.py` | Registered diarize router |
| `api/src/routers/tts.py` | Added `speaker_wav` param, per-speaker voice map from diarization |
| `api/src/routers/translate.py` | Speaker label propagation into cached translations |
| `api/src/routers/download.py` | Registry-first resolution to skip YouTube API calls |
| `api/src/services/alignment_service.py` | `.strip()` defense for HF token |
| `api/src/services/download_engine.py` | Cookie file support for yt-dlp |
| `api/src/services/tts_service.py` | `speaker_wav` and `voice_map` passthrough |

### Frontend Files Edited

| File | What was changed |
|------|-----------------|
| `frontend/src/lib/types.ts` | Added `DiarizeResponse`, `"diarize"` to `PipelineStage` |
| `frontend/src/lib/api.ts` | Added `diarizeVideo()`, `speakerWav` param to `synthesizeSpeech()` |
| `frontend/src/hooks/use-pipeline.ts` | Wired diarize stage, pass `speakerWav` when voice cloning enabled |
| `frontend/src/components/pipeline-table.tsx` | Added diarize row with `UsersIcon` |
| `frontend/src/components/pipeline-status-bar.tsx` | Added diarize status message |

### Config Files Edited

| File | What was changed |
|------|-----------------|
| `docker-compose.yml` | Uncommented `cookies.txt` volume mount |

---

## 5. Limitations

1. **Language fixed to EN → ES.** Alignment heuristics are Spanish-calibrated. Other languages need re-tuning.
2. **Translation quality.** argostranslate (OpenNMT) is fully offline but lower quality than GPT-4/DeepL.
3. **No lip-sync.** Segment-level duration matching only, no phoneme-level alignment.
4. **Voice cloning needs GPU.** CPU fallback (Coqui TTS) = single fixed voice, no cloning.
5. **Diarization needs HF token.** pyannote.audio is gated; missing/malformed token = silent failure.
6. **TTS is slow.** ~10-15s per segment on GPU. A 7-minute video takes 15-20 minutes.
7. **YouTube blocked from cloud.** Datacenter IPs always get bot-detected. Workaround: pre-place files via scp.
8. **`global_align_dp` is not true DP.** Pairwise gap splitting, not globally optimal.
9. **Intelligibility metric is a placeholder.** Returns hardcoded 0.8 when Whisper is available.
10. **No VAD integration in TTS.** Silence regions not passed to alignment = no gap-shift actions.

---

## 6. What Could Be Done with More Resources

### With a Dedicated GPU Server

| What | Why it helps |
|------|-------------|
| Separate GPUs for Whisper and Chatterbox | No more CUDA memory contention crashes |
| Whisper-large-v3 instead of medium | Better transcription → better alignment |
| Run VAD (Silero) and pass silence to alignment | Unlocks `gap_shift` — borrows from silence instead of stretching |
| Train a duration predictor on Chatterbox output | Replaces heuristic with learned model |

### With More Time

| What | Why it helps |
|------|-------------|
| True DP for `global_align_dp` | Globally optimal gap allocation |
| Real STT round-trip intelligibility | TTS → Whisper → WER instead of placeholder |
| LLM-based translation shortening (Gemini/GPT) | Much better than rule-based truncation |
| WebSocket progress streaming | Real-time feedback during 15-min TTS runs |

### Research Directions

- **Isochronous MT:** Fine-tune NMT with segment-length tags so translations naturally fit the source window. Would reduce dependence on time-stretching.
- **Duration-Controlled TTS:** Specify target duration as TTS input. Would eliminate pyrubberband time-stretch entirely.
- **LLM Re-ranking:** For `request_shorter` segments, an LLM generates shorter translation candidates ranked by duration fitness. `GEMINI_API_KEY` was provisioned for this but not fully integrated.
- **Syllable-Aware Stretching:** Stretch only vowels, leave consonants untouched → more natural sounding.

---

## 7. Technology Stack

| Component | Technology |
|-----------|-----------|
| Speech-to-Text | Whisper (`faster-whisper-medium`) via speaches |
| Translation | argostranslate (OpenNMT, offline) |
| TTS | Chatterbox (zero-shot voice cloning, GPU) |
| Speaker Diarization | pyannote.audio 3.1 |
| Voice Resolution | `resolve_speaker_wav()` — 3-level fallback |
| Time-Stretching | pyrubberband (0.75×–1.25×) |
| Re-ranking | Rule-based + MarianMT + argos + truncation |
| Video Processing | ffmpeg (stream-copy remux) |
| Evaluation | `clip_evaluation_report` + `dubbing_scorecard` |
| Backend | FastAPI + Pydantic (Python 3.11) |
| Frontend | Next.js 14 + shadcn/ui + Tailwind |
| Containers | Docker Compose (`nvidia` / `cpu` profiles) |

---

## 8. Repository Structure

```
foreign-whispers/
├── api/src/
│   ├── main.py                     # App factory, router registration
│   ├── core/config.py              # Settings (diarizations_dir, speakers_dir, hf_token)
│   ├── core/video_registry.py      # video_registry.yml loader
│   ├── routers/
│   │   ├── download.py             # yt-dlp + registry-first resolution
│   │   ├── transcribe.py           # Whisper STT
│   │   ├── diarize.py              # pyannote diarization + speaker clips  [NEW]
│   │   ├── translate.py            # argostranslate + speaker label propagation
│   │   ├── tts.py                  # TTS + voice map + speaker_wav param
│   │   ├── stitch.py               # ffmpeg remux + rolling VTT
│   │   └── eval.py                 # Alignment metrics
│   ├── schemas/
│   │   └── diarize.py              # DiarizeResponse  [NEW]
│   ├── services/
│   │   ├── tts_engine.py           # Chatterbox client, time-stretch, alignment
│   │   ├── alignment_service.py    # VAD, diarization (.strip() fix), evaluation
│   │   ├── download_engine.py      # yt-dlp + cookiefile support
│   │   └── tts_service.py          # speaker_wav + voice_map passthrough
│   └── inference/                  # Whisper/TTS backend abstraction
├── foreign_whispers/
│   ├── alignment.py                # _estimate_duration, global_align, global_align_dp
│   ├── reranking.py                # get_shorter_translations (4 strategies)
│   ├── evaluation.py               # clip_evaluation_report, dubbing_scorecard
│   ├── diarization.py              # assign_speakers, extract_speaker_clips
│   ├── voice_resolution.py         # resolve_speaker_wav  [NEW]
│   ├── vad.py                      # Silero VAD wrapper
│   └── client.py                   # FWClient HTTP SDK
├── frontend/src/
│   ├── hooks/use-pipeline.ts       # Pipeline orchestration + diarize stage
│   ├── lib/api.ts                  # API client (diarizeVideo, speakerWav param)
│   ├── lib/types.ts                # DiarizeResponse, PipelineStage
│   └── components/                 # pipeline-table, pipeline-status-bar, etc.
├── notebooks/                      # 8 integration notebooks
├── tests/                          # 28 test files
├── docker-compose.yml              # nvidia + cpu profiles, cookies mount
├── video_registry.yml              # Video catalog
└── REPORT.md                       # This document
```
