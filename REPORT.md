# Foreign Whispers: Automated Video Dubbing Pipeline
### Project Report — Natural Language Processing

---

## Abstract

Foreign Whispers is a fully automated end-to-end video dubbing pipeline that transforms English-language YouTube videos into Spanish-dubbed output with synchronized translated captions. The system chains five stages — video acquisition, speech-to-text transcription, machine translation, text-to-speech synthesis, and video rendering — into a cohesive service exposed through a Next.js web interface and a FastAPI backend. A core research contribution is the temporal alignment subsystem that attempts to fit target-language speech into the same time windows as the original, addressing the fundamental problem that Spanish typically runs 15–25% longer than English when spoken aloud. The pipeline was tested on *60 Minutes* interview segments and successfully produces dubbed video with translated WebVTT subtitle tracks.

---

## 1. Motivation and Problem Statement

Automated dubbing is significantly harder than subtitling. Subtitles can appear and disappear at any time; dubbed audio must match the lip movements and silences of the original speaker. When a phrase takes 4 seconds to say in English but 5.5 seconds in Spanish, one of three things must happen: the Spanish speech must be sped up (which sounds unnatural beyond ~25%), the translation must be shortened (which risks losing meaning), or the audio must drift out of sync with the video. Professional dubbing studios resolve this through months of script adaptation by human dubbing directors — Foreign Whispers attempts a computational solution.

The project was implemented locally on a CPU-only development machine for iteration and testing, with GPU-accelerated inference (NVIDIA GPU for both Whisper and Chatterbox TTS) used for actual pipeline runs via Docker Compose.

---

## 2. System Architecture

### 2.1 High-Level Design

The system follows a strict pipeline architecture where each stage is independently addressable via REST API and caches its output to disk. This design means any stage can be re-run in isolation without re-running the full pipeline — critical when debugging or when GPU resources are shared.

```
YouTube URL
    │
    ▼
[1] Download          yt-dlp → MP4 + line-delimited JSON captions
    │
    ▼
[2] Transcribe        OpenAI Whisper (via speaches GPU container) → JSON segments
    │
    ▼
[3] Translate         argostranslate (offline OpenNMT) → translated JSON segments
    │
    ▼
[4] Synthesize TTS    Chatterbox (GPU, voice cloning) → time-aligned WAV
    │
    ▼
[5] Stitch            ffmpeg audio remux → dubbed MP4 + WebVTT captions
```

### 2.2 Service Architecture

The system is decomposed into four Docker containers communicating over the host network:

| Container | Technology | Port | Role |
|-----------|-----------|------|------|
| `foreign-whispers-stt` | speaches (Faster-Whisper CUDA) | 8000 | GPU Whisper inference |
| `foreign-whispers-tts` | Chatterbox TTS API | 8020 | GPU voice synthesis |
| `foreign-whispers-api` | FastAPI (CPU-only) | 8080 | Pipeline orchestrator |
| `foreign-whispers-frontend` | Next.js | 8501 | Web UI |

The API container is intentionally CPU-only — it delegates all heavy inference to the two GPU containers via HTTP. This separation means the orchestration logic can be developed and tested without touching GPU resources, and GPU containers can be restarted independently when CUDA errors occur (which happened in practice — detailed in Section 6).

### 2.3 Backend Layer Architecture

The FastAPI backend follows a strict layered architecture to keep concerns separated:

```
routers/          HTTP boundary — request parsing, response formatting
services/         Business logic — pipeline orchestration, caching
inference/        Backend abstraction — local vs remote model switching
foreign_whispers/ Library — alignment algorithms, evaluation metrics
```

The `inference/` layer uses an abstract base class (`WhisperBackend`) that lets the system switch between local Whisper (CPU, for development) and the remote speaches GPU container via a single environment variable (`FW_WHISPER_API_URL`). The same pattern applies to TTS — Chatterbox (GPU server) with Coqui TTS (CPU) as a fallback.

---

## 3. Pipeline Components

### 3.1 Stage 1: Video Download

Videos are fetched using `yt-dlp` with an optional `cookies.txt` for authenticated requests. The download stage captures both the MP4 video and auto-generated English captions. On environments where YouTube bot-detection blocks automated downloads (e.g., shared compute clusters), the pipeline supports **pre-placed files**: if an MP4 or caption file already exists in `pipeline_data/api/videos/` or `pipeline_data/api/youtube_captions/`, the download stage detects the existing file and skips the network request entirely.

YouTube captions are distributed in a WebVTT rolling-window format where each phrase appears in two or three consecutive overlapping windows. The download stage normalizes this into a clean line-delimited JSON format, deduplicating overlapping segments. This normalization was found to be critical — without it, 168 input "segments" for a 7-minute clip were actually only ~96 unique phrases duplicated by the rolling-window format, which caused downstream TTS to generate audio for tiny 10ms marker segments and produce silent output.

**Caption fast-path:** The pipeline exposes a `use_youtube_captions=true` flag that uses the downloaded captions as the transcription source, bypassing the GPU Whisper stage. This was essential for testing during development and for submissions where GPU availability was uncertain.

### 3.2 Stage 2: Speech-to-Text Transcription

Transcription uses **OpenAI Whisper** via the `speaches` server (an OpenAI-compatible API wrapper around Faster-Whisper). The `Systran/faster-whisper-medium` model was chosen as a balance between accuracy and inference speed on a single NVIDIA GPU shared with the TTS container.

Key engineering challenges encountered:
- **Model lifecycle management:** speaches does not auto-load models on startup. The API's FastAPI `lifespan` hook pre-loads the model by calling `POST /api/ps/{model_id}` with a 6-attempt retry loop to handle the Docker startup race condition where speaches may not be ready when the API starts.
- **speaches v0.8.0 breaking change:** An upgrade to speaches v0.8.0 introduced a breaking change in the model card format, causing all transcription requests to return HTTP 500 with "invalid model card" errors. The fix required calling `DELETE /v1/models/{id}` followed by `POST /v1/models/{id}` to delete and re-download the model, plus adding this recovery sequence to both the startup hook and the per-request retry path.
- **GPU memory contention:** Both the Whisper (speaches) and TTS (Chatterbox) containers share the same NVIDIA GPU. Under heavy load, one container's CUDA state could corrupt the other's, triggering `CUDA error: device-side assert triggered`. Recovery required restarting both GPU containers.
- **Auto-unloading prevention:** speaches defaults to unloading idle models after 300 seconds. This was disabled by setting `WHISPER__TTL=-1` in docker-compose, preventing mid-session transcription failures.

### 3.3 Stage 3: Machine Translation

Translation uses **argostranslate**, a Python wrapper around OpenNMT neural machine translation models. All translation runs fully offline — no API keys, no network requests, no usage costs. The English → Spanish model is downloaded once and cached locally.

The output preserves Whisper's segment boundary timestamps, attaching each translated phrase to the same `[start, end]` time window as the source. This timestamp preservation is what makes temporal alignment possible in Stage 4.

### 3.4 Stage 4: TTS Synthesis with Temporal Alignment

This is the most technically complex stage and the core research contribution of the project.

#### 4a. TTS Engine: Chatterbox with Voice Cloning

Speech synthesis uses **Chatterbox**, an open-source zero-shot TTS model capable of voice cloning from a short reference audio clip. The system:
1. Runs **speaker diarization** (pyannote.audio) on the source video to identify which speaker is talking in each segment
2. Extracts **reference audio clips** (3–15 seconds) for each identified speaker
3. Uses these clips as voice references when synthesizing the Spanish translation, so the dubbed voice sounds similar to the original speaker

The Chatterbox API accepts the reference WAV via a multipart upload to `/v1/audio/speech/upload`, where Chatterbox performs zero-shot voice cloning to generate Spanish speech in the source speaker's vocal style.

#### 4b. Temporal Alignment

The alignment problem: Spanish is spoken approximately 15–25% slower than English per unit of information. A segment that takes 4 seconds in English may take 4.9 seconds in Spanish, leaving 0.9 seconds of audio that must fit into the original 4-second window — or be allowed to spill into adjacent silence.

The `foreign_whispers.alignment` module implements a **greedy left-to-right scheduler** that processes segments in order and makes per-segment decisions:

| Action | Condition | Result |
|--------|-----------|--------|
| `accept` | TTS fits within 110% of source window | Play at natural speed |
| `mild_stretch` | TTS fits within 140% of source window | Time-stretch via pyrubberband |
| `gap_shift` | Overflow fits in adjacent silence gap | Extend into silence, accumulate drift |
| `request_shorter` | No gap available | Flag for LLM re-ranking (future work) |
| `fail` | Nothing works | Log and continue |

Duration estimation uses a **syllable-rate heuristic** rather than character count, because Spanish character count is distorted by accents, digraphs, and punctuation. The heuristic estimates 5.2 syllables/second and adds penalties for punctuation pauses (0.12s per comma, 0.22s per sentence-ending mark) and inter-word micro-pauses.

Time-stretching is performed by **pyrubberband** (a Python binding to the Rubber Band audio time-stretching library), constrained to a 0.75×–1.25× window. Audio stretched or compressed beyond this range becomes noticeably distorted. Segments where the natural TTS audio is less than 50% of the source window are padded with silence rather than stretched (extreme stretching to fill long pause windows sounds unnatural).

### 3.5 Stage 5: Video Rendering

The final stage uses **ffmpeg** to replace the original English audio track with the synthesized Spanish WAV, using stream copy (`-c:v copy`) to avoid re-encoding the video track. This is a lossless remux — the original video quality is preserved exactly, and the operation takes seconds regardless of video length.

WebVTT subtitle files are generated from the translated segments and served via the API's `/api/captions/{id}` endpoint, rendered in the browser via the HTML5 `<track>` element.

---

## 4. Web Interface

The frontend is built with **Next.js** and **shadcn/ui**, providing a pipeline control panel where the user can:
- Select a video from the registered catalog
- Step through each pipeline stage with a single click
- Watch the original and dubbed video side-by-side with optional subtitle overlays
- Download the final dubbed MP4

Pipeline state is managed by a custom React hook (`use-pipeline.ts`) that orchestrates sequential API calls and tracks stage completion. The frontend proxies all API requests through Next.js rewrites to avoid CORS complexity.

**Infrastructure note:** TTS synthesis for a 96-segment video takes 15–20 minutes on the available GPU. The Next.js proxy timeout was set to 30 minutes (`proxyTimeout: 1_800_000`) to accommodate long-running TTS requests without the connection being dropped.

---

## 5. Evaluation

The `foreign_whispers.evaluation` module provides quantitative metrics for alignment quality:

| Metric | Description |
|--------|-------------|
| `mean_abs_duration_error_s` | Mean absolute difference between predicted and actual segment duration |
| `pct_severe_stretch` | Percentage of segments with stretch factor > 1.4× (audibly distorted) |
| `n_gap_shifts` | Number of segments that borrowed time from adjacent silence |
| `total_cumulative_drift_s` | End-to-end timing drift from gap-shifts accumulating |

The `clip_evaluation_report()` function consumes the `.align.json` sidecar files written alongside each TTS WAV and produces these metrics for A/B comparison between aligned and baseline (unconstrained) runs.

A `FailureAnalysis` system categorizes the dominant failure mode of each clip into: `duration_overflow`, `cumulative_drift`, `stretch_quality`, or `ok`, pointing to the most impactful corrective action.

---

## 6. Challenges and Engineering Decisions

### 6.1 YouTube Bot Detection
YouTube's automated download detection blocked yt-dlp requests from shared compute cluster IP ranges. The solution was to support pre-placed files: videos and captions manually placed in the pipeline directories are detected and used directly, bypassing the download stage.

### 6.2 YouTube Caption Format
YouTube's auto-generated captions use a "rolling window" format that presents each phrase multiple times in overlapping time windows. For a 7-minute video, this produced 168 apparent segments when only 96 were unique. Without normalization, the pipeline generated TTS audio for 72 spurious 10ms "marker" segments — audio that was then compressed to near-silence and mixed into the output. The fix was a minimum segment duration filter (`_MIN_SEGMENT_SEC = 0.3`) applied after deduplication.

### 6.3 GPU Resource Contention
Running two GPU containers (Whisper STT and Chatterbox TTS) on a single NVIDIA GPU caused intermittent CUDA errors. When Chatterbox exhausted GPU memory mid-generation, the CUDA device entered an error state that propagated to subsequent speaches requests, causing transcription failures even though speaches itself had not misbehaved. Recovery required restarting both containers. For production, these should run on separate GPUs or be serialized.

### 6.4 speaches Model Lifecycle
The speaches STT server does not auto-load models on startup and has a default 300-second idle TTL that unloads models. This caused mysterious 404 errors on transcription requests. Mitigations: (1) `WHISPER__TTL=-1` disables the idle unload; (2) the API pre-loads the model during startup; (3) per-request retry logic detects 404/500 and reloads the model before retrying once.

### 6.5 Docker Startup Race Condition
The API container started before speaches was ready, causing the startup pre-load to fail. Fixed by adding a 6-attempt retry loop with 5-second delays to the startup hook, and `depends_on: whisper-gpu: condition: service_healthy` in docker-compose so the API waits for speaches to pass its healthcheck before starting.

### 6.6 Alignment Quality
The alignment system works well for segments with adjacent silence to borrow from. The weakest point is the syllable-rate heuristic: Spanish speech rate varies significantly between speakers and styles, so the 5.2 syllables/second calibration is a rough approximation. Segments where the predicted duration is very different from the actual TTS output may be stretched beyond the comfortable range, producing slightly robotic-sounding speech. This is inherent to post-hoc time-stretching and is the motivation for duration-controlled TTS generation (future work).

---

## 7. Limitations

1. **Language pair is fixed to English → Spanish.** argostranslate supports additional language pairs, but the alignment heuristics (syllable rate, punctuation pauses) are calibrated for Spanish. Extending to other languages requires re-calibration.

2. **Translation quality is limited by argostranslate (OpenNMT).** Being a fully offline model, it produces lower-quality translations than GPT-4 or DeepL, particularly for idiomatic phrases and domain-specific vocabulary.

3. **No lip-sync.** The alignment system does not attempt lip synchronization. It targets segment-level isochrony (matching the duration of each speech segment) but does not track individual phoneme boundaries. True lip-sync requires visual speech analysis beyond the scope of this project.

4. **Single-speaker voice cloning per segment.** If two speakers overlap within a single Whisper segment, only one voice reference is used. Whisper's segmentation rarely splits overlapping speech.

5. **TTS speed.** Chatterbox with voice cloning takes approximately 10–15 seconds per segment on an NVIDIA GPU. A 96-segment 7-minute video takes 15–20 minutes end-to-end. This is impractical for real-time use but acceptable as a batch pipeline.

6. **No translation isochrony.** The translation stage does not attempt to produce Spanish text with durations matching the English source. This is the most impactful future improvement — isochronous machine translation (Lakew et al., 2021) would reduce the burden on time-stretching by producing translations that naturally fit the source timing.

7. **YouTube captions as default transcription.** To reduce pipeline time, the system defaults to using YouTube's auto-generated captions rather than running Whisper. These captions are sometimes inaccurate or improperly timed, which affects translation and TTS alignment quality.

---

## 8. Future Work

### Isochronous Machine Translation
Replace argostranslate with a length-aware translation model. Lakew et al. (2021, 2022) demonstrated that fine-tuning NMT models with segment-length tags produces translations that naturally fit source-language speaking durations, substantially reducing the need for post-hoc time-stretching.

### Duration-Controlled TTS
Microsoft Research's Total-Duration-Aware TTS (arXiv 2406.04281) and IndexTTS2 (2025) support specifying the target duration as an input, producing speech that fits the window naturally rather than stretching it after the fact.

### LLM Re-ranking for Short Translations
For segments where the greedy aligner flags `request_shorter`, an LLM could generate multiple shorter translation candidates ranked by duration fitness. This is a prompt-engineering approximation of isochronous MT that requires no model fine-tuning.

### Syllable-Aware Stretching
Rubber band time-stretching applied uniformly across a segment distorts all phonemes equally. A better approach (VideosDubber, Wu et al. 2023) stretches only vowel durations and leaves consonants untouched, producing more natural-sounding speed adjustments.

### On-Screen vs Off-Screen Isochrony
Federico et al. (2020) observed that on-screen speech (where the speaker's lips are visible) requires tight timing alignment (±100 ms), while off-screen narration tolerates ±300 ms of drift. A vision model could classify each segment and apply different alignment thresholds accordingly. 60 Minutes interviews are predominantly off-screen, so this distinction would have a significant effect.

---

## 9. Technology Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Speech-to-Text | OpenAI Whisper (`faster-whisper-medium`) | Via speaches OpenAI-compatible server |
| Machine Translation | argostranslate (OpenNMT) | Fully offline, English → Spanish |
| TTS | Chatterbox (zero-shot voice cloning) | GPU server, voice cloning from speaker clips |
| Speaker Diarization | pyannote.audio 3.1 | Identifies per-speaker time intervals |
| Time-Stretching | pyrubberband (Rubber Band Library) | Constrained to 0.75×–1.25× |
| Video Processing | ffmpeg | Stream-copy remux (lossless) |
| Backend | FastAPI + Pydantic | Python 3.11, uv package manager |
| Frontend | Next.js 14 + shadcn/ui | TypeScript, Tailwind CSS |
| Containerization | Docker Compose | Profiles: `nvidia` (GPU), `cpu` (dev) |
| Settings | Pydantic Settings (`FW_` prefix) | All config via environment variables |

---

## 10. Running the Pipeline

### Prerequisites
- Docker with NVIDIA Container Toolkit (for GPU profile)
- `HF_TOKEN` environment variable (for pyannote diarization model)

### Start all services
```bash
docker compose --profile nvidia up -d
```

Open **http://localhost:8501** in your browser.

### For CPU-only development
```bash
docker compose --profile cpu up -d
```
The CPU profile runs the API and frontend without the GPU containers. Whisper and Chatterbox must be provided externally, or the pipeline stages can be tested individually via the REST API with pre-computed outputs.

### Pre-placing videos (bypassing YouTube download)
Place files directly into the pipeline directories before starting the download stage:
```
pipeline_data/api/videos/<Video Title>.mp4
pipeline_data/api/youtube_captions/<Video Title>.txt
```
The download endpoint detects existing files and returns immediately without calling yt-dlp.

---

## 11. Repository Structure

```
foreign-whispers/
├── api/src/                     # FastAPI backend
│   ├── main.py                  # App factory, lazy model loading, speaches pre-load
│   ├── core/config.py           # Pydantic settings (FW_ env prefix)
│   ├── routers/                 # Route handlers (download, transcribe, translate,
│   │                            #   tts, stitch, diarize, eval)
│   ├── services/                # Business logic (tts_engine, translation_engine, …)
│   └── inference/               # Whisper backend abstraction (local / remote)
├── foreign_whispers/            # Alignment + evaluation library
│   ├── alignment.py             # SegmentMetrics, decide_action, global_align
│   ├── evaluation.py            # clip_evaluation_report
│   ├── diarization.py           # pyannote.audio wrapper
│   ├── reranking.py             # Failure analysis, translation re-ranking stub
│   └── vad.py                   # Silero VAD wrapper
├── frontend/                    # Next.js + shadcn/ui web interface
├── pipeline_data/api/           # Runtime artifacts (volume-mounted, not committed)
│   ├── videos/                  # Source MP4s
│   ├── youtube_captions/        # Normalized caption JSON
│   ├── transcriptions/whisper/  # Whisper output JSON
│   ├── translations/argos/      # argostranslate output JSON
│   ├── tts_audio/chatterbox/    # Synthesized WAV per config hash
│   ├── dubbed_captions/         # Target-language WebVTT
│   ├── dubbed_videos/           # Final dubbed MP4 per config hash
│   └── speakers/                # Extracted reference voice clips
├── docker-compose.yml           # All services, nvidia + cpu profiles
├── docs/
│   └── tts-temporal-alignment-research.md   # Literature survey
└── video_registry.yml           # Video catalog (single source of truth)
```

---

## References

1. Lakew, S. M., Federico, M., et al. (2021). *Isochrony-Aware Neural MT for Automatic Dubbing*. arXiv:2112.08548.
2. Lakew, S. M., Federico, M., et al. (2022). *Isometric Machine Translation for Automatic Dubbing*. ICASSP 2022.
3. Federico, M., et al. (2020). *Evaluating and Optimizing Prosodic Alignment for Automatic Dubbing*. Interspeech 2020.
4. Wu, S., Guo, H., Tan, X., et al. (2023). *VideoDubber: Machine Translation with Speech-Aware Length Control*. AAAI 2023.
5. Effendi, J., Virkar, Y., et al. (2022). *Duration Modeling of Neural TTS for Automatic Dubbing*. ICASSP 2022.
6. Microsoft Research (2024). *Total-Duration-Aware Duration Modeling for TTS*. arXiv:2406.04281.
7. Choi, Y., Kim, J., et al. (2025). *Dub-S2ST: Textless Speech-to-Speech Translation for Seamless Dubbing*. EMNLP 2025.
