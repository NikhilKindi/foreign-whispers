"""Speaker diarization using pyannote.audio.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M2-align).

Optional dependency: pyannote.audio
    pip install pyannote.audio
Requires accepting the pyannote/speaker-diarization-3.1 licence on HuggingFace
and providing an HF token.  Returns empty list with a warning if the dep is
absent or the token is missing.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MIN_CLIP_DURATION_S = 3.0
TARGET_CLIP_DURATION_S = 8.0
MAX_CLIP_DURATION_S = 15.0


def diarize_audio(audio_path: str, hf_token: str | None = None) -> list[dict]:
    """Return speaker-labeled intervals for *audio_path*.

    Returns:
        List of ``{start_s: float, end_s: float, speaker: str}``.
        Empty list when pyannote.audio is absent, token is missing, or diarization fails.
    """
    if not hf_token:
        logger.warning("No HF token provided — diarization skipped.")
        return []

    try:
        from pyannote.audio import Pipeline
    except (ImportError, TypeError):
        logger.warning("pyannote.audio not installed — returning empty diarization.")
        return []

    try:
        pipeline    = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        diarization = pipeline(audio_path)
        return [
            {"start_s": turn.start, "end_s": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
    except Exception as exc:
        logger.warning("Diarization failed for %s: %s", audio_path, exc)
        return []


def assign_speakers(
    segments: list[dict],
    diarization: list[dict],
) -> list[dict]:
    """Assign a speaker label to each transcription segment.

    For each segment, finds the diarization interval with the greatest
    temporal overlap and copies its speaker label. If diarization is
    empty, all segments default to ``SPEAKER_00``.

    Args:
        segments: Whisper-style ``[{id, start, end, text, ...}]``.
        diarization: pyannote-style ``[{start_s, end_s, speaker}]``.

    Returns:
        New list of segment dicts, each with an added ``speaker`` key.
        Original list is not mutated.
    """
    result = []
    for seg in segments:
        new_seg = dict(seg)
        best_speaker = "SPEAKER_00"
        best_overlap = 0.0
        for d in diarization:
            overlap = max(0.0, min(seg["end"], d["end_s"]) - max(seg["start"], d["start_s"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d["speaker"]
        new_seg["speaker"] = best_speaker
        result.append(new_seg)
    return result


def _pick_best_segments(
    speaker_segments: list[dict],
) -> list[dict]:
    """Select the best contiguous segments for a speaker reference clip.

    Strategy: sort by duration descending, greedily accumulate segments
    until we reach TARGET_CLIP_DURATION_S.  Prefers fewer, longer segments
    for a cleaner voice sample (less cross-talk risk).
    """
    sorted_segs = sorted(speaker_segments, key=lambda s: s["end_s"] - s["start_s"], reverse=True)
    picked = []
    total = 0.0
    for seg in sorted_segs:
        dur = seg["end_s"] - seg["start_s"]
        if dur < 0.5:
            continue
        picked.append(seg)
        total += dur
        if total >= TARGET_CLIP_DURATION_S:
            break
    # Sort by time so concatenation is chronological
    picked.sort(key=lambda s: s["start_s"])
    return picked


def extract_speaker_clips(
    audio_path: str,
    diarization: list[dict],
    output_dir: Path,
    target_language: str = "es",
) -> dict[str, str]:
    """Extract per-speaker reference WAV clips from the source audio.

    Uses diarization timestamps to cut the cleanest segments for each
    speaker, concatenates them into a single reference clip (5-15s),
    and saves to output_dir/{target_language}/{SPEAKER_XX}.wav.

    Args:
        audio_path: Path to the source audio WAV (16kHz mono).
        diarization: pyannote-style [{start_s, end_s, speaker}].
        output_dir: Root speakers directory (e.g. pipeline_data/speakers/).
        target_language: Language code for subdirectory (e.g. "es").

    Returns:
        Dict mapping speaker ID to the relative WAV path
        (e.g. {"SPEAKER_00": "es/SPEAKER_00.wav"}).
    """
    if not diarization:
        return {}

    try:
        from pydub import AudioSegment
    except ImportError:
        logger.warning("pydub not installed — speaker clip extraction skipped.")
        return {}

    try:
        source_audio = AudioSegment.from_wav(audio_path)
    except Exception as exc:
        logger.warning("Failed to load audio %s: %s", audio_path, exc)
        return {}

    # Group diarization segments by speaker
    speaker_segments: dict[str, list[dict]] = {}
    for seg in diarization:
        spk = seg["speaker"]
        speaker_segments.setdefault(spk, []).append(seg)

    lang_dir = output_dir / target_language
    lang_dir.mkdir(parents=True, exist_ok=True)

    voice_map: dict[str, str] = {}

    for speaker, segments in speaker_segments.items():
        best = _pick_best_segments(segments)
        if not best:
            continue

        # Extract and concatenate the chosen segments
        combined = AudioSegment.empty()
        for seg in best:
            start_ms = int(seg["start_s"] * 1000)
            end_ms = int(seg["end_s"] * 1000)
            end_ms = min(end_ms, len(source_audio))
            if start_ms >= end_ms:
                continue
            combined += source_audio[start_ms:end_ms]

        total_s = len(combined) / 1000.0
        if total_s < MIN_CLIP_DURATION_S:
            logger.info(
                "Speaker %s has only %.1fs of audio, below %.1fs minimum — skipping.",
                speaker, total_s, MIN_CLIP_DURATION_S,
            )
            continue

        # Trim to MAX_CLIP_DURATION_S
        if total_s > MAX_CLIP_DURATION_S:
            combined = combined[: int(MAX_CLIP_DURATION_S * 1000)]

        clip_path = lang_dir / f"{speaker}.wav"
        combined.export(str(clip_path), format="wav")
        rel_path = f"{target_language}/{speaker}.wav"
        voice_map[speaker] = rel_path
        logger.info(
            "Extracted %.1fs reference clip for %s → %s",
            len(combined) / 1000.0, speaker, rel_path,
        )

    return voice_map
