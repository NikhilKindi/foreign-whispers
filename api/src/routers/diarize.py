"""POST /api/diarize/{video_id} - speaker diarization (issue fw-lua)."""

import json
import subprocess

from fastapi import APIRouter, HTTPException

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.schemas.diarize import DiarizeResponse
from api.src.services.alignment_service import AlignmentService
from foreign_whispers.diarization import assign_speakers, extract_speaker_clips

router = APIRouter(prefix="/api")

_alignment_service = AlignmentService(settings=settings)


@router.post("/diarize/{video_id}", response_model=DiarizeResponse)
async def diarize_endpoint(video_id: str):
    """Run speaker diarization on a video's audio track.

    Steps:
    1. Extract audio from video via ffmpeg
    2. Run pyannote diarization
    3. Merge speaker labels into transcription
    4. Cache and return speaker segments
    """
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    diar_dir = settings.diarizations_dir
    diar_dir.mkdir(parents=True, exist_ok=True)
    diar_path = diar_dir / f"{title}.json"

    if diar_path.exists():
        data = json.loads(diar_path.read_text())

        # Back-fill voice clips for caches created before voice cloning
        if not data.get("voice_map"):
            audio_path = diar_dir / f"{title}.wav"
            if audio_path.exists() and data.get("segments"):
                voice_map = extract_speaker_clips(
                    audio_path=str(audio_path),
                    diarization=data["segments"],
                    output_dir=settings.speakers_dir,
                )
                data["voice_map"] = voice_map
                diar_path.write_text(json.dumps(data))

        return DiarizeResponse(
            video_id=video_id,
            speakers=data.get("speakers", []),
            segments=data.get("segments", []),
            skipped=True,
        )

    video_path = settings.videos_dir / f"{title}.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {title}.mp4")

    audio_path = diar_dir / f"{title}.wav"
    result = subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-y", str(audio_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"ffmpeg audio extraction failed: {result.stderr[:500]}")

    diar_segments = _alignment_service.diarize(str(audio_path))
    speakers = sorted(set(s["speaker"] for s in diar_segments)) if diar_segments else ["SPEAKER_00"]

    # Extract per-speaker reference clips for voice cloning
    voice_map = extract_speaker_clips(
        audio_path=str(audio_path),
        diarization=diar_segments,
        output_dir=settings.speakers_dir,
    )

    cache = {"speakers": speakers, "segments": diar_segments, "voice_map": voice_map}
    diar_path.write_text(json.dumps(cache))

    transcript_path = settings.transcriptions_dir / f"{title}.json"
    if transcript_path.exists():
        transcript = json.loads(transcript_path.read_text())
        labeled = assign_speakers(transcript.get("segments", []), diar_segments)
        transcript["segments"] = labeled
        transcript_path.write_text(json.dumps(transcript))

    return DiarizeResponse(video_id=video_id, speakers=speakers, segments=diar_segments)
