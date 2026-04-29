"""POST /api/translate/{video_id} — translation endpoint (issue c0m)."""

import json
import pathlib

from fastapi import APIRouter, HTTPException, Query

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.services.translation_service import TranslationService

router = APIRouter(prefix="/api")

_translation_service = TranslationService(ui_dir=settings.data_dir)


@router.post("/translate/{video_id}")
async def translate_endpoint(
    video_id: str,
    target_language: str = Query(default="es"),
):
    """Translate a single video's transcript (fixes issue 5ss — no directory sweep)."""
    raw_dir = settings.transcriptions_dir
    out_dir = settings.translations_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    out_path = out_dir / f"{title}.json"

    src_path = raw_dir / f"{title}.json"
    transcript = json.loads(src_path.read_text())

    # Skip if already translated — but propagate speaker labels from transcription
    if out_path.exists():
        data = json.loads(out_path.read_text())
        src_segs = transcript.get("segments", [])
        out_segs = data.get("segments", [])
        if src_segs and out_segs and "speaker" in src_segs[0] and "speaker" not in out_segs[0]:
            for i, seg in enumerate(out_segs):
                if i < len(src_segs):
                    seg["speaker"] = src_segs[i].get("speaker", "SPEAKER_00")
            data["segments"] = out_segs
            out_path.write_text(json.dumps(data))
        return {
            "video_id": video_id,
            "target_language": target_language,
            "text": data.get("text", ""),
            "segments": data.get("segments", []),
        }

    _translation_service.install_language_pack("en", target_language)
    translated = _translation_service.translate_transcript(transcript, "en", target_language)

    out_path.write_text(json.dumps(translated))

    return {
        "video_id": video_id,
        "target_language": target_language,
        "text": translated.get("text", ""),
        "segments": translated.get("segments", []),
    }
