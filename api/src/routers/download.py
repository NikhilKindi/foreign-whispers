"""POST /api/download — download YouTube video + captions (issue by5)."""

import json
import pathlib

from fastapi import APIRouter, HTTPException, Request

from api.src.core.config import settings
from api.src.core.video_registry import get_video
from api.src.schemas.download import CaptionSegment, DownloadRequest, DownloadResponse
from api.src.services.download_service import DownloadService

router = APIRouter(prefix="/api")

_download_service = DownloadService(ui_dir=settings.data_dir)


@router.post("/download", response_model=DownloadResponse)
async def download_endpoint(body: DownloadRequest):
    """Download video and captions, returning video_id and caption segments."""
    import re
    # Try to resolve from the registry first (avoids calling YouTube API)
    m = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})", body.url)
    video_id_from_url = m.group(1) if m else None
    entry = get_video(video_id_from_url) if video_id_from_url else None

    if entry:
        video_id = entry.id
        title = entry.title
        stem = entry.title
    else:
        video_id, title = _download_service.get_video_info(body.url)
        entry = get_video(video_id)
        stem = entry.title if entry else title.replace(":", "")

    videos_dir = settings.videos_dir
    captions_dir = settings.youtube_captions_dir
    videos_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    video_path = videos_dir / f"{stem}.mp4"
    caption_path = captions_dir / f"{stem}.txt"

    # Skip re-download if both files exist
    if not video_path.exists():
        _download_service.download_video(body.url, str(videos_dir), stem)

    if not caption_path.exists():
        _download_service.download_caption(body.url, str(captions_dir), stem)

    segments = _download_service.read_caption_segments(caption_path)

    return DownloadResponse(
        video_id=video_id,
        title=title,
        caption_segments=segments,
    )
