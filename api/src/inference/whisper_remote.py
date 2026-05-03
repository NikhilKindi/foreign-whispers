"""Remote Whisper backend — delegates to an OpenAI-compatible HTTP endpoint."""

from __future__ import annotations

import logging

import requests

from api.src.inference.base import WhisperBackend

logger = logging.getLogger(__name__)


class RemoteWhisperBackend(WhisperBackend):
    """Sends audio to ``{api_url}/v1/audio/transcriptions`` via HTTP POST."""

    def __init__(self, api_url: str) -> None:
        # Strip trailing slash for consistent URL building.
        self._api_url = api_url.rstrip("/")

    def transcribe(self, audio_path: str) -> dict:
        """POST the audio file to the remote Whisper service."""
        url = f"{self._api_url}/v1/audio/transcriptions"
        logger.info("Remote Whisper transcription: POST %s", url)

        import mimetypes
        mime = mimetypes.guess_type(audio_path)[0] or "application/octet-stream"
        filename = audio_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

        with open(audio_path, "rb") as f:
            response = requests.post(
                url,
                files={"file": (filename, f, mime)},
                data={
                    "model": "Systran/faster-whisper-medium",
                    "response_format": "verbose_json",
                },
                timeout=600,
            )

        response.raise_for_status()
        return response.json()

    def __repr__(self) -> str:
        return f"<RemoteWhisperBackend url={self._api_url!r}>"
