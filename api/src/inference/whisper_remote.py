"""Remote Whisper backend — delegates to an OpenAI-compatible HTTP endpoint."""

from __future__ import annotations

import logging

import requests

from api.src.inference.base import WhisperBackend

logger = logging.getLogger(__name__)


class RemoteWhisperBackend(WhisperBackend):
    """Sends audio to ``{api_url}/v1/audio/transcriptions`` via HTTP POST."""

    def __init__(self, api_url: str, model_name: str = "Systran/faster-whisper-medium") -> None:
        self._api_url = api_url.rstrip("/")
        self._model_name = model_name

    def transcribe(self, audio_path: str) -> dict:
        """POST the audio file to the remote Whisper service."""
        url = f"{self._api_url}/v1/audio/transcriptions"
        logger.info("Remote Whisper transcription: POST %s", url)

        import mimetypes
        mime = mimetypes.guess_type(audio_path)[0] or "application/octet-stream"
        filename = audio_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

        response = self._post_transcription(url, audio_path, filename, mime)

        if response.status_code in (404, 500):
            # Model was unloaded (idle TTL expired) — reload it and retry once.
            logger.warning(
                "Whisper returned %s; reloading model '%s' and retrying.",
                response.status_code, self._model_name,
            )
            self._reload_model()
            response = self._post_transcription(url, audio_path, filename, mime)

        response.raise_for_status()
        return response.json()

    def _post_transcription(self, url: str, audio_path: str, filename: str, mime: str):
        with open(audio_path, "rb") as f:
            return requests.post(
                url,
                files={"file": (filename, f, mime)},
                data={"model": self._model_name, "response_format": "verbose_json"},
                timeout=600,
            )

    def _reload_model(self) -> None:
        """Ensure the model is downloaded and loaded in speaches memory."""
        import urllib.parse
        model_id = urllib.parse.quote(self._model_name, safe="")
        download_url = f"{self._api_url}/v1/models/{model_id}"
        delete_url = f"{self._api_url}/v1/models/{model_id}"
        load_url = f"{self._api_url}/api/ps/{model_id}"
        try:
            r = requests.post(download_url, timeout=300)
            if r.status_code == 500:
                # Broken model cache (speaches v0.8.0 breaking change)
                logger.warning("Broken model cache; deleting and re-downloading '%s'.", self._model_name)
                requests.delete(delete_url, timeout=30)
                requests.post(download_url, timeout=300)
            requests.post(load_url, timeout=30)
            import time
            time.sleep(2)
        except Exception as exc:
            logger.warning("Failed to reload Whisper model: %s", exc)

    def __repr__(self) -> str:
        return f"<RemoteWhisperBackend url={self._api_url!r}>"
