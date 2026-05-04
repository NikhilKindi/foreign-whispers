"""Foreign Whispers FastAPI application."""

import logging
import urllib.error
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.src.core.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lazy model loading — models are loaded on first use, not at startup.

    This avoids blocking startup in Docker Compose where Whisper and TTS
    inference may be handled by separate containers (speaches, Chatterbox).
    """
    app.state._whisper_model = None
    app.state._tts_model = None

    # Pre-load the Whisper model in speaches so /v1/audio/transcriptions is ready.
    # speaches v0.8.0+: first POST /v1/models/{id} to download, then POST /api/ps/{id} to load.
    if settings.whisper_api_url:
        import time as _time
        import urllib.parse
        import urllib.request
        base = settings.whisper_api_url.rstrip("/")
        model_id = urllib.parse.quote(settings.whisper_model, safe="")
        download_url = f"{base}/v1/models/{model_id}"
        load_url = f"{base}/api/ps/{model_id}"
        delete_url = f"{base}/v1/models/{model_id}"

        def _speaches_request(url: str, method: str, timeout: int = 30):
            req = urllib.request.Request(url, data=b"", method=method)
            return urllib.request.urlopen(req, timeout=timeout)

        for attempt in range(1, 7):
            try:
                # Ensure model is downloaded (no-op if already present)
                try:
                    _speaches_request(download_url, "POST", timeout=300)
                    logger.info("Whisper model '%s' downloaded in speaches.", settings.whisper_model)
                except urllib.error.HTTPError as exc:
                    if exc.code == 500:
                        # Broken model cache (v0.8.0 breaking change) — delete and re-download
                        logger.warning("Broken model cache detected; deleting and re-downloading '%s'.", settings.whisper_model)
                        try:
                            _speaches_request(delete_url, "DELETE")
                        except Exception:
                            pass
                        _speaches_request(download_url, "POST", timeout=300)
                        logger.info("Whisper model '%s' re-downloaded.", settings.whisper_model)
                # Load model into memory
                try:
                    _speaches_request(load_url, "POST")
                    logger.info("Whisper model '%s' loaded in speaches.", settings.whisper_model)
                except urllib.error.HTTPError as exc:
                    if exc.code == 409:
                        logger.info("Whisper model '%s' already loaded in speaches.", settings.whisper_model)
                    else:
                        logger.warning("Speaches returned %s loading model.", exc.code)
                break
            except Exception as exc:
                if attempt < 6:
                    logger.info("Speaches not ready yet (attempt %s/6), retrying in 5s…", attempt)
                    _time.sleep(5)
                else:
                    logger.warning("Could not pre-load Whisper model after 6 attempts: %s", exc)

    logger.info("Application ready (models will load on first use).")

    # Configure Logfire if a write token is available
    if settings.logfire_write_token:
        try:
            import logfire
            logfire.configure(
                write_token=settings.logfire_write_token,
                service_name="foreign-whispers",
            )
            logfire.instrument_fastapi(app)
            logger.info("Logfire tracing enabled.")
        except ImportError:
            logger.info("Logfire not installed — tracing disabled.")

    yield

    # Cleanup
    if app.state._whisper_model is not None:
        del app.state._whisper_model
    if app.state._tts_model is not None:
        del app.state._tts_model
    logger.info("Models unloaded.")


def get_whisper_model(app):
    """Lazy-load Whisper backend on first use.

    Uses the remote speaches GPU container when whisper_api_url is configured,
    otherwise falls back to local Whisper.
    """
    if app.state._whisper_model is None:
        if settings.whisper_api_url:
            from api.src.inference.whisper_remote import RemoteWhisperBackend
            app.state._whisper_model = RemoteWhisperBackend(settings.whisper_api_url, settings.whisper_model)
            logger.info("Using remote Whisper at %s (model: %s)", settings.whisper_api_url, settings.whisper_model)
        else:
            logger.info("Loading local Whisper model (%s)...", settings.whisper_model)
            import whisper
            app.state._whisper_model = whisper.load_model(settings.whisper_model)
            logger.info("Whisper model loaded.")
    return app.state._whisper_model


def get_tts_model(app):
    """Lazy-load TTS engine on first use.

    Uses the Chatterbox GPU server (with retry + Coqui fallback) — same
    engine that tts_engine.py uses internally via _get_tts_engine().
    """
    if app.state._tts_model is None:
        from api.src.services.tts_engine import _get_tts_engine
        app.state._tts_model = _get_tts_engine()
        logger.info("TTS engine ready: %r", app.state._tts_model)
    return app.state._tts_model


def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI instance."""
    app = FastAPI(
        title=settings.app_title,
        lifespan=lifespan,
    )

    if settings.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    from api.src.routers.download import router as download_router
    from api.src.routers.transcribe import router as transcribe_router
    from api.src.routers.translate import router as translate_router
    from api.src.routers.tts import router as tts_router
    from api.src.routers.stitch import router as stitch_router

    app.include_router(download_router)
    app.include_router(transcribe_router)
    app.include_router(translate_router)
    app.include_router(tts_router)
    app.include_router(stitch_router)
    from api.src.routers.eval import router as eval_router
    app.include_router(eval_router)
    from api.src.routers.diarize import router as diarize_router
    app.include_router(diarize_router)

    @app.get("/healthz")
    async def healthz():
        """Health check endpoint."""
        return {"status": "ok"}

    @app.get("/api/videos")
    async def list_videos():
        """Return the video catalog from video_registry.yml."""
        from api.src.core.video_registry import get_all_videos
        return [
            {"id": v.id, "title": v.title, "url": v.url}
            for v in get_all_videos()
        ]

    return app


app = create_app()
