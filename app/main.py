"""
fake-subsai-asr-gateway — Bazarr ASR Gateway with EN→LV Translation Chain

Provides Bazarr-compatible ASR endpoints backed by the existing Latvian
infrastructure (asr-transcription-lv + back-translator-lv).

Endpoint overview:
  POST /detect-language    — openai-whisper-asr-webservice compat: detect audio lang
  POST /asr               — Smart routing: src×target lang matrix → SRT/VTT/TXT/JSON
  POST /asr-translate-lv  — EN audio → LV SRT  (Whisper + NLLB batch MT)
  POST /asr-lv-native     — LV audio → LV SRT  (dedicated Latvian Whisper)
  GET  /health            — Service health + dependency status
  GET  /status            — Human-readable pipeline status page (auto-refresh 3s)

Dependency URLs (configurable via env vars):
  ASR_URL            — asr-transcription-lv REST endpoint
  BACK_TRANSLATOR_URL — back-translator-lv REST endpoint

Protocol compatibility:
  - Implements openai-whisper-asr-webservice API shape for Bazarr's whisperai provider
  - POST /detect-language returns {"detected_language": "english", "language_code": "en"}
  - POST /asr accepts audio_file (primary) or file (legacy alias) form field
  - Query params: task, language (target subtitle lang), output (srt|vtt|txt|json), encode
  - ISO 639-2 three-letter codes (lav, eng, etc.) are normalized to 639-1 at entry

All heavy lifting (GPU Whisper, GPU NLLB) happens in the downstream services.
This gateway is pure HTTP coordination — no GPU, no models.

Host port: 9001
"""

from __future__ import annotations

import asyncio
import collections
import ctypes
import datetime
import functools
import gc
import html
import io
import json
import logging
import os
import struct
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from math import gcd
from typing import Annotated, Optional

import httpx
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from forced_aligner_client import ENABLE_FORCED_ALIGN, ForcedAlignerClient
from mt_translator import MTTranslator
from scipy.signal import resample_poly
from subtitle_utils import segments_to_srt, segments_to_vtt
from vocal_isolator_client import ENABLED as VOCAL_ISO_ENABLED
from vocal_isolator_client import VocalIsolatorClient

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("fake-subsai-gateway")

# ── Pipeline serialization ──────────────────────────────────────────────────────
# Bazarr fires concurrent requests for batch operations (e.g. 6 episodes of a TV
# show simultaneously). Without serialization, each request buffers its own
# audio + holds memory + competes for GPU/CPU, which on a 12 GB RAM host means
# kernel swap thrashing → freeze.
# This lock ensures only one /asr-family request processes at a time. Bazarr's
# 3600s gateway timeout is generous enough for 6+ requests in series.
# /detect-language is NOT serialized — it's fast (<2s with truncation) and
# Bazarr depends on it returning quickly to even attempt /asr.
_pipeline_lock = asyncio.Lock()

# ── Pipeline status tracking ────────────────────────────────────────────────────
# All updates are safe under FastAPI's single-threaded async event loop.
# No threading.Lock needed — dict/list mutations inside async def are atomic.

# Currently-processing request (None = IDLE).
# Keys: endpoint, video_file, started_at (float monotonic), stage (str)
_in_progress_request: dict | None = None

# Waiting requests (in FIFO order they acquired the queue spot).
# Each entry: {endpoint, video_file, queued_at (float monotonic)}
_queue: list[dict] = []

# Ring buffer of the last 20 completed requests.
# Each entry: {endpoint, video_file, started_at, ended_at, status, stage_at_failure}
_recent_completions: collections.deque = collections.deque(maxlen=20)

# ── Configuration ──────────────────────────────────────────────────────────────

# asr-transcription-lv: host port 8101, container port 8011
ASR_URL = os.environ.get("ASR_URL", "http://localhost:8101")

# back-translator-lv: host port 8104, container port 8008
BACK_TRANSLATOR_URL = os.environ.get("BACK_TRANSLATOR_URL", "http://localhost:8104")

# gpu-supervisor-lv: host port 8202 (gateway uses network_mode: host)
GPU_SUPERVISOR_URL = os.environ.get("GPU_SUPERVISOR_URL", "http://localhost:8202")

# HTTP client timeout for ASR calls (long — 2-hr movie can take 10+ min on P4 GPU)
ASR_TIMEOUT = float(os.environ.get("ASR_TIMEOUT_SECONDS", "900.0"))

# HTTP client timeout for translation calls
MT_TIMEOUT = float(os.environ.get("MT_TIMEOUT_SECONDS", "30.0"))

# Optional API key authentication.
# When API_KEY env var is set (non-empty), every request to a non-public endpoint
# must include header X-API-Key: <value>. When unset (default), the gateway is
# open — suitable for a private homelab network behind a firewall. Set this
# whenever the gateway is reachable from an untrusted network.
API_KEY = os.getenv("API_KEY", "")

# Paths that bypass API key enforcement so health checks, docs, and the status
# board remain reachable for monitoring without leaking credentials.
_API_KEY_EXEMPT_PATHS: frozenset[str] = frozenset(
    {
        "/health",
        "/status",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
)

# ── Language code normalization ────────────────────────────────────────────────
#
# Bazarr uses ISO 639-2 three-letter codes internally (lav, eng, etc.) but
# Whisper and openai-whisper-asr-webservice use ISO 639-1 two-letter codes
# (lv, en, etc.). Bazarr's whisperai provider converts before calling us, but
# we normalize defensively at every entry point so both forms work.
#
# NLLB-200 uses its own FLORES-200 codes (eng_Latn, lvs_Latn, etc.):
# Whisper 639-1 → NLLB FLORES-200 mapping is in _WHISPER_TO_NLLB below.

_ISO3_TO_ISO1: dict[str, str] = {
    "lav": "lv",  # Latvian
    "eng": "en",  # English
    "rus": "ru",  # Russian
    "fra": "fr",  # French (bibliographic)
    "fre": "fr",  # French (terminology)
    "deu": "de",  # German (bibliographic)
    "ger": "de",  # German (terminology)
    "spa": "es",  # Spanish
    "ita": "it",  # Italian
    "por": "pt",  # Portuguese
    "jpn": "ja",  # Japanese
    "zho": "zh",  # Chinese (bibliographic)
    "chi": "zh",  # Chinese (terminology)
    "lit": "lt",  # Lithuanian
    "est": "et",  # Estonian
    "fin": "fi",  # Finnish
    "pol": "pl",  # Polish
    "ukr": "uk",  # Ukrainian
}

# Maps Whisper's 2-letter code → NLLB-200 FLORES-200 code for back-translator
_WHISPER_TO_NLLB: dict[str, str] = {
    "en": "eng_Latn",
    "lv": "lvs_Latn",
    "ru": "rus_Cyrl",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "lt": "lit_Latn",
    "et": "est_Latn",
    "fi": "fin_Latn",
    "pl": "pol_Latn",
    "uk": "ukr_Cyrl",
    "ja": "jpn_Jpan",
    "zh": "zho_Hans",
}

# Maps Whisper 2-letter code → human-readable language name (for detect-language)
_LANG_CODE_TO_NAME: dict[str, str] = {
    "af": "afrikaans",
    "ar": "arabic",
    "hy": "armenian",
    "az": "azerbaijani",
    "be": "belarusian",
    "bs": "bosnian",
    "bg": "bulgarian",
    "ca": "catalan",
    "zh": "chinese",
    "hr": "croatian",
    "cs": "czech",
    "da": "danish",
    "nl": "dutch",
    "en": "english",
    "et": "estonian",
    "fi": "finnish",
    "fr": "french",
    "gl": "galician",
    "de": "german",
    "el": "greek",
    "he": "hebrew",
    "hi": "hindi",
    "hu": "hungarian",
    "is": "icelandic",
    "id": "indonesian",
    "it": "italian",
    "ja": "japanese",
    "kn": "kannada",
    "kk": "kazakh",
    "ko": "korean",
    "lv": "latvian",
    "lt": "lithuanian",
    "mk": "macedonian",
    "ms": "malay",
    "mr": "marathi",
    "mi": "maori",
    "ne": "nepali",
    "no": "norwegian",
    "fa": "persian",
    "pl": "polish",
    "pt": "portuguese",
    "ro": "romanian",
    "ru": "russian",
    "sr": "serbian",
    "sk": "slovak",
    "sl": "slovenian",
    "es": "spanish",
    "sw": "swahili",
    "sv": "swedish",
    "tl": "tagalog",
    "ta": "tamil",
    "th": "thai",
    "tr": "turkish",
    "uk": "ukrainian",
    "ur": "urdu",
    "vi": "vietnamese",
    "cy": "welsh",
    "yi": "yiddish",
    "nn": "norwegian nynorsk",
    "nb": "norwegian bokmal",
    "ka": "georgian",
    "sq": "albanian",
    "eu": "basque",
    "lb": "luxembourgish",
    "mg": "malagasy",
    "mt": "maltese",
    "mn": "mongolian",
    "my": "burmese",
    "km": "khmer",
    "lo": "lao",
    "si": "sinhala",
    "gu": "gujarati",
    "bn": "bengali",
    "pa": "punjabi",
    "te": "telugu",
    "ml": "malayalam",
    "am": "amharic",
    "tk": "turkmen",
    "uz": "uzbek",
}


def _normalize_lang(code: str | None) -> str | None:
    """
    Normalize an ISO 639-2 three-letter code to ISO 639-1 two-letter code.
    Two-letter codes and None pass through unchanged.
    Unknown codes pass through unchanged (let Whisper handle or reject).
    """
    if code is None:
        return None
    code = code.strip().lower()
    return _ISO3_TO_ISO1.get(code, code)


def _format_output(
    segments: list[dict],
    output_format: str,
    detected_language: str | None = None,
) -> Response:
    """
    Render segments in the requested output format and return a Response with
    the correct Content-Type header.

    Supported formats (matching openai-whisper-asr-webservice):
      srt  → application/x-subrip
      vtt  → text/vtt
      txt  → text/plain (plain concatenated text, no timestamps)
      json → application/json (Whisper-style JSON with segments array)
    """
    fmt = (output_format or "srt").lower()

    if fmt == "srt":
        body = segments_to_srt(segments)
        return Response(content=body, media_type="application/x-subrip")

    if fmt == "vtt":
        body = segments_to_vtt(segments)
        return Response(content=body, media_type="text/vtt")

    if fmt == "txt":
        lines = [seg.get("text", "").strip() for seg in segments if seg.get("text", "").strip()]
        body = "\n".join(lines)
        return Response(content=body, media_type="text/plain")

    if fmt == "json":
        payload = {
            "text": " ".join(
                seg.get("text", "").strip() for seg in segments if seg.get("text", "").strip()
            ),
            "language": detected_language or "",
            "segments": segments,
        }
        return Response(content=json.dumps(payload), media_type="application/json")

    # Unknown format — fall back to SRT (safe default for Bazarr)
    log.warning("Unknown output format %r, falling back to SRT", output_format)
    body = segments_to_srt(segments)
    return Response(content=body, media_type="application/x-subrip")


# ── Application state ──────────────────────────────────────────────────────────

_asr_client: httpx.AsyncClient | None = None
_mt: MTTranslator | None = None

# ── Lifespan ───────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise shared HTTP client and MT translator; release on shutdown."""
    global _asr_client, _mt

    log.info("Gateway starting — ASR_URL=%s  BACK_TRANSLATOR_URL=%s", ASR_URL, BACK_TRANSLATOR_URL)

    _asr_client = httpx.AsyncClient(timeout=httpx.Timeout(ASR_TIMEOUT))
    _mt = MTTranslator(url=BACK_TRANSLATOR_URL, timeout=MT_TIMEOUT)

    log.info("Gateway ready on port 9001.")
    yield

    await _asr_client.aclose()
    _asr_client = None
    _mt = None
    log.info("Gateway shutdown complete.")


# ── GPU supervisor session helper ──────────────────────────────────────────────


class GpuYieldError(Exception):
    """
    Raised when the GPU supervisor defers a Tier 3 claim because a higher-priority
    service is currently active (Policy B, 2026-05-06).

    The caller should propagate this as HTTP 503 with Retry-After so Bazarr
    retries on its natural schedule rather than stacking requests.
    """

    def __init__(self, retry_after: int, active: list[str]) -> None:
        self.retry_after = retry_after
        self.active = active
        super().__init__(f"GPU busy: {active}")


async def _supervisor_post(
    client: httpx.AsyncClient,
    path: str,
) -> httpx.Response:
    """POST to the supervisor; returns the raw response (no raise_for_status)."""
    return await client.post(f"{GPU_SUPERVISOR_URL}{path}")


@asynccontextmanager
async def _supervisor_session(service_names: list[str]) -> AsyncGenerator[None, None]:
    """
    Claim GPU services from supervisor on entry; release on exit (always).

    Usage:
        async with _supervisor_session(["asr-transcription-lv", "back-translator-lv"]):
            # both services are claimed for the duration

    Error handling:
      - 503 with reason=tier3_yield: raises GpuYieldError (Policy B). Any claims
        already acquired before the yield are released before raising so we don't
        leak refcounts. Callers must convert GpuYieldError to HTTP 503 for Bazarr.
      - Connection errors / timeouts / other 5xx: log warning and proceed (graceful
        degradation — supervisor unreachable doesn't block the pipeline).

    The yield-aware path is intentional: when omnivoice-lv or fluency-gate is active
    (user is in an interactive session), Bazarr's Tier 3 pipeline waits politely.
    Bazarr's natural retry loop handles the 503 + Retry-After header.
    """
    claimed: list[str] = []
    async with httpx.AsyncClient(timeout=5.0) as sup_client:
        for name in service_names:
            try:
                resp = await _supervisor_post(sup_client, f"/claim/{name}")
                if resp.status_code == 503:
                    body = (
                        resp.json().get("detail", {})
                        if resp.headers.get("content-type", "").startswith("application/json")
                        else {}
                    )
                    if isinstance(body, dict) and body.get("reason") == "tier3_yield":
                        # Release any services claimed before this point
                        for already_claimed in claimed:
                            try:
                                await _supervisor_post(sup_client, f"/release/{already_claimed}")
                                log.info(
                                    "supervisor: released %s (pre-yield cleanup)", already_claimed
                                )
                            except Exception as rel_exc:
                                log.warning(
                                    "supervisor: cleanup release %s failed: %s",
                                    already_claimed,
                                    rel_exc,
                                )
                        raise GpuYieldError(
                            retry_after=body.get("retry_after_seconds", 60),
                            active=body.get("active_higher_priority", []),
                        )
                    # Other 503 — degrade gracefully
                    log.warning("supervisor: /claim/%s returned 503 (non-yield) — proceeding", name)
                else:
                    resp.raise_for_status()
                    claimed.append(name)
                    log.info("supervisor: claimed %s", name)
            except GpuYieldError:
                raise
            except Exception as exc:
                log.warning("supervisor: could not claim %s (%s) — proceeding", name, exc)

    try:
        yield
    finally:
        async with httpx.AsyncClient(timeout=5.0) as sup_client:
            for name in claimed:
                try:
                    r = await _supervisor_post(sup_client, f"/release/{name}")
                    r.raise_for_status()
                    log.info("supervisor: released %s", name)
                except Exception as exc:
                    log.warning("supervisor: could not release %s (%s)", name, exc)


# ── FastAPI application ────────────────────────────────────────────────────────

app = FastAPI(
    title="fake-subsai-asr-gateway",
    description=(
        "Bazarr-compatible ASR gateway (openai-whisper-asr-webservice protocol). "
        "Routes subtitle requests to asr-transcription-lv (Whisper) and "
        "back-translator-lv (NLLB-200).\n\n"
        "**Endpoints:**\n\n"
        "- `POST /detect-language` — Detect audio language (Bazarr calls this first)\n"
        "- `POST /asr` — Smart routing: transcribe/translate based on src×target lang\n"
        "- `POST /asr-translate-lv` — EN audio → LV SRT (Whisper + MT chain)\n"
        "- `POST /asr-lv-native` — LV audio → LV SRT (Latvian Whisper)\n"
        "- `GET /health` — Dependency health check"
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Optional API key middleware ────────────────────────────────────────────────


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Why: Allow the same Docker image to run in two trust modes — open (homelab
    LAN behind a firewall) or authenticated (any deployment exposed to an
    untrusted network) — controlled by a single env var, with no code change.
    What: If API_KEY env var is non-empty, require header X-API-Key on every
    request whose path is not in _API_KEY_EXEMPT_PATHS (/health, /status, /docs,
    /redoc, /openapi.json). Mismatches return HTTP 401. If API_KEY is empty,
    every request is allowed through (current homelab behaviour).
    Test: Set API_KEY=secret, POST /asr without the header → expect 401; POST
    with X-API-Key: secret → expect normal pipeline response. Leave API_KEY
    unset and POST /asr without the header → expect normal pipeline response.
    """
    if API_KEY and request.url.path not in _API_KEY_EXEMPT_PATHS:
        if request.headers.get("X-API-Key", "") != API_KEY:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


# ── Exception handler ──────────────────────────────────────────────────────────


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled exception for %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. See service logs for details."},
    )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _get_client() -> httpx.AsyncClient:
    if _asr_client is None:
        raise HTTPException(status_code=503, detail="Gateway not ready.")
    return _asr_client


def _get_mt() -> MTTranslator:
    if _mt is None:
        raise HTTPException(status_code=503, detail="MT translator not ready.")
    return _mt


def _downsample_vocals_for_whisper(vocals_wav: bytes) -> bytes:
    """
    Passthrough / sanity-check after the architectural fix (2026-05-06).

    vocal-isolator-lv now performs the 44.1 kHz stereo float32 → 16 kHz mono
    int16 conversion internally when the caller sends output_format=int16_mono_16000.
    The gateway therefore receives an already-correct WAV and this function
    should be a no-op for all normal traffic.

    Sanity-check: read the WAV header via soundfile.  If the audio is already
    16 kHz mono int16 (the expected post-fix format), return it unchanged.
    If somehow it arrives in the old 44.1 kHz stereo float32 format (e.g. a
    legacy cache hit from before the fix, or a fallback caller that didn't send
    output_format), perform the conversion here and log a warning so the
    discrepancy is visible in logs without breaking the pipeline.
    """
    info = sf.info(io.BytesIO(vocals_wav))
    if info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16":
        # Already in the correct format — passthrough, no allocation.
        return vocals_wav

    # Unexpected format — defensive fallback (preserves rollback safety).
    log.warning(
        "_downsample_vocals_for_whisper: received unexpected format "
        "samplerate=%d channels=%d subtype=%s size=%d bytes — "
        "performing in-gateway downsample (check vocal-isolator output_format param)",
        info.samplerate,
        info.channels,
        info.subtype,
        len(vocals_wav),
    )
    audio, src_sr = sf.read(io.BytesIO(vocals_wav), dtype="float32", always_2d=True)
    # audio shape: (samples, channels)

    # Mono-mix: average channels
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1, keepdims=True)

    # Downsample to 16 kHz using polyphase resampling
    target_sr = 16000
    if src_sr != target_sr:
        g = gcd(src_sr, target_sr)
        audio = resample_poly(audio, target_sr // g, src_sr // g, axis=0).astype(np.float32)

    # Convert float32 → int16 (Whisper native)
    audio_int16 = (audio.flatten() * 32767.0).clip(-32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    sf.write(buf, audio_int16, target_sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# Chunk size for sending audio to ASR.  5 min at 16 kHz mono int16 ≈ 9.6 MB —
# well under the 200 MB ASR body limit.  Overlap avoids cutting words at chunk
# boundaries; overlapping segments are deduplicated after stitching.
_WHISPER_CHUNK_SECONDS = 300.0  # 5 minutes
_WHISPER_OVERLAP_SECONDS = 2.0  # 2-second cross-fade zone


async def _whisper_chunked(
    audio_16k_int16_wav: bytes,
    language: Optional[str],
    task: str,
    client: httpx.AsyncClient,
) -> tuple[list[dict], str]:
    """
    Split already-downsampled 16 kHz mono int16 WAV into ~5-min chunks,
    transcribe each with _call_asr, adjust timestamps to absolute, and
    return (concatenated_segments, detected_language).

    Uses a 2-second overlap between chunks so Whisper doesn't drop words at
    boundaries; segments that fall entirely within the overlap region of the
    previous chunk are discarded after stitching.

    Updates _in_progress_request["stage"] to "whisper-chunk-N/T" while running
    so the /status page shows per-chunk progress.
    """
    audio, sr = sf.read(io.BytesIO(audio_16k_int16_wav), dtype="int16", always_2d=False)
    total_samples = len(audio)
    chunk_samples = int(_WHISPER_CHUNK_SECONDS * sr)
    overlap_samples = int(_WHISPER_OVERLAP_SECONDS * sr)
    step_samples = chunk_samples - overlap_samples

    # Pre-compute chunk count for status labels
    n_chunks = max(1, (total_samples + step_samples - 1) // step_samples)

    all_segments: list[dict] = []
    detected_lang: str = ""
    chunk_idx = 0
    chunk_start = 0

    while chunk_start < total_samples:
        chunk_end = min(chunk_start + chunk_samples, total_samples)
        chunk_audio = audio[chunk_start:chunk_end]

        # Update status page with per-chunk progress
        if _in_progress_request is not None:
            _in_progress_request["stage"] = f"whisper-chunk-{chunk_idx + 1}/{n_chunks}"

        # Encode chunk as WAV
        chunk_buf = io.BytesIO()
        sf.write(chunk_buf, chunk_audio, sr, format="WAV", subtype="PCM_16")
        chunk_wav = chunk_buf.getvalue()

        chunk_offset_s = chunk_start / sr
        log.info(
            "whisper-chunk %d/%d  [%.1f-%.1fs  %.1fs total]  size=%d bytes",
            chunk_idx + 1,
            n_chunks,
            chunk_offset_s,
            chunk_end / sr,
            total_samples / sr,
            len(chunk_wav),
        )

        chunk_segments, chunk_lang = await _call_asr(client, chunk_wav, "chunk.wav", language, task)

        # Keep the first non-empty detected language
        if not detected_lang and chunk_lang:
            detected_lang = chunk_lang

        # Shift chunk-relative timestamps → absolute file timestamps
        for seg in chunk_segments:
            seg["start"] = seg.get("start", 0.0) + chunk_offset_s
            seg["end"] = seg.get("end", 0.0) + chunk_offset_s
            for word in seg.get("words", []) or []:
                word["start"] = word.get("start", 0.0) + chunk_offset_s
                word["end"] = word.get("end", 0.0) + chunk_offset_s

        # Dedup: drop segments from the overlap zone already covered by the
        # previous chunk (i.e. start time is before the previous chunk's last
        # segment end, with 0.5 s tolerance).
        if all_segments and chunk_idx > 0:
            last_end = all_segments[-1].get("end", 0.0)
            chunk_segments = [s for s in chunk_segments if s.get("start", 0.0) >= last_end - 0.5]

        all_segments.extend(chunk_segments)

        chunk_idx += 1
        if chunk_end >= total_samples:
            break
        chunk_start += step_samples

    return all_segments, detected_lang


async def _call_asr(
    client: httpx.AsyncClient,
    audio_bytes: bytes,
    filename: str,
    language: str | None,
    task: str = "transcribe",
) -> tuple[list[dict], str]:
    """
    Call asr-transcription-lv POST /transcribe and return (segments, detected_lang).

    Each segment is a dict with at least {start: float, end: float, text: str}.
    Pass language=None to let Whisper auto-detect the source language.
    Returns (segments list, detected_language_code) where the code is ISO 639-1.
    Raises HTTPException on failure.
    """
    # If the caller passed raw PCM (encode=false, Bazarr convention), wrap it in a
    # minimal RIFF/WAVE header so soundfile/pydub in asr-transcription-lv can decode it.
    wav_bytes = _ensure_wav(audio_bytes)
    files = {"file": (filename, wav_bytes, "audio/wav")}
    data: dict[str, str] = {
        "words": "true",  # enables per-word DTW timestamps for tighter cue boundaries
        "vad_filter": "true",
        "task": task,
    }
    # Pass language=auto to trigger Whisper's auto-detect; omit if None/auto
    if language is not None and language.lower() not in {"", "auto", "detect"}:
        data["language"] = language

    try:
        response = await client.post(f"{ASR_URL}/transcribe", files=files, data=data)
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        log.error("ASR timeout after %.0f s: %s", ASR_TIMEOUT, exc)
        raise HTTPException(status_code=504, detail="ASR service timed out.") from exc
    except httpx.HTTPStatusError as exc:
        log.error("ASR HTTP error %d: %s", exc.response.status_code, exc.response.text[:200])
        raise HTTPException(
            status_code=502,
            detail=f"ASR service returned {exc.response.status_code}.",
        ) from exc
    except httpx.RequestError as exc:
        log.error("ASR request error: %s", exc)
        raise HTTPException(status_code=502, detail="Cannot reach ASR service.") from exc

    payload = response.json()
    detected_lang = payload.get("language", "")  # ISO 639-1 two-letter code from Whisper
    segments: list[dict] = payload.get("segments", [])
    if not segments and payload.get("text"):
        # Fallback: wrap full text as a single segment (no timing info)
        segments = [{"start": 0.0, "end": 0.0, "text": payload["text"]}]
    return segments, detected_lang


async def _transcribe_and_translate_to_lv(
    client: httpx.AsyncClient,
    mt: MTTranslator,
    audio_bytes: bytes,
    filename: str,
    src_lang: str | None,
    asr_task: str = "transcribe",
) -> list[dict]:
    """
    Transcribe audio (in src_lang) then MT-translate all segments to Latvian.

    This is the shared EN→LV (and other→LV) pipeline:
    1. Whisper transcription in src_lang (or auto-detect) via chunked path
    2. Batch NLLB translation of each segment cue → Latvian
    Timecodes are preserved exactly — translation is cue-by-cue, not concatenated.
    """
    segments, detected = await _whisper_chunked(
        audio_bytes, language=src_lang, task=asr_task, client=client
    )
    actual_src = src_lang or detected  # use detected if src not specified
    log.info(
        "_transcribe_and_translate_to_lv: %d segments, src=%s, detected=%s",
        len(segments),
        src_lang,
        detected,
    )

    # Determine NLLB source code — fall back to eng_Latn if unknown
    nllb_src = _WHISPER_TO_NLLB.get(actual_src or "", "eng_Latn")
    log.info("MT: %s → lvs_Latn (%d cues)", nllb_src, len(segments))

    # Translate cue-by-cue (in batches internally) — timecodes unchanged.
    # Run in executor: MTTranslator uses synchronous `requests`, which would
    # otherwise block the asyncio event loop for the full MT duration (and
    # while holding a supervisor claim).
    loop = asyncio.get_running_loop()
    translated_segments = await loop.run_in_executor(
        None,
        functools.partial(
            mt.translate_segments,
            segments,
            source_lang=nllb_src,
            target_lang="lvs_Latn",
        ),
    )
    return translated_segments


_ALLOWED_AUDIO_EXT = {"flac", "m4a", "mp3", "mp4", "ogg", "wav", "webm"}


def _safe_audio_filename(upload: UploadFile) -> str:
    """
    Return a filename guaranteed to have a recognised audio extension.

    Bazarr sends the multipart field with no filename (or an empty/extensionless
    name) so the downstream ASR service sees '.' as the extension and rejects
    with HTTP 415. We fall back to 'audio.wav' because Bazarr always pre-extracts
    to WAV before calling us.
    """
    raw = (upload.filename or "").strip()
    ext = os.path.splitext(raw)[1].lower().lstrip(".")
    return raw if ext in _ALLOWED_AUDIO_EXT else "audio.wav"


def _wrap_pcm_as_wav(
    pcm_bytes: bytes,
    sample_rate: int = 16000,
    channels: int = 1,
    bits: int = 16,
) -> bytes:
    """Wrap raw PCM bytes as a minimal WAV file (44-byte RIFF/WAVE header + PCM data).

    Assumes 16-bit signed little-endian mono at 16 kHz — the format that
    openai-whisper-asr-webservice mandates when encode=false is passed.
    """
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    data_size = len(pcm_bytes)
    chunk_size = 36 + data_size  # 4 (WAVE) + 24 (fmt chunk) + 8 (data header) + data_size
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits,
        b"data",
        data_size,
    )
    return header + pcm_bytes


def _ensure_wav(raw: bytes) -> bytes:
    """Pass through real RIFF/WAVE containers; wrap raw PCM as WAV otherwise.

    Bazarr's whisperai provider sends encode=false, meaning the upload bytes are
    raw signed 16-bit little-endian PCM at 16 kHz mono with NO RIFF header.
    asr-transcription-lv (soundfile/pydub) requires a container format, so we
    prepend the minimal 44-byte WAV header here before forwarding.
    """
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return raw  # already a valid WAV container — pass through unchanged
    return _wrap_pcm_as_wav(raw)


def _truncate_audio_for_detection(raw: bytes, seconds: int = 30) -> bytes:
    """Return at most `seconds` of audio for language detection.

    Whisper only uses the first ~30 s of audio for language identification.
    Sending the full file (e.g. a 144 MB 75-minute WAV) forces asr-transcription-lv
    to transcribe the ENTIRE movie before returning, causing Bazarr's 30-second
    timeout to fire. Truncating here keeps the round-trip well under 5 seconds.

    Handles both raw PCM (16 kHz s16le mono, no header) and RIFF/WAVE containers.
    For WAV files the RIFF ChunkSize and data SubChunk2Size fields are patched
    to reflect the new, shorter data length so the downstream decoder is happy.
    """
    PCM_SAMPLE_RATE = 16000
    PCM_BYTES_PER_SAMPLE = 2  # s16le
    PCM_CHANNELS = 1
    pcm_byte_budget = seconds * PCM_SAMPLE_RATE * PCM_BYTES_PER_SAMPLE * PCM_CHANNELS

    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        # WAV container — find the "data" chunk and slice its payload.
        # Most WAV files have only a fmt + data chunk, but we search for the
        # "data" marker defensively in case there are LIST/INFO chunks before it.
        idx = raw.find(b"data")
        if idx == -1 or idx + 8 > len(raw):
            # Malformed or exotic WAV: best-effort — take header + budget of payload.
            return raw[: 44 + pcm_byte_budget]
        data_start = idx + 8  # skip "data" (4 bytes) + SubChunk2Size (4 bytes)
        slice_end = min(data_start + pcm_byte_budget, len(raw))
        sliced_data = raw[data_start:slice_end]

        # Reconstruct: everything up to and including "data" tag + new sizes.
        header_prefix = raw[: idx + 4]  # bytes 0 … "data"
        new_data_size = len(sliced_data).to_bytes(4, "little")  # SubChunk2Size
        new_riff_size = (36 + len(sliced_data)).to_bytes(4, "little")  # ChunkSize
        rebuilt = header_prefix + new_data_size + sliced_data
        # Patch bytes 4-8 (RIFF ChunkSize) in the reconstructed buffer.
        rebuilt = rebuilt[:4] + new_riff_size + rebuilt[8:]
        return rebuilt

    # Raw PCM (no header) — just slice the byte stream.
    return raw[:pcm_byte_budget]


def _strip_wav_header(raw: bytes) -> bytes:
    """Return the raw PCM payload from a RIFF/WAVE container.

    Searches for the 'data' chunk and returns everything after its 8-byte header
    (4-byte tag + 4-byte size).  If the bytes are already raw PCM (no RIFF magic),
    they are returned unchanged.  Malformed WAV falls back to skipping 44 bytes
    (standard header length) rather than crashing.
    """
    if not (len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"):
        return raw  # already raw PCM
    idx = raw.find(b"data")
    if idx == -1 or idx + 8 > len(raw):
        # Malformed: skip the standard 44-byte header and hope for the best
        return raw[44:]
    return raw[idx + 8 :]  # skip 'data' tag (4) + SubChunk2Size (4)


def _slice_pcm_for_segment(
    raw_pcm: bytes,
    start_s: float,
    end_s: float,
    sample_rate: int = 16000,
    channels: int = 1,
    bits: int = 16,
) -> bytes:
    """Slice raw PCM bytes to the [start_s, end_s] window of the audio timeline.

    Timestamps are in seconds (float).  The byte range is aligned to 2-byte
    sample boundaries to avoid producing a half-sample artifact.  Returns an
    empty bytes object if the range falls completely outside the buffer.
    """
    bytes_per_sec = sample_rate * channels * bits // 8
    # Align to 2-byte sample boundary (mask off the LSB)
    start_byte = int(start_s * bytes_per_sec) & ~1
    end_byte = min(int(end_s * bytes_per_sec) & ~1, len(raw_pcm))
    if start_byte >= end_byte:
        return b""
    return raw_pcm[start_byte:end_byte]


# ── Forced-alignment post-processor ───────────────────────────────────────────

# Minimum segment duration (seconds) worth sending to forced-aligner.
# Very short segments produce negligible gain but add per-request overhead.
_MIN_ALIGN_DURATION: float = 0.3

# Maximum concurrent forced-align requests.  The aligner is CPU-bound;
# over-subscribing wastes time to context switching without throughput gain.
_ALIGN_CONCURRENCY: int = 4


async def _refine_segments_with_alignment(
    segments: list[dict],
    raw_audio: bytes,
    sample_rate: int = 16000,
) -> list[dict]:
    """Post-process Whisper segments by refining word timestamps via forced-aligner-lv.

    For each segment whose source language is Latvian:
      1. Slice the corresponding PCM window from raw_audio.
      2. Wrap the slice as a WAV file.
      3. POST to forced-aligner-lv /align with the segment text.
      4. On success, replace segment["words"] with aligned words shifted to
         absolute timeline coordinates (aligner returns relative-to-slice times).
      5. On any failure or "poor" quality rating, keep Whisper's original words.

    Runs up to _ALIGN_CONCURRENCY requests concurrently to bound total latency.
    Logs a summary line at INFO level when complete.

    raw_audio may be a RIFF/WAVE container or raw PCM.  The RIFF header is
    stripped before slicing so byte offsets map correctly to PCM samples.
    """
    if not ENABLE_FORCED_ALIGN:
        return segments

    # Strip RIFF header once so _slice_pcm_for_segment works on pure PCM bytes
    raw_pcm = _strip_wav_header(raw_audio)

    sem = asyncio.Semaphore(_ALIGN_CONCURRENCY)
    aligner = ForcedAlignerClient()

    counters: dict[str, int] = {
        "skipped_short": 0,
        "aligned_good": 0,
        "aligned_acceptable": 0,
        "fallback": 0,
    }

    async def _refine_one(seg: dict) -> dict:
        """Attempt to refine one segment; return original on any failure path."""
        duration = float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))
        if duration < _MIN_ALIGN_DURATION:
            counters["skipped_short"] += 1
            return seg

        text = seg.get("text", "").strip()
        if not text:
            counters["skipped_short"] += 1
            return seg

        async with sem:
            start_s = float(seg.get("start", 0.0))
            end_s = float(seg.get("end", 0.0))
            slice_pcm = _slice_pcm_for_segment(raw_pcm, start_s, end_s, sample_rate)
            if not slice_pcm:
                counters["fallback"] += 1
                return seg

            slice_wav = _wrap_pcm_as_wav(slice_pcm, sample_rate=sample_rate)
            words = await aligner.align(slice_wav, text)

        if words is None:
            counters["fallback"] += 1
            return seg

        # Determine quality for counters (aligner client already filtered "poor")
        # We infer quality from the fact that the client returned words at all.
        # Track as "good" — acceptable would require reading the raw response,
        # which the client abstracts away.  Both good+acceptable are counted together
        # since the client accepts both and only rejects "poor".
        # We split the counter into aligned_good for now; acceptable tracking would
        # require surfacing the raw quality field from the client.
        counters["aligned_good"] += 1

        # Shift relative timestamps → absolute timeline coordinates
        adjusted_words = [
            {**w, "start": w["start"] + start_s, "end": w["end"] + start_s} for w in words
        ]
        return {**seg, "words": adjusted_words}

    t0 = time.monotonic()
    refined: list[dict] = list(await asyncio.gather(*(_refine_one(s) for s in segments)))
    elapsed = time.monotonic() - t0

    total = len(segments)
    log.info(
        "forced-align summary: %d segments, %d aligned, %d short/empty skipped, "
        "%d fallback to whisper, total +%.1fs",
        total,
        counters["aligned_good"],
        counters["skipped_short"],
        counters["fallback"],
        elapsed,
    )

    await aligner.close()
    return refined


async def _isolate_vocals_if_appropriate(
    audio_bytes: bytes,
    src_lang_hint: str = "lv",
    duration_s: float = 0,
) -> tuple[bytes, str]:
    """Run audio through vocal-isolator-lv before Whisper if conditions are met.

    Returns (audio_to_use, status) where status is one of:
      'cache_hit'  — isolator returned a cached result (fast)
      'cache_miss' — isolator processed fresh (slow, GPU)
      'skipped'    — isolation disabled or audio too short
      'failed'     — isolator error; caller should use original audio

    Side-effects: stops and restarts asr-transcription-lv around the isolator
    call to free VRAM for Demucs (~2.5 GB FP16). This is intentionally naive —
    worst case a cache-hit result costs an unnecessary 30-second restart cycle.
    Throwaway code until issue #118 lands.
    """
    min_audio_s = float(os.environ.get("MIN_AUDIO_S_FOR_ISOLATION", "30"))

    if not VOCAL_ISO_ENABLED:
        return (audio_bytes, "skipped")
    if duration_s and duration_s < min_audio_s:
        log.info(
            "vocal-isolator: skipping (duration=%.1fs < min=%.0fs)",
            duration_s,
            min_audio_s,
        )
        return (audio_bytes, "skipped")

    isolator = VocalIsolatorClient()
    try:
        # 2026-05-06: eviction removed. Chunked Demucs uses ~3 GB peak per chunk,
        # which fits alongside ASR (1.9 GB) + back-translator (2.8 GB) + omnivoice
        # (when not running) on the 12 GB GPU. The previous full-file Demucs
        # needed 5.94 GB output buffer, hence the eviction; that's no longer true.
        # Removing eviction also eliminates a race condition where concurrent
        # /detect-language requests hit a stopped ASR and returned 502 to Bazarr
        # ("Language detection failed").
        # Wrap raw PCM (Bazarr's encode=false) as WAV so soundfile can decode.
        wav_audio = _ensure_wav(audio_bytes)

        # vocal-isolator is now claimed by the outer _supervisor_session in the
        # /asr handler (hoisted before asr-transcription/back-translator so the
        # Tier 3 yield check only fires against other-workflow services, not our own).
        result = await isolator.isolate(wav_audio)

        if result is None:
            log.warning("vocal-isolator failed; falling back to original audio")
            return (audio_bytes, "failed")

        vocals, cache_status = result
        log.info(
            "vocal-isolator: cache_%s, original_size=%d, vocals_size=%d",
            cache_status,
            len(audio_bytes),
            len(vocals),
        )
        return (vocals, f"cache_{cache_status}")
    finally:
        await isolator.close()


def _get_audio_field(
    audio_file: UploadFile | None,
    file: UploadFile | None,
) -> UploadFile:
    """
    Return the first non-None upload field.
    Accepts 'audio_file' (openai-whisper-asr-webservice) or 'file' (legacy).
    Raises 422 if neither is provided.
    """
    upload = audio_file or file
    if upload is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "No audio file provided. "
                "Send as form field 'audio_file' (preferred) or 'file' (legacy)."
            ),
        )
    return upload


# ── Endpoints ──────────────────────────────────────────────────────────────────


def _finalize_pipeline_request(
    endpoint: str,
    video_file: str,
    started_at: float,
    status: str,
) -> None:
    """Why: The three /asr-family endpoints share an identical post-pipeline
    cleanup block (ring-buffer append, _in_progress_request reset, gc/malloc_trim
    to drop multi-GB audio buffers, queue cleanup). Centralising it eliminates
    three copies that drifted on previous edits and made future changes risky.
    What: Captures stage_at_failure from the in-progress request, appends a
    completion record to _recent_completions, clears _in_progress_request, then
    forces a gc + glibc malloc_trim to return arena memory to the OS so RSS
    drops back to baseline between requests.
    Test: Set _in_progress_request to a dict with stage='whisper'; call with
    status='failed'; assert _recent_completions[0]['stage_at_failure'] ==
    'whisper' and _in_progress_request is None. Call with status='complete';
    assert stage_at_failure is None.
    """
    global _in_progress_request
    ended_at = time.monotonic()
    stage_at_failure = (
        _in_progress_request.get("stage", "?")
        if status == "failed" and _in_progress_request is not None
        else None
    )
    _recent_completions.appendleft(
        {
            "endpoint": endpoint,
            "video_file": video_file,
            "started_at": started_at,
            "ended_at": ended_at,
            "status": status,
            "stage_at_failure": stage_at_failure,
            "duration_sec": ended_at - started_at,
        }
    )
    _in_progress_request = None

    # Free Python objects + return glibc arena memory to OS so RSS drops back
    # to baseline between requests. Without this, the vocals response buffer
    # (up to ~2.4 GB for 2-hr films) stays pinned in Python's heap pool.
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as 'Xm Ys' or 'Xs'."""
    secs = int(seconds)
    if secs < 60:
        return f"{secs}s"
    return f"{secs // 60}m {secs % 60:02d}s"


def _fmt_time(ts: float) -> str:
    """Format a monotonic timestamp as HH:MM:SS (wall clock approximation)."""
    # Convert monotonic to wall time by anchoring to now
    wall = datetime.datetime.now() - datetime.timedelta(seconds=time.monotonic() - ts)
    return wall.strftime("%H:%M:%S")


@app.get("/status", summary="Pipeline status board (HTML, auto-refresh 3s)", tags=["health"])
async def status_page() -> HTMLResponse:
    """
    Human-readable pipeline status page.

    Shows current in-flight request, queue of waiting requests, recent
    completions, and dependency health.  Auto-refreshes every 3 seconds.
    Designed to be readable on a phone or terminal browser (dark, monospace).
    No external dependencies — pure inline HTML/CSS.
    """
    client = _get_client()
    now = time.monotonic()

    # ── Dependency health (best-effort, short timeout) ─────────────────────
    dep_urls = [
        ("asr-transcription-lv", f"{ASR_URL}/health"),
        ("back-translator-lv", f"{BACK_TRANSLATOR_URL}/health"),
        (
            "vocal-isolator-lv",
            f"{os.environ.get('VOCAL_ISOLATOR_URL', 'http://localhost:8106')}/health",
        ),
        (
            "forced-aligner-lv",
            f"{os.environ.get('FORCED_ALIGNER_URL', 'http://localhost:8102')}/health",
        ),
    ]
    dep_results: dict[str, str] = {}
    for name, url in dep_urls:
        try:
            r = await client.get(url, timeout=2.0)
            dep_results[name] = "ok" if r.is_success else f"http_{r.status_code}"
        except Exception:
            dep_results[name] = "unreachable"

    # ── State summary line ─────────────────────────────────────────────────
    if _in_progress_request is not None:
        elapsed = now - _in_progress_request["started_at"]
        state_str = f"PROCESSING ({_fmt_duration(elapsed)} elapsed)"
    else:
        state_str = "IDLE"

    queue_len = len(_queue)
    queue_str = f"QUEUE: {queue_len} waiting" if queue_len else "QUEUE: empty"

    # ── Current request block ──────────────────────────────────────────────
    # All dynamic values are html.escape()d before interpolation because
    # video_file comes from Bazarr (user-controlled filename) and stage labels
    # contain runtime-formatted strings. Escaping closes XSS via crafted paths.
    if _in_progress_request:
        ip = _in_progress_request
        elapsed = now - ip["started_at"]
        current_html = f"""<div class="section">
<div class="section-title">CURRENT REQUEST</div>
  Endpoint: {html.escape(str(ip['endpoint']))}
  Video:    {html.escape(str(ip.get('video_file') or '(unknown)'))}
  Stage:    {html.escape(str(ip.get('stage', '—')))}
  Started:  {html.escape(_fmt_time(ip['started_at']))} ({html.escape(_fmt_duration(elapsed))} ago)
</div>"""
    else:
        current_html = '<div class="section"><div class="section-title">CURRENT REQUEST</div>  —— IDLE ——\n</div>'

    # ── Queue block ────────────────────────────────────────────────────────
    if _queue:
        rows = []
        for i, entry in enumerate(_queue, 1):
            waited = now - entry["queued_at"]
            vf = (str(entry.get("video_file") or "(unknown)"))[:60]
            rows.append(
                f"  {i}. {html.escape(str(entry['endpoint']))}  {html.escape(vf)}   "
                f"(waiting {html.escape(_fmt_duration(waited))})"
            )
        queue_html = (
            '<div class="section"><div class="section-title">QUEUE (in order)</div>\n'
            + "\n".join(rows)
            + "\n</div>"
        )
    else:
        queue_html = '<div class="section"><div class="section-title">QUEUE</div>  (empty)\n</div>'

    # ── Recent completions block ───────────────────────────────────────────
    if _recent_completions:
        rows = []
        for entry in _recent_completions:
            duration = entry["ended_at"] - entry["started_at"]
            vf = (str(entry.get("video_file") or "(unknown)"))[:55]
            status = str(entry.get("status", "?")).upper()
            status_class = (
                "ok" if status == "COMPLETE" else ("warn" if status == "DEFERRED" else "err")
            )
            if status == "FAILED":
                stage_info = f"  stage={html.escape(str(entry.get('stage_at_failure', '?')))}"
            else:
                stage_info = ""
            rows.append(
                f'  {html.escape(_fmt_time(entry["started_at"]))}  {html.escape(str(entry["endpoint"]))}  '
                f'{html.escape(vf)}   '
                f'<span class="{status_class}">{html.escape(status)}</span>{stage_info}  '
                f'{html.escape(_fmt_duration(duration))}'
            )
        completions_html = (
            '<div class="section"><div class="section-title">RECENT COMPLETIONS (last 20)</div>\n'
            + "\n".join(rows)
            + "\n</div>"
        )
    else:
        completions_html = '<div class="section"><div class="section-title">RECENT COMPLETIONS</div>  (none yet)\n</div>'

    # ── Dependencies block ─────────────────────────────────────────────────
    dep_rows = []
    for name, result in dep_results.items():
        cls = "ok" if result == "ok" else "err"
        dep_rows.append(
            f'  {html.escape(str(name)):<28} <span class="{cls}">{html.escape(str(result))}</span>'
        )
    deps_html = (
        '<div class="section"><div class="section-title">DEPENDENCIES</div>\n'
        + "\n".join(dep_rows)
        + "\n</div>"
    )

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="3">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ASR Gateway Status</title>
<style>
  body {{
    background: #0d0d0d; color: #c8c8c8;
    font-family: 'Courier New', Courier, monospace;
    font-size: 14px; line-height: 1.6;
    margin: 0; padding: 16px;
  }}
  h1 {{ color: #e0e0e0; font-size: 16px; margin: 0 0 4px 0; letter-spacing: 2px; }}
  .subtitle {{ color: #555; font-size: 12px; margin-bottom: 20px; }}
  .state-line {{ font-size: 15px; color: #f0c060; margin-bottom: 16px; font-weight: bold; }}
  .section {{ margin-bottom: 20px; white-space: pre; }}
  .section-title {{ color: #7090b0; font-size: 12px; letter-spacing: 1px;
                    border-bottom: 1px solid #222; margin-bottom: 6px; padding-bottom: 2px; }}
  .ok {{ color: #50c050; }}
  .warn {{ color: #f0c060; }}
  .err {{ color: #e05050; }}
</style>
</head>
<body>
<h1>GATEWAY STATUS</h1>
<div class="subtitle">auto-refresh 3s &mdash; {time.strftime("%Y-%m-%d %H:%M:%S")}</div>
<div class="state-line">State: {html.escape(state_str)} &nbsp;&nbsp;|&nbsp;&nbsp; {html.escape(queue_str)}</div>
{current_html}
{queue_html}
{completions_html}
{deps_html}
</body>
</html>"""
    return HTMLResponse(content=page_html)


@app.get("/health", summary="Health check", tags=["health"])
async def health() -> JSONResponse:
    """
    Check gateway health and downstream dependency reachability.

    Tests both asr-transcription-lv and back-translator-lv health endpoints.
    """
    client = _get_client()
    results: dict[str, str] = {}

    for name, url in [("asr", ASR_URL), ("back_translator", BACK_TRANSLATOR_URL)]:
        try:
            r = await client.get(f"{url}/health", timeout=5.0)
            results[name] = "ok" if r.is_success else f"http_{r.status_code}"
        except Exception as exc:
            results[name] = f"unreachable ({type(exc).__name__})"

    overall = "ok" if all(v == "ok" for v in results.values()) else "degraded"
    return JSONResponse({"status": overall, "dependencies": results})


@app.post(
    "/detect-language",
    summary="Detect audio language (openai-whisper-asr-webservice compatible)",
    tags=["detection"],
)
async def detect_language(
    audio_file: Annotated[
        Optional[UploadFile],
        File(description="Audio file (preferred field name for Bazarr compatibility)"),
    ] = None,
    file: Annotated[
        Optional[UploadFile],
        File(description="Audio file (legacy alias for audio_file)"),
    ] = None,
    encode: Annotated[
        bool,
        Query(description="Ignored (Bazarr always pre-encodes to WAV before calling us)."),
    ] = True,
    video_file: Annotated[
        Optional[str],
        Query(description="Original video path metadata from Bazarr (logged only, not read)."),
    ] = None,
) -> JSONResponse:
    """
    Detect the spoken language in an audio file.

    Returns the openai-whisper-asr-webservice response shape:
      {"detected_language": "english", "language_code": "en"}

    Bazarr's whisperai provider calls this endpoint BEFORE requesting transcription.
    Without a valid response here, Bazarr logs "WhisperAI returned empty language code"
    and the entire subtitle generation pipeline fails.

    Implementation: truncates audio to the first 30 seconds, then sends that
    slice to asr-transcription-lv with language=auto. Whisper only uses the first
    ~30 s for language identification, so the truncation is lossless for detection
    while keeping the round-trip well under Bazarr's 30-second timeout. Returns
    the detected ISO 639-1 code and its human-readable English name.

    Accepts Bazarr's extra query params (encode, video_file) gracefully:
    - encode: ignored, we receive pre-encoded WAV
    - video_file: original media path for logging context only (not read from disk)
    """
    upload = _get_audio_field(audio_file, file)
    client = _get_client()
    audio_bytes = await upload.read()
    filename = _safe_audio_filename(upload)

    log.info(
        "/detect-language  file=%s size=%d bytes  video_file=%s",
        filename,
        len(audio_bytes),
        video_file,
    )

    # Truncate to first 30 seconds before sending to ASR.
    # Whisper only needs ~30 s for language detection; forwarding a full 75-min
    # movie caused Bazarr's 30-second timeout to fire every time.
    audio_bytes = _truncate_audio_for_detection(audio_bytes, seconds=30)
    log.info("/detect-language  truncated to %d bytes for fast detection", len(audio_bytes))

    # Call ASR with language=None → triggers Whisper auto-detection
    _segments, detected_code = await _call_asr(
        client, audio_bytes, filename, language=None, task="transcribe"
    )

    detected_code = detected_code or "en"  # safe fallback
    detected_name = _LANG_CODE_TO_NAME.get(detected_code, detected_code)

    log.info("/detect-language  result: code=%s name=%s", detected_code, detected_name)
    return JSONResponse(
        {
            "detected_language": detected_name,
            "language_code": detected_code,
        }
    )


@app.post(
    "/asr",
    summary="Smart ASR routing: transcribe/translate → SRT/VTT/TXT/JSON",
    tags=["transcription"],
)
async def asr_smart(
    # Audio upload: accept 'audio_file' (Bazarr/openai-whisper-asr-webservice) or
    # 'file' (legacy internal callers). Both are optional so FastAPI doesn't 422
    # before we can give a helpful error.
    audio_file: Annotated[
        Optional[UploadFile],
        File(description="Audio file (openai-whisper-asr-webservice field name)"),
    ] = None,
    file: Annotated[
        Optional[UploadFile],
        File(description="Audio file (legacy alias; kept for backwards compat)"),
    ] = None,
    # Query params (openai-whisper-asr-webservice spec)
    task: Annotated[
        str,
        Query(description="'transcribe' or 'translate'. Default: transcribe."),
    ] = "transcribe",
    language: Annotated[
        Optional[str],
        Query(
            description=(
                "Target subtitle language in ISO 639-1 (en, lv) or ISO 639-2 (eng, lav). "
                "When provided, this is the DESIRED OUTPUT language — the gateway "
                "detects the source language from the audio and routes accordingly. "
                "Omit to transcribe in whatever language Whisper detects."
            )
        ),
    ] = None,
    output: Annotated[
        str,
        Query(description="Output format: srt | vtt | txt | json. Default: srt."),
    ] = "srt",
    encode: Annotated[
        bool,
        Query(description="Ignored (we always receive pre-encoded WAV from Bazarr)."),
    ] = True,
    video_file: Annotated[
        Optional[str],
        Query(description="Original video path metadata from Bazarr (logged only, not read)."),
    ] = None,
) -> Response:
    """
    Smart ASR endpoint with source×target language routing matrix.

    Bazarr calls this once per requested subtitle language, passing the audio
    WAV bytes (already extracted by Bazarr via ffmpeg) as 'audio_file'.
    The 'language' query param is the TARGET subtitle language, not the source.

    Routing matrix (src = Whisper-detected source, target = 'language' param):
      src=en, target=en  → Whisper transcribe EN → EN SRT
      src=en, target=lv  → Whisper transcribe EN → MT EN→LV → LV SRT
      src=lv, target=lv  → Whisper transcribe LV → LV SRT
      src=lv, target=en  → Whisper task=translate (EN output) → EN SRT
      src=*, target=en   → Whisper task=translate → EN SRT
      src=*, target=lv   → Whisper transcribe + MT src→LV (best effort)
      target=None        → Transcribe in detected source language

    ISO 639-2 codes (lav, eng) are normalized to 639-1 (lv, en) at entry.

    Accepts Bazarr's extra query params (encode, video_file) gracefully:
    - encode: ignored, we receive pre-encoded WAV
    - video_file: original media path for logging context only (not read from disk)
    """
    global _in_progress_request, _queue

    upload = _get_audio_field(audio_file, file)

    # Normalize target language: 3-letter → 2-letter (no I/O — safe before lock)
    target_lang = _normalize_lang(language)
    asr_task = (task or "transcribe").lower()
    output_fmt = (output or "srt").lower()

    # Queue this request while the pipeline is busy.
    # audio_bytes is NOT read here — each queued request keeps its multipart body
    # spooled on disk (Starlette SpooledTemporaryFile) so we don't accumulate
    # 6 × 144 MB = 864 MB of audio buffers in RAM before the first lock is acquired.
    queue_entry: dict = {
        "endpoint": "/asr",
        "video_file": video_file or upload.filename or "(unknown)",
        "queued_at": time.monotonic(),
    }
    _queue.append(queue_entry)
    if _pipeline_lock.locked():
        log.info("/asr  queueing — pipeline busy  video_file=%s", video_file)

    result: Response | None = None
    _status = "complete"
    audio_bytes: bytes = b""
    audio_for_asr: bytes = b""
    filename: str = "(unknown)"
    try:
        async with _pipeline_lock:
            # Remove ourselves from the waiting queue
            try:
                _queue.remove(queue_entry)
            except ValueError:
                pass

            # Read audio NOW — only one request holds this buffer at a time,
            # so peak RAM = 1 × buffer instead of N × buffer.
            audio_bytes = await upload.read()
            filename = _safe_audio_filename(upload)
            client = _get_client()
            mt = _get_mt()

            started_at = time.monotonic()
            _in_progress_request = {
                "endpoint": "/asr",
                "video_file": video_file or filename,
                "started_at": started_at,
                "stage": "init",
            }

            log.info(
                "/asr  file=%s size=%d target_lang=%s task=%s output=%s  video_file=%s",
                filename,
                len(audio_bytes),
                target_lang,
                asr_task,
                output_fmt,
                video_file,
            )

            try:
                # Determine if vocal isolation will run; claim it FIRST if so.
                # Why first: vocal-isolator is Tier 3, which yields to active Tier 2/1
                # services. Claiming it before asr-transcription/back-translator means
                # the yield check only fires against services from OTHER workflows
                # (e.g. omnivoice during a study session), not against this same /asr
                # request's own downstream claims.
                #
                # Policy B (2026-05-06): GpuYieldError means a Tier 3 claim was
                # deferred by the supervisor. Re-raise it so it propagates out of
                # the pipeline lock and is caught by the outer GpuYieldError handler.
                duration_estimate_s = len(audio_bytes) / 32000  # 16 kHz mono int16
                min_isolation_s = float(os.environ.get("MIN_AUDIO_S_FOR_ISOLATION", "30"))
                will_isolate = (
                    target_lang == "lv"
                    and VOCAL_ISO_ENABLED
                    and duration_estimate_s >= min_isolation_s
                )
                services_to_claim: list[str] = []
                if will_isolate:
                    services_to_claim.append("vocal-isolator-lv")
                services_to_claim.extend(["asr-transcription-lv", "back-translator-lv"])

                async with _supervisor_session(services_to_claim):
                    # ── Vocal isolation pass (LV target only) ─────────────────────────────
                    if target_lang == "lv":
                        _in_progress_request["stage"] = "vocal-isolation"
                        approx_duration_s = len(audio_bytes) / 32000
                        audio_for_asr, iso_status = await _isolate_vocals_if_appropriate(
                            audio_bytes, src_lang_hint="lv", duration_s=approx_duration_s
                        )
                    else:
                        audio_for_asr = audio_bytes
                        iso_status = "skipped"

                    # ── Downsample vocals to 16 kHz mono int16 before forwarding to ASR ──
                    # vocal-isolator returns 44.1 kHz stereo float32 (Demucs training
                    # distribution).  At that format a 75-min movie is 1.52 GB — above the
                    # 200 MB ASR body limit.  Whisper uses only 16 kHz mono int16, so
                    # downsampling is lossless from Whisper's perspective and reduces size ~10x.
                    # We only downsample when isolation actually ran (cache_hit / cache_miss);
                    # plain WAV from Bazarr is already 16 kHz int16 and needs no conversion.
                    if iso_status in ("cache_hit", "cache_miss"):
                        log.info(
                            "whisper: downsampling %d-byte vocals (44.1kHz stereo float32) for Whisper",
                            len(audio_for_asr),
                        )
                        _in_progress_request["stage"] = "downsample"
                        audio_for_asr = _downsample_vocals_for_whisper(audio_for_asr)
                        log.info(
                            "whisper: downsampled to %d bytes (16kHz mono int16)",
                            len(audio_for_asr),
                        )

                    # ── Case 1: Explicit Whisper translate task ──────────────────────────
                    if asr_task == "translate":
                        segments, detected = await _whisper_chunked(
                            audio_for_asr, language=None, task="translate", client=client
                        )
                        log.info(
                            "/asr translate done: %d segs, detected src=%s, isolation=%s",
                            len(segments),
                            detected,
                            iso_status,
                        )
                        result = _format_output(segments, output_fmt, detected)
                    else:
                        # ── Case 2: Transcribe task — detect source, then route ──────────
                        segments, detected_src = await _whisper_chunked(
                            audio_for_asr, language=None, task="transcribe", client=client
                        )
                        detected_src = detected_src or "en"
                        log.info(
                            "/asr detected src=%s target=%s segs=%d",
                            detected_src,
                            target_lang,
                            len(segments),
                        )

                        # ── Forced-alignment refinement (Latvian audio only) ─────────────
                        if detected_src == "lv" and target_lang != "en":
                            _in_progress_request["stage"] = "forced-align"
                            segments = await _refine_segments_with_alignment(
                                segments, audio_bytes, sample_rate=16000
                            )

                        # No target specified
                        if target_lang is None:
                            log.info(
                                "/asr complete: isolation=%s, segments=%d",
                                iso_status,
                                len(segments),
                            )
                            result = _format_output(segments, output_fmt, detected_src)

                        # Same source and target
                        elif target_lang == detected_src:
                            log.info(
                                "/asr complete: isolation=%s, segments=%d",
                                iso_status,
                                len(segments),
                            )
                            result = _format_output(segments, output_fmt, detected_src)

                        # Target is English
                        elif target_lang == "en":
                            if detected_src != "en":
                                _in_progress_request["stage"] = "whisper-translate"
                                segments, _ = await _whisper_chunked(
                                    audio_for_asr,
                                    language=detected_src,
                                    task="translate",
                                    client=client,
                                )
                            log.info(
                                "/asr complete: isolation=%s, segments=%d",
                                iso_status,
                                len(segments),
                            )
                            result = _format_output(segments, output_fmt, "en")

                        # Target is Latvian
                        elif target_lang == "lv":
                            _in_progress_request["stage"] = "mt-translate"
                            # _transcribe_and_translate_to_lv calls _call_asr internally.
                            # Pass the already-downsampled audio_for_asr directly — it's
                            # already 16 kHz mono int16 WAV so _call_asr/_ensure_wav handle
                            # it correctly without any double-conversion.
                            translated = await _transcribe_and_translate_to_lv(
                                client,
                                mt,
                                audio_for_asr,
                                filename,
                                src_lang=detected_src,
                                asr_task="transcribe",
                            )
                            log.info(
                                "/asr complete: isolation=%s, segments=%d",
                                iso_status,
                                len(translated),
                            )
                            result = _format_output(translated, output_fmt, "lv")

                        # Fallback
                        else:
                            log.warning(
                                "/asr unsupported target_lang=%s — returning source language %s",
                                target_lang,
                                detected_src,
                            )
                            log.info(
                                "/asr complete: isolation=%s, segments=%d",
                                iso_status,
                                len(segments),
                            )
                            result = _format_output(segments, output_fmt, detected_src)

                    _in_progress_request["stage"] = "done"
                # end async with _supervisor_session

            except GpuYieldError:
                # Tier 3 yield (Policy B) — not a pipeline failure, just a deferral.
                # Mark as deferred so the ring buffer reflects accurately, then re-raise
                # for the outer handler to convert to HTTP 503 for Bazarr.
                _status = "deferred"
                raise
            except Exception:
                _status = "failed"
                raise
            finally:
                # Drop multi-GB audio buffers before _finalize_pipeline_request
                # forces gc + malloc_trim — otherwise the trim cannot reclaim them.
                audio_bytes = None  # type: ignore[assignment]
                audio_for_asr = None  # type: ignore[assignment]  # vocals buffer (up to ~2.4 GB)
                _finalize_pipeline_request(
                    endpoint="/asr",
                    video_file=video_file or filename,
                    started_at=started_at,
                    status=_status,
                )
    except GpuYieldError as yield_err:
        # Policy B (2026-05-06): Tier 3 service was deferred because a higher-priority
        # service is active. Return 503 + Retry-After to Bazarr so it retries naturally.
        log.info(
            "/asr deferred: GPU busy with %s — Bazarr will retry in %ds",
            yield_err.active,
            yield_err.retry_after,
        )
        return JSONResponse(
            status_code=503,
            content={
                "detail": "GPU busy with higher-priority services. Try again later.",
                "active_services": yield_err.active,
            },
            headers={"Retry-After": str(yield_err.retry_after)},
        )
    finally:
        # Belt-and-suspenders: ensure queue_entry is not left in _queue on any exception
        try:
            _queue.remove(queue_entry)
        except ValueError:
            pass

    if result is None:
        raise HTTPException(status_code=500, detail="Pipeline produced no result")
    return result


@app.post(
    "/asr-translate-lv",
    summary="EN audio → LV SRT (Whisper + NLLB-200 MT chain)",
    tags=["translation"],
)
async def asr_translate_lv(
    audio_file: Annotated[
        Optional[UploadFile],
        File(description="English-audio file (preferred field name)"),
    ] = None,
    file: Annotated[
        Optional[UploadFile],
        File(description="English-audio file (legacy alias)"),
    ] = None,
    language: Annotated[
        Optional[str],
        Form(description="Source audio language. Default 'en'. ISO 639-1 or 639-2."),
    ] = "en",
    output: Annotated[
        Optional[str],
        Query(description="Output format: srt | vtt | txt | json. Default: srt."),
    ] = "srt",
) -> Response:
    """
    EN audio → LV SRT subtitle generation.

    Pipeline:
      1. Transcribe English audio via asr-transcription-lv (Whisper large-v3)
      2. Batch-translate all segments EN→LV via back-translator-lv (NLLB-200 1.3B)
         — cue-by-cue translation preserves all timecodes exactly
      3. Format translated segments in requested output format

    Latency estimate (2-hr movie, Tesla P4):
      - Whisper ASR:        5-15 min
      - Batch MT (~1000 segs): ~30-90 s  (GPU batching, cache hits)
      - Total:              6-17 min

    On MT failure for individual segments: text is prefixed with [MT-FAIL]
    so the subtitle track remains complete (just untranslated for that line).
    """
    global _in_progress_request, _queue

    upload = _get_audio_field(audio_file, file)

    # Normalize: 3-letter → 2-letter, default to en (no I/O — safe before lock)
    asr_lang = _normalize_lang(language or "en") or "en"
    output_fmt = (output or "srt").lower()

    # Queue without reading audio — keep multipart body spooled on disk until
    # we hold the lock, preventing N × 144 MB RAM accumulation while queued.
    queue_entry: dict = {
        "endpoint": "/asr-translate-lv",
        "video_file": upload.filename or "(unknown)",
        "queued_at": time.monotonic(),
    }
    _queue.append(queue_entry)
    if _pipeline_lock.locked():
        log.info("/asr-translate-lv  queueing — pipeline busy  file=%s", upload.filename)

    result: Response | None = None
    _status = "complete"
    audio_bytes: bytes = b""
    filename: str = "(unknown)"
    try:
        async with _pipeline_lock:
            try:
                _queue.remove(queue_entry)
            except ValueError:
                pass

            # Read audio inside the lock — only one request holds this buffer at a time.
            audio_bytes = await upload.read()
            filename = _safe_audio_filename(upload)
            client = _get_client()
            mt = _get_mt()

            started_at = time.monotonic()
            _in_progress_request = {
                "endpoint": "/asr-translate-lv",
                "video_file": filename,
                "started_at": started_at,
                "stage": "whisper",
            }

            log.info(
                "/asr-translate-lv  file=%s lang=%s output=%s size=%d bytes",
                filename,
                asr_lang,
                output_fmt,
                len(audio_bytes),
            )

            try:
                # Claim asr-transcription-lv + back-translator-lv via supervisor.
                # No vocal-isolator claim here: this endpoint does not invoke
                # _isolate_vocals_if_appropriate (Whisper runs directly on the
                # incoming WAV). If isolation is added later, prepend
                # "vocal-isolator-lv" to services_to_claim when it will run.
                services_to_claim: list[str] = [
                    "asr-transcription-lv",
                    "back-translator-lv",
                ]
                async with _supervisor_session(services_to_claim):
                    # Transcribe + MT-translate to LV (timecodes preserved cue-by-cue)
                    translated_segments = await _transcribe_and_translate_to_lv(
                        client, mt, audio_bytes, filename, src_lang=asr_lang
                    )
                    log.info("/asr-translate-lv  complete — %d segments", len(translated_segments))
                    _in_progress_request["stage"] = "done"
                    result = _format_output(translated_segments, output_fmt, "lv")
            except GpuYieldError:
                # Tier 3 yield (Policy B): mark as deferred, re-raise for outer
                # handler to convert to HTTP 503 + Retry-After for Bazarr.
                _status = "deferred"
                raise
            except Exception:
                _status = "failed"
                raise
            finally:
                audio_bytes = None  # type: ignore[assignment]
                _finalize_pipeline_request(
                    endpoint="/asr-translate-lv",
                    video_file=filename,
                    started_at=started_at,
                    status=_status,
                )
    except GpuYieldError as yield_err:
        # Policy B (2026-05-06): Tier 3 service was deferred because a higher-priority
        # service is active. Return 503 + Retry-After to Bazarr so it retries naturally.
        log.info(
            "/asr-translate-lv deferred: GPU busy with %s — Bazarr will retry in %ds",
            yield_err.active,
            yield_err.retry_after,
        )
        return JSONResponse(
            status_code=503,
            content={
                "detail": "GPU busy with higher-priority services. Try again later.",
                "active_services": yield_err.active,
            },
            headers={"Retry-After": str(yield_err.retry_after)},
        )
    finally:
        try:
            _queue.remove(queue_entry)
        except ValueError:
            pass

    if result is None:
        raise HTTPException(status_code=500, detail="Pipeline produced no result")
    return result


@app.post(
    "/asr-lv-native",
    summary="LV audio → LV SRT (dedicated Latvian Whisper model)",
    tags=["transcription"],
)
async def asr_lv_native(
    audio_file: Annotated[
        Optional[UploadFile],
        File(description="Latvian-audio file (preferred field name)"),
    ] = None,
    file: Annotated[
        Optional[UploadFile],
        File(description="Latvian-audio file (legacy alias)"),
    ] = None,
    language: Annotated[
        Optional[str],
        Form(description="Language code. Default 'lv'. ISO 639-1 or 639-2."),
    ] = "lv",
    output: Annotated[
        Optional[str],
        Query(description="Output format: srt | vtt | txt | json. Default: srt."),
    ] = "srt",
) -> Response:
    """
    LV audio → LV SRT subtitle generation.

    Routes to asr-transcription-lv which is loaded with the Latvian-fine-tuned
    whisper-lv-ct2 model (Whisper large-v3, int8 quantised for Tesla P4).

    Superior to the generic Whisper for Latvian audio — higher accuracy on
    Latvian phonology, proper noun handling, and diacritics.

    Latency: ~5-15 min for 2-hr film on Tesla P4.
    """
    global _in_progress_request, _queue

    upload = _get_audio_field(audio_file, file)

    # Normalize language (no I/O — safe before lock)
    lang = _normalize_lang(language or "lv") or "lv"
    output_fmt = (output or "srt").lower()

    # Queue without reading audio — keep multipart body spooled on disk until
    # we hold the lock, preventing N × 144 MB RAM accumulation while queued.
    queue_entry: dict = {
        "endpoint": "/asr-lv-native",
        "video_file": upload.filename or "(unknown)",
        "queued_at": time.monotonic(),
    }
    _queue.append(queue_entry)
    if _pipeline_lock.locked():
        log.info("/asr-lv-native  queueing — pipeline busy  file=%s", upload.filename)

    result: Response | None = None
    _status = "complete"
    audio_bytes: bytes = b""
    filename: str = "(unknown)"
    try:
        async with _pipeline_lock:
            try:
                _queue.remove(queue_entry)
            except ValueError:
                pass

            # Read audio inside the lock — only one request holds this buffer at a time.
            audio_bytes = await upload.read()
            filename = _safe_audio_filename(upload)
            client = _get_client()

            started_at = time.monotonic()
            _in_progress_request = {
                "endpoint": "/asr-lv-native",
                "video_file": filename,
                "started_at": started_at,
                "stage": "whisper",
            }

            log.info(
                "/asr-lv-native  file=%s lang=%s output=%s size=%d bytes",
                filename,
                lang,
                output_fmt,
                len(audio_bytes),
            )

            try:
                # Claim asr-transcription-lv via supervisor. No back-translator-lv
                # (no MT step in this endpoint). No vocal-isolator-lv (this endpoint
                # does not invoke _isolate_vocals_if_appropriate). forced-aligner-lv
                # is CPU-only and not under supervisor control.
                services_to_claim: list[str] = ["asr-transcription-lv"]
                async with _supervisor_session(services_to_claim):
                    segments, detected = await _whisper_chunked(
                        audio_bytes, language=lang, task="transcribe", client=client
                    )
                    log.info("/asr-lv-native  done — %d segments", len(segments))

                    # Refine word timestamps via forced-aligner-lv (Latvian audio only endpoint)
                    _in_progress_request["stage"] = "forced-align"
                    segments = await _refine_segments_with_alignment(
                        segments, audio_bytes, sample_rate=16000
                    )

                    _in_progress_request["stage"] = "done"
                    result = _format_output(segments, output_fmt, detected or lang)
            except GpuYieldError:
                # Tier 3 yield (Policy B): mark as deferred, re-raise for outer
                # handler to convert to HTTP 503 + Retry-After for Bazarr.
                _status = "deferred"
                raise
            except Exception:
                _status = "failed"
                raise
            finally:
                audio_bytes = None  # type: ignore[assignment]
                _finalize_pipeline_request(
                    endpoint="/asr-lv-native",
                    video_file=filename,
                    started_at=started_at,
                    status=_status,
                )
    except GpuYieldError as yield_err:
        # Policy B (2026-05-06): Tier 3 service was deferred because a higher-priority
        # service is active. Return 503 + Retry-After to Bazarr so it retries naturally.
        log.info(
            "/asr-lv-native deferred: GPU busy with %s — Bazarr will retry in %ds",
            yield_err.active,
            yield_err.retry_after,
        )
        return JSONResponse(
            status_code=503,
            content={
                "detail": "GPU busy with higher-priority services. Try again later.",
                "active_services": yield_err.active,
            },
            headers={"Retry-After": str(yield_err.retry_after)},
        )
    finally:
        try:
            _queue.remove(queue_entry)
        except ValueError:
            pass

    if result is None:
        raise HTTPException(status_code=500, detail="Pipeline produced no result")
    return result


# ── Lingarr text-translation proxy ─────────────────────────────────────────────


async def _translate_proxy(request: Request, path: str) -> Response:
    """Why: Lingarr (and other callers) hit back-translator-lv at :8104 directly,
    bypassing the GPU supervisor entirely. Routing through a gateway proxy lets
    the supervisor mediate claim/release for the text-translation path the same
    way it does for ASR. Factored here so /translate and /translate/batch share
    a single implementation — they differ only in the upstream path suffix.
    What: Claims back-translator-lv via _supervisor_session, forwards the request
    body to f"{BACK_TRANSLATOR_URL}{path}", and returns the response verbatim
    (body, status_code, content-type). Releases the service after the response.
    Test: POST /translate with {"text": "hi", "source_lang": "eng_Latn",
    "target_lang": "lvs_Latn"}; assert status mirrors back-translator's and the
    JSON has a "translation" key. POST /translate/batch with a list payload;
    assert it reaches back-translator-lv's /translate/batch. With a
    higher-priority service active, assert a 503 with Retry-After is returned.
    """
    client = _get_client()

    # Read body once — must be reusable for forwarding even if claim succeeds.
    body = await request.body()
    content_type = request.headers.get("content-type", "application/json")

    try:
        async with _supervisor_session(["back-translator-lv"]):
            try:
                upstream = await client.post(
                    f"{BACK_TRANSLATOR_URL}{path}",
                    content=body,
                    headers={"content-type": content_type},
                    timeout=MT_TIMEOUT,
                )
            except httpx.TimeoutException as exc:
                log.error("%s proxy: back-translator timeout: %s", path, exc)
                raise HTTPException(
                    status_code=504,
                    detail="Back-translator timed out.",
                ) from exc
            except httpx.RequestError as exc:
                log.error("%s proxy: back-translator unreachable: %s", path, exc)
                raise HTTPException(
                    status_code=502,
                    detail="Cannot reach back-translator.",
                ) from exc

            # Forward response verbatim — body, status, content-type.
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "application/json"),
            )
    except GpuYieldError as yield_err:
        log.info(
            "%s deferred: GPU busy with %s — caller will retry in %ds",
            path,
            yield_err.active,
            yield_err.retry_after,
        )
        return JSONResponse(
            status_code=503,
            content={
                "detail": "GPU busy with higher-priority services. Try again later.",
                "active_services": yield_err.active,
            },
            headers={"Retry-After": str(yield_err.retry_after)},
        )


@app.post(
    "/translate",
    summary="Text translation proxy (claims back-translator-lv via supervisor)",
    tags=["translation"],
)
async def translate_proxy(request: Request) -> Response:
    """Why: Lingarr's single-text translation path must go through the supervisor.
    What: Thin wrapper that delegates to _translate_proxy with path='/translate'.
    Test: POST {"text": "hi", "source_lang": "eng_Latn", "target_lang": "lvs_Latn"}
    and assert the response mirrors back-translator-lv's /translate output.
    """
    return await _translate_proxy(request, "/translate")


@app.post(
    "/translate/batch",
    summary="Batch text translation proxy (claims back-translator-lv via supervisor)",
    tags=["translation"],
)
async def translate_batch_proxy(request: Request) -> Response:
    """Why: Lingarr's batch translation path (many cues in one request) must also
    go through the supervisor — otherwise long batch jobs at :8104 starve
    higher-priority services holding their own claims.
    What: Thin wrapper that delegates to _translate_proxy with path='/translate/batch'.
    Test: POST a batch payload and assert it reaches back-translator-lv's
    /translate/batch endpoint with status/body forwarded verbatim. With a
    higher-priority service active, assert a 503 with Retry-After is returned.
    """
    return await _translate_proxy(request, "/translate/batch")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9001)
