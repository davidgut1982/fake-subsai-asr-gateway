"""Client for vocal-isolator-lv service. Handles audio isolation requests."""

import logging
import os
from typing import Optional

import httpx

# Gateway uses network_mode: host, so vocal-isolator is reached via host-mapped port.
# host port 8106 (updated from 8105 which is taken by fluency-gate-roberta-lv).
ISOLATOR_URL = os.environ.get("VOCAL_ISOLATOR_URL", "http://localhost:8106")
ENABLED = os.environ.get("ENABLE_VOCAL_ISOLATION", "true").lower() == "true"
TIMEOUT_S = float(os.environ.get("VOCAL_ISOLATOR_TIMEOUT_S", "1800"))  # 30 min for full movies

# Default skip_denoise to True: DeepFilterNet can cause memory blowup on long films
# and Demucs alone produces sufficient quality for phase 1.  Set env var to "false"
# to re-enable once chunked denoise is implemented (phase 2).
_SKIP_DENOISE_DEFAULT: bool = (
    os.environ.get("VOCAL_ISOLATOR_SKIP_DENOISE", "true").lower() != "false"
)

log = logging.getLogger("fake-subsai-gateway")


class VocalIsolatorClient:
    def __init__(self, base_url: str = ISOLATOR_URL, timeout: float = TIMEOUT_S):
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def isolate(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        skip_denoise: bool = _SKIP_DENOISE_DEFAULT,
        output_format: str = "int16_mono_16000",
    ) -> Optional[tuple[bytes, str]]:
        """Send audio to vocal-isolator-lv and return (vocals_bytes, cache_status).

        cache_status is 'hit' or 'miss'. Returns None on any failure so callers
        can fall back to the original audio without crashing the ASR pipeline.

        output_format controls the encoding returned by vocal-isolator:
          'int16_mono_16000' (default) — 16 kHz mono int16 WAV; ~10x smaller
              than the raw Demucs output; Whisper-native.  The gateway never
              holds the 1.52 GB float32 blob in memory.
          'float32_stereo_44100' — Demucs native; for callers that need it.
        """
        if not ENABLED:
            return None
        try:
            files = {"audio_file": (filename, audio_bytes, "audio/wav")}
            data = {
                "skip_denoise": str(skip_denoise).lower(),
                "output_format": output_format,
            }
            r = await self._client.post("/isolate", files=files, data=data)
            if r.status_code != 200:
                log.warning(
                    "vocal-isolator returned %d: %s", r.status_code, r.text[:200]
                )
                return None
            cache_status = r.headers.get("X-Cache", "unknown")
            return (r.content, cache_status)
        except Exception as e:
            log.warning(
                "vocal-isolator request failed: %s: %s", type(e).__name__, e
            )
            return None

    async def health(self) -> bool:
        """Return True if the isolator service is healthy and model is loaded."""
        try:
            r = await self._client.get("/health", timeout=5.0)
            return r.status_code == 200 and r.json().get("model_loaded", False)
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()
