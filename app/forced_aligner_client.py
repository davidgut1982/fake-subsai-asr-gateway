"""
forced_aligner_client — HTTP client for forced-aligner-lv.

Provides word-level timestamp refinement for Latvian audio segments.
The forced-aligner-lv service uses torchaudio MMS_FA (lang=lav) and accepts
multipart-form requests with an audio WAV slice and transcript text.

API contract (POST /align):
  Form fields:
    audio              — WAV file bytes (multipart)
    text               — Latvian transcript string
    confidence_threshold — float (optional, default 0.5)
  Response JSON:
    {
      "words": [{"word": str, "start": float, "end": float, "confidence": float}, ...],
      "alignment_quality": "good" | "acceptable" | "poor",
      "audio_duration": float
    }

Returned word timestamps are RELATIVE to the audio slice start.
Callers must add segment.start to convert to absolute timeline offsets.

Feature flag: ENABLE_FORCED_ALIGN env var (default "true").
Kill-switch: set ENABLE_FORCED_ALIGN=false to bypass entirely.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

log = logging.getLogger("fake-subsai-gateway.aligner")

# ── Configuration ──────────────────────────────────────────────────────────────

# Internal Docker DNS name used when both services share a Docker bridge network.
# With host-network mode on the gateway, use the host-mapped port instead.
FORCED_ALIGNER_URL: str = os.environ.get(
    "FORCED_ALIGNER_URL", "http://localhost:8102"
)

ENABLE_FORCED_ALIGN: bool = (
    os.environ.get("ENABLE_FORCED_ALIGN", "true").lower().strip() == "true"
)

# Per-request timeout in seconds.  The aligner runs on CPU and can be slow for
# long segments; 30 s is generous but prevents indefinite hangs.
_ALIGNER_TIMEOUT: float = float(os.environ.get("FORCED_ALIGNER_TIMEOUT", "30.0"))

# Confidence threshold sent to the aligner service.
_CONFIDENCE_THRESHOLD: str = "0.5"


# ── Client ─────────────────────────────────────────────────────────────────────


class ForcedAlignerClient:
    """
    Async HTTP client for the forced-aligner-lv /align endpoint.

    Usage:
        client = ForcedAlignerClient()
        words = await client.align(wav_bytes, "labdien kā jums klājas")
        if words is not None:
            # use refined word timings (relative to wav slice start)
            ...
        await client.close()
    """

    def __init__(
        self,
        base_url: str = FORCED_ALIGNER_URL,
        timeout: float = _ALIGNER_TIMEOUT,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
        )

    async def align(
        self,
        wav_bytes: bytes,
        text: str,
    ) -> Optional[list[dict]]:
        """
        POST /align with wav_bytes + text.  Returns word list or None on failure.

        Returns None when:
          - ENABLE_FORCED_ALIGN is false
          - text is blank
          - HTTP error or network timeout
          - alignment_quality == "poor" (aligner's own classification)

        Word dicts contain at least: {word, start, end, confidence}.
        Timestamps are RELATIVE to the wav slice — caller adjusts to absolute.
        """
        if not ENABLE_FORCED_ALIGN:
            return None

        text = text.strip()
        if not text:
            return None

        try:
            files = {"audio": ("audio.wav", wav_bytes, "audio/wav")}
            data = {
                "text": text,
                "confidence_threshold": _CONFIDENCE_THRESHOLD,
            }
            response = await self._client.post("/align", files=files, data=data)

            if response.status_code != 200:
                log.debug(
                    "forced-aligner returned HTTP %d for text %r",
                    response.status_code, text[:40],
                )
                return None

            payload = response.json()
            quality = payload.get("alignment_quality", "good")
            if quality == "poor":
                log.debug(
                    "forced-aligner quality=poor for text %r — discarding",
                    text[:40],
                )
                return None

            words = payload.get("words")
            if not words:
                return None

            return words  # type: ignore[return-value]

        except httpx.TimeoutException:
            log.debug("forced-aligner timed out for text %r", text[:40])
            return None
        except Exception as exc:  # noqa: BLE001
            log.debug("forced-aligner error: %s", exc)
            return None

    async def close(self) -> None:
        """Release the underlying httpx connection pool."""
        await self._client.aclose()
