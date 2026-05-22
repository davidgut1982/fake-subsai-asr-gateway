"""
MTTranslator — HTTP client for back-translator-lv (Component 2G).

Translates Whisper subtitle segments from English to Latvian via NLLB-200.
Uses the /translate/batch endpoint for efficiency (single GPU forward pass
for all segments, 3-8x faster than sequential /translate calls).

API reference (back-translator-lv):
  POST /translate        — single text, JSON {text, source_lang, target_lang}
  POST /translate/batch  — batch texts, JSON {texts, source_lang, target_lang}
  Response key: "translation" (single) | "translations" (batch)

Language codes (NLLB-200 / FLORES-200):
  eng_Latn — English
  lvs_Latn — Latvian Standard (Latin script)
  NOTE: lav_Latn resolves to <unk> in the NLLB tokenizer — always use lvs_Latn.
"""
from __future__ import annotations

import logging
from typing import Any

import requests

log = logging.getLogger("fake-subsai-gateway.mt")

# Back-translator API allows max 64 texts per batch (enforced by Pydantic model)
_BATCH_SIZE = 32  # conservative; stay well under the 64 limit


class MTTranslator:
    """
    Synchronous HTTP wrapper around back-translator-lv.

    Uses batch translation by default for efficiency. Falls back to sequential
    single-segment calls only if the batch endpoint is unavailable.
    """

    def __init__(
        self,
        url: str = "http://localhost:8104",
        timeout: float = 30.0,
    ) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

    # ── Public API ─────────────────────────────────────────────────────────────

    def translate_segments(
        self,
        segments: list[dict[str, Any]],
        source_lang: str = "eng_Latn",
        target_lang: str = "lvs_Latn",
    ) -> list[dict[str, Any]]:
        """
        Translate a list of Whisper-style segments, preserving timing metadata.

        Each segment must have at least {"start": float, "end": float, "text": str}.
        Returns a new list with the same structure but text replaced by translation.

        Empty-text segments are passed through unchanged (silence / non-speech).
        On batch failure, falls back to sequential single-segment translation.
        On per-segment failure, preserves original text prefixed with [MT-FAIL].
        """
        if not segments:
            return []

        # Separate empty from non-empty segments (preserve indices)
        indices_to_translate: list[int] = []
        texts_to_translate: list[str] = []
        for i, seg in enumerate(segments):
            text = seg.get("text", "").strip()
            if text:
                indices_to_translate.append(i)
                texts_to_translate.append(seg["text"])

        if not texts_to_translate:
            return [dict(s) for s in segments]

        # Translate in batches
        translated_texts = self._translate_batch_chunked(
            texts_to_translate, source_lang, target_lang
        )

        # Reconstruct segments with translated text
        out = [dict(s) for s in segments]
        for idx, translated in zip(indices_to_translate, translated_texts):
            out[idx] = {**out[idx], "text": translated}
        return out

    def translate_single(
        self,
        text: str,
        source_lang: str = "eng_Latn",
        target_lang: str = "lvs_Latn",
    ) -> str:
        """
        Translate a single text string.

        Returns original text (prefixed with [MT-FAIL]) on error so callers
        always get a string back and can produce a complete (if degraded) SRT.
        """
        if not text.strip():
            return text
        try:
            r = requests.post(
                f"{self.url}/translate",
                json={
                    "text": text,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                },
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json().get("translation", text)
        except requests.RequestException as exc:
            log.warning("MT single-translate failed: %s — returning original", exc)
            return f"[MT-FAIL] {text}"

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _translate_batch_chunked(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        """
        Translate texts in _BATCH_SIZE chunks via /translate/batch.

        Falls back to sequential /translate on batch endpoint failure.
        """
        results: list[str] = []
        for chunk_start in range(0, len(texts), _BATCH_SIZE):
            chunk = texts[chunk_start : chunk_start + _BATCH_SIZE]
            chunk_results = self._translate_one_batch(chunk, source_lang, target_lang)
            results.extend(chunk_results)
        return results

    def _translate_one_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        """
        POST to /translate/batch.  Returns a list of translated strings in order.

        On HTTP error or timeout, falls back to sequential single-segment calls.
        """
        try:
            r = requests.post(
                f"{self.url}/translate/batch",
                json={
                    "texts": texts,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                },
                timeout=self.timeout,  # per-batch flat timeout (not scaled by size)
            )
            r.raise_for_status()
            translations = r.json().get("translations", [])
            if len(translations) == len(texts):
                return translations
            # Mismatched count — fall back
            log.warning(
                "Batch returned %d translations for %d inputs — falling back to sequential",
                len(translations), len(texts),
            )
        except requests.RequestException as exc:
            log.warning("Batch MT failed (%s) — falling back to sequential", exc)

        # Sequential fallback
        return [self.translate_single(t, source_lang, target_lang) for t in texts]
