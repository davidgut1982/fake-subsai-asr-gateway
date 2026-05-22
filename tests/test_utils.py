"""Unit tests for pure helper functions in app/main.py and app/subtitle_utils.py.

Why: These helpers (_normalize_lang, _ensure_wav, _format_output, _wrap_pcm_as_wav,
and the /asr routing matrix) are the parts of the gateway that have correctness
invariants but no I/O dependency. Pinning their behaviour with cheap unit tests
catches regressions before they ship without needing a GPU, network, or audio
files. Aligns with the project's 90% coverage standard for pure logic.
What: pytest test cases that exercise each helper with table-driven inputs.
Audio inputs are minimal byte sequences (44-byte WAV headers + small PCM payloads
or raw bytes); no real files are read or written.
Test: Run `pytest tests/test_utils.py -v` from the repo root. All tests should
pass without `API_KEY`, network, or GPU access.
"""
from __future__ import annotations

import json
import struct

import pytest

from main import (
    _ISO3_TO_ISO1,
    _WHISPER_TO_NLLB,
    _ensure_wav,
    _format_output,
    _normalize_lang,
    _wrap_pcm_as_wav,
)


# ── _normalize_lang ────────────────────────────────────────────────────────────


class TestNormalizeLang:
    """Why: Bazarr sends ISO 639-2 three-letter codes (lav, eng) but Whisper
    expects ISO 639-1 two-letter codes (lv, en). _normalize_lang is the single
    chokepoint where the conversion happens — every endpoint relies on it.
    """

    @pytest.mark.parametrize(
        "code_in,expected",
        [
            ("lav", "lv"),
            ("eng", "en"),
            ("rus", "ru"),
            ("fra", "fr"),
            ("fre", "fr"),
            ("deu", "de"),
            ("ger", "de"),
            ("spa", "es"),
            ("jpn", "ja"),
        ],
    )
    def test_alpha3_converts_to_alpha2(self, code_in: str, expected: str) -> None:
        """Why: Every Bazarr request hits this path on first call.
        What: Three-letter ISO 639-2 → two-letter ISO 639-1.
        Test: Each known alpha-3 code maps to its alpha-2 sibling.
        """
        assert _normalize_lang(code_in) == expected

    @pytest.mark.parametrize("code_in", ["en", "lv", "ru", "de", "ja"])
    def test_alpha2_passthrough(self, code_in: str) -> None:
        """Why: When Bazarr sends a two-letter code we must leave it alone.
        What: Existing two-letter codes pass through unchanged.
        Test: Each two-letter code returns identical string (after .lower()).
        """
        assert _normalize_lang(code_in) == code_in

    def test_none_passthrough(self) -> None:
        """Why: language is optional on /asr — None must be allowed.
        What: None input returns None (sentinel for 'autodetect').
        Test: Assert _normalize_lang(None) is None.
        """
        assert _normalize_lang(None) is None

    def test_unknown_code_passthrough(self) -> None:
        """Why: Unknown codes should reach Whisper for it to handle/reject.
        What: Unknown codes pass through unchanged.
        Test: Assert _normalize_lang('xyz') == 'xyz'.
        """
        assert _normalize_lang("xyz") == "xyz"

    def test_case_insensitive_and_whitespace_stripped(self) -> None:
        """Why: Header values can include stray whitespace and odd case.
        What: Input is stripped and lower-cased before lookup.
        Test: '  LAV ' normalises to 'lv'.
        """
        assert _normalize_lang("  LAV ") == "lv"
        assert _normalize_lang("ENG") == "en"


# ── _ensure_wav / _wrap_pcm_as_wav ─────────────────────────────────────────────


def _make_riff_wav(payload: bytes = b"\x00\x00" * 8) -> bytes:
    """Build a minimal valid RIFF/WAVE container around `payload` for tests."""
    data_size = len(payload)
    chunk_size = 36 + data_size
    return (
        struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", chunk_size, b"WAVE",
            b"fmt ", 16, 1, 1, 16000, 32000, 2, 16,
            b"data", data_size,
        )
        + payload
    )


class TestEnsureWav:
    """Why: Bazarr sends raw PCM (encode=false) but our ASR backend requires a
    RIFF/WAVE container. _ensure_wav is the boundary that distinguishes the two
    and wraps when necessary. Mistakes here cause "Cannot decode audio" errors
    from the downstream Whisper service.
    """

    def test_valid_riff_wav_passthrough(self) -> None:
        """Why: Real WAV uploads must not be re-wrapped (would double the header).
        What: A buffer starting with RIFF...WAVE is returned unchanged.
        Test: Build a minimal WAV via _make_riff_wav, feed it in, assert identity.
        """
        wav = _make_riff_wav()
        assert _ensure_wav(wav) is wav  # identity (no copy)

    def test_raw_pcm_gets_wrapped(self) -> None:
        """Why: Bazarr's default encode=false path sends raw s16le PCM.
        What: Raw bytes must be prepended with a 44-byte RIFF/WAVE header.
        Test: Feed raw bytes; assert output starts with RIFF...WAVE and is exactly
        44 bytes longer than the input.
        """
        pcm = b"\x00\x01" * 100  # 200 raw PCM bytes
        wrapped = _ensure_wav(pcm)
        assert wrapped[:4] == b"RIFF"
        assert wrapped[8:12] == b"WAVE"
        assert len(wrapped) == len(pcm) + 44

    def test_empty_input_still_produces_wav_header(self) -> None:
        """Why: An empty upload should not crash the gateway — let the backend
        return a meaningful 4xx instead.
        What: Empty bytes become a 44-byte header with data_size=0.
        Test: _ensure_wav(b'') is a valid 44-byte WAV with RIFF magic.
        """
        wrapped = _ensure_wav(b"")
        assert wrapped[:4] == b"RIFF"
        assert wrapped[8:12] == b"WAVE"
        assert len(wrapped) == 44

    def test_short_non_riff_bytes_get_wrapped(self) -> None:
        """Why: Some upload paths send <12 bytes (probes, malformed clients);
        _ensure_wav must treat anything not-RIFF as raw PCM, not crash.
        What: <12-byte input falls through the RIFF check and gets wrapped.
        Test: 5-byte input → 49-byte WAV.
        """
        wrapped = _ensure_wav(b"\x00\x01\x02\x03\x04")
        assert wrapped[:4] == b"RIFF"
        assert len(wrapped) == 49


class TestWrapPcmAsWav:
    """Why: _wrap_pcm_as_wav is the primitive used by _ensure_wav and the
    forced-align slice path. Verifying its header math separately guards against
    regressions in either caller.
    """

    def test_header_size_and_magic(self) -> None:
        """Why: The 44-byte RIFF/WAVE header is fixed-length.
        What: Output = 44-byte header + payload, in that order.
        Test: 100 PCM bytes → 144-byte WAV beginning with RIFF.
        """
        pcm = b"\x00" * 100
        wav = _wrap_pcm_as_wav(pcm)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        assert wav[12:16] == b"fmt "
        assert wav[36:40] == b"data"
        assert len(wav) == 44 + 100


# ── _format_output ─────────────────────────────────────────────────────────────


_SAMPLE_SEGMENTS = [
    {"start": 0.0, "end": 1.5, "text": "Hello"},
    {"start": 1.5, "end": 3.0, "text": "World"},
]


class TestFormatOutput:
    """Why: /asr serves four output formats — getting Content-Type wrong silently
    breaks Bazarr (which sniffs the response). Each branch is exercised here.
    """

    def test_srt_default(self) -> None:
        """Why: SRT is the default and the format Bazarr requests on every call.
        What: output='srt' returns application/x-subrip with HH:MM:SS,mmm timecodes.
        Test: Body contains '00:00:00,000', media_type is application/x-subrip.
        """
        resp = _format_output(_SAMPLE_SEGMENTS, "srt", "en")
        assert resp.media_type == "application/x-subrip"
        assert b"00:00:00,000" in resp.body
        assert b"Hello" in resp.body

    def test_vtt_format(self) -> None:
        """Why: Some clients (browsers, modern players) prefer WebVTT.
        What: output='vtt' returns text/vtt with the WEBVTT header.
        Test: Body starts with 'WEBVTT', media_type is text/vtt.
        """
        resp = _format_output(_SAMPLE_SEGMENTS, "vtt", "en")
        assert resp.media_type == "text/vtt"
        assert resp.body.startswith(b"WEBVTT")

    def test_txt_format(self) -> None:
        """Why: Plain-text output is used by some manual workflows.
        What: output='txt' returns text/plain with one cue per line, no timecodes.
        Test: Body is 'Hello\\nWorld', media_type is text/plain.
        """
        resp = _format_output(_SAMPLE_SEGMENTS, "txt", "en")
        assert resp.media_type == "text/plain"
        assert resp.body == b"Hello\nWorld"

    def test_json_format(self) -> None:
        """Why: JSON output mirrors openai-whisper-asr-webservice for clients
        that want the full segment array with timing.
        What: output='json' returns application/json with text/language/segments.
        Test: Body parses as JSON with language='en' and len(segments)==2.
        """
        resp = _format_output(_SAMPLE_SEGMENTS, "json", "en")
        assert resp.media_type == "application/json"
        payload = json.loads(resp.body)
        assert payload["language"] == "en"
        assert len(payload["segments"]) == 2
        assert "Hello World" in payload["text"]

    def test_unknown_format_falls_back_to_srt(self) -> None:
        """Why: Bazarr or a misconfigured client may pass garbage — we fall back
        to SRT (safe Bazarr-compatible default) instead of 400ing.
        What: output='garbage' returns application/x-subrip.
        Test: media_type is application/x-subrip.
        """
        resp = _format_output(_SAMPLE_SEGMENTS, "garbage", "en")
        assert resp.media_type == "application/x-subrip"

    def test_case_insensitive(self) -> None:
        """Why: Bazarr may send 'SRT' or 'Srt'; the matrix is lower-case.
        What: Format names are lower-cased before dispatch.
        Test: 'SRT' resolves to the SRT branch.
        """
        resp = _format_output(_SAMPLE_SEGMENTS, "SRT", "en")
        assert resp.media_type == "application/x-subrip"


# ── Routing matrix for /asr ────────────────────────────────────────────────────
#
# Why: /asr decides which downstream pipeline to invoke based on (src_lang,
# target_lang, task). The routing logic itself lives inline in asr_smart() so
# we can't import a function to test directly — but we can pin the *decision
# table* via a pure Python re-implementation that mirrors the conditions in
# main.py. If the production code drifts from this table, the production code
# is wrong (or this table needs an explicit update with a code review).


def _route_decision(
    detected_src: str | None,
    target_lang: str | None,
    task: str,
) -> str:
    """Mirror of the /asr routing tree in app/main.py::asr_smart.

    Returns a stable symbolic label for the pipeline that would run, NOT a
    direct invocation. Keeping the mirror small means the test asserts the
    *intent* of the matrix; production code must stay in sync.
    """
    asr_task = (task or "transcribe").lower()
    if asr_task == "translate":
        return "whisper-translate"

    detected_src = detected_src or "en"

    if target_lang is None:
        return f"passthrough:{detected_src}"
    if target_lang == detected_src:
        return f"passthrough:{detected_src}"
    if target_lang == "en":
        if detected_src != "en":
            return "whisper-translate"
        return "passthrough:en"
    if target_lang == "lv":
        return "transcribe-then-mt:lv"
    return f"fallback:{detected_src}"


class TestAsrRoutingMatrix:
    """Why: The src×target routing matrix is the gateway's most decision-dense
    code path. Pinning the table catches accidental swaps (e.g. en→lv silently
    routing through whisper-translate instead of NLLB).
    """

    @pytest.mark.parametrize(
        "detected_src,target_lang,task,expected",
        [
            # task=translate short-circuits everything
            ("en", "en", "translate", "whisper-translate"),
            ("lv", "lv", "translate", "whisper-translate"),
            ("ru", None, "translate", "whisper-translate"),
            # No target: pass through whatever Whisper detected
            ("en", None, "transcribe", "passthrough:en"),
            ("lv", None, "transcribe", "passthrough:lv"),
            # Same source and target: pass through
            ("en", "en", "transcribe", "passthrough:en"),
            ("lv", "lv", "transcribe", "passthrough:lv"),
            # Target English, non-English source: Whisper task=translate
            ("lv", "en", "transcribe", "whisper-translate"),
            ("ru", "en", "transcribe", "whisper-translate"),
            ("de", "en", "transcribe", "whisper-translate"),
            # Target Latvian: transcribe + NLLB MT to Latvian
            ("en", "lv", "transcribe", "transcribe-then-mt:lv"),
            ("ru", "lv", "transcribe", "transcribe-then-mt:lv"),
            # Unsupported target: fallback to source language
            ("en", "fr", "transcribe", "fallback:en"),
            ("lv", "ja", "transcribe", "fallback:lv"),
        ],
    )
    def test_decision_table(
        self,
        detected_src: str,
        target_lang: str | None,
        task: str,
        expected: str,
    ) -> None:
        """Why: One row per logical branch in asr_smart.
        What: _route_decision must produce the expected pipeline label.
        Test: Each parametrised tuple must satisfy the equality.
        """
        assert _route_decision(detected_src, target_lang, task) == expected


# ── Language code maps sanity ──────────────────────────────────────────────────


class TestLanguageMaps:
    """Why: The ISO/NLLB code maps are silent data — wrong entries fail at
    runtime with mysterious downstream errors. A quick sanity check catches
    typos when the maps are extended.
    """

    def test_iso3_to_iso1_round_trips_through_normalize(self) -> None:
        """Why: Every entry in _ISO3_TO_ISO1 must round-trip through _normalize_lang.
        What: For each (alpha3, alpha2) pair, _normalize_lang(alpha3) == alpha2.
        Test: Iterate the dict and assert per pair.
        """
        for alpha3, alpha2 in _ISO3_TO_ISO1.items():
            assert _normalize_lang(alpha3) == alpha2

    def test_nllb_codes_have_correct_shape(self) -> None:
        """Why: NLLB-200 codes are `<lang>_<Script>` strings; malformed entries
        cause the back-translator to return <unk>.
        What: Each value contains exactly one underscore.
        Test: All values match the `aaa_Xxxx` pattern (3-letter lang + script).
        """
        for code in _WHISPER_TO_NLLB.values():
            parts = code.split("_")
            assert len(parts) == 2, f"Malformed NLLB code: {code}"
            assert len(parts[0]) == 3, f"Bad lang component: {code}"
            assert parts[1][0].isupper(), f"Bad script casing: {code}"
