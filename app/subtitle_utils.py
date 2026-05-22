"""
subtitle_utils — SRT and VTT subtitle formatting utilities.

Converts Whisper-style segment lists to SubRip (SRT) or WebVTT (VTT) format.
Word-level timestamps (words=True) are used when present to tighten cue
boundaries and split long segments at natural word boundaries.

SRT format spec:
  <index>
  <HH:MM:SS,mmm> --> <HH:MM:SS,mmm>
  <text>
  <blank line>

VTT format spec:
  WEBVTT
  <blank line>
  <HH:MM:SS.mmm> --> <HH:MM:SS.mmm>
  <text>
  <blank line>
"""
from __future__ import annotations

import re

# ── Constants ──────────────────────────────────────────────────────────────────

# Maximum cue duration before splitting (seconds).
# Segments longer than this produce a single on-screen line for too long, making
# the viewer feel the subtitle is lagging even with accurate boundaries.
MAX_CUE_DURATION = 7.0

# Minimum cue duration (seconds).  Sub-second cues are nearly unreadable;
# collapse them into the neighbouring cue instead of emitting tiny fragments.
MIN_CUE_DURATION = 1.0

# Punctuation at which we prefer to split long segments.
_HARD_SPLIT_RE = re.compile(r"[.!?]")
_SOFT_SPLIT_RE = re.compile(r"[,;]")


# ── Timestamp formatting ───────────────────────────────────────────────────────


def format_timestamp_srt(seconds: float) -> str:
    """
    Convert seconds (float) to SRT timestamp format: HH:MM:SS,mmm.

    Examples:
        0.0     → "00:00:00,000"
        65.5    → "00:01:05,500"
        3723.75 → "01:02:03,750"
    """
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# Keep legacy alias so existing callers don't break.
format_timestamp = format_timestamp_srt


def format_timestamp_vtt(seconds: float) -> str:
    """
    Convert seconds (float) to WebVTT timestamp format: HH:MM:SS.mmm.

    Same as SRT but uses '.' instead of ',' as the millisecond separator.
    """
    return format_timestamp_srt(seconds).replace(",", ".")


# ── Timing helpers ─────────────────────────────────────────────────────────────


def _normalise_segment_timing(seg: dict, index: int) -> tuple[float, float]:
    """Return (start, end) seconds with synthetic fallback for zero-timing segments."""
    start = float(seg.get("start", 0.0))
    end = float(seg.get("end", 0.0))

    # Handle degenerate timing (both zero → synthetic 2s window)
    if start == 0.0 and end == 0.0:
        start = float(index - 1) * 2.0
        end = start + 2.0

    # Guard against end <= start
    if end <= start:
        end = start + 2.0

    return start, end


def _word_boundaries(seg: dict) -> tuple[float, float] | None:
    """
    Extract (start, end) from the segment's words array.

    Returns None if the words array is absent or empty.  Falls back gracefully
    so callers can use seg["start"]/seg["end"] when word timing is unavailable.
    """
    words = seg.get("words")
    if not words:
        return None
    try:
        w_start = float(words[0]["start"])
        w_end = float(words[-1]["end"])
    except (KeyError, TypeError, ValueError):
        return None
    if w_end <= w_start:
        return None
    return w_start, w_end


def _cue_boundaries(seg: dict, index: int) -> tuple[float, float]:
    """
    Determine the tightest available (start, end) for a segment cue.

    Priority:
      1. words[0]["start"] / words[-1]["end"]  — per-word DTW timing (tightest)
      2. seg["start"] / seg["end"]             — cross-attention estimate (fallback)
      3. Synthetic 2-second window             — degenerate / missing timing
    """
    wb = _word_boundaries(seg)
    if wb is not None:
        start, end = wb
        if end > start:
            return start, end
    return _normalise_segment_timing(seg, index)


# ── Segment splitting ──────────────────────────────────────────────────────────


def _make_seg_from_words(words: list[dict], original_seg: dict) -> dict:
    """
    Build a new segment dict from a slice of word dicts.

    Inherits all keys from original_seg and overrides start/end/text/words.
    """
    text = " ".join(w.get("word", w.get("text", "")).strip() for w in words).strip()
    return {
        **original_seg,
        "start": float(words[0]["start"]),
        "end": float(words[-1]["end"]),
        "text": text,
        "words": words,
    }


def _find_split_index(words: list[dict]) -> int:
    """
    Find the best word index to split a list of words into two halves.

    Search order:
      1. Last word ending with hard punctuation (.!?) in the left half
      2. Last word ending with soft punctuation (,;) in the left half
      3. Midpoint by word index

    Returns the 0-based index of the LAST word in the first half (inclusive).
    The second half starts at index + 1.
    """
    n = len(words)
    mid = n // 2

    # Search left half (exclusive of the very last position so both halves have ≥1 word)
    best_hard = -1
    best_soft = -1
    for i in range(mid - 1, 0, -1):
        token = words[i].get("word", words[i].get("text", "")).strip()
        if _HARD_SPLIT_RE.search(token):
            best_hard = i
            break
        if _SOFT_SPLIT_RE.search(token) and best_soft == -1:
            best_soft = i

    if best_hard > 0:
        return best_hard
    if best_soft > 0:
        return best_soft
    # Fall back to midpoint, but ensure both halves have at least 1 word
    return max(1, mid - 1)


def _split_long_segment(seg: dict) -> list[dict]:
    """
    Recursively split a segment whose word-based duration exceeds MAX_CUE_DURATION.

    Rules:
    - Only splits when words array is present and has ≥2 words.
    - Recursively splits each half until all sub-segments are ≤ MAX_CUE_DURATION.
    - Never produces a cue shorter than MIN_CUE_DURATION; if splitting would do so,
      returns the segment unsplit.
    - Edge cases:
        * Single word (cannot split): return as-is.
        * words array absent/empty: return as-is (caller falls back to seg boundary).
        * Word timestamps missing/malformed: return as-is.
        * Resulting half < MIN_CUE_DURATION: abort split, return parent unsplit.
    """
    words = seg.get("words")
    if not words or len(words) < 2:
        return [seg]

    try:
        duration = float(words[-1]["end"]) - float(words[0]["start"])
    except (KeyError, TypeError, ValueError):
        return [seg]

    if duration <= MAX_CUE_DURATION:
        return [seg]

    split_idx = _find_split_index(words)

    left_words = words[: split_idx + 1]
    right_words = words[split_idx + 1 :]

    # Safety: both halves must be non-empty
    if not left_words or not right_words:
        return [seg]

    # Abort if either half would be below the minimum readable duration
    try:
        left_dur = float(left_words[-1]["end"]) - float(left_words[0]["start"])
        right_dur = float(right_words[-1]["end"]) - float(right_words[0]["start"])
    except (KeyError, TypeError, ValueError):
        return [seg]

    if left_dur < MIN_CUE_DURATION or right_dur < MIN_CUE_DURATION:
        return [seg]

    left_seg = _make_seg_from_words(left_words, seg)
    right_seg = _make_seg_from_words(right_words, seg)

    # Recurse on each half
    return _split_long_segment(left_seg) + _split_long_segment(right_seg)


def _expand_segments(segments: list[dict]) -> list[dict]:
    """
    Apply word-boundary tightening and long-segment splitting to the full list.

    For each segment:
      - If it has a words array and its duration exceeds MAX_CUE_DURATION, split it.
      - Otherwise pass through unchanged (cue boundaries tightened later via
        _cue_boundaries when actually writing the SRT/VTT lines).

    Returns a new flat list of segments in monotonic start-time order.
    """
    result: list[dict] = []
    for seg in segments:
        result.extend(_split_long_segment(seg))
    return result


# ── Public formatters ──────────────────────────────────────────────────────────


def segments_to_srt(segments: list[dict]) -> str:
    """
    Convert a list of Whisper-style segments to SRT-formatted string.

    Each segment should have:
        start (float): start time in seconds
        end   (float): end time in seconds
        text  (str):   subtitle text
        words (list, optional): per-word dicts with start/end/word fields

    When words are present:
      - Cue boundaries use words[0]["start"] and words[-1]["end"] (DTW-accurate).
      - Segments longer than MAX_CUE_DURATION (7 s) are split at word boundaries.

    Segments with empty text are skipped.
    If start == end == 0 (fallback / timing-less segment), a 2-second
    synthetic duration is assigned per segment index.

    Returns an empty string if there are no displayable segments.
    """
    expanded = _expand_segments(segments)

    lines: list[str] = []
    index = 1

    for seg in expanded:
        text = seg.get("text", "").strip()
        if not text:
            continue  # skip silence / empty segments

        start, end = _cue_boundaries(seg, index)

        lines.append(str(index))
        lines.append(f"{format_timestamp_srt(start)} --> {format_timestamp_srt(end)}")
        lines.append(text)
        lines.append("")  # blank separator line
        index += 1

    return "\n".join(lines)


def segments_to_vtt(segments: list[dict]) -> str:
    """
    Convert a list of Whisper-style segments to WebVTT-formatted string.

    Same structure as segments_to_srt but uses WebVTT header and '.' millisecond
    separator in timestamps.  No numeric cue identifiers (not required by VTT spec).

    When words are present:
      - Cue boundaries use words[0]["start"] and words[-1]["end"] (DTW-accurate).
      - Segments longer than MAX_CUE_DURATION (7 s) are split at word boundaries.

    Returns a minimal "WEBVTT\n\n" if there are no displayable segments.
    """
    expanded = _expand_segments(segments)

    lines: list[str] = ["WEBVTT", ""]
    index = 1

    for seg in expanded:
        text = seg.get("text", "").strip()
        if not text:
            continue

        start, end = _cue_boundaries(seg, index)

        lines.append(f"{format_timestamp_vtt(start)} --> {format_timestamp_vtt(end)}")
        lines.append(text)
        lines.append("")
        index += 1

    return "\n".join(lines)
