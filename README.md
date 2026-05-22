# fake-subsai-asr-gateway

FastAPI gateway on `latvian-vm` (192.168.1.11) port **9001**. Acts as a thin HTTP
coordinator between Bazarr's `whisperai` provider and the backend ML services. It has
no GPU and loads no models — it adapts request formats, routes calls, and reassembles
responses.

## Architecture

```
Bazarr (192.168.1.10, Plex host)
  └─ whisperai provider
       └─ POST http://192.168.1.11:9001/{endpoint}

fake-subsai-asr-gateway  (latvian-vm, host network, port 9001)
  ├─ POST /detect-language   → asr-transcription-lv:8101  (first 30 s only)
  ├─ POST /asr               → asr-transcription-lv:8101  → EN SRT
  ├─ POST /asr-translate-lv  → asr-transcription-lv:8101 + back-translator-lv:8104 → LV SRT
  └─ POST /asr-lv-native     → asr-transcription-lv:8101 (language=lv) → LV SRT

asr-transcription-lv  port 8101 (internal): faster-whisper large-v3 int8_float16, RTX 3060
back-translator-lv    port 8104 (internal): NLLB-200-distilled-600M, EN→LV
```

---

## Endpoints

All endpoints accept `multipart/form-data`. The primary audio field name is `audio_file`
(openai-whisper-asr-webservice standard); `file` is accepted as a legacy alias.

All endpoints also accept (and require) the following query parameters that Bazarr sends
on every call:

| Query param | Type | Default | Notes |
|-------------|------|---------|-------|
| `task` | str | `transcribe` | `transcribe` or `translate` |
| `language` | str | None | ISO 639-1 or ISO 639-2 |
| `output` | str | `srt` | Response format — gateway always returns SRT |
| `encode` | bool | True | `false` = upload is raw s16le PCM (see below) |
| `video_file` | str | None | Path on Bazarr's host — logged only, never opened |

---

### `POST /detect-language`

Called by Bazarr before every transcription. Returns the detected language of the audio.

**Request**: multipart `audio_file` (full audio; gateway truncates to first 30 s before forwarding)

**Response**:
```json
{
    "detected_language": "english",
    "language_code": "en"
}
```

`language_code` is ISO 639-1. Bazarr checks this field by exact key name — any other name is
treated as absent and triggers "WhisperAI returned empty language code".

**Timeout note**: Bazarr has a hardcoded ~30 s timeout on this call. The gateway truncates the
forwarded audio to the first 30 seconds (960 000 bytes at s16le 16 kHz mono) so the ASR backend
finishes well within that window. This truncation is applied only here — transcription endpoints
receive the full audio.

---

### `POST /asr`

Transcribes audio and returns EN SRT subtitles. This is the primary Bazarr transcription endpoint.

**Request**: multipart `audio_file`

**Response**: `text/plain`, SRT format

**Pipeline**: audio → asr-transcription-lv (Whisper, language=en) → EN segments → SRT

---

### `POST /asr-translate-lv`

Transcribes English audio, translates to Latvian, returns LV SRT.

**Request**: multipart `audio_file`

**Response**: `text/plain`, SRT format

**Pipeline**:
```
audio → asr-transcription-lv (language=en) → EN segments
      → back-translator-lv /translate/batch (eng_Latn → lvs_Latn, batch=32)
      → LV segments → LV SRT
```

MT failure per segment: text returned as `[MT-FAIL] <original EN>`. Track remains complete.

---

### `POST /asr-lv-native`

Transcribes Latvian audio directly and returns LV SRT.

**Request**: multipart `audio_file`

**Response**: `text/plain`, SRT format

**Pipeline**: audio → asr-transcription-lv (language=lv) → LV segments → LV SRT

---

### `GET /health`

Returns status of gateway and downstream dependencies.

**Response**:
```json
{
    "status": "healthy",
    "dependencies": {
        "asr-transcription-lv": "ok",
        "back-translator-lv": "ok"
    }
}
```

---

## Audio format handling — the `encode=false` contract

Bazarr always sends `?encode=false`. Per the openai-whisper-asr-webservice spec this means:

> The uploaded bytes are **raw signed 16-bit little-endian PCM at 16 kHz mono** with no
> RIFF/WAVE container header.

The gateway detects this by inspecting the first 4 bytes of the upload. If they are not
`RIFF` (+ `WAVE` at offset 8), it wraps the bytes in a 44-byte WAV header before forwarding
to the ASR backend (which requires a container format).

**How to verify raw PCM in the field**: divide file size by 32,000 (bytes/second at s16le
16 kHz mono). If the result matches audio duration with no overhead, it is raw PCM.
Example: 144,574,042 bytes ÷ 32,000 = 4,517.9 s = 75 min 17 s (a real 75-minute movie).

Because the gateway buffers the entire upload to inspect it, the container memory limit is
set to **1024 MB** (raised from 256 MB during the 2026-05-05 debugging session).

---

## Integration with Bazarr

**Gateway URL** (from Bazarr's host): `http://192.168.1.11:9001`

No path prefix is needed. Bazarr's whisperai provider appends the endpoint paths itself.

### Bazarr Settings → Providers → WhisperAI

| Field | Value |
|-------|-------|
| Endpoint | `http://192.168.1.11:9001` |
| Response format | SRT |
| Timeout | Default (transcription); leave as-is |

### What Bazarr calls (per subtitle file)

1. `POST /detect-language` — language detection (~700 ms with truncation)
2. `POST /asr` — transcription for first requested language
3. `POST /asr` (or alternate endpoint) — transcription for second language, if configured

### Configuring the EN→LV chain

To get Latvian subtitles for English movies, add a second Whisper ASR provider in Bazarr
pointed at `/asr-translate-lv`:

1. Bazarr → Settings → Subtitles → Providers → Add
2. Provider: **WhisperAI**
3. Endpoint: `http://192.168.1.11:9001`
4. Under Advanced, override the endpoint path to `/asr-translate-lv` if supported,
   or configure a separate language profile that routes to this gateway for Latvian.

**Alternative**: call the endpoint directly for manual generation:
```bash
curl -X POST \
  -F "audio_file=@movie_audio.wav" \
  "http://192.168.1.11:9001/asr-translate-lv?task=transcribe&language=en&output=srt&encode=false" \
  -o movie.lv.srt
```

---

## Known limitations

### Bazarr refuses EN→LV translation for English-tagged source files

Bazarr's whisperai provider contains a hardcoded check: if the source audio track has an
`eng` (English) language metadata tag, the provider will not request non-English output and
raises "Only translations to English supported!" before calling the gateway.

This is a restriction in Bazarr's code, not the gateway. The gateway itself handles EN→LV
correctly. Options:

1. **Strip the audio language tag** — use `mkvpropedit` before Bazarr scans the file:
   ```bash
   mkvpropedit movie.mkv --edit track:a1 --set language=und
   ```
2. **Use a different Bazarr provider** — a provider that does not have this restriction
   can call the gateway's `/asr-translate-lv` endpoint directly.
3. **Generate LV subtitles manually** — `curl` the gateway endpoint directly and import
   the resulting `.srt` file into Bazarr.

### MT quality

NLLB-200-distilled-600M (EN→LV) produces grammatically coherent subtitles but the output
is noticeably machine-translated. BLEU ~16-17 on EN→LV. Adequate for comprehension, not
literary quality.

### Single-host memory constraint

The gateway buffers each full audio upload in RAM before forwarding (required by the raw-PCM
detection logic). For a 2-hour movie this is approximately 230 MB. The container memory limit
is 1024 MB, so up to 4 concurrent requests of that size would be the practical ceiling.
In practice Bazarr serializes subtitle requests so this is not an issue.

---

## Performance (RTX 3060, latvian-vm)

| Step | 2-hour movie |
|------|-------------|
| Language detection (`/detect-language`, 30-s audio) | ~700 ms |
| Whisper ASR (EN or LV) | 10–20 min |
| Batch MT ~1000 segments (NLLB-200) | 30–90 s |
| SRT formatting | < 1 s |
| **Total EN→LV** | **11–21 min** |

---

## Error handling

| Error | Cause | Resolution |
|-------|-------|------------|
| `[MT-FAIL] <text>` in SRT | NLLB batch translation failed for segment | EN text preserved; subtitle track stays complete |
| HTTP 504 from gateway | ASR backend exceeded `ASR_TIMEOUT_SECONDS` | Increase env var (default 900 s); check ASR container |
| HTTP 503 from gateway | Backend dependency unreachable | Check `GET /health`; restart relevant container |
| "WhisperAI returned empty language code" (Bazarr) | `/detect-language` returned wrong shape or timed out | Check gateway logs; verify `language_code` field present |
| "Only translations to English supported!" (Bazarr) | Source file has `eng` audio language tag | See Known limitations above |

---

## Dependencies

| Service | Internal port | Purpose |
|---------|--------------|---------|
| `asr-transcription-lv` | 8101 | Whisper large-v3 CT2 (all audio transcription) |
| `back-translator-lv` | 8104 | NLLB-200-distilled-600M EN→LV translation |

Check status:
```bash
curl http://192.168.1.11:9001/health
```

---

## Source files

```
/srv/latvian_learning/tilts-system/docker/fake-subsai-asr-gateway/
├── Dockerfile
├── docker-compose.yml        # memory_limit: 1024m (raised 2026-05-05)
├── requirements.txt
├── README.md                 # this file
├── README-LV-SUBTITLES.md   # older setup guide (pipeline architecture)
└── app/
    ├── main.py               # FastAPI endpoints, audio format helpers
    ├── mt_translator.py      # MTTranslator: batch EN→LV via back-translator-lv
    └── subtitle_utils.py     # segments_to_srt(), format_timestamp()
```

## Rebuild after code changes

```bash
cd /srv/latvian_learning/tilts-system/docker/fake-subsai-asr-gateway
docker compose up -d --build
```

---

## Debugging reference (KB)

- `kb_2f4b7ad8f9ef` — Full 6-bug debugging narrative (2026-05-05): field names, missing
  endpoints, raw PCM detection, audio truncation for language detection, byte math
- `kb_6ae675394b36` — openai-whisper-asr-webservice protocol gotchas: quick reference
- `kb_0fd3e6f1360b` — Earlier research: SubsAI vs whisper-asr-webservice (2025-12-12)
- `kb_72380ed314ce` — Alignment, audio preprocessing, and the Demucs path (2026-05-05)
- `kb_00ddfc9b350b` — CTC vs HMM forced alignment: when to use which
- `kb_61ab7c3b6e79` — Demucs htdemucs for ASR preprocessing: model selection, parameters, GPU budget

---

## word_timestamps and forced-alignment integration (2026-05-05)

### The timestamp drift problem

faster-whisper's default segment-level timestamps drift **2–10 seconds** on long-form
archival audio. This is not a quirk or edge case: on a 75-minute 1972 mono SDTV Latvian
film, one test line appeared in the SRT at `00:00:10` but was actually spoken between
`01:08` and `03:55` — approximately 3 minutes of drift. This is caused by Whisper's
cross-attention-based timestamp estimation, which produces plausible-looking but
acoustically unconstrained timestamps.

This is **language-agnostic** — the same drift occurs on English, Latvian, or any other
language. It is a property of the estimation algorithm, not the language model.

### Fix: word_timestamps=True (words=True in the ASR backend)

Pass `words=True` in the `/transcribe` request to asr-transcription-lv. This activates
per-word DTW (Dynamic Time Warping) alignment, which post-hoc aligns each decoded word
to the waveform. Measured accuracy: **±200–500 ms** vs 2–10 s for segment-level.

**Cost**: ~20-30% additional transcription time (negligible for batch use).

**SRT cue construction**: use `segment.words[0].start` and `segment.words[-1].end`
instead of the segment-level `segment.start` / `segment.end`.

**Cue length**: split segments longer than ~7 seconds at word boundaries regardless of
timestamp accuracy — this is a viewer readability constraint, not a timing issue.

### forced-aligner-lv (MMS_FA) — what it is and what it actually does

The `forced-aligner-lv` service uses **torchaudio MMS_FA** (Meta's Massively Multilingual
Speech Forced Alignment), a wav2vec2-based CTC aligner. It is **not** classical Montreal
Forced Aligner (MFA). The distinction matters for understanding its failure modes.

MMS_FA takes Whisper's decoded text and the audio waveform and refines the word timestamps
using CTC forced alignment. It reports a per-segment confidence score. Segments with "poor"
confidence should fall back to the Whisper word_timestamps values.

**Real-world results on 75-min 1972 SDTV film**:

| Metric | Value |
|--------|-------|
| Total Whisper segments | 111 |
| MMS_FA accepted (good quality) | 34 (31%) |
| Fell back to Whisper DTW | 77 (69%) |
| Improvement per accepted cue | +40 ms tighter (median) |

The 69% fallback is correct behavior. MMS_FA is honest about its limits: 1972 SDTV audio
is outside its training distribution (modern clean speech). The fallback to word_timestamps
still produces substantially better output than the default segment-level estimates.

On modern HD content, MMS_FA acceptance rate is expected to reach 80–95%.

### Volume permissions fix (forced-aligner-lv)

The service was in `degraded` state for 27 days due to a permissions issue.

**Root cause**: container runs as `appuser`; the `latvian_models_data` Docker volume was
owned by `root:root 0755`. The model download on first start failed silently.

**Fix** (run on the host before restarting the container):
```bash
chmod 777 /var/lib/docker/volumes/latvian_models_data/_data/
docker compose restart forced-aligner-lv
```

The ~1.2 GB MMS_FA model then downloaded successfully on restart.

### The Demucs preprocessing path (research phase — not yet implemented)

The 69% MMS_FA fallback rate traces to Whisper hallucinating text on music/ambient
sections of the mixed audio. The fix is upstream: strip music before Whisper sees the
audio using **Demucs `htdemucs`** vocal isolation.

**Decision rule for preprocessing**:
- Modern HD film (clean audio): Whisper + word_timestamps only (~7 min per 75-min film)
- Archival/SDTV film (music+dialogue mix): DeepFilterNet 3 → Demucs htdemucs → Whisper
  + word_timestamps (~25-30 min per 75-min film)

**GPU constraint**: Demucs requires ~3 GB VRAM; asr-transcription-lv holds ~1.9 GB.
They cannot coexist. Operational pattern: stop asr-transcription-lv → run Demucs
(~20 min) → restart asr-transcription-lv → run Whisper+alignment.

**Critical parameters**:
- Model: `htdemucs` (not `htdemucs_ft` — 4x slower for no benefit on archival audio)
- Never use Spleeter — it introduces phase artifacts that break forced alignment
- `--two-stems=vocals` does NOT speed up processing (always runs full 4-stem internally)
- Cache vocal stems by SHA-256 content hash; ~150 MB per 75-min film

See `kb_61ab7c3b6e79` for full operational detail on Demucs parameters and GPU budget,
and `kb_72380ed314ce` for the complete preprocessing pipeline design.
