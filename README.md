# fake-subsai-asr-gateway

A thin FastAPI gateway that adapts the [openai-whisper-asr-webservice][whisper-ws]
HTTP protocol — the one Bazarr's `whisperai` provider speaks — to an arbitrary
backend Whisper transcription service plus an optional machine-translation
service. It has no GPU and loads no models; it routes calls, reshapes payloads,
serialises concurrent requests so a small host doesn't thrash, and reassembles
SRT/VTT/TXT/JSON responses.

The gateway is what stands between Bazarr and your own self-hosted ASR + MT
stack. Anything that speaks the asr-transcription-lv `/transcribe` shape and
optionally the back-translator-lv `/translate/batch` shape can sit behind it.

[whisper-ws]: https://github.com/ahmetoner/whisper-asr-webservice

---

## What it is good for

- **Bazarr drop-in**: Point Bazarr's WhisperAI provider at this gateway and
  Bazarr's standard `/asr` + `/detect-language` flow Just Works.
- **Source × target language routing**: One `/asr` endpoint detects the source
  language and dispatches to the right pipeline (passthrough transcribe,
  Whisper task=translate to English, or transcribe + MT to any target).
- **Pipeline serialisation**: Concurrent Bazarr requests are queued on disk
  (Starlette spooled multipart) so memory peaks at one request's worth of audio
  instead of N × audio.
- **Optional VAD + vocal isolation + forced alignment**: Each post-processor is
  a separate sidecar service the gateway can call when configured.
- **GPU supervisor integration**: Coordinates with an external supervisor so
  multiple GPU services on the same card don't trample each other.

---

## Architecture

```
Bazarr (whisperai provider)
  │  POST http://your-host-ip:9001/{detect-language,asr,…}
  ▼
fake-subsai-asr-gateway (this repo)
  ├─ POST /detect-language       → ASR backend (first 30 s only)
  ├─ POST /asr                   → ASR backend → SRT/VTT/TXT/JSON
  ├─ POST /asr-translate-lv      → ASR backend + MT backend → translated SRT
  ├─ POST /asr-lv-native         → ASR backend (language=lv) → LV SRT
  ├─ POST /translate             → MT backend (proxy, supervisor-mediated)
  ├─ POST /translate/batch       → MT backend (proxy, supervisor-mediated)
  ├─ GET  /health                → dependency reachability JSON
  └─ GET  /status                → human-readable status board (auto-refresh 3s)
  ▼
Configurable backends (HTTP):
  • ASR_URL              — Whisper service (asr-transcription-lv shape)
  • BACK_TRANSLATOR_URL  — NLLB-200 service (back-translator-lv shape)
  • VOCAL_ISOLATOR_URL   — Demucs isolation service (optional)
  • FORCED_ALIGNER_URL   — MMS_FA word-timestamp refinement (optional)
  • GPU_SUPERVISOR_URL   — claim/release coordinator (optional)
```

The gateway is intentionally I/O-bound — every heavy operation is a downstream
HTTP call. The container does not require a GPU, never loads a model, and is
small enough to run with a 3.5 GB memory limit alongside the rest of the stack.

---

## Quickstart

```bash
# 1. Clone and adjust the env block in docker-compose.yml to point at your
#    ASR + MT services (defaults assume host-mapped ports on localhost).
git clone https://github.com/davidgut1982/fake-subsai-asr-gateway.git
cd fake-subsai-asr-gateway

# 2. Build and start
docker compose up -d --build

# 3. Verify
curl -sf http://localhost:9001/health
open http://localhost:9001/status   # human-readable status board
```

The gateway listens on port `9001`. Bazarr should be pointed at
`http://your-host-ip:9001` in the WhisperAI provider settings.

---

## Endpoints

All endpoints accept `multipart/form-data`. The primary audio field name is
`audio_file` (openai-whisper-asr-webservice standard); `file` is accepted as a
legacy alias. Query parameters Bazarr sends on every call:

| Query param | Type | Default | Notes |
|-------------|------|---------|-------|
| `task` | str | `transcribe` | `transcribe` or `translate` |
| `language` | str | None | ISO 639-1 or ISO 639-2 (target subtitle language) |
| `output` | str | `srt` | Response format — `srt`, `vtt`, `txt`, or `json` |
| `encode` | bool | `true` | `false` → upload is raw s16le PCM (Bazarr default) |
| `video_file` | str | None | Path on the caller's host — logged only, never opened |

### `POST /detect-language`

Called by Bazarr before every transcription. Returns the detected language of
the audio.

**Request**: multipart `audio_file` (full audio; the gateway truncates to the
first 30 s before forwarding so Bazarr's ~30 s timeout is comfortably met).

**Response**:
```json
{ "detected_language": "english", "language_code": "en" }
```

`language_code` is ISO 639-1. Bazarr checks this field by exact key name — any
other shape triggers "WhisperAI returned empty language code".

### `POST /asr`

Smart routing. Detects the source language (or accepts an explicit one) and
dispatches to the matching pipeline:

| src (detected) | target (`language` param) | Pipeline |
|----------------|---------------------------|----------|
| `en` | `en` | Whisper transcribe → EN |
| `en` | `lv` | Whisper transcribe → NLLB EN→LV → LV |
| `lv` | `lv` | Whisper transcribe (LV) → LV |
| `lv` | `en` | Whisper `task=translate` → EN |
| `*`  | `en` | Whisper `task=translate` → EN |
| `*`  | `lv` | Whisper transcribe → NLLB src→LV (best effort) |
| any  | (none) | Transcribe in detected source language |

ISO 639-2 codes (`lav`, `eng`) are normalised to 639-1 (`lv`, `en`) at entry.

### `POST /asr-translate-lv`

EN audio → LV subtitles via Whisper + NLLB-200 batch MT. Timecodes are
preserved cue-by-cue. On MT failure for a segment, the cue is prefixed with
`[MT-FAIL]` so the subtitle track remains complete.

### `POST /asr-lv-native`

LV audio → LV subtitles via a dedicated Latvian Whisper model. Adds
forced-aligner post-processing for tighter word boundaries.

### `POST /translate` and `POST /translate/batch`

Thin proxies to the MT backend. The gateway claims/releases the MT service via
the GPU supervisor on each call so long batch jobs do not starve higher-priority
services holding their own GPU claims. Use these when a client (e.g. Lingarr)
would otherwise hit the MT service directly and bypass coordination.

### `GET /health`

```json
{ "status": "ok", "dependencies": { "asr": "ok", "back_translator": "ok" } }
```

### `GET /status`

Auto-refreshing (3 s) HTML status board showing the current in-flight request,
the queue of waiting requests, the last 20 completions, and dependency health.
Designed to be readable on a phone or terminal browser.

---

## Configuration

All configuration is environment variables. Defaults assume the gateway runs in
`network_mode: host` next to its backends.

| Variable | Default | Purpose |
|----------|---------|---------|
| `ASR_URL` | `http://localhost:8101` | Whisper backend (openai-whisper-asr-webservice `/transcribe`) |
| `BACK_TRANSLATOR_URL` | `http://localhost:8104` | NLLB MT backend (`/translate`, `/translate/batch`) |
| `ASR_TIMEOUT_SECONDS` | `900` | Per-request timeout for ASR calls (long for full films) |
| `MT_TIMEOUT_SECONDS` | `30` | Per-batch timeout for MT calls |
| `VOCAL_ISOLATOR_URL` | `http://localhost:8106` | Optional Demucs isolation service |
| `ENABLE_VOCAL_ISOLATION` | `true` | Set `false` to bypass vocal isolation entirely |
| `VOCAL_ISOLATOR_TIMEOUT_S` | `1800` | 30-minute ceiling for full-film isolation |
| `MIN_AUDIO_S_FOR_ISOLATION` | `30` | Skip isolation for clips shorter than this |
| `FORCED_ALIGNER_URL` | `http://localhost:8102` | Optional MMS_FA aligner service |
| `ENABLE_FORCED_ALIGN` | `true` | Set `false` to bypass alignment |
| `GPU_SUPERVISOR_URL` | `http://localhost:8202` | Optional GPU claim/release coordinator |
| `API_KEY` | *(empty)* | When set, require `X-API-Key` header on every non-public endpoint |

### Authentication

`API_KEY` controls a tiny middleware:

- **Empty (default)** — every endpoint is open. Suitable when the gateway is
  reachable only from a trusted LAN.
- **Non-empty** — every request whose path is not in `{/health, /status, /docs,
  /redoc, /openapi.json}` must carry `X-API-Key: <value>`. Mismatches return
  HTTP 401.

Use the second mode whenever the gateway is exposed beyond a trusted network.

---

## Audio format — the `encode=false` contract

Bazarr always sends `?encode=false`. Per the openai-whisper-asr-webservice spec
the uploaded bytes are **raw signed 16-bit little-endian PCM at 16 kHz mono**
with no RIFF/WAVE container header.

The gateway detects this by inspecting the first 4 bytes of the upload. If they
are not `RIFF` (plus `WAVE` at offset 8), it wraps the bytes in a 44-byte WAV
header before forwarding to the ASR backend (which requires a container format).

How to verify raw PCM in the field: divide file size by 32,000 bytes/s (s16le
16 kHz mono). If the result matches the audio duration with no overhead, it is
raw PCM. Example: 144,574,042 bytes ÷ 32,000 = 4,517.9 s = 75 min 17 s.

---

## GPU supervisor integration

When `GPU_SUPERVISOR_URL` points at a running supervisor, the gateway calls
`POST /claim/<service>` before invoking each GPU-bound dependency
(`asr-transcription-lv`, `back-translator-lv`, `vocal-isolator-lv`) and
`POST /release/<service>` afterwards. The supervisor implements:

- **Refcount-based eviction** so multiple callers can share a service.
- **Tiered priority** so a high-priority interactive workload temporarily
  defers low-priority batch ones (the supervisor returns HTTP 503 with a
  `reason: tier3_yield` body and a `Retry-After` value; the gateway converts
  that into HTTP 503 + `Retry-After` for Bazarr, which then retries naturally).

If the supervisor is unreachable the gateway degrades gracefully — it logs a
warning per claim attempt and proceeds without coordination. Set
`GPU_SUPERVISOR_URL` to an unreachable hostname to disable coordination
entirely.

A compatible supervisor implementation lives at
[davidgut1982/gpu-supervisor](https://github.com/davidgut1982/gpu-supervisor).

---

## Performance notes

Wall-clock latency for a 2-hour film, measured on a single Tesla P4 (8 GB)
shared between Whisper int8_float16, NLLB-200-distilled-600M, and Demucs:

| Step | 2-hour film |
|------|-------------|
| `/detect-language` (30 s of audio) | ~700 ms |
| Whisper ASR (EN or LV, int8_float16) | 10–20 min |
| Batch MT, ~1000 segments (NLLB-200) | 30–90 s |
| SRT formatting | <1 s |
| **Total EN → LV** | **11–21 min** |

Memory budget for a 2-hour film, mapped to the `mem_limit: 3584m` in
`docker-compose.yml`:

| Source | Approximate size |
|--------|------------------|
| Vocals response from vocal-isolator (FP32) | ~2.4 GB |
| Input audio buffer (raw PCM) | ~150 MB |
| WAV-wrapped copy | ~150 MB |
| Python + httpx + framework overhead | ~500 MB |
| Margin | ~400 MB |

Pipeline serialisation (`asyncio.Lock`) ensures only one request holds these
buffers at a time — concurrent requests are queued on disk by Starlette.

---

## Error handling

| Error | Cause | Resolution |
|-------|-------|------------|
| `[MT-FAIL] <text>` in a cue | NLLB batch translation failed for that segment | Source text preserved; subtitle track stays complete |
| HTTP 504 from `/asr` family | ASR backend exceeded `ASR_TIMEOUT_SECONDS` | Increase the env var, check backend health |
| HTTP 502 from `/asr` family | ASR or MT backend unreachable | Check `GET /health`, restart the relevant container |
| HTTP 503 + `Retry-After` | Supervisor deferred a Tier 3 claim because a higher-priority service is active | Caller (Bazarr) retries naturally on the header value |
| HTTP 401 `Unauthorized` | `API_KEY` is set and the request did not include a matching `X-API-Key` | Provide the header or unset `API_KEY` |
| "WhisperAI returned empty language code" (Bazarr) | `/detect-language` returned the wrong shape or timed out | Check gateway logs, verify the JSON shape |
| "Only translations to English supported!" (Bazarr) | The source file has an `eng` audio language tag | See Known limitations |

---

## Known limitations

### Bazarr refuses EN→LV translation for English-tagged source files

Bazarr's whisperai provider contains a hardcoded check: if the source audio
track has an `eng` (English) language metadata tag, the provider will not
request non-English output. The check fires before the gateway is called.
Options:

1. **Strip the audio language tag** with `mkvpropedit` before Bazarr scans:
   ```bash
   mkvpropedit movie.mkv --edit track:a1 --set language=und
   ```
2. **Call `/asr-translate-lv` directly** from any client that does not impose
   the same restriction.
3. **Generate subtitles manually**:
   ```bash
   curl -X POST \
     -F "audio_file=@movie.wav" \
     "http://your-host-ip:9001/asr-translate-lv?task=transcribe&language=en&output=srt&encode=false" \
     -o movie.lv.srt
   ```

### MT quality

NLLB-200-distilled-600M (EN→LV) produces grammatically coherent subtitles but
the output is noticeably machine-translated. BLEU ~16-17 on EN→LV. Adequate for
comprehension, not for literary quality.

---

## Project layout

```
fake-subsai-asr-gateway/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── app/
    ├── main.py                  # FastAPI endpoints, routing, format helpers
    ├── mt_translator.py         # Batch MT client (back-translator-lv shape)
    ├── subtitle_utils.py        # segments_to_srt(), segments_to_vtt()
    ├── vocal_isolator_client.py # Optional Demucs-isolation HTTP client
    └── forced_aligner_client.py # Optional MMS_FA aligner HTTP client
```

Rebuild after code changes:

```bash
docker compose up -d --build
```

---

## License

See repository for license terms.
