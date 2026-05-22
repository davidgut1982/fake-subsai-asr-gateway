# Latvian Subtitle Generation — Setup Guide

Gateway: `fake-subsai-asr-gateway` on port **9001**

## New Endpoints (2026-05-04)

| Endpoint | Input | Output | Pipeline |
|---|---|---|---|
| `POST /asr` | Any audio + `language=en` | EN SRT | Whisper (existing Bazarr path, unchanged) |
| `POST /asr-translate-lv` | EN audio | LV SRT | Whisper + NLLB-200 MT chain |
| `POST /asr-lv-native` | LV audio | LV SRT | Latvian Whisper (whisper-lv-ct2) |

All endpoints accept multipart form-data with `file` (audio) + optional `language`.

## Architecture

```
Bazarr → fake-subsai-asr-gateway:9001
         │
         ├─ POST /asr (language=en, task=transcribe)
         │       └─ asr-transcription-lv:8101/transcribe → EN segments → EN SRT
         │
         ├─ POST /asr-translate-lv
         │       ├─ asr-transcription-lv:8101/transcribe (language=en) → EN segments
         │       ├─ back-translator-lv:8104/translate/batch (eng_Latn→lvs_Latn)
         │       └─ LV segments → LV SRT
         │
         └─ POST /asr-lv-native
                 └─ asr-transcription-lv:8101/transcribe (language=lv) → LV segments → LV SRT
```

The gateway has no GPU and no ML models — it is a pure HTTP coordinator.

## Bazarr Configuration

### Mode 1: EN audio → EN subtitles (closed captions)

Already working. Bazarr's existing Whisper ASR provider pointing at:
```
http://localhost:9001/asr
```
with language set to `English`.

### Mode 2: EN audio → LV subtitles (translation)

Add a **second** Whisper ASR provider entry in Bazarr:

1. Open Bazarr → Settings → Subtitles → Whisper ASR
2. Add a new provider with:
   - **URL**: `http://localhost:9001/asr-translate-lv`
     (or `http://192.168.1.11:9001/asr-translate-lv` from the network)
   - **Language**: Latvian
3. Create a language profile for Latvian using this provider.
4. Apply the Latvian language profile to movies in your library.

When Bazarr requests Latvian subtitles for an English-audio movie, it will POST
the audio to `/asr-translate-lv` and receive an LV-formatted SRT back.

### Mode 3: LV audio → LV subtitles

For movies with Latvian audio tracks:

- **URL**: `http://localhost:9001/asr-lv-native`
- **Language**: Latvian

This routes to the `whisper-lv-ct2` model (Whisper large-v3 fine-tuned for
Latvian, int8-quantised for Tesla P4). Better Latvian accuracy than generic
Whisper.

## Performance Estimates (Tesla P4 GPU)

| Step | 2-hr movie |
|---|---|
| Whisper ASR (EN or LV) | 5–15 min |
| Batch MT ~1000 segments | 30–90 s |
| SRT formatting | < 1 s |
| **Total EN→LV** | **6–17 min** |

MT uses the `/translate/batch` endpoint (batch size 32) — single GPU pass per
chunk, cache hits are free. A 2-hr movie has roughly 800–1200 subtitle segments.

## Error Handling

- **MT failure per segment**: text is returned as `[MT-FAIL] <original EN text>`.
  The subtitle track remains complete — only the failed lines stay in English.
- **ASR timeout**: HTTP 504. Increase `ASR_TIMEOUT_SECONDS` env var if needed
  (default 900 s / 15 min).
- **Dependency down**: `GET /health` shows which dependency is unreachable.

## Dependencies

| Service | Host Port | Purpose |
|---|---|---|
| `asr-transcription-lv` | 8101 | Whisper large-v3 CT2 (Latvian model) |
| `back-translator-lv` | 8104 | NLLB-200-distilled-1.3B (EN↔LV MT) |

Both must be running. Check status:
```bash
curl http://localhost:9001/health
```

## Source Files

```
/srv/latvian_learning/tilts-system/docker/fake-subsai-asr-gateway/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README-LV-SUBTITLES.md  ← this file
└── app/
    ├── main.py             # FastAPI endpoints
    ├── mt_translator.py    # MTTranslator: batch EN→LV via back-translator-lv
    └── subtitle_utils.py   # segments_to_srt(), format_timestamp()
```

## Rebuild After Code Changes

```bash
cd /srv/latvian_learning/tilts-system/docker/fake-subsai-asr-gateway
docker compose build
docker compose up -d
```

Or rebuild and restart in one step:
```bash
docker compose up -d --build
```

## Open Items for User Testing

1. **Configure Bazarr** — add 2nd language profile (Latvian → `/asr-translate-lv`)
2. **Test with a real EN movie** — full pipeline end-to-end with speech
3. **Test with a real LV movie** — `/asr-lv-native` path
4. **Quality review** — NLLB-200 BLEU 16.52 on EN→LV is functional but not
   literary. Subtitle text will be grammatically correct but may sound
   machine-translated compared to professional subtitles.
5. **If MT latency is too slow** — the batch endpoint already provides 3-8x
   speedup. Further options: reduce `num_beams` (default 4→2 in back-translator
   request) or upgrade to back-translator-ct2-1b3 if it's faster on this GPU.
