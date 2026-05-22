# Research: Vocal Isolation + Whisper ASR — Existing Open-Source Solutions

**Date**: 2026-05-05  
**Context**: Bazarr subtitle generation for Latvian movie/TV content. Single RTX 3060 12GB. Need Demucs/UVR vocal isolation before Whisper to suppress hallucinations on music sections.  
**Search engines used**: Exa + Tavily (cross-validated)

---

## A. Existing Solutions Found

### 1. ventura8/Whisper-Pro-ASR

**GitHub**: https://github.com/ventura8/Whisper-Pro-ASR  
**Docker Hub**: https://hub.docker.com/r/ventura8/whisper-pro-asr  
**Stars**: 1 (tiny project, single contributor)  
**Last commit**: 2026-02-01 (v1.0.0 release)  
**License**: MIT

This is the closest match to exactly what we built. It is a drop-in replacement for `ahmetoner/whisper-asr-webservice`, explicitly targeting Bazarr integration, implementing the identical `/asr` and `/detect-language` endpoints, and adding **UVR/MDX-NET vocal isolation as a first-class preprocessing stage** that runs before every transcription. It uses `faster-whisper` with `Systran/faster-whisper-large-v3` by default, supports int8/float16 compute types, and ships a complete `docker-compose.yml`.

The preprocessing pipeline: FFmpeg normalize → UVR MDX-NET vocal isolation (GPU) → faster-whisper inference. This is precisely our pipeline. Key env vars: `ENABLE_VOCAL_SEPARATION=true`, `VOCAL_SEPARATION_MODEL=UVR-MDX-NET-Inst_HQ_3.onnx`, `ASR_DEVICE=CUDA`. The architecture diagram in the README matches our design almost exactly.

**Fit assessment**: Architecture match is near-perfect. Bazarr-compatible protocol. faster-whisper backend. UVR (not Demucs — uses MDX-NET ONNX models) for isolation. No forced alignment. No mention of Latvian specifically but uses large-v3 which covers it. **Main concern**: 1 star, 1 contributor, last push was February 2026. Effectively an experiment rather than a maintained project.

---

### 2. EtienneAb3d/WhisperHallu

**GitHub**: https://github.com/EtienneAb3d/WhisperHallu  
**Stars**: 349  
**Last commit**: 2024-11-12  
**License**: implicit (research/commercial — "demonstration of our know-how")

The original proof-of-concept that named the problem. It combines Demucs or Spleeter for vocal extraction, Silero VAD for silence removal, ffmpeg loudness normalization, a speech compressor, and a "voice marker" hallucination-detection trick (inject inaudible markers; if Whisper echoes them back, transcription is valid). Supports openai-whisper and faster-whisper backends.

The hallucination marker trick is clever and unique to this project. However it explicitly notes V3 "seems bad with music" and defaults to V2. The repo has no Docker support. No HTTP API — it is a Python library you call directly. No chunking documentation for long-form (75-min) files. A user in issues confirmed a 4GB WAV "would just spin forever." **13 open issues, no Docker, not Bazarr-compatible.** The author's own karaoke project (karaok-AI) uses it as a dependency.

**Fit assessment**: Conceptually the right idea, but a research tool not a service. Would require substantial wrapping to become a Bazarr backend.

---

### 3. LunarCommand/audio-refinery

**GitHub**: https://github.com/LunarCommand/audio-refinery  
**Stars**: 0 (brand new, Feb–March 2026)  
**Last commit**: 2026-03-06  
**License**: MIT

The most technically sophisticated pipeline found. Implements what the author calls the "Ghost Track strategy": run all AI models (Demucs, Pyannote diarization, WhisperX) against the clean vocal stem extracted by `htdemucs --two-stems=vocals`, then apply resulting timestamps/text back to original. Includes Wav2Vec2 forced alignment via WhisperX for word-level timestamps. Batch pipeline with per-file VRAM cleanup. Designed for 24GB GPUs with all models resident simultaneously.

Handles long-form via `--segment N` flag on Demucs and batch-size controls on WhisperX. The cleanup logic deletes `no_vocals.wav` immediately after separation and `vocals.wav` after transcription, bounding scratch to ~400MB per file. Includes GPU temperature monitoring and thermal shutdown.

**Fit assessment**: Best architecture among all candidates for the underlying pipeline logic. However: no HTTP API, no Bazarr integration, no SRT output, designed for building RAG databases not subtitle files, needs 24GB VRAM (we have 12GB), and uses PyAnnote diarization (HuggingFace gated access required). Not usable directly.

---

### 4. McCloudS/subgen

**GitHub**: https://github.com/McCloudS/subgen  
**Stars**: ~1000+ (active project)  
**Bazarr-compatible**: Yes (implements whisper-asr-webservice protocol)  
**Last commit**: Active as of 2026

This is the project Bazarr's wiki actually points to as the canonical whisper-asr-webservice backend. Uses `faster-whisper` + `stable-ts` under the hood. Speaks the exact Bazarr whisper protocol. Ships Docker image with GPU support. **No vocal isolation whatsoever.** Handles chunking via stable-ts. Actively maintained. The Bazarr wiki page for the Whisper provider says "Bazarr's Whisper provider communicates with SubGen."

**Fit assessment**: The production-grade Bazarr backend, but it does not solve the music hallucination problem at all.

---

### 5. absadiki/subsai

**GitHub**: https://github.com/absadiki/subsai  
**Stars**: 1648  
**Forks**: 140  
**Last commit**: Active  
**License**: GPL-3

A subtitle generation tool (Web UI + CLI + Python package) supporting openai-whisper, faster-whisper, whisperX, stable-ts, whisper.cpp, and HuggingFace Transformers backends — all switchable. Includes subtitle format conversion and auto-sync. Ships Docker image. **No vocal isolation preprocessing.** This is the "SubsAI" project implied by the Bazarr provider name — the provider is named `whisperai` and the service was historically called SubsAI, but the actual Bazarr integration now points at SubGen and ahmetoner's webservice, not this package.

**Fit assessment**: Good multi-backend Whisper wrapper but lacks vocal isolation. The Docker image is available but this is not the Bazarr protocol backend.

---

### 6. ahmetoner/whisper-asr-webservice

**GitHub**: https://github.com/ahmetoner/whisper-asr-webservice  
**Stars**: Several thousand (major project)  
**Last commit**: Active, 321 commits  
**License**: MIT

The reference implementation of the Bazarr whisper protocol. Supports openai_whisper and faster_whisper backends. **No vocal isolation.** The issues list shows active discussion of adding diarization, sentiment, and VAD features, but no vocal isolation PRs. No fork in the network adds vocal isolation based on search results — ventura8/Whisper-Pro-ASR appears to be a from-scratch reimplementation that is *compatible with* this protocol, not a fork of it.

---

### 7. Nightingale (rzru/nightingale)

**GitHub**: https://github.com/rzru/nightingale  
**Stars**: modest (newer project, March 2026)  
**License**: MIT

A karaoke desktop app (Tauri/Rust + React) that implements the exact pipeline: UVR Karaoke model or Demucs for vocal isolation → WhisperX large-v3 for transcription with forced Wav2Vec2 alignment → word-level synchronized playback. Proves the pipeline works end-to-end for long-form audio. However: desktop app, not a service, no HTTP API, no Bazarr integration, music-focused (assumes songs, not TV dialogue).

**Fit assessment**: Useful as proof that UVR + WhisperX pipeline works at word-level alignment. Architecture patterns are directly applicable but the project itself is not adoptable.

---

### 8. karaok-AI (EtienneAb3d/karaok-AI)

Mentioned in WhisperHallu README. Uses WhisperHallu + WhisperTimeSync as dependencies for karaoke lyric extraction. Essentially confirms the Demucs → WhisperHallu → timestamp sync approach works, but inherits all of WhisperHallu's limitations (no service API, no Docker, research-grade).

---

## B. Recommendation

**ADOPT ventura8/Whisper-Pro-ASR as the reference implementation, but do not run it unmodified.**

It is the closest architectural match: Bazarr protocol compliant, faster-whisper backend, UVR vocal isolation as first-class preprocessing, Docker Compose example, NVIDIA CUDA support, configurable isolation model. The problem is that it has 1 star, was last updated in February 2026, and appears to be one person's proof-of-concept rather than a production system.

**Recommended integration path**: Use it as a template, not a dependency. The key insight from its codebase:

1. UVR MDX-NET (ONNX runtime, not Demucs) is faster for VRAM-constrained single-GPU deployments. On a 3060 12GB, running Demucs htdemucs and faster-whisper large-v3 sequentially (not simultaneously) is the safer approach. UVR MDX-NET models are lighter.
2. The ENABLE_VOCAL_SEPARATION toggle pattern is the right design: make it a feature flag so you can A/B test with and without.
3. Shared media volume path (pass filename rather than upload bytes) is architecturally correct for large movie files.

**Specific delta from SubGen** (the canonical Bazarr backend, which has real maintenance): SubGen lacks the UVR/Demucs preprocessing stage. If SubGen added a `VOCAL_SEPARATION=true` env var calling into `audio-separator` (the Python UVR wrapper), it would be exactly what we need. That gap is the one thing our custom service addresses that the ecosystem has not solved for Bazarr specifically.

---

## C. Anti-Patterns to Avoid

**1. Spleeter as the isolation model.** Spleeter (Deezer) is deprecated, no longer maintained, uses TensorFlow 1.x under the hood, and has severe quality limitations on old/SDTV audio compared to Demucs htdemucs or UVR MDX-NET. WhisperHallu mentions it as a fallback; do not use it.

**2. Processing the full 75-minute file as a single tensor.** The WhisperHallu issues confirm a 4GB WAV (approximately 75 min at 16kHz/16-bit mono) would "spin forever" without chunking. Demucs has a `--segment` flag (default: model-dependent, ~8 seconds for htdemucs). Without it, a 75-min movie will OOM on 12GB VRAM during the separation stage alone. Both audio-refinery and Nightingale use `--segment 40` or similar.

**3. Running Demucs AND faster-whisper-large-v3 in VRAM simultaneously on 12GB.** audio-refinery requires 24GB for this reason. The correct pattern on 12GB: run Demucs, unload it from VRAM, then load faster-whisper. Sequential, not concurrent.

**4. Using Whisper V3 with music-contaminated audio without preprocessing.** WhisperHallu explicitly states "V3 seems bad with music" and defaults to V2. UVR/Demucs preprocessing first neutralizes this — but if preprocessing fails or is skipped, use large-v2, not large-v3.

**5. Word timestamps from Whisper alone.** Whisper's word-level timestamps have known drift on long-form audio. Both audio-refinery and Nightingale use WhisperX + Wav2Vec2 forced alignment on top. For tight subtitle sync (which Bazarr needs), segment-level timestamps from Whisper are usually acceptable; word-level alignment is optional but requires a language-specific Wav2Vec2 model. For Latvian, check HuggingFace for `facebook/wav2vec2-large-xlsr-53-latvian` or similar — if none exists, WhisperX will fall back to Whisper's raw timestamps.

**6. No VRAM cleanup between stages.** audio-refinery documents this explicitly: delete ghost-track stems per-file, call `torch.cuda.empty_cache()` between stages.

---

## D. Patterns to Steal

**1. The "Ghost Track" strategy (audio-refinery).** Run AI against the clean vocal stem; apply resulting metadata back to the original audio for output. Never feed the original mixed audio to Whisper.

**2. `--two-stems=vocals` Demucs flag.** Do not run the full 4-stem separation (drums + bass + other + vocals) and then throw away 3 stems. `--two-stems=vocals` gives you `vocals.wav` + `no_vocals.wav` with the same quality and uses less scratch space.

**3. ENABLE_VOCAL_SEPARATION as a feature toggle (ventura8/Whisper-Pro-ASR).** Make it possible to disable in the docker-compose env for testing/debugging without rebuilding the image.

**4. Volume-mapping for local file access (ventura8/Whisper-Pro-ASR).** If Bazarr and the ASR service share the same media volume, the ASR service can read the file path directly instead of receiving it over HTTP. This is crucial for 75-min movies (several GB of uncompressed WAV is expensive to upload over localhost loopback).

**5. Per-file scratch cleanup with configurable keep flag.** audio-refinery's `--keep-scratch` pattern: default to deleting intermediate stems, but allow keeping them for debugging.

**6. Demucs segment parameter in docker-compose env.** `DEMUCS_SEGMENT=40` as a tunable. The right value on 12GB VRAM depends on the model (htdemucs: 40s is safe; htdemucs_ft: lower is needed).

**7. VAD as a pre-filter, not a replacement.** WhisperX uses VAD (Silero) to chunk the audio into speech segments before batched Whisper inference. This is not the same as vocal isolation — it handles silence/music *detection* but not *removal*. Use both: Demucs/UVR to remove music from the waveform, then VAD to skip silence segments. WhisperHallu layers both in sequence.

**8. Probabilistic multi-zone language detection (ventura8/Whisper-Pro-ASR).** For detect-language, sample multiple zones of the audio (beginning, middle, end) and use voting consensus rather than only the first 30 seconds. Bazarr defaults to 30s for language detection; for Latvian films the first 30s may be a title sequence with music.

---

## E. Honest Assessment

We built the right thing. No existing open-source project simultaneously satisfies all five of our constraints:

1. Bazarr `whisperai` provider protocol (POST /asr, POST /detect-language)
2. Vocal isolation before transcription (not just VAD)
3. faster-whisper backend with int8/float16
4. Docker Compose deployable on a single RTX 3060 12GB
5. Latvian language support

**ventura8/Whisper-Pro-ASR** comes closest (satisfies 1, 2, 3, 4 and implicitly 5 via large-v3) but is a 1-star experiment with no community. **SubGen** satisfies 1, 3, 4, 5 but not 2 — no vocal isolation. **audio-refinery** solves the core audio problem (2, 3, 5) beautifully but satisfies none of 1, 4 (requires 24GB GPU).

The 5 sequential bugs we hit are not evidence of a wrong approach — they are normal integration bugs for a pipeline with 4 models (FFmpeg, Demucs, faster-whisper, and the HTTP layer). The patterns above, especially the VRAM sequential loading and the ghost-track cleanup, should resolve the OOM class of bugs. The hallucination bugs are solved by the Demucs preprocessing itself.

The architecture is validated by multiple independent projects reaching the same conclusion. The implementation is ours to maintain.

---

## References

- ventura8/Whisper-Pro-ASR: https://github.com/ventura8/Whisper-Pro-ASR
- LunarCommand/audio-refinery: https://github.com/LunarCommand/audio-refinery
- EtienneAb3d/WhisperHallu: https://github.com/EtienneAb3d/WhisperHallu
- McCloudS/subgen: https://github.com/McCloudS/subgen
- absadiki/subsai: https://github.com/absadiki/subsai
- ahmetoner/whisper-asr-webservice: https://github.com/ahmetoner/whisper-asr-webservice
- rzru/nightingale: https://github.com/rzru/nightingale
- WhisperHallu hallucination discussion: https://github.com/openai/whisper/discussions/679
- Bazarr Whisper provider: http://wiki.bazarr.media/Additional-Configuration/Whisper-Provider/
- Bazarr whisperai.py: https://github.com/morpheus65535/bazarr/blob/master/custom_libs/subliminal_patch/providers/whisperai.py
- DigitalOcean Whisper+Spleeter karaoke tutorial: https://www.digitalocean.com/community/tutorials/how-to-make-karaoke-videos-using-whisper-and-spleeter-ai-tools
