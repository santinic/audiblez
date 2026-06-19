# MOSS resident pipe co-process — impl spec

Design spec for the MOSS engine integration (ADR 0001/0002/0003). **Gated on the Phase-0 C++
spike.** Grounded against the real fork source `/home/caio/Projects/llama.cpp-moss/tools/tts/
run-moss-tts-delay.cpp` (branch `moss-tts-firstclass`); both feasibility claims were adversarially
confirmed (2026-06-18). Re-anchor by symbol, not line number, if the fork moves.

## Current binary shape (one-shot)
`main()` text branch (@2545) → `moss_generate_from_text` → **loads the model 3×**: vocab-only
backbone (@2009), full backbone + per-prompt context (@1845/1866), audio-decoder (@1297); a 4th
load (encoder, @1221) when `--reference-audio`. Writes a 16-bit PCM WAV via `save_wav16` (@1150),
exits. No stdin loop, no stdout PCM path.

## The `--serve` patch (the real work)
All in `run-moss-tts-delay.cpp`:
- **Startup, once:** `llama_backend` init; load full backbone (mparams @1841-1843, **not** vocab_only);
  create backbone ctx sized **worst-case** (`n_ctx = MAX_SENTENCE_FRAMES + max_new_tokens + 8`,
  `n_batch = MAX_SENTENCE_FRAMES`); load audio-decoder + ctx (`moss_init_audio_context`,
  `n_ctx = MAX_RAW_FRAMES`); load encoder + ctx if a default ref/clone is in play.
- **Eliminate the vocab-only load:** `moss_build_prompt_input` takes only a `const llama_vocab*`
  (@1451) obtainable from the live backbone via `llama_model_get_vocab` (@1850/2014). One persistent
  backbone serves prompt-build + generation.
- **Refactor** `moss_generate_from_prompt` (@1845) and `moss_decode_audio_llama` (@1297) to split
  *load* from *body* (body takes ctx+model args). Keep the one-shot `--text` path routing through the
  same body fns so its behavior is unchanged.
- **Per request:** read one JSON line from stdin → `llama_memory_clear(llama_get_memory(ctx), true)`
  on **each** reused ctx (backbone + audio; audio ctx is non-causal, `llama_set_causal_attn(ctx,false)`
  @532, so it also needs the clear or sentence N+1 is KV-contaminated by N) → **re-seed** `moss_rng`
  from the request seed (@1890; without reseed audio is non-reproducible) → `moss_build_prompt_input`
  → gen loop (@1875-1921) → decode body (@1315-1334) → `save_wav16` to the request's `wav_out` →
  print one JSON response line → flush.
- **LOG pollution fix:** the synth-path `LOG()` macro prints to **stdout** (common/log.cpp:81-85, used
  e.g. @1924). Redirect those to `LOG_INF` (stderr) so stdout stays a clean JSON control channel.
  (Audio is file-handoff, not piped, so only the control channel needs protecting.)

## Wire protocol — "MOSS resident pipe v1"
Decided: **16-bit WAV file-handoff** (not float32/stdout — no audible gain after the m4b AAC pass,
and audiblez's chapter wavs are 16-bit anyway; reuses `save_wav16` untouched).

- **Channels:** stdin = 1 JSON request/line; stdout = 1 JSON response/line (control only); stderr =
  llama logs — **drain continuously** (thread or per-child logfile, rotate/cap) or a full pipe
  deadlocks the child. Audio is **not** on the pipe — child writes a temp WAV, parent reads via
  `soundfile.read(dtype='float32')` (the `moss_bench.py:101` path).
- **REQUEST** p→c: `{"v":1,"id":N,"op":"synth","text":"...","wav_out":"/abs/req-N.wav","seed":S,`
  `"sampling":{"text_temperature":1.5,"text_top_k":50,"audio_temperature":1.7,"audio_top_p":0.8,`
  `"audio_top_k":25,"audio_repetition_penalty":1.0},"max_new_tokens":2048}`. Parent **unlinks
  `wav_out` before sending** (stale-WAV guard). **No `speed` field** — speed is applied downstream
  (see below).
- **OK** c→p: `{"v":1,"id":N,"status":"ok","wav_out":"...","frames":F,"sample_rate":24000}`. `F>0`
  required; parent validates id + file exists + frame count. Empty audio → `error` code
  `empty_audio`, **never** ok-with-zero (so `np.zeros(0)` is never cached).
- **ERROR** c→p: `{"v":1,"id":N,"status":"error","code":"gen_failed|empty_audio|bad_request","message":"..."}`.
- **SHUTDOWN:** `{"op":"shutdown"}` → `{"status":"bye"}` exit 0; stdin EOF is the backstop.

## Failure vs death (two-axis) — lives inside the synth closure
- **(A) `status:"error"` line = real failure** → closure raises → `_synth_one_or_silence` (core.py:556)
  dead-letters + splices silence. Child stays healthy.
- **(B) EOF on stdout OR `proc.poll() is not None` OR response-timeout** (deadline scaled to sentence
  length, generous floor ~30s) = **death** → restart + re-dispatch the **same** sentence (infra, not
  the sentence — do **not** dead-letter the first death).
- **Poison-pill guard:** `GGML_ASSERT`→`abort()` kills the child with **no error line** every time, so
  the closure counts **consecutive deaths per sentence**; after **K=2** → dead-letter + silence
  (raised as a `_PERMANENT_SYNTH_ERRORS` type so `_retry` at core.py:513 doesn't re-spin a
  child-killing sentence). Without it, one bad sentence is an infinite crash loop.
- **Global circuit-breaker:** abort the run if the restart rate exceeds a threshold — a systemic crash
  (VRAM exhaustion mid-book) must not silently dead-letter many good sentences as silence.
- Restart logic owns the `Popen`; `synth(text,speed)->list[np.ndarray]@24kHz` stays intact;
  cli/ui/core + `_synth_one_or_silence`/`_synth_batch` unchanged. Closure returns `(synth, close)`;
  `core.main` tears MOSS down in a `finally`.

## Speed — atempo at chapter assembly
MOSS has **no speed knob** (no `--speed`; autoregressive). Decided: synth at 1.0; apply ffmpeg
`atempo` (pitch-preserving) **once per chapter** after concat. → MOSS sentence cache stays
**speed-agnostic** (reused across speed changes); **speed enters the chapter `.sig`** so a change
re-stretches. Never let speed silently no-op (it's a banned silent-wrongness trap — speed is in the
key).

## Cache
- MOSS sentence key adds **seed + all sampling params** (or pin them as constants folded into
  `CACHE_VERSION`). `engine` is already in `make_key` → MOSS (`llamacpp`) can never collide with
  Kokoro (`torch`/`mlx`); **no `CACHE_VERSION` bump needed for engine separation** — bump only if
  pinning the sampling defaults.
- `repo_id` for MOSS = a stable id hashing **all three GGUF identities** (backbone/decoder/encoder) —
  each independently affects the waveform.
- speed **not** in the MOSS sentence key (synth@1.0; speed handled at chapter assembly).

## VRAM (open — measure on-box)
Lazy load on first sentence. Resident **~9.4GB text-only / ~11.5GB cloning** (3 models co-resident)
— but these are **summed-from-logs estimates, not a measured co-resident peak**. The current one-shot
**destructs the backbone (@1922) before the decoder loads (@1950)**, so all-three-co-resident is
**new, unexercised** behavior. The bench's 8.25GB was a sequential single-model peak. Measure the real
simultaneous peak before committing to keeping the encoder permanently loaded (alternative:
load/free encoder on demand for cloning).

## Phase-0 spike gate
Pass when: model loads **once**; per-sentence (resident) marginal RTF approaches the gen+decode
**~0.168** (if it measures ~0.29 the loop failed to amortize); KV-clear + RNG-reseed correct
(sentence N+1 not contaminated, reproducible with seed); co-resident VRAM fits 16GB; cloning
(encoder + per-request reference) works. ⚠️ "toward 0.168" is a **biased-subtraction estimate**
(`moss_bench.py:105-109` bias note — the binary emits no per-phase timers) — verify on the real
resident build, don't quote it as achieved.
