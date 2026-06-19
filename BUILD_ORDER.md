# BUILD ORDER — committed sequence (post-digest 2026-06-18)

The decisions extracted from grilling `CROSS_PROJECT_ANALYSIS.md`. This is the *committed*
plan; the analysis is the source material. Glossary: `CONTEXT.md`. Rationale for the
surprising calls: `docs/adr/0001..0003`.

## Headline reframe
**Engine-first.** MOSS (llama.cpp-Vulkan, zero-shot cloning) is the headline and the new
**quality default**; Kokoro is the fast fallback. Multi-speaker is **phase-2**, riding MOSS
cloning. (ADR 0001, 0003.)

---

## Phase 0 — opening move (parallel)

**A. MOSS C++ feasibility spike** *(gates the whole headline)*
- Patch the OpenMOSS fork's `llama-moss-tts` to loop resident on stdin (sentence → wav, no
  reload). Measure on a real chapter: model loads **once**; per-sentence (resident) marginal RTF
  approaches the gen+decode **~0.168** (≈ Kokoro) — if it measures ~0.29 the loop failed to
  amortize the load. **Measure co-resident VRAM fits 16GB** (est. ~9.4GB text / ~11.5GB clone —
  unmeasured; the bench's 8.25GB was a *sequential single-model* peak, not the resident peak).
- Also confirm **cloning** works (`--audio-encoder-model` + `--reference-audio`, 24 kHz mono).
- Plain-narration quality is NOT a spike gate — settled by the listening test (ADR 0003).
- **Fail → fallback:** chapter-granularity subprocess (chapter-level cache/resilience only).

**B. Tier-1 quick wins** *(independent, no decisions — apply each verdict's `must_fix`)*
1. **sentence-level-edit-resynth** — add post-override chapter-text hash to `_render_signature`,
   move its computation inside the chapter loop. Closes the confirmed silent-wrongness bug
   (`.sig` omits chapter text, core.py:877). Core-fix scope only; `--resynth-chapters` escape
   hatch is a separate follow-on. One-time chapter regen on existing books is acceptable.
2. **sml-silence-tags** — `[break]`/`[pause]`/`[pause:N]`, the clean independent slice the
   verdict split out. (Voice-switch tags defer to phase-2.)
3. **batch-folder-voicemap** — `--books-dir` + `--voice-map`, cli.py only. Fix the `file_path`
   (not `epub_file_path`) kwarg crash bug; drop the overstated per-book MIOpen-warmup claim.
4. **spaCy-model constant** — dedupe the `xx_ent_wiki_sm` literals; prefer a local const in
   core.py (not an import from doctor.py). No `config.py`.

## Phase 1 — pay down the seam
**Entry-point unification (refactor #2, M)** — collapse cli/ui/core onto one params object so a
signature change can't silently miss `ui.py`. Do this **before** MOSS piles `clone_ref_wav`
through all three entry points.

## Phase 2 — MOSS integration (the headline XL)
**Full spec: [`docs/moss-coprocess-spec.md`](docs/moss-coprocess-spec.md). Decisions resolved in the
2026-06-18 grounding (both feasibility claims adversarially confirmed).**
- **C++ `--serve` patch first** (the Phase-0 spike productionized): hoist the 3 model loads to
  startup, eliminate the vocab-only load, `llama_memory_clear` + RNG-reseed per request, worst-case
  context sizing. Refactor `moss_generate_from_prompt`/`moss_decode_audio_llama` to split load from body.
- Resident **pipe co-process** (ADR 0002): one child per run, lazy load, model stays in VRAM,
  idle-release. Keeps per-sentence cache + resilience.
- **Transport = 16-bit WAV file-handoff**; control = newline-JSON on stdout (synth-path `LOG()`→stderr).
- **Speed = ffmpeg `atempo` at chapter assembly** (MOSS has no speed knob); MOSS sentence cache
  speed-agnostic, speed → chapter `.sig`.
- **Failure model** in the synth closure: error→dead-letter+silence; death→restart+re-dispatch; K=2
  deaths/sentence→dead-letter; global circuit-breaker.
- New `engine='llamacpp'` dispatch branch + `_build_llamacpp_synth` closure; returns `(synth, close)`,
  torn down in `core.main` `finally`.
- **Cache key:** MOSS adds seed + sampling params + all three GGUF identities (backbone/decoder/encoder).
  **Do NOT bump CACHE_VERSION** for engine separation — `engine` is already in the key, so MOSS can't
  collide with Kokoro entries (bump only if pinning sampling defaults).
- **doctor:** `check_backend` for engine `llamacpp` → `shutil.which('llama-moss-tts')` (NOT
  `llama-cli`) + verify backbone/decoder (encoder warn-if-absent when cloning).
- **Default selection:** extend auto-select to prefer MOSS when present, else Kokoro (ADR 0003).
- **Engine-registry (refactor #3)** folds in here — MOSS is the trigger that finally justifies it.
- Real binary/build facts are in findings.md §MOSS repro; the spec's `llama-cli` / stdout-PCM /
  `--output-format` / single-GGUF details are **fabricated — ignore them.**

## Phase 3 — Tier-2 standouts
- **cancel-mid-run** (cancel half only) — highest-value GUI gap; worth more with long resident
  MOSS runs. `threading.Event` polled per chapter/sentence; fix the 4 verdict bugs
  (RESOURCE_HINTS keys, CORE_FINISHED-vs-CANCELLED ownership, on_core_cancelled teardown,
  empty-return branch). Hermetic tests in test_resilience.py/test_synth.py, not test_main.py.
- **output-formats/stereo** — `--format`/`--channels`; put the capability table in a stdlib-only
  module (NOT imported from heavy core.py); extend test_m4b.py.
- **multi-format-input** (Calibre) — `formats.normalize_to_epub`; keep original `file_path` for
  naming/lexicon (only a local var carries the converted epub); mtime-check the conversion cache
  (no stale reuse); put `check_calibre` in doctor.py.

## Phase 4 — core.py decomposition (refactor #1, L)
**After MOSS lands** (never concurrent with the XL). Peel the import-heavy synth/pipeline core
from the light orchestration, once MOSS's reshaped seams are known.

## Phase 5 — multi-speaker (phase-2 headline)
**ADR 0004 + [`docs/phase2-multispeaker-spec.md`](docs/phase2-multispeaker-spec.md). The grounding
SOLVED chapter-alignment the doc called unsolved.**
- **Alignment = strategy-d** (recorded-boundary bucketing): audiblez owns booknlp input construction,
  records each chapter's char-range, buckets booknlp sentences by char-offset. Exact, not heuristic.
  Landmines: **char offsets not bytes**; key on `sentence_id` value; **layer-2** needs new plumbing
  (carry spaCy `start_char` through `_split_into_sentences`/`split_long_sentence`).
- **Cast = `book.cast.json`** discriminated-union sidecar, **MOSS-default / Kokoro-ids-optional**.
- Threading: `synth(text,speed,voice_spec)` through `_synth_one_or_silence`/`_synth_batch`;
  `pack_sentences` breaks batches on speaker change; per-sentence cast → `make_key` (bump
  `CACHE_VERSION`) + `_render_signature`/`.sig` (clone_ref = WAV **content** hash).
- **Real residual risk = attribution** (span→character), upstream of this design. English-only (booknlp).
- CLI-only v1. Replaces the old "sml voice-switch + per-sentence cache override" enabler framing.

---

## Dropped / deferred
- **translation-pre-tts** — DROPPED. Gating design bug (source-lang inverted → no-op), 1 French
  voice, broken JA G2P.
- **audiobookshelf-metadata** — DEFERRED behind a proof gate: an `ffprobe` round-trip test must
  show the iTunes atoms actually embed before any work (verdict: payoff unverified).
- **ocr-image-pdf**, **refactor-engine-registry as a standalone** — stay deferred (the latter is
  absorbed into Phase 2).
- **GUI precision dropdown** — do not expose (fp16 flat / bf16 breaks on gfx1101).
