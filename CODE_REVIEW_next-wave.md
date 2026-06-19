# Code Review — `feature/next-wave`

**Scope:** all changes vs `main` (committed + working tree) plus new untracked files
(`audiblez/gpu.py`, `tools/*.py`, tests). ~1,720 insertions across `core.py` (+485),
`ui.py` (+146), `lexicon.py`, `cache.py`, `doctor.py`, `cli.py`, `gpu.py`, README, CI.

**Method:** 13 specialist finder angles (line-by-line, removed-behavior, cross-file
contract, concurrency, audio/DSP, cache, GPU/ROCm, resilience, security, lexicon,
reuse/simplify, efficiency, altitude/hygiene) → adversarial 1-vote verification →
completeness critic → re-verification. 57 raw candidates → **55 verified**
(45 CONFIRMED, 10 PLAUSIBLE), consolidated below into ~30 distinct issues.

Overall: the branch is a large, well-documented feature wave (cache, lexicon, trailer,
resilience, ETA, GPU tuning, doctor). Code quality is high and the docstrings are
unusually good. The real risks cluster in two places: **(a) "looks done but isn't"
failure modes** — a degraded run that reports success (silence chapters, dead-letter,
zeroed chapter markers), and **(b) cache/lexicon correctness corners** that can serve
wrong audio. Nothing here is a crash-on-happy-path; the danger is silent wrongness.

---

## P1 — High (fix before merge)

### 1. Silent/failed chapter passes validation, is muxed, then skipped forever
`core.py:694` (`is_valid_chapter_wav`) · CONFIRMED
`is_valid_chapter_wav` is **duration-only**. A chapter whose sentences all fail is
replaced by `_silence_for()` ≈ `len(text)/15` s of zeros, while the validation floor is
`len(text)/60` s — so the all-silence wav is ~4× over the floor and validates as **True**.
Consequences: the silent chapter is written, muxed into the `.m4b`, and on the next run
the skip path (`core.py:195`) sees it "already exists", so the dead-letter unlink +
regeneration (`core.py:217-218`) never runs — **the failed narration can never be
recovered by re-running.**
**Fix:** reject near-silent content (sample RMS/peak, or fraction of zero samples), or
only write a chapter wav when its dead-letter count is 0 (write `.partial` otherwise) so a
failed chapter never validates.

### 2. Synth cache key omits `precision` → wrong-precision audio served on GPU
`core.py:461-466` (`_cache_key_fields`) + `cache.py:27` (`make_key`) · CONFIRMED (×2 angles)
`--cache --precision fp32` then `--cache --precision bf16` (or vice-versa) produces the
**identical key**, so `SynthCache.get` returns the other precision's waveform. The code
itself documents the contract it breaks: cache.py:8 "the key… captures everything that
changes the waveform" and core.py:307 "Lower precision changes the waveform"
(`gpu.autocast_context` returns a real `torch.autocast` for cuda/rocm fp16/bf16). Bounded
to GPU torch backends — i.e. exactly the target hardware (RX 7800 XT / ROCm).
**Fix:** thread `precision` into `_cache_key_fields` and `make_key`'s payload (normalize to
`fp32` on cpu/mlx/mps so the cache doesn't fragment needlessly).

---

## P2 — Medium

### 3. Recurring "output_folder not threaded into secondary entry points" (altitude pattern)
Three concrete instances of the same shape — a secondary entry point hardcodes `'.'`
instead of the run's real folder:
- **`core.py:562` `make_trailer`** loads the lexicon from `output_folder='.'`, so with
  `-o mydir` the trailer auditions **un-corrected pronunciations** while the real run uses
  `mydir/book.lexicon.json` — defeating the trailer's stated purpose. (CONFIRMED, ×3)
- **`doctor.py:157` `deep_check`** writes the TunableOp CSV to `'.'` while `main()` writes
  it under `output_folder`, so `--doctor --deep --tune -o out` warms the wrong location and
  the real run re-tunes from scratch (and drops the stray `tunableop_results0.csv` now in
  your tree). (CONFIRMED, ×2)
- **`doctor.py:156` `deep_check`** builds the synth at default `fp32` regardless of
  `--precision`, so `--doctor --deep --precision bf16` validates a path the real run won't
  use. (CONFIRMED)
**Fix:** add `output_folder`/`precision` params to `make_trailer`, `run_doctor`,
`deep_check`, and forward `args.output`/`args.precision` from `cli.py`.

### 4. Empty-string lexicon key garbles the entire chapter
`lexicon.py:57,71` · CONFIRMED (×2 angles)
`_active` filters only on the value (`v and v != k`), so `{'': 'X'}` survives and
`re.sub(r'\b\b', 'X', text)` inserts `X` at **every word boundary**:
`apply_lexicon('hello world', {'':'X'})` → `'XhelloX XworldX'`. Reachable via a hand- or
GUI-edited sidecar (the JSON editor does no key validation).
**Fix:** `return {k: v for k, v in mapping.items() if k.strip() and v and v != k}`.

### 5. Chained lexicon replacement corrupts longer rules
`lexicon.py:70-71` · CONFIRMED
Sequential `re.sub` passes mean a shorter active rule re-matches inside a **longer rule's
replacement output**: `{'AIME':'AI-me','AI':'ay-eye'}` turns "AIME" into "ay-eye-me".
Longer-first ordering only protects the *input*, not prior replacements' output.
**Fix:** single alternation pass — `re.compile(r'\b(?:' + '|'.join(map(re.escape, sorted(active, key=len, reverse=True))) + r')\b').sub(lambda m: active[m.group(0)], text)`.
This also removes the `.replace('\\', ...)` backslash hack.

### 6. ffmpeg concat injection via control chars in epub-derived wav names
`core.py:188` + `core.py:665` (`_escape_concat_path`) + `core.py:772/777` · CONFIRMED
`xhtml_file_name` sanitizes space/`/`/`\` but **not newline**; `_escape_concat_path`
escapes only single quotes; the concat demuxer runs with `-safe 0` and is line-oriented.
A crafted epub spine name containing `\n` splits the `file '...'` directive across lines —
mux corruption/abort (DoS), or fiddly arbitrary-file inclusion. Note `_escape_ffmetadata`
(`core.py:678`) *does* strip newlines — inconsistent defense.
**Fix:** strip control chars at `core.py:188` (`re.sub(r'[\x00-\x1f/\\]', '_', name)`) and/or
reject them in `_escape_concat_path`.

### 7. `_retry` has no error classification or backoff; batch failure amplifies work
`core.py:396-404,425-438` · CONFIRMED
`_retry` calls `fn()` `retries+1` times back-to-back (zero delay, bare `except Exception`).
Deterministic bad input is retried identically to transient faults; OOM is retried
immediately while memory is still held. In `_synth_batch`, one bad sentence fails the whole
batch `retries+1` times, then each sibling `retries+1` times → `3 + 3N` synth calls.
**Fix:** classify exceptions (skip retry for input/value errors), add short backoff for
transient ones, and don't re-synth siblings that already succeeded.

### 8. Dead-letter is write-only → a degraded audiobook reports success
`core.py:407-412` · CONFIRMED
Failed sentences are appended to `<book>.failed.jsonl`, but **no code path ever reads it**,
there's no end-of-run summary, and the CLI exits 0. A post-preflight runtime fault
(driver, phonemizer) can dead-letter everything, ship a silent/gappy `.m4b`, fire
`CORE_FINISHED`, and exit 0. (Per-sentence red warnings do print, so it's not totally
silent — but nothing persistent/programmatic.)
**Fix:** tally failures per chapter + end of run, surface `.failed.jsonl` in `--merge`, and
exit non-zero on any dead-letter so scripts/CI can detect a degraded run.

### 9. Missing ffprobe zeroes ALL chapter markers (silent nav loss)
`core.py:741` (`create_index_file`) · CONFIRMED
This branch added `FileNotFoundError` to `probe_duration`'s catch (`core.py:688`) so it now
returns `None` instead of raising. `create_index_file` does `duration = probe_duration(c) or 0.0`,
so with ffprobe absent **every** chapter gets `START=END=0`. The doctor only marks missing
ffprobe as `warn` ("will fall back to file size"), so the run proceeds and ships an `.m4b`
with destroyed chapter navigation and a success message. Unlike `is_valid_chapter_wav`,
`create_index_file` has no size fallback — proving this is an unintended regression.
**Fix:** distinguish `None` from `0.0`; if any chapter is unprobeable, omit `[CHAPTER]`
markers or raise an actionable "install ffprobe" error. Don't coalesce `None` to `0.0`.

### 10. Chapter-skip ignores lexicon changes → stale wavs reused silently
`core.py:191-200` · CONFIRMED
The default (non-cache) resume path validates only duration, never content. Edit
`<book>.lexicon.json` to fix a name and re-run: every existing chapter validates and is
skipped, and the `.m4b` is rebuilt from **old-pronunciation** wavs with no warning. (The
`--cache` path keys on lexicon-transformed text and would regenerate; the default path
doesn't.)
**Fix:** fold `lexicon.fingerprint(book_lexicon)` (+ voice/speed/precision) into the wav
filename or a sidecar fingerprint checked in `is_valid_chapter_wav`; at minimum warn when
skipping with an active lexicon.

### 11. Empty `valid_wavs` never posts `CORE_FINISHED` → GUI locks permanently
`core.py:244-248` · CONFIRMED
The new `else` branch only prints "No valid chapter audio…" and posts nothing. `core.main`
returns normally, so `CoreThread.run`'s except-reset never fires, and `on_core_finished`
(the only handler that re-enables the start button / params / checkboxes) is bound solely to
`CORE_FINISHED`. Reachable when every selected chapter is empty/`<10` chars — UI stuck,
must kill the app.
**Fix:** post a terminal event in the `else` too (or move the post out of the `if valid_wavs`).

### 12. `make_trailer` never registers espeak (unlike `main`)
`core.py:560` · PLAUSIBLE
`main()` calls `set_espeak_library()` before building the synth; `make_trailer` and the GUI
audition path go straight to `build_synthesizer` + synth. If espeak-ng isn't on the default
loader path, the audition feature — whose whole job is to de-risk a long run — silently
produces broken phonemization or crashes. (PLAUSIBLE: depends on runtime espeak
discoverability.)
**Fix:** call `set_espeak_library()` inside `build_synthesizer()` so every synth entry point
is covered.

### 13. Bare invocation flipped from CPU to auto-GPU with no runtime fallback
`cli.py:108-112` + `backends.default_backend()` · PLAUSIBLE
`audiblez book.epub` previously pinned CPU ("unchanged default behavior"); it now
auto-selects rocm/cuda/mps. On a gfx1101 GPU that passes `torch.cuda.is_available()` but
faults without `HSA_OVERRIDE_GFX_VERSION`, there's **no runtime GPU→CPU fallback**:
`build_synthesizer`/first synth either aborts the run or per-sentence HIP errors get
swallowed into `_silence_for` (silent `.m4b`). Related: `--doctor` with no `-b` still checks
`cpu` (`cli.py:63`), green-lighting a backend the real run won't use.
**Fix:** wrap synth construction + a tiny real synth in try/except and fall back to CPU with
a warning on the *auto-selected* path; make `--doctor` mirror `default_backend()`. (Given
your MEMORY note that eager-ROCm is slow/unstable, consider excluding rocm from
auto-select, or keep CPU as the bare default and just print "GPU detected; pass -b rocm".)

---

## P3 — Low (quality, latent, and polish)

**Caching**
- `cache.py:62` `get()` swallows all exceptions, never deletes the corrupt entry, catches
  bare `Exception` (mask). Narrow to `(ValueError, OSError, EOFError)`, log once, `unlink`. (CONFIRMED)
- `cache.py:70` `np.save` is non-atomic → torn file on crash / concurrent runs to a shared
  output dir. Write to temp + `os.replace`. Self-heals on next run, so low. (PLAUSIBLE)
- `cache.py:42` no size bound / eviction / `--cache-clear`; grows monotonically (already
  ~89 MB for one test book) and old `CACHE_VERSION`/`repo_id` entries never pruned. (CONFIRMED)
- `core.py:508` cache persists `np.zeros(0)` when synth returns `[]` without raising →
  permanently caches "no audio" for that sentence (no dead-letter, no warning). (PLAUSIBLE)
- `core.py:464` `repo_id` hardcoded in 3 places (`build_synthesizer`, `_build_mlx_synth`,
  `_cache_key_fields`); a one-sided bump serves stale cache. Hoist to constants. (CONFIRMED)
- *(My cross-check)* `cache.py:67` `put()` only catches `OSError` despite the docstring
  "cache failures never abort a run". Harmless today (callers always pass an ndarray), but
  widen to `Exception` for the stated contract. (latent)

**GPU / backend**
- `backends.py:60-66` `default_backend()` prefers torch-MPS over native MLX on Apple Silicon
  — the slower path the README itself calls inferior. Reorder `mlx` before `mps`. (CONFIRMED)
- `doctor.py:126-129` `check_backend` reports `OK` purely on `is_available()`; a present-but-
  unusable GPU still passes the fast `--doctor`. Mitigated by `--deep`. Consider a tiny real
  matmul probe. (PLAUSIBLE)

**Resilience / ETA / concurrency**
- `core.py:171` EWMA never bootstraps from the first real sample — the flat 500/50 prior
  dominates the ETA for ~15 batches (ETA can read 10× too optimistic early). Seed
  `chars_per_sec=None` so the first sample replaces the prior. (CONFIRMED)
- `core.py:523-527` failure/retry/silence batches feed full char-count with inflated elapsed
  into the EWMA, transiently dragging the rate down. Pass `elapsed=0` on the fallback path
  (same convention as cache hits). (CONFIRMED)
- `ui.py:712-714` progress events carry the **live, still-mutating** `stats` object across
  the thread boundary → torn multi-field reads on the GUI thread. Snapshot scalars or pass a
  shallow copy. Cosmetic today. (CONFIRMED)

**Audio assembly**
- `core.py:713` `is_valid_chapter_wav(w, None)` (assembly/merge) checks only readable+
  non-empty, so a truncated wav from a crash-mid-write can be muxed via `--merge`. (PLAUSIBLE;
  the finder's "header lies about duration" mechanism is wrong — soundfile/ffprobe report
  bytes-present — but the too-short-passes outcome is real.)
- `core.py:243` assembly drops omitted/short chapters and still fires `CORE_FINISHED`,
  shipping an `.m4b` missing content on a success signal. Partly intended resilience (it
  prints warnings + supports `--merge`); consider a distinct "degraded" signal. (PLAUSIBLE)

**Lexicon**
- `lexicon.py:20` `_PROPER_RE` is ASCII-only (`[A-Z][a-z]{2,}`) → accented proper nouns
  (Muñoz, André, Björn) are never auto-seeded, though a hand-added key still applies. (CONFIRMED)
- `lexicon.py:71` matching is case-sensitive → case variants (lowercase prose, all-caps
  headings) aren't respelled. Document or `re.IGNORECASE` for acronym keys. (PLAUSIBLE)
- *(My cross-check)* `\b…\b` silently no-ops on punctuation-containing terms a user might add
  (e.g. `C++`, `U.S.A.`) because `\b` doesn't anchor next to non-word chars. (latent)

**Trailer**
- `core.py:565-571` "Chapter N" uses the `enumerate` position, not `chapter.chapter_index`,
  and advances `n` past `<10`-char skipped chapters → spoken numbers don't match the
  chapters being auditioned (the trailer's whole point). Use `chapter.chapter_index` or the
  `sampled` counter. (CONFIRMED ×2)
- `core.py:557-559` `make_trailer` re-reads the epub + extracts all chapters even when
  `selected_chapters` is supplied (the common GUI path) — multi-second wasted parse on the
  UI thread. Guard with `if not selected_chapters`. (CONFIRMED)
- `core.py:489-491` (via `make_trailer`) spaCy-parses each chapter's *entire* text to keep 2
  sentences. Minor vs synth cost; truncate text before parsing. (CONFIRMED)

**GUI**
- `ui.py:648` the GUI **never wires the cache** — README and `cache.py` both promise "reuse
  on Preview/re-run", but no GUI path constructs a `SynthCache`. Either expose a checkbox or
  fix the docs. (CONFIRMED)
- `ui.py:544` every preview/audition click rebuilds the synthesizer (reloads Kokoro-82M).
  Cache the synth keyed by `(voice, backend, precision)`. The new Audition button makes this
  easy to spam. (CONFIRMED)
- `ui.py:604-607` first open of the Pronunciations editor seeds by regexing the whole book
  **on the UI thread** → freezes on large books. Move off-thread or precompute at open_epub. (CONFIRMED)

**Startup / efficiency**
- `core.py:119` + `core.py:177` the espeak library is globbed twice per run (preflight, then
  `set_espeak_library`). Resolve once / memoize `find_espeak_library`. (CONFIRMED)
- `core.py:243` + `core.py:741` every chapter wav is ffprobe'd twice at assembly (validate,
  then index). Thread the measured duration through. (CONFIRMED, minor)

**Conventions / tests / hygiene**
- **`.gitignore`** adds only `.direnv`. The new code generates ~123 `test/e2e_out/.audiblez_cache/*.npy`,
  `tunableop_results0.csv`, and `test/gpu.log` — **none ignored** (`git check-ignore` → NOT
  IGNORED). A `git add -A` commits them. Add `.audiblez_cache/`, `tunableop_results*.csv`,
  `*.log`, `test/e2e_out/`. (CONFIRMED — medium hygiene)
- `cli.py:73,123,138` `import os` three times inside `cli_main`; hoist to module scope. (CONFIRMED)
- `core.py:40/694` `VALIDATION_MAX_CHARS_PER_SEC` (a max rate) is passed as parameter
  `min_chars_per_sec` — contradictory naming invites an off-direction edit. (CONFIRMED)
- `core.py:456-463` retry/silence/dead-letter triplet is copy-pasted byte-for-byte into the
  cache branch; extract `_synth_sentence_or_silence(...)`. (CONFIRMED)
- `core.py:811` chapter-wav filename scheme is encoded twice (writer in `main`, glob+regex
  reader in `find_chapter_wavs`); a future format change silently breaks `--merge`. Share one
  builder. (PLAUSIBLE)
- `cli.py:122` `--trailer` skips the doctor preflight that `main()` runs → opaque kokoro
  traceback instead of the actionable "Preflight failed" message. (CONFIRMED)
- `core.py:821` `merge_chapters` and the CLI auto-select wiring have no test. (CONFIRMED)
- `test/test_main.py:67-69` the only decode/playback assertion is now opt-in
  (`AUDIBLEZ_TEST_PLAY`); remaining check is just `size > 1024`. Low impact (it only ran
  locally with a fixture, never in CI), but replace with a non-interactive `ffprobe`
  duration probe so a >1KB-but-undecodable `.m4b` still fails. (CONFIRMED)

---

## P-tools — Benchmark scripts (`tools/`)
These drive your TTS model A/B selection, so a wrong *number* is the worst outcome. The WER
edit-distance math in `score_wer.py` is **correct** (standard Levenshtein, correct empty-ref
guard, no off-by-one). The harness has two bias bugs that systematically favor one model:

- **`tts_ab.py:331-340` `--make-ref` overwrites the human reference with Kokoro audio** ·
  HIGH. Lines 35-37 explicitly warn `ref_human.wav` must be a *real human* clip ("a
  Kokoro-synthesized reference would… bias the A/B toward Kokoro"), but `make_reference()`
  synthesizes with Kokoro and writes straight to `REF_WAV`. The docstring even names a
  different file (`ref_voice.wav`). Following the documented `--make-ref` workflow makes every
  voice-cloner clone synthetic Kokoro audio → skews naturalness/WER toward Kokoro, the
  opposite of intent. **Fix:** write `--make-ref` to a distinct file (never `ref_human.wav`),
  or drop it and require a real clip; reconcile the docstring.
- **`moss_bench.py:95,106-107` MOSS RTF uses a different method than `tts_ab.py` RTF** ·
  HIGH. MOSS reports `rtf_gen_decode = (p_med - t_med)/audio_s` (baseline-subtracted), while
  `tts_ab.py:373` reports full resident-synth time / audio_s (no subtraction). Both land in
  the same `ab_out/` for side-by-side comparison, so lining up `moss.json` against
  `kokoro.json`/`fish.json` systematically favors MOSS. **Fix:** report a comparable resident
  RTF for MOSS (or clearly mark `rtf_gen_decode` as non-comparable and surface `rtf_total`).
- **`moss_bench.py:95` baseline subtraction over-subtracts** · MEDIUM. The TINY ("Hello.")
  run has its own gen+decode cost subtracted from the passage, understating MOSS's true
  gen+decode; the `max(0.0, …)` clamp can yield `0.0` → `rtf 0.0`. **Fix:** parse llama.cpp's
  `load/prompt eval/eval time` from stderr (already captured) instead of wall-time subtraction.
- **`tts_ab.py:73-80,376` `vram_gb` not comparable across runtimes** · MEDIUM.
  `torch.cuda.max_memory_allocated()` is ~0 for llama.cpp-Vulkan (Orpheus) and non-torch
  allocators, so cross-model VRAM comparison is meaningless. Use `rocm-smi`/`nvidia-smi` or
  document the limitation.
- **`tts_ab.py:363-373` RTF mixes median time with last-run duration** · LOW-MEDIUM. `audio_s`
  is the final run's duration but `median_s` is the median across runs; for stochastic-length
  models (Fish, VoxCPM2) this divides a median time by a non-median duration. **Fix:** compute
  per-run RTF and take the median.
- **`score_wer.py:32-33` normalization can create phantom WER deltas** · LOW. `re.sub(r'[^a-z0-9 ]','',…)`
  deletes hyphens/punctuation without a space ("well-known"→"wellknown") and keeps digits, so
  "nineteen" vs "19" counts as an error. **Fix:** replace non-alphanumerics with a space and
  normalize spelled-out vs digit numbers.

Subprocess handling in `moss_bench.py` is clean (list argv, no `shell=True`, temp cleanup).

---

## Recommended fix order
1. **#1, #2** (P1) — silent-chapter validation + precision cache key. Both serve a
   "complete" audiobook that's actually wrong; cheap fixes.
2. **#9, #8, #10, #11** — the other "reports success while degraded / stuck" set.
3. **#4, #5, #6** — lexicon corruption + concat injection (small, self-contained).
4. **#13 + #3 + #12** — auto-GPU fallback and the output_folder/precision threading; these
   touch CLI/doctor/trailer signatures, so batch them.
5. **`.gitignore`** now (one line of risk standing between you and committing 123 blobs).
6. P3 quality items opportunistically.
