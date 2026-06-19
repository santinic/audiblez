# audiblez — Implementation Plan

## Goal
Sequence the next wave of audiblez improvements into PR-sized, test-backed phases.
- Candidates: ADHD ideation → `findings.md` §2.
- Sequencing: specialist debate (5 specialists → cross-examination), synthesized + red-teamed → `findings.md` §3.

## Principles / Constraints
- Phases are PR-sized and independently shippable; risky changes behind a flag/constant.
- Every phase has concrete acceptance criteria + at least one test.
- **Phases 3→4→5 all edit `gen_audio_segments` (core.py:290–319) — keep them STRICTLY SEQUENTIAL** (no parallel branches; they collide).
- Preserve audiblez's offline single-process simplicity (no daemon / web rewrite).
- KEYSTONE (content-addressed cache) is **downgraded to a gated, optional, later layer** — its cache key is a correctness landmine and its durability win is mostly covered by Phases 2 + 5.

## Phases

### Phase 0 — Ideation + debate-driven planning — Status: complete
- [x] Read source; capture architecture (findings.md §1) + LSP-verified synth-path blast radius
- [x] ADHD divergent ideation → keystone insight + shortlist (findings.md §2)
- [x] Specialist debate → positions + cross-examination (recovered from workflow journal)
- [x] Synthesize + red-team + reconcile (done in-session; findings.md §3) → phases below

### Phase 1 — Preflight & fail-loud (DOCTOR) — Status: complete — PR: S
- **Goal:** never die 40 min into a run on a missing/misconfigured dep; fix the silent espeak fall-through.
- **Tasks:**
  - Make `set_espeak_library` (core.py:68–98) fail LOUD: raise a clear error instead of `traceback.print_exc()` + silent return; callers surface it before synthesis.
  - Add `audiblez --doctor` (cli.py): FAST checks — ffmpeg & ffprobe on PATH (`shutil.which`), espeak-ng `.so` resolvable, spaCy `xx_ent_wiki_sm` present, selected backend device available (`backends.available_backends()`). Print a red/green table; exit nonzero on any red.
  - `--doctor --deep` (optional): build the synthesizer + synth one word (slower — loads the model; NOT part of the fast path).
  - Run the fast preflight at the start of `main()` and before GUI Start.
- **Files:** `audiblez/cli.py`, `audiblez/core.py` (set_espeak_library), optional `audiblez/doctor.py`.
- **Depends on:** none.
- **Acceptance:** with ffmpeg removed from PATH, `--doctor` exits nonzero naming ffmpeg; a real run aborts in <5s with an actionable message; espeak failure now raises; unit test mocks `shutil.which`/spaCy/backends to assert red/green.

### Phase 2 — Validate-before-skip (corruption guard) — Status: complete — PR: S
- **Goal:** stop reusing / shipping truncated or zero-byte chapter wavs from a prior crash.
- **Tasks:** at core.py:162, gate the skip on a real check — `soundfile` readable AND duration plausible vs expected (`len(text)/chars_per_sec`), reusing `probe_duration` (:430, already returns None if ffprobe missing → fall back to soundfile/size). If invalid: delete + regenerate, never add an unreadable wav to `chapter_wav_files`.
- **Files:** `audiblez/core.py` (main skip branch + small helper).
- **Depends on:** none (independent of Phase 1).
- **Acceptance:** truncate a chapter wav → re-run regenerates it (not skipped, not piped into m4b); zero-byte wav test asserts regeneration; a valid wav is still skipped (no regression); works with ffprobe absent.

### Phase 3 — Honest ETA (measured throughput + heartbeat) — Status: complete — PR: S
- **Goal:** replace the fantasy `GPU/CPU_CHARS_PER_SEC` constants with measured chars/sec; make a hang observable.
- **Tasks:** capture per-sentence/chapter timing (value already computed at :186 and discarded), maintain an EWMA, feed `stats.chars_per_sec` (replacing the flat use at :145); recompute ETA at :315 from the rolling rate; emit a heartbeat log line every K sentences (index, chars/sec, chapter). Keep the flat constant only as the initial prior.
- **Files:** `audiblez/core.py` (:33–35, :142–150, gen_audio_segments :309–319).
- **Depends on:** none, but **edits `gen_audio_segments` → must precede Phase 4 (sequential).**
- **Acceptance:** ETA converges to real machine speed within the first chapter; heartbeat appears every K sentences; unit test feeds synthetic timings and asserts the EWMA + ETA math.

### Phase 4 — Batching (the throughput win) — Status: complete — PR: M
- **Goal:** feed Kokoro multi-sentence loads via the already-wired `\n\n\n` split_pattern.
- **Tasks:** in `gen_audio_segments` (:296–319), pack consecutive sentences up to `MAX_SENTENCE_LENGTH` joined by `\n\n\n` and call `synth` once per batch (closures at :270/:286 split internally); move stats accounting from per-sentence (:313) to per-batch (update the Phase-3 ETA code accordingly); behind a `batch_max_chars` constant (set tiny to disable → revert). Preserve order for concatenation; keep the non-English `split_long_sentence` path correct.
- **Files:** `audiblez/core.py` (gen_audio_segments).
- **Depends on:** Phase 3 (same function).
- **Acceptance:** measured chars/sec materially higher on a multi-chapter sample vs batch-disabled; audio ordering equivalent; `test_main` + `test_synth` pass; test asserts batches pack ≤ `MAX_SENTENCE_LENGTH` and a too-long sentence falls back per-sentence.

### Phase 5 — Sentence resilience (retry · dead-letter · partial m4b) — Status: complete — PR: M (2–3 sub-PRs)
- **Goal:** one bad sentence/batch must not nuke a chapter; a late failure still yields a playable file.
- **Tasks:**
  - 5a: wrap `synth()` in the loop (:311) with bounded retries; on persistent failure splice silence (`np.zeros`) + record to `<chapter>.failed.jsonl` (dead-letter). On a batched call, fall back to per-sentence.
  - 5b: incremental "merge-completed-chapters-now" — a CLI flag / on-failure path that runs `create_m4b` over finished chapters, so a crash in a late chapter or the final ffmpeg pass (:194–195) still yields a playable m4b.
- **Files:** `audiblez/core.py` (gen_audio_segments loop; main finalize; create_m4b call site); `audiblez/cli.py` (--merge).
- **Depends on:** Phase 4 (same loop; batching raises the blast radius).
- **Acceptance:** inject a synth exception on one sentence → chapter completes with silence + a dead-letter entry, run does not abort; kill before final mux → `--merge` produces a playable m4b of completed chapters; tests for retry exhaustion + dead-letter write.

### Phase 6 — User-facing audition (TRAILER + VOICE-AUDITION) — Status: complete — PR: M
- **Goal:** let users hear that detection/voice/pronunciation are right BEFORE the hour; fix the opaque 54-code picker.
- **Tasks:**
  - TRAILER: `make_trailer()` in core.py modeled on `gen_text` (:322) — synth opening 1–2 sentences/chapter (`max_sentences` cap), join with `np.zeros` gaps + spoken "Chapter N" labels; `--trailer` CLI flag; GUI "Preview whole book" button on the existing Preview worker pattern (ui.py:496–538).
  - VOICE-AUDITION: GUI control to speak THIS book's first paragraph in any selected voice (reuse Preview path); turns the dropdown (ui.py:311) into an audible casting step.
- **Files:** `audiblez/core.py` (make_trailer), `audiblez/cli.py` (--trailer), `audiblez/ui.py` (buttons).
- **Depends on:** Phase 1 ONLY (without DOCTOR the trailer throws / emits silence on first click).
- **Note:** cheap + low coupling — **can be pulled forward to Phase 2 if user-facing validation is the priority over raw throughput** (the "trust-first" camp's argument).
- **Acceptance:** `--trailer book.epub` produces a ≲2-min `trailer.wav` covering each detected chapter; a misdetected chapter is audibly revealed; voice-audition plays the chosen voice on the book's text; the preview thread never blocks the UI.

### Phase 7 — Pronunciation lexicon (book-scoped overrides) — Status: complete — PR: M (scoped, lower priority)
- **Goal:** fix recurring names/acronyms once; reuse everywhere.
- **Tasks:** a book-scoped sidecar (term → respelling/phonemes), auto-seeded from capitalized proper nouns/acronyms, applied before phonemization; GUI-editable. Record its text transform so a FUTURE cache key (Phase 8) can capture it (it changes the waveform).
- **Files:** new `audiblez/lexicon.py`; `core.py` (apply pre-phonemization); `ui.py` (editor).
- **Depends on:** Phase 1; coordinate with Phase 8 key design.
- **Acceptance:** a seeded override changes pronunciation consistently across chapters; round-trips through the sidecar; test asserts the transformed text reaching synth.

### Phase 8 — KEYSTONE (content-addressed cache) — Status: slice 1 complete (spike resolved → findings.md §4) — PR: L
- **Spike FIRST (decision gates):**
  1. Granularity vs batching: per-sentence keys give edit-aware reuse but don't match the batched synth unit. Likely answer: cache per-sentence, batch only the cache-misses.
  2. Versioned key must capture: engine + **model repo_id AND quantization** (torch `hexgrad/Kokoro-82M` :266 vs MLX `mlx-community/Kokoro-82M-bf16` :281), spaCy model version, `MAX_SENTENCE_LENGTH`, lang-split path (:296), voice, speed, text.
  3. Bit-identical concatenation seams for any resume use.
- **Then:** ship slice 1 = opt-in cache-only decorator (NO resume/correctness claim), behind `--cache`; prove re-run/Preview hit rate before anything depends on it. Later, optionally: WAL mid-chapter resume, streaming, edit-aware re-render, parallel sharding.
- **Depends on:** Phase 4 (granularity), Phase 7 (lexicon transform in key).

## Wave 2 — GPU performance review (RX 7800 XT / gfx1101, ROCm 7.2) — Status: planned

> Source-verified review of the synth hot path (findings.md §5–§6). Headline correction:
> Phase-4 "batching" amortizes Python/g2p overhead but is **batch-size-1 on the GPU** —
> the model hardcodes `input_ids=[[0,*ids,0]]` (model.py:131). So the GPU is under-fed and
> the wins here are (1) tuning the recurring batch-1 GEMMs and (2) lower precision — NOT
> more batching. Each phase is flag/env-gated and independently revertible.

> **MEASURED OUTCOME (findings §7):** benchmarked on the real RX 7800 XT. fp16/bf16 and
> TunableOp all came out flat (~1.00–1.01×, within noise) on Kokoro-82M — it's too small to
> be compute-bound; bf16 also breaks the audio (sim 0.06). **Only Phase 11 (warm cache)
> pays off** (~13× first-call). 9 & 10 are shipped correct + flag-gated + default-off for
> other hardware/models, but warn and are documented as "no gain here."

### Phase 9 — TunableOp (recurring-GEMM autotune) — Status: done (measured flat, kept opt-in) — PR: S
- **Goal:** the highest win/risk lever for batch-1 repeated inference; zero code risk.
- **Tasks:** opt-in `--tune` (or `AUDIBLEZ_TUNE=1`) that sets `PYTORCH_TUNABLEOP_ENABLED=1`,
  `PYTORCH_TUNABLEOP_TUNING=1`, and a **persisted** `PYTORCH_TUNABLEOP_FILENAME` under the
  output dir (so tunings survive across runs); must be set **before** torch builds any GEMM
  (env in cli/`main()` entry, before `build_synthesizer`). Print a one-line note that first
  run is slower (tuning) and the CSV is reused after. DOCTOR: surface tuning state + CSV path.
- **Files:** `audiblez/cli.py`, `audiblez/core.py` (set env early), `audiblez/doctor.py`.
- **Depends on:** none. **GPU-only** (no-op on cpu/mlx → guard on `backends.is_gpu`).
- **Acceptance:** with `--tune`, first run writes a `tunableop_results.csv`; second run loads
  it (no re-tune, validator line matches); measured chars/sec (Phase-3 EWMA) ≥ baseline on a
  multi-chapter sample; flag off → byte-identical to today. Test: env-set assertion + CSV path.

### Phase 10 — Mixed-precision autocast (fp16/bf16, STFT-safe) — Status: done (measured flat + lossy, kept opt-in w/ warning) — PR: M
- **Goal:** the only real per-call GPU speedup left; feed RDNA3 fp16 matmul throughput.
- **Tasks:** wrap the torch synth call in `torch.autocast(device_type='cuda', dtype=…)` behind
  a `--precision {fp32,fp16,bf16}` flag (default fp32 = today). Keep istftnet/`custom_stft`
  math in fp32 (autocast usually excludes FFT, but verify). Wrap at the `synth` closure in
  `build_synthesizer` (core.py:316) — torch backends only.
- **Files:** `audiblez/core.py` (build_synthesizer torch path), `audiblez/cli.py`.
- **Depends on:** none code-wise, but **A/B the audio**: ship a tiny waveform-similarity check
  (RMS/where-diff vs an fp32 reference clip) so a regression is caught, not shipped.
- **Acceptance:** fp16 run is measurably faster on the RX 7800 XT; an automated similarity
  check vs fp32 reference stays within tolerance (else the flag is documented as "audition
  first"); bf16 path works as the safe fallback; default (fp32) unchanged. **Quality-gated.**

### Phase 11 — Warm caches via DOCTOR `--deep` (MIOpen / first-run stall) — Status: done (★ the real win — ~13× first-call) — PR: S
- **Goal:** remove the first-run conv-kernel compile stall (MIOpen) from the real conversion.
- **Tasks:** extend DOCTOR `--deep` to pre-warm: build synth + synth one short utterance so
  MIOpen compiles & caches kernels (`~/.cache/miopen`) and, if `--tune` is on, seeds the
  TunableOp CSV — before the user's hour-long run. Document the warm-up cost.
- **Files:** `audiblez/doctor.py`.
- **Depends on:** Phase 1 (DOCTOR exists), composes with Phase 9.
- **Acceptance:** after `--doctor --deep`, the first real chapter shows no extra multi-second
  stall vs a steady-state chapter; `~/.cache/miopen` populated. (Hard to unit-test → assert
  the warm-up code path runs + a manual timing note.)

### Explicitly NOT in this wave (verified high-risk / low-ROI)
- **True `[N,L]` GPU batching** — needs model surgery: `forward_with_tokens` duration/alignment
  collapses the batch (`squeeze()` model.py:109; `pred_aln_trg` :110-113). High correctness
  risk on a vendored model for uncertain gain. Deferred unless profiling demands it.
- **`torch.inference_mode()` wrap** — redundant; forward is already `@torch.no_grad()`
  (model.py:86) + istftnet `no_grad` blocks. Negligible delta. Dropped.
- **`torch.compile`/inductor** — RDNA3 Triton coverage uneven; not worth the fragility now.

## Wave 3 — Alternative TTS engines: StyleTTS2 · VoxCPM2 · CosyVoice2 — Status: planned

> Goal: validate (install on ROCm + prove they synthesize) and A/B these 3 permissive models
> (findings §8) against Kokoro on the RX 7800 XT, THEN integrate the winners. User decisions:
> isolated venvs (protect `.venv` torch 2.12.1+rocm7.2); validate/A-B first; wire reference-audio.
> Voice strategy: generate ONE clean reference clip with Kokoro (af_sky) → all cloning models
> clone the SAME voice → equal-voice quality A/B; `--ref-audio` overrides.

### Phase 12 — A/B harness + adapter contract — Status: complete — PR: S
- **Goal:** a standalone, venv-agnostic harness that runs the same passage through any engine and
  emits a wav + metrics (load time, steady-state RTF via warmup+median, VRAM).
- **Tasks:** `tools/tts_ab.py` (outside the package). Adapter contract:
  `synth(text, ref_wav=None, ref_text=None, device='cuda') -> np.float32 mono @ 24kHz`. Dispatch on
  `--model {kokoro,styletts2,voxcpm2,cosyvoice2}`; each adapter imports its libs LAZILY (so the
  script runs in whichever venv has that model). Fixed test passage + `torch.cuda.synchronize()`
  timing. Generate the shared reference clip with Kokoro (`tools/ref_voice.wav` + its transcript).
- **Acceptance:** `python tools/tts_ab.py --model kokoro` (main .venv) writes `ab_out/kokoro.wav` +
  `kokoro.json` (rtf, load_s, sr). Reference clip exists. No package code touched.

### Phase 13 — ROCm install recipes (bounded research) — Status: planned — PR: n/a
- **Goal:** exact, current per-model install recipe + minimal snippet + reference-audio API, from
  PRIMARY docs, so installs don't fail-by-guessing. Parallel workflow (3 agents, schema'd).
- **Acceptance:** findings §9 has, per model: pip/git steps that DON'T pull CUDA torch (use the
  existing rocm torch / `--no-deps` where needed), the import + synth call, and how to pass a
  reference wav. Note any torch-version conflict with 2.12.1+rocm7.2.

### Phase 14 — StyleTTS2 (MIT, non-AR) — isolated venv + validate + A/B — Status: complete (works; WER 0.0; but ~9× slower than Kokoro on ROCm) — PR: M
- **Tasks:** `.venv-styletts2`; install (repo not pip-packaged → git + reqs, reuse rocm torch);
  LibriTTS checkpoint; adapter; synth the passage on GPU; A/B vs Kokoro. **Lowest risk per §8.**
- **Acceptance:** produces 24kHz audio on the RX 7800 XT; `ab_out/styletts2.wav` + metrics; `.venv` untouched.

### Phase 15 — VoxCPM2 (Apache, diffusion-AR, SOTA-tier) — Status: complete (works; WER 0.0; but ~31× slower than Kokoro, 6.6GB, 48kHz) — PR: M
- **Tasks:** `.venv-voxcpm2`; `pip install voxcpm` (guard torch); clone ref voice; adapter; synth + A/B.
- **Acceptance:** produces 24kHz audio on GPU; `ab_out/voxcpm2.wav` + metrics; `.venv` untouched.
- **Risk:** README says CUDA≥12.0 — verify ROCm runs (findings §8 verifier said base diffusion path runs on ROCm).

### Phase 16 — CosyVoice2 (Apache, best-maintained) — Status: complete (works; WER 0.0; ~15× slower; hardest install) — PR: M
- **Tasks:** `.venv-cosyvoice2`; install (git + requirements, ROCm guide from §8); CosyVoice2-0.5B;
  zero-shot clone of ref voice; adapter; synth + A/B.
- **Acceptance:** produces 24kHz audio on GPU; `ab_out/cosyvoice2.wav` + metrics; `.venv` untouched.

### Phase 17 — A/B comparison report + decision — Status: complete (findings §10) — PR: S
- **Result:** all 3 work on ROCm + WER 0.0, but 9× / 15× / 31× slower than Kokoro (StyleTTS2 /
  CosyVoice2 / VoxCPM2). Recommendation: keep Kokoro default; only ADD an opt-in "quality mode"
  (StyleTTS2 best trade) — Phase 18 — if the user wants higher naturalness at a large speed cost.
  Decision pending the user's listen of `ab_out/*.wav` (naturalness is the only open axis).
- **Tasks:** table of load time, steady RTF (vs Kokoro 0.87s baseline), VRAM, install pain, subjective
  notes; samples in `ab_out/` for the user to listen. Recommend which (if any) to integrate.
- **Acceptance:** findings §10 comparison + a clear go/no-go per model.

### Phase 18 — Integrate winner(s) as audiblez engines — Status: deferred (post-A/B)
- New `--model` axis through `build_synthesizer`/`main`/`gen_text`/`make_trailer` (+`--ref-audio`),
  modeled on the mlx engine path (core.py:322); optional dep groups in pyproject; tests. Only after Phase 17.

## Wave 4 — AMD-first TTS research (ROCm/Vulkan-optimized runtimes) — Status: complete (findings §12-§14)

> RESULT (sobering, evidence-backed): Kokoro is actually #1-tier on BLIND naturalness ELO; most
> "better-feeling" models don't beat it blind. ONNX-on-AMD-Linux is dead (ROCm EP removed, DirectML
> Windows-only, sherpa no AMD EP). torch.compile = dead end for TTS on gfx1101 (§13). The ONE viable
> upgrade = Orpheus-3B (Apache) for EXPRESSIVENESS (not proven MOS win), LLM on llama.cpp-Vulkan (real
> AMD accel) + SNAC decoder. Cheap pre-step: try other Kokoro voices / voice-blending.

> User: Kokoro naturalness is too low; wants better-sounding TTS that is OPTIMIZED for AMD
> (ROCm/Vulkan) — not the PyTorch-eager-on-ROCm path that made Wave 3 models 9-31× slow.
> Reframe: the bottleneck is the RUNTIME, not the model. Hunt AMD-first inference paths:
> llama.cpp (Vulkan/HIP), ONNX Runtime (ROCm/MIGraphX/DirectML), sherpa-onnx, Vulkan-native (MLC/TVM,
> ncnn). Bar: naturalness ABOVE Kokoro AND real AMD acceleration (not eager-pytorch). License/offline noted.
- **Phase 19 — multi-agent research** (workflow): sweep by runtime → per-candidate dossier → adversarial
  verify (real-AMD-accel? quality>Kokoro? license/offline). → findings §12.
- **Phase 20 — validate top picks** in the A/B harness on the RX 7800 XT (extends Wave 3). → findings §13.
- **Phase 21 — "PyTorch on AMD" research** (workflow): can the eager-ROCm path itself be made fast on
  gfx1101 (torch.compile/Inductor, AOTriton flash-attn, hipBLASLt, TunableOp, RDNA3 support tier)? If
  yes, it could speed the Wave 3 models without switching runtime. → findings §14.

## Wave 5 — Orpheus-3B prototype on llama.cpp-Vulkan + SNAC (user chose option 2) — Status: prototype WORKING (findings §15)
> RESULT: Orpheus-3B runs with CONFIRMED Vulkan GPU offload on the RX 7800 XT (escapes eager-PyTorch).
> RTF 1.35 = fastest of the quality models, but still ~9× Kokoro / slower than realtime (10h book ≈ 13.5h).
> WER 0.016 (perfect intelligibility). DECISIVE open Q = does it SOUND better than Kokoro → user must
> listen to ab_out/orpheus.wav vs ab_out/kokoro.wav. If yes → integrate as opt-in expressive mode (Phase 23).
- **Goal:** prove the ONE viable upgrade — Orpheus-3B (expressive, Apache) with the LLM on
  llama.cpp-Vulkan (real AMD accel, escapes eager-pytorch) + SNAC decoder — and A/B vs Kokoro on the
  RX 7800 XT (speed + WER + sample). Isolated `.venv-orpheus`; main `.venv` untouched.
- **Phase 22:** recon (Vulkan present? GGUF repo? snac/orpheus-cpp on pip?) → build llama-cpp-python
  (Vulkan) → Orpheus GGUF + SNAC → harness adapter → A/B. → findings §15.

## Wave 6 — Fish-Speech / OpenAudio S1-mini (user chose #1-quality, license now irrelevant) — Status: BLOCKED
- Built adapter + HF-authed gated download of s1-mini; fixed torchaudio→torchcodec (soundfile patch for
  paths + BytesIO). **Blocked: tokenizer won't load (s1-mini weights vs S2-pro repo HEAD = version
  mismatch)** → findings §17. Fix = checkout matching fish-speech version (re-clone w/ history) OR pin
  transformers. Escalated to user: pursue version checkout, or conclude (GPU proven + Orpheus works).
- **KEY user-validated finding → §16:** GPU is near-idle (1–11%) during PyTorch-ROCm runs; these models
  are batch-1 AR (CPU-bound serial loop), so the GPU is loaded-but-starved. Architectural, not config.

## Wave 8 — Code-review fixes (deep 13-angle review → CODE_REVIEW_next-wave.md) — Status: in_progress

> 61 verified findings consolidated into the phases below. core.py edits are STRICTLY
> SEQUENTIAL (shared file). tools/ track is independent → delegated in parallel.

### Phase 24 — P1 correctness — Status: complete
- [x] Silent/failed chapter: skip-path treats a chapter with a non-empty `.failed.jsonl` as incomplete → regenerates on re-run (assembly still muxes partials in-run). (core.py:191)
- [x] Cache key omits precision → add `precision` to `_cache_key_fields` + `make_key` (normalized to fp32 off-GPU). (core.py:461, cache.py:27)
- [x] Tests for both.

### Phase 25 — Degraded-run honesty — Status: complete
- [x] Dead-letter surfacing: main() returns failure count; end-of-run summary; CLI exits nonzero. (core.py, cli.py)
- [x] ffprobe markers: `create_index_file` uses a robust duration (ffprobe→soundfile) so markers aren't zeroed when ffprobe is absent. (core.py:728)
- [x] Always fire CORE_FINISHED (GUI never locks on empty valid_wavs). (core.py:246)
- [x] Chapter-skip signature: sidecar `.sig` (lexicon fp + speed + precision) → re-run after a lexicon/speed/precision change regenerates. (core.py:191)

### Phase 26 — Lexicon correctness — Status: complete
- [x] Drop empty/whitespace keys in `_active`. Single-pass alternation in `apply_lexicon` (fixes chained corruption + backslash hack). Unicode-aware seed regex. Document case-sensitivity. (lexicon.py)

### Phase 27 — Security — Status: complete
- [x] Sanitize control chars (newline) in epub-derived wav names + harden `_escape_concat_path`. (core.py:188,658)

### Phase 28 — Resilience / retry / cache hardening — Status: complete
- [x] `_retry`: skip non-transient errors, add backoff. Batch: single attempt then per-sentence. EWMA: elapsed=0 on fallback. Don't cache zero-length. cache.get cleanup + narrow except + unlink. Atomic `np.save` (temp+replace). `clear()` + `--cache-clear`.

### Phase 29 — output_folder/precision threading + espeak — Status: complete
- [x] make_trailer/deep_check/run_doctor take output_folder + precision; cli forwards args.output/args.precision. set_espeak_library memoized + called in build_synthesizer (covers trailer/audition; kills double glob).

### Phase 30 — GPU/backend safety — Status: complete
- [x] Auto-select runtime probe (tiny matmul) → fall back to cpu with warning. mlx before mps. --doctor checks default_backend() when no -b.

### Phase 31 — ETA/concurrency polish — Status: complete
- [x] EWMA bootstraps from first real sample. Progress events carry a stats snapshot (no torn reads).

### Phase 32 — Trailer / GUI / perf — Status: complete
- [x] Trailer chapter numbering = sampled+1 (matches final m4b). Skip epub re-read when selected_chapters given. Truncate text before spaCy parse. --trailer runs preflight. GUI caches built synth. on_edit_lexicon seeds off-UI-thread/busy cursor. Fix README/cache Preview-reuse doc.

### Phase 33 — Cleanup / conventions — Status: complete
- [x] Hoist `import os` in cli.py. Rename min/max chars param. repo_id constants. Extract `_synth_one_or_silence`. Shared chapter-wav-name builder.

### Phase 34 — tools/ benchmark fixes (delegated, independent) — Status: complete
- [x] make-ref→ref_voice.wav; MOSS vs tts_ab RTF comparability; baseline over-subtract; vram cross-runtime; RTF median/last; score_wer normalization.

### Phase 35 — Tests + full suite green — Status: complete
- [x] merge_chapters + auto-select-fallback tests; decode→ffprobe probe; new-behavior tests; whole suite passes.

## Deferred / Dead (traps — do not build)
STREAMING + LIVE-TICKER (leans on KEYSTONE + wxPython rewrite; also a throughput trap — small low-latency chunks fight large batched loads) · PIPELINE-STATUS (#34; folds into the Phase-3 heartbeat + a future manifest) · auto multi-voice/dialogue routing (#23) · pluggable multi-format IR (#25) · headless daemon + web (#27) · always-on Whisper proofread (#28) · prefetch thread (#6) · quorum/adaptive batch (#31) · thermal/circadian throttle (#32) · gamification (#11) · cron narrator (#13).

## Open Questions
- KEYSTONE granularity + versioned key (the Phase-8 spike) — the one real unresolved design decision.
- Should TRAILER (Phase 6) move to Phase 2? Recommend **yes** if user-facing validation outranks throughput.
- DOCTOR `--deep` (synth one word) loads the full model — keep it opt-in so the default preflight stays fast.

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Debate workflow synthesize-agent hung ~26 min (structured-output retry loop, 541KB transcript) | 1 | Stopped the workflow; recovered the 10 completed debate results from the journal; did synthesize/red-team/reconcile in-session |
