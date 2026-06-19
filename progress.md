# audiblez — Progress Log

## Session 2026-06-17

| Step | Action | Result |
|------|--------|--------|
| 1 | Read source (core.py, ui.py, cli.py, backends.py, pyproject, docs) | Architecture captured → findings.md §1 |
| 2 | ADHD divergent ideation (workflow `wxvp2l8vs`, 11 agents) | 35 ideas → keystone insight + shortlist → findings.md §2 |
| 3 | Launched specialist debate (workflow `wq0j8pnlh`, ~13 agents) | RUNNING — produces the reconciled sequenced plan |
| 4 | Invoked planning-with-files; created task_plan.md, findings.md, progress.md | done |

| 5 | Demonstrated LSP tools (9 ops on core.py) | Verified synth-path blast radius → findings.md §1 |
| 6 | Debate `wq0j8pnlh` synthesize-agent hung (26 min, 541KB) | Stopped it; recovered 10 debate results from journal |
| 7 | Did synthesize + red-team + reconcile in-session | 8-phase plan written → task_plan.md; debate resolution → findings.md §3 |

### Current state
- Branch: `feature/next-wave`. **Phases 1–8 all implemented, tested, and committed** (one
  atomic commit per phase + a docs/CI commit). Working tree clean apart from planning files.
- New modules: `audiblez/doctor.py`, `lexicon.py`, `cache.py`. New tests: test_doctor,
  test_validation, test_eta, test_batching, test_resilience, test_trailer, test_lexicon,
  test_cache (wired into CI). New CLI flags: --doctor/--deep, --merge, --trailer,
  --seed-lexicon, --cache. GUI: trailer + audition + pronunciation-editor buttons.
- KEYSTONE spike resolved in findings.md §4; slice 1 (opt-in cache) shipped.

### Verification
- Runnable-here suites all green (test_doctor/lexicon/cache run fully; numpy-only).
- Heavy-stack tests skip cleanly here and run in CI (deps present). The only `discover`
  failures are the PRE-EXISTING unguarded test_main/test_find_chapters collection errors
  (present on `main`; CI exercises those paths via the real e2e conversion).

### Next action
- Done. Optional follow-ups (explicitly deferred): batch the cache-misses, WAL resume,
  streaming, parallel sentence sharding (findings.md §4); guard test_main/test_find_chapters imports.

## Session 2026-06-18 — GPU performance review (RX 7800 XT)

| Step | Action | Result |
|------|--------|--------|
| 1 | Set up direnv + zsh hook + `.envrc` → existing `.venv` | Auto-activates ROCm venv on `cd`; `.direnv` gitignored |
| 2 | Verified GPU: torch 2.12.1+rocm7.2, HIP 7.2, `cuda.is_available()=True`, gfx1101 native | No `HSA_OVERRIDE` needed (commented in `.envrc` as fallback) |
| 3 | ctx7 research: torch/ROCm configs for the setup | → findings.md §5 (TunableOp, autocast, inference_mode) |
| 4 | Loaded LSP (pyright); confirmed live on core.py | pyright can't resolve venv imports → read kokoro source directly |
| 5 | Source-audited installed kokoro pipeline.py + model.py | → findings.md §6 (corrections) |
| 6 | Wrote Wave 2 plan (Phases 9–11) | → task_plan.md |

### Key finding (corrects the old plan)
**Phase-4 "batching" is NOT GPU batching.** `KPipeline.__call__` iterates segments in a
Python for-loop (pipeline.py:369); `KModel.forward` hardcodes batch=1 (`[[0,*ids,0]]`,
model.py:131). So the GPU runs batch-size-1 forwards regardless of `batch_max_chars`. The
real RX 7800 XT wins are: **(9) TunableOp** [top — tunes recurring batch-1 GEMMs, env-only,
zero risk], **(10) fp16/bf16 autocast** [only per-call GPU lever; quality-gated], **(11)
warm MIOpen cache via DOCTOR --deep**. Dropped: `inference_mode` (redundant w/ existing
`@torch.no_grad()` model.py:86); true `[N,L]` batching (model surgery, alignment collapses
batch — high risk); `torch.compile` (RDNA3 Triton uneven).

### Wave 2 implemented + benchmarked (Phases 9–11)
| Step | Action | Result |
|------|--------|--------|
| 7 | New `audiblez/gpu.py` (TunableOp config, autocast CM, waveform_similarity) | import-light, torch lazy |
| 8 | Threaded `tune`/`precision` through core.main/build_synthesizer/gen_text/make_trailer + cli `--tune`/`--precision` | backward-compatible kwargs (default off/fp32) |
| 9 | DOCTOR `--deep` now warms MIOpen + seeds TunableOp CSV | `deep: synth + warm caches` ✓ |
| 10 | `test/test_gpu.py` (14 tests) + full suite | 71 tests OK (numpy-only here; torch-autocast test runs in venv) |
| 11 | **Benchmarked on real RX 7800 XT** | → findings.md §7 |

### MEASURED outcome (the honest result — overturns the hypotheses)
On Kokoro-82M / RX 7800 XT, steady-state medians: fp32 0.873s, fp16 0.863s (1.01×, sim
0.865), bf16 0.868s (1.01×, sim **0.059 = broken**), TunableOp 0.870s (no gain; CSV writes
fine). **All compute-precision/GEMM levers are flat — the model is too small + batch-1
(memory/launch-bound), not compute-bound.** The ONLY real win is **Phase 11: warming the
MIOpen cache removes a ~11s→0.87s (~13×) first-call stall.** Phases 9 & 10 shipped correct,
flag-gated, default-off, and now WARN (bf16 known-broken) — kept for bigger models / other HW.

### Next action
- Implementation complete + measured. NOT yet committed (working tree has the changes +
  planning files). Recommend: commit Phase 11 as the real win; keep 9/10 as documented
  opt-in. Real throughput ceiling = batch-1 architecture (model surgery, deferred).

## Session 2026-06-18 (cont.) — Wave 3: alternative TTS engines A/B

Built `tools/tts_ab.py` (engine A/B harness) + `tools/score_wer.py` (Whisper WER proxy) +
`tools/ref_human.wav` (real human reference; advisor caught that a Kokoro-synth reference would
bias the test). Installed all 3 models in ISOLATED uv venvs (main `.venv` untouched). Recipes via
workflow `wf_c04fa83b-721` → findings §9. Results → findings §10.

| Model | RTF | vs Kokoro | WER | VRAM | install |
|-------|-----|-----------|-----|------|---------|
| Kokoro (baseline) | 0.155 | 1× | 0.016 | 1.49GB | (in) |
| StyleTTS2 | 1.40 | ~9× slower | 0.0 | 1.76GB | medium (weights_only patch, punkt_tab, style-reuse) |
| CosyVoice2 | 2.40 | ~15× slower | 0.0 | 3.0GB | hard (minimal reqs, torchaudio.load→soundfile patch, path-not-tensor) |
| VoxCPM2 | 4.87 | ~31× slower | 0.0 | 6.6GB | medium (optimize=False on ROCm, 48kHz) |

**VERDICT:** all 3 work on ROCm + perfectly intelligible, but all are multiples slower than Kokoro
and slower-than-real-time (a 10h book: Kokoro ~1.5h vs 14h/24h/48h). Naturalness is the only axis
they might win — USER must judge `ab_out/*.wav`. Recommend: keep Kokoro default; only ADD opt-in
"quality mode" (StyleTTS2 = best trade) as Phase 18 if the user accepts the speed cost.

### Errors / retries (Wave 3)
| Error | Resolution |
|-------|-----------|
| StyleTTS2: torch≥2.6 weights_only=True rejects ckpt | patch torch.load default→False (trusted) |
| StyleTTS2: NLTK punkt_tab missing | nltk.download('punkt_tab') |
| CosyVoice2: openai-whisper/grpcio-tools no cp312 wheel, C-build fails | minimal inference reqs; whisper --no-deps |
| CosyVoice2: torchaudio 2.11 forces CUDA-only torchcodec on .load | monkeypatch torchaudio.load → soundfile |
| CosyVoice2: passed prompt as tensor; API wants a file PATH | pass str(ref_wav) path |

### Next action
- Wave 3 validation DONE. Awaiting user's listen of `ab_out/*.wav` to decide Phase 18 (integrate
  StyleTTS2 as opt-in `--model` quality backend) or stop. Nothing committed; isolated venvs +
  /home/caio/Projects/CosyVoice are outside the repo / gitignored.

## Session 2026-06-18 (cont.) — Wave 4: AMD-first + PyTorch-on-AMD research

3 background workflows (102 + 11 + 102 agents) + on-box A/B tests. Findings §12 (AMD-first TTS),
§14 (PyTorch-on-AMD), §13 (torch.compile A/B), §11b (Chatterbox-server), §11c (lemonade llamacpp-rocm).

**Bottom line (evidence-backed, sobering):**
- Kokoro is actually TOP-TIER on BLIND naturalness (TTS-Arena ELO ~1056); Chatterbox/Piper/Kitten/
  sherpa do NOT beat it blind. The "bad quality" feel = flat EMOTION + maybe voice choice, not low rank.
- No open TTS is simultaneously (a) clearly>Kokoro blind, (b) AMD-accel on Linux/ROCm, (c) permissive.
  ONNX-on-AMD-Linux dead (ROCm EP removed 1.23; DirectML Win-only; sherpa no AMD EP). F5/Llasa = NC + off-platform.
- torch.compile = dead end for TTS on gfx1101 (Kokoro 0.155→0.152; StyleTTS2 1.40→1.54; vocoder crashes
  under Inductor). fp16/TunableOp also flat (Wave 2). No PyTorch-ROCm lever speeds these models here.
- ONE real path: **Orpheus-3B (Apache)** — expressive (not proven higher MOS), LLM on **llama.cpp-Vulkan**
  (real AMD accel, escapes eager-pytorch) + SNAC decoder. Runtime = lemonade llamacpp-rocm / Vulkan.

### Next action
- Decision point for user: (a) cheap — try other Kokoro voices / voice-blending; (b) prototype
  Orpheus-3B-GGUF on llama.cpp-Vulkan + SNAC and A/B vs Kokoro (the one viable upgrade); (c) accept
  Kokoro is the right call and stop. Nothing committed. New harness models: kokoro-compiled,
  styletts2-compiled (both flat — keep as evidence or drop).

### Errors / retries
| Error | Attempt | Resolution |
|-------|---------|------------|
| Debate `synthesize` agent hung in a structured-output retry loop (~26 min, 541KB transcript) | 1 | Stopped workflow via TaskStop; recovered the 10 completed position+rebuttal results from the run journal; ran synthesize/red-team/reconcile in-session against source in context |

#### Lesson (workflow design)
An `agentType: 'Explore'` or open-ended "read the source and validate everything" agent has no tight stop condition and can balloon. For a single structured-output synthesis step, give the inputs inline (don't make it re-read), and prefer a bounded prompt over an exploratory agent.

## Session 2026-06-18 (cont.) — Wave 6: get s1-mini working (option A)

| Step | Action | Result |
|------|--------|--------|
| 1 | Re-diagnosed the fish tokenizer blocker | Root cause = repo HEAD too **NEW**, not too old: HEAD `tokenizer.py:57` hardcodes `AutoTokenizer.from_pretrained` (HF format); s1-mini ships only `tokenizer.tiktoken`. `dual_ar` "Transformers out of date" = red herring (custom model type, never in transformers). |
| 2 | `git fetch --unshallow --tags` fish-speech | Got history+tags. s1-era commit = `781bf1c` "Finetune support of OpenAudio-S1" (2025-10-20). AutoTokenizer landed later in `b72bcb3`/`daa9b4f` "S2 beta" (2026-03-10). |
| 3 | Coupling check + install-mode check | Glue (content_sequence/conversation/inference) changed in S2-beta → one-file swap unsafe. Install = PEP 660 editable mapping pkg-root→worktree → source checkout needs **no reinstall**. All adapter APIs (`TTSInferenceEngine`, `launch_thread_safe_queue`, decoder `load_model`, `ServeTTSRequest/ReferenceAudio`, `modded_dac_vq`) exist at `781bf1c`. tiktoken 0.13.0 installed. |
| 4 | `git checkout 781bf1c -- fish_speech tools` (keep venv+pyproject) | Source now s1-era + internally consistent. Tokenizer smoke test: `FishTokenizer.from_pretrained(s1-mini)` encode/decode round-trips "Hello world, this is a test." ✓ **Blocker resolved.** |
| 5 | Run `tts_ab.py --model fish --runs 1` (attempt 1) | FAIL: `torchaudio.list_audio_backends` removed in torchaudio 2.9 (s1-era code predates it; HEAD fixed in `75a9afe`). Fixed by stubbing it in adapter monkeypatch (only used to pick ffmpeg vs soundfile; we force soundfile). Other call site = `tools/vqgan/extract_vq.py`, not in synth path. |
| 6 | Re-run (attempt 3, absolute paths) | **s1-mini WORKS end-to-end through generation**: loaded (102s), encoded ref (5.33s), DualAR LLM **generated 463 tokens @9.78 tok/s on GPU, 5.40 GB VRAM**, started DAC decode (VQ `[10,462]`). |
| 7 | Terminal/desktop crashed during DAC decode | GPU reset took down the user's session. Cause = eager-ROCm fragility at decode: experimental mem-efficient SDPA on RDNA3 + MIOpen GEMM workspace fallbacks (`provided ptr:0 size:0`), on the **display GPU**. `ab_out/fish.wav` NOT written. GPU recovered (idle, 3.4GB = desktop). No zombie procs. |

| 8 | User chose GPU retry; re-ran | **SUCCESS.** Warm MIOpen kernels cleared the decode (no crash). Wrote `ab_out/fish.wav` (19.9s @ 44.1kHz). Metrics: RTF **19.4** (synth 386s; gen-only ~43s ≈ RTF 2.2, decode ~340s = MIOpen fallback), VRAM 7.47GB. |
| 9 | WER gate (`score_wer.py`) | fish **WER 0.032** — genuine intelligible speech, voice-cloned the reference. ✅ |
| 10 | Recorded results | findings §17 (BLOCKED→RESOLVED), TTS_MODELS_COMPARISON.md (+fish row), memory updated. |

### Wave 6 status — ✅ COMPLETE
- **s1-mini WORKS on the RX 7800 XT.** WER 0.032, `ab_out/fish.wav`. Version blocker resolved via s1-era source checkout (`781bf1c`) — no reinstall. Corrected diagnosis: HEAD was too NEW (AutoTokenizer), not too old.
- **But not viable as a daily driver:** RTF 19.4 (~125× Kokoro; ~14× even gen-only) AND the first run crashed the desktop (display-GPU reset during cold-kernel DAC decode). Confirms [[audiblez-amd-tts-runtime-reality]].
- **Only open question = naturalness vs Kokoro** (the user's actual complaint). Decide by EAR: `ab_out/fish.wav` vs `ab_out/kokoro.wav`. WER can't rank naturalness.

## Wave 7 — MOSS-TTS (MossTTSDelay 8B) on llama.cpp-Vulkan — 🏆 THE WIN (2026-06-18)

User asked to vet OpenMOSS/MOSS-TTS, then "test the 4B Local first". Pivoted (with user OK) to the
**8B Delay** because the 4B `MossTTSLocal-v1.5` is PyTorch-only (no GGUF) → eager-ROCm slow band; only
the 8B Delay has a first-class llama.cpp path = the one fast AMD runtime.

Steps:
1. Built OpenMOSS/llama.cpp fork (`moss-tts-firstclass`) with `-DGGML_VULKAN=ON` → `llama-moss-tts` + `llama-quantize`.
2. Downloaded HF `OpenMOSS-Team/MOSS-TTS` (8B, 16 GB) + `MOSS-Audio-Tokenizer` (6.7 GB); base model not gated.
3. Converted HF→f16 GGUF (17 GB; needed `sentencepiece` isolated; BPE vocab fallback). Pre-quant `MOSS-TTS-GGUF`
   repo is NOT compatible w/ first-class binary. Audio tokenizer → encoder/decoder GGUFs (`--encoder/decoder-outfile`).
4. Quantized Q5_K_M (6.06 GB). q5_K of the 32 audio output heads did NOT hurt (WER 0.000 == f16).
5. CPU f16 smoke (WER 0.0) → isolates conversion. GPU Q5 smoke (WER 0.0, no crash, ~8.25 GB VRAM, coopmat).
6. `tools/moss_bench.py`: RTF_gen+decode **0.168** (Kokoro 0.155), RTF_total (incl per-call load) 0.29, WER 0.000.

**Verdict:** MOSS breaks the speed/quality trade-off — first quality model ~as fast as Kokoro on this box,
expressive 8B, Apache-2.0, no crash. **Only open question = naturalness by EAR: `ab_out/moss.wav` vs `ab_out/kokoro.wav`.**
Docs updated: `TTS_MODELS_COMPARISON.md` (moss row + verdict), `findings.md` §18 (RESOLVED + full repro), memory.

## Wave 7 status — ✅ COMPLETE (pending only the subjective ear test)

## Session 2026-06-18 (cont.) — Wave 8: implement deep code-review fixes

Deep 13-angle code review of `feature/next-wave` (→ `CODE_REVIEW_next-wave.md`, 61 verified
findings) → now implementing ALL fixes. core.py edits serialized; `tools/` track delegated to a
parallel agent (independent files). Phases 24–35 in task_plan.md.

| Step | Action | Result |
|------|--------|--------|
| 1 | Ran the review (workflow `wf_35ea940d-e89`, 13 specialists + verify + completeness) + a tools-only pass | 61 findings → CODE_REVIEW_next-wave.md |
| 2 | Implemented all P1/P2/P3 audiblez/ fixes + tools/ (parallel agent) | Phases 24–35 done |
| 3 | Test suite | **146 pass, 10 skipped** (was 120) — 26 new tests added |
| 4 | CLI smokes | --doctor now checks the auto-selected backend (rocm), --cache-clear works, --help shows new flags |

### Wave 8 — what changed (by file)
- **core.py:** silent/failed-chapter skip guard (dead-letter + `.sig` render signature); precision in cache key; `_robust_duration` so chapter markers aren't zeroed without ffprobe; dead-letter surfaced (per-chapter + end-of-run summary, `main()` returns failure count); always fire `CORE_FINISHED`; `_retry` error classification + backoff; `_synth_one_or_silence` (dedup) + don't-cache empty/failed; EWMA bootstraps from first real sample; progress event carries a stats snapshot; ffmpeg concat control-char sanitization + shared `_chapter_wav_name`; espeak registered in `build_synthesizer` (covers trailer/audition); trailer: output_folder lexicon scope, skip epub re-read, prefix-only parse, consecutive numbering; repo_id constants.
- **cache.py:** precision in `make_key`; atomic write (temp+os.replace); corrupt-entry cleanup + narrow except; `clear()`.
- **lexicon.py:** drop empty/blank keys; single-pass alternation (kills chained corruption + backslash hack); Unicode-aware proper-noun seeding.
- **doctor.py / cli.py / backends.py:** output_folder+precision threaded into deep_check/run_doctor; `--doctor` checks default_backend(); auto-select GPU probe → CPU fallback; mlx>mps; `import os` hoisted; `--cache-clear`; nonzero exit on degraded run; `--trailer` runs preflight.
- **ui.py:** cache built synth across preview/audition; busy cursor on lexicon seed.
- **.gitignore / README / docs:** ignore generated artifacts (.npy/.csv/.log/.sig/.failed.jsonl/.audiblez_cache/test/e2e_out); fix "Preview reuse" claim; document --cache-clear + auto-GPU default.
- **tools/ (agent):** `--make-ref`→ref_voice.wav (stops biasing the A/B); MOSS RTF comparability bracket + 0.0-clamp guard; vram caveat; self-consistent RTF; WER normalization (space-split + 0-99 number folding).

### NOT committed — working tree only (awaiting user go-ahead to commit).

---

## Session — MOSS GUI epic + coarse mode (issues #2, #5, #6, #7, #8, #9) — 2026-06-19

Implemented all 6 open GitHub issues (MOSS-into-GUI epic + Phase-2 backend remainder).

| Issue | What landed | Verification |
|------|-------------|--------------|
| #7 | `backends.engine_choices()` (ordered engines + availability + reason from `moss_paths`); GUI renders disabled "MOSS (unavailable — …)" radio | 5 hermetic tests in test_backends.py |
| #8 | `core.MOSS_DEFAULT_VOICE` sentinel on the MOSS-no-clone cache voice axis; GUI greys voice dropdown + "loading model" busy state under MOSS | updated test_synth.py + 3 tests in test_cache.py |
| #9 | GUI Clone-reference control (file dialog + clear + WAV header validation + encoder-absent gating); clone_ref threaded into audition (`_cached_synth` key), render (CoreThread), trailer + `make_trailer`/cli parity | clone_ref parity grep clean; ui imports + handlers present |
| #6 | wired prebuilt `--serve` binary via `AUDIBLEZ_MOSS_BIN` in .envrc (binary already built at ~/Projects/llama.cpp-moss/build-vulkan/bin) | moss_status()=='present'; --doctor OK; 1-sentence synth = 3.68s audio |
| #2 | coarse-chunk mode now WIRED (was unwired stub): `gen_audio_segments(coarse=)` → `chunking.pack_chunks` @ 16.32s cap, per-chunk cache/failure unit, MOSS-gated in main; coarse folded into chapter `.sig` so a coarse↔per-sentence toggle busts resume (story 11); CLI help updated | 4 synth-path tests in test_batching.py + 2 `.sig` tests in test_cache.py; real-binary smoke = 1 request/chunk, 6.32s audio |
| #5 | PRD umbrella — satisfied by #6/#7/#8/#9 | — |

**Tests:** full suite 313 → **327 pass, 10 skipped**; ruff clean.

**GUI control wiring verified on the live display (18/18 programmatic checks):** MOSS radio present+enabled;
voice dropdown greys under MOSS + re-enables on switch-back; clone control shows/hides; WAV validation
accepts a real wav + rejects a non-wav; clone_ref threads through under MOSS + is suppressed for Kokoro.

**Committed:** branch `feat/moss-gui-integration`, commit `1017fcf` (9 files: backends/cli/core/ui + 4 test
files + progress.md). NOT pushed; no PR. `.envrc` MOSS wiring is gitignored (local-only — re-add after a
fresh clone). Pre-existing `tools/tts_ab.py` edit + untracked `CROSS_PROJECT_ANALYSIS.md` /
`test/MOSS_TEST_AUDIT.md` left untouched (not part of this work).

**Docs:** README "Choosing a backend" gained a MOSS row + a "MOSS — the higher-quality engine" subsection
(install, `--clone-ref`, `--coarse`, GUI clone control); `--help` block updated for `-b moss` / `--clone-ref` / `--coarse`.

**Still human-gated (irreducible):** by-ear AUDIO quality (does the MOSS/cloned voice sound right; coarse WER)
— no display or automation can judge perception. 22s coarse cap stays gated on an on-box WER check (16.32s is the validated default).
