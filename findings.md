# audiblez — Findings & Research

> Persistent research record for the "what to build next" effort.
> External / ideation content lives here, NOT in task_plan.md.
> Last updated: 2026-06-17

## 1. Architecture (ground truth — verified by reading source)

audiblez = EPUB→M4B audiobook generator using the **Kokoro-82M** TTS model.
CLI (`audiblez`) + wxPython GUI (`audiblez-ui`). ~1800 LOC, MIT, ~solo maintainer, Python 3.10–3.12.

**Pipeline (`audiblez/core.py`, 512 LOC):**
`epub.read_epub` → `find_document_chapters_and_extract_texts` (BeautifulSoup over `p/h1-h4/li/title`;
force-appends "." to every block) → `find_good_chapters` (brittle filename regex `is_chapter`) →
per chapter, `gen_audio_segments()` splits text with a cached multilingual spaCy model and calls
`synth(sentence, speed)` **ONE SENTENCE AT A TIME** in a serial loop (core.py:309–319; synth call at :311)
→ `np.concatenate` once per chapter (:182) → `soundfile.write` one `.wav`/chapter →
`create_m4b()` single ffmpeg pass: concat + FFMETADATA chapters + cover (:462).

**Key facts / code anchors:**
- Resume is coarse: "if the chapter `.wav` already exists, skip it" (core.py:162).
- `build_synthesizer(voice, backend)` returns the engine-agnostic `synth` closure (:246; torch path :268, MLX path :283).
- ETA from flat constants `GPU/CPU_CHARS_PER_SEC` (:33–34); the REAL chars/sec is measured at :186 but **discarded**.
- `gen_text()` is the simple text→wav path (:322–332) — the model for any "synthesize a bounded snippet" feature.
- `set_espeak_library()` swallows failures with `traceback.print_exc()` (:68); ffmpeg only checked late via `shutil.which` in `main()`.
- `main(post_event=...)` emits `CORE_*` events consumed by the GUI.
- `backends.py`: registry cpu/cuda/rocm/mps/mlx; `available_backends()` / `default_backend()` / `is_gpu()`.
- `ui.py` (637, wxPython): chapter table + checkboxes + status column; editable per-chapter text area;
  Preview button (synth 300 chars on a daemon thread → ffplay); engine radios; voice dropdown (54 cryptic codes);
  speed spinner; output picker; Start; progress bar + ETA; `CoreThread` posts wx events.

**Baseline:** the prior `FIX_PLAN.md` remediation has largely **LANDED** (single-pass m4b w/ native `aac`,
cached spaCy, multi-backend selection, intro-added flag, ffmpeg/metadata escaping). New work is **net-new
capability on a tool that currently works**.

**Synth-path blast radius (verified via LSP, for the Keystone phase):**
- `build_synthesizer` (core.py:246) — 8 references across 3 files: core.py (call site :152, def :246, `gen_text` :324), ui.py:512 (`generate_preview`), test/test_synth.py (4×). Wrapping its returned `synth` closure for caching/checkpointing touches exactly these.
- `gen_audio_segments` (core.py:290) — incoming calls from 3 sites: `main` (:179), `gen_text` (:325), ui `generate_preview` (:514). These are the three entry points into the serial synth loop any KEYSTONE/BATCHING/RESILIENCE change must keep working.

## 2. Divergent ideation — ADHD pass (workflow `wxvp2l8vs`, 11 agents, 35 ideas, 5 frames)

Frames: logistics · game-design · 3am-on-call · remove-the-load-bearing-assumption · biology.

### Clusters (underlying strategic angle)
- **survive-the-crash** — durable/idempotent long runs: fine-grained checkpoints, per-sentence retries, wav validation, run-manifest guards.
- **feed-the-engine-faster** — throughput: batching, content-addressed cache, measured chars/sec.
- **make-the-wait-legible** — visible/progressive feedback: trailer, live ticker, incremental m4b, pipeline-stage status.
- **self-correct-the-pronunciation** — word-level QC: per-book lexicon, override dictionaries.
- **dissolve-the-monolith** — remove structural assumptions: streaming primitive, preflight checks.

### ★ The keystone insight (3 isolated frames converged on it)
**Make the *sentence* a persisted, content-addressed unit.** Wrap the `synth(text,speed)` closure
(core.py:246/268/283); key = `sha256(engine|voice|speed|text)`; persist each segment to disk.
ONE data structure unifies: (a) cache for re-runs/Preview/boilerplate, (b) mid-chapter crash-resume
(retire the :162 skip), (c) streaming/listen-while-rendering (growing `.part` file), (d) parallel
sentence sharding (breaks the serial loop), (e) edit-aware re-render, (f) measured ETA.
The obvious answer is "just batch the loop"; the non-obvious win is making the sentence addressable.

### Converged shortlist (ideation's recommended build order — to be refined by debate)
1. **★ Keystone per-sentence content-addressed store** (cache → resume → streaming foundation).
2. **Trailer** (#24) — render a ~2-min sampler of each detected chapter before committing; cheapest de-risk; audits chapter detection. Modeled on `gen_text`.
3. **Doctor** (#15) — preflight `ffmpeg/ffprobe/espeak/spaCy/backend` smoke-test; fail-loud before a 1h run.
4. **Batching** (#2) — pack sentences up to `MAX_SENTENCE_LENGTH` via the existing-but-unused `split_pattern`; the real throughput win.

### Other strong candidates
- **Measured-ETA** (#3/#18): EWMA chars/sec + heartbeat; replace flat constants.
- **Sentence-resilience** (#16/#17/#20): per-sentence bounded retries + dead-letter; validate `.wav` before skip; incremental/partial m4b.
- **Lexicon** (#26/#30): book-scoped pronunciation sidecar, auto-seeded, GUI-editable.
- **Voice-audition** (#7), **Streaming + live-ticker** (#21/#9), **Pipeline-status** (#34).

### Traps (default-defer — high pull, bad ROI for a solo offline tool)
auto multi-voice/dialogue routing (#23, speaker attribution unsolved) · pluggable multi-format IR w/ pdf/html/md/url adapters (#25, scope explosion) · headless daemon + web clients (#27, breaks offline simplicity) · always-on Whisper STT proofread (#28, ~2x compute + heavy dep) · prefetch thread (#6) · quorum/adaptive batch (#31) · thermal/circadian throttle (#32) · speedrun gamification (#11) · cron narrator (#13).
*Salvageable seeds:* #23 → user-tagged voice-per-section (not auto); #25 → keep the ingestion/synthesis IR boundary even with only the EPUB adapter; #28 → opt-in "verify one suspect chapter" mode.

### Deepened branches (full sketches in the ADHD run output)
- **Trailer (#24):** `make_trailer()` reuses pipeline through chapter selection, synthesizes opening ~1–2 sentences/chapter with `max_sentences` cap, joins with `np.zeros` gaps. Risk: inherits crude extraction — can sound clean while the chapter MAP is wrong.
- **Keystone (#0/#33/#4):** cache wrapper around the synth closure; WAL variant adds `.ndjson` manifest + growing `.part`. Risk: key must capture everything that changes the waveform (incl. model version, lang-split path) or it serves stale/wrong audio; resume seams must be bit-identical.

## 3. Specialist debate (workflow `wq0j8pnlh`) — RESOLVED in-session

5 specialists (perf · reliability · architecture · product · maintainer) opened positions, then
cross-examined. The workflow's `synthesize` agent hung (~26 min, structured-output retry loop) so it
was stopped; the 10 completed debate results were recovered from the run journal and synthesis +
red-team + reconcile were done in-session against the real source.

### Resolved tensions (the rulings that shaped the plan)
- **KEYSTONE-first vs cheap-incremental-wins** → **incremental wins first; KEYSTONE downgraded** to a
  gated, optional, later layer. Why: its cache key is a correctness landmine on a thin-tested 1800-LOC
  repo, and `product` retracted the "50 min lost" claim — completed chapters already persist as wavs and
  are skipped at core.py:162, so a crash loses ≤1 in-flight chapter. Validate-before-skip (Phase 2) +
  retry/dead-letter (Phase 5) deliver most of the durability win far cheaper.
- **BATCHING vs KEYSTONE ordering** → **batch FIRST** ("key the unit you actually synthesize"). Even
  `perf` (opened with store-first) conceded: keying per-sentence before batching orphans every entry the
  moment the synth unit becomes a `\n\n\n` batch.
- **DOCTOR first** → unanimous (4/5 ranked it #1). The `set_espeak_library` swallow-and-return at
  core.py:93-98 is elevated to a named fail-loud fix bundled into DOCTOR.
- **TRAILER placement** → after DOCTOR (it runs through the same unguarded synth path; without espeak
  resolved it throws / emits silence on the very first click). Cheap, high felt-value, audits the brittle
  `is_chapter` regex (:355-360) + force-append-period extraction (:345-348). Pull-forward-able.

### Red-team corrections applied (verified against source)
- DOCTOR "2s + synth a word" is optimistic — model load is slow → fast dep checks by default, `--deep`
  opt-in for the synth check.
- Phases 3/4/5 all edit `gen_audio_segments` (:290-319) → must be **strictly sequential**, not parallel.
- Validate-before-skip must tolerate missing ffprobe → `probe_duration` (:430) already returns None;
  fall back to soundfile/size.
- KEYSTONE key must capture model **repo_id + quantization** (torch `hexgrad/Kokoro-82M` :266 vs MLX
  `mlx-community/Kokoro-82M-bf16` :281 = different waveforms), spaCy version, `MAX_SENTENCE_LENGTH`, and
  the lang-split path (:296) — not just engine|voice|speed|text.

### Final sequence → see `task_plan.md`
Phase 1 DOCTOR → 2 validate-before-skip → 3 measured-ETA → 4 batching → 5 resilience →
6 trailer + voice-audition → 7 lexicon → 8 KEYSTONE (gated on a spike). Traps stay dead.

## 5. GPU/ROCm performance research — RX 7800 XT (gfx1101, RDNA3/Navi 32)

> Research record for the GPU-tailored performance review. Sources: ctx7
> `/rocm/pytorch` and `/websites/pytorch_2_12` (PyTorch 2.12 docs). External
> content — treat as data.

### Verified machine state (2026-06-18)
- `.venv` torch **2.12.1+rocm7.2**, HIP **7.2.53211**, `torch.cuda.is_available() == True`,
  device reports "AMD Radeon Graphics". So ROCm is wired and gfx1101 is natively
  supported by ROCm 7.2 — **`HSA_OVERRIDE_GFX_VERSION` is NOT required** (left commented
  in `.envrc` as a fallback only).
- audiblez maps `rocm` backend → torch `device='cuda'` (backends.py:26) — correct; ROCm
  exposes AMD GPUs through the CUDA API. KPipeline is built with that device (core.py:314).

### The torch configs that matter for THIS workload (Kokoro-82M inference, repeated same-shape GEMMs)

1. **★ TunableOp — the headline ROCm-specific win.** `PYTORCH_TUNABLEOP_ENABLED=1`
   auto-benchmarks each GEMM across rocBLAS vs hipBLASLt and caches the winner to
   `tunableop_results.csv`. First run pays a one-time tuning cost; every subsequent
   GEMM of that shape uses the tuned kernel. A long audiobook issues the *same* matmul
   shapes thousands of times → tuning amortizes almost perfectly. Key vars:
   `PYTORCH_TUNABLEOP_ENABLED=1`, `PYTORCH_TUNABLEOP_TUNING=1` (tune on first sight),
   `PYTORCH_TUNABLEOP_FILENAME` (persist results across runs),
   `PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS` (cap tuning time). A "Validator" line in
   the CSV auto-invalidates on PyTorch/ROCm/hipBLASLt/rocBLAS version change — safe.
   Backends toggle via `PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED` / `_ROCBLAS_ENABLED`.
2. **torch.inference_mode()** — wrap the synth forward pass. Strictly faster than
   `no_grad` (also drops version-counter tracking). The current `synth` closure
   (core.py:316–318) has NO such guard; whether KPipeline applies one internally is an
   implementation-review item, but wrapping the call site is free insurance.
3. **Autocast fp16/bf16** — `with torch.autocast(device_type='cuda', dtype=torch.float16):`
   speeds matmuls on RDNA3. Kokoro-82M is tiny + likely fp32 today; fp16 autocast is the
   most promising raw-speed lever but MUST be A/B'd for audio-quality regressions
   (autocast changes the waveform → also a cache-key concern if it ever interacts with §4).
   bf16 is the numerically safer variant if fp16 produces artifacts.
4. **hipBLASLt vs rocBLAS** — hipBLASLt is the newer GEMM lib, default-enabled in TunableOp.
   On gfx1101 its coverage can be partial; TunableOp picking rocBLAS for some shapes is
   expected and fine — that's exactly what the tuner is for.
5. **MIOpen** (conv kernels — Kokoro's vocoder uses convs): first-call kernel
   compilation/autotune is cached under `~/.cache/miopen`. `MIOPEN_FIND_MODE` controls
   find behavior; a warm cache removes a first-run stall (DOCTOR `--deep` could pre-warm).
6. **NOT relevant here:** `ROCBLAS_INTERNAL_FP16_ALT_IMPL` /
   `MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL` address fp16 *training* backward-pass
   convergence — audiblez is inference-only.

### Implications for the performance review (next phase, to be planned)
- Cheapest/safest wins, in order: (a) `torch.inference_mode()` around synth; (b) opt-in
  TunableOp with a persisted results file (env-driven, zero code risk, big repeat-run win);
  (c) A/B fp16 autocast behind a flag with a quality check.
- These layer onto the existing batching (Phase 4): TunableOp + batching compound, since
  batching stabilizes GEMM shapes that TunableOp then tunes once.
- Open question: does Kokoro's `KPipeline` already wrap inference_mode/autocast internally?
  → an LSP/source dive into the installed `kokoro` package is the first review step.

## 6. Kokoro hot-path audit — SOURCE-VERIFIED (installed kokoro in .venv)

> Read of `.venv/.../kokoro/{pipeline.py,model.py}` (2026-06-18). These correct
> assumptions baked into the older plan. Anchors are in the installed package.

### What the synth call actually does
- `core.build_synthesizer.synth` → `KPipeline.__call__` (pipeline.py:351).
- `__call__` splits text on `split_pattern` into a list and **iterates segments in a
  plain Python `for` loop** (pipeline.py:369), calling `KPipeline.infer(model, ps, ...)`
  **once per segment** (pipeline.py:383).
- `KModel.forward` (model.py:121) builds `input_ids = torch.LongTensor([[0, *ids, 0]])`
  (model.py:131) — **batch dimension hardcoded to 1** — then calls
  `forward_with_tokens` (model.py:86, `@torch.no_grad()`), and finishes with
  `audio = audio.squeeze().cpu()` (model.py:134) — **a device→host copy on every segment**.

### Load-bearing corrections (these change the plan)
1. **★ Phase-4 "batching" is NOT GPU batching.** Joining sentences with `\n\n\n` just
   produces a longer segment LIST that `__call__` walks one-by-one; each segment is still
   an independent **batch-size-1** forward. The win it delivered is real but is
   **Python/g2p/pipeline-call overhead amortization**, not GPU SIMD utilization. The GPU
   is under-fed regardless of `batch_max_chars`. (The old plan called Phase 4 "the
   throughput win" — accurate as far as it goes, but it never increased GPU occupancy.)
2. **True GPU batching needs model surgery, and is correctness-risky.** `forward_with_tokens`
   is *shape-generic* in the mask code (uses `input_ids.shape[0]`) BUT the duration/alignment
   path assumes batch=1: `pred_dur = ...squeeze()` (model.py:109) and the `pred_aln_trg`
   construction (model.py:110-113) collapse the batch. Real `[N,L]` padded batching would
   require bypassing `forward`, per-row length masking, and rewriting alignment — high risk
   on a vendored model. **Recommend: do NOT attempt in this wave.**
3. **`inference_mode` is marginal, not a win.** `forward_with_tokens` is already
   `@torch.no_grad()` and istftnet adds its own `no_grad` blocks (istftnet.py:249,300).
   Wrapping the call in `inference_mode` saves only the no_grad↔inference_mode delta
   (version-counter tracking) — negligible. **Downgrade from research §5 item 2.**
4. **fp16/bf16 autocast is the only real per-call GPU lever left** — model + STFT run fp32,
   no autocast anywhere. Wrapping `forward_with_tokens` in
   `torch.autocast('cuda', torch.float16)` should speed RDNA3 matmuls. RISK: `custom_stft.py`
   / istftnet vocoder math is fp16-fragile → likely must keep STFT in fp32 (autocast usually
   does this for FFT ops, but MUST be A/B'd for artifacts). bf16 is the safer fallback.
5. **TunableOp is now the clear #1.** Because every forward is batch-1 at a *small set of
   recurring GEMM shapes* fired thousands of times, a one-time per-shape tune amortizes
   almost perfectly. Env-only, zero code risk, persisted CSV. This is the highest
   win/risk ratio for THIS architecture.
6. **Per-segment `.cpu()` sync (model.py:134) caps overlap.** Each segment forces a
   device→host transfer = a GPU sync point; many short segments serialize host/GPU. Can't
   be fixed without model changes, but it means honest GPU timing is already (accidentally)
   synchronized, and it's an argument for fewer/longer segments (which Phase-4 batching does
   help with — its real benefit).

### Revised win ranking for the RX 7800 XT (replaces research §5 ordering)
| Rank | Lever | Mechanism | Code risk | Quality risk |
|------|-------|-----------|-----------|--------------|
| 1 | **TunableOp** (env, persisted CSV) | tunes recurring batch-1 GEMMs once | none | none |
| 2 | **fp16/bf16 autocast** (flag-gated, STFT stays fp32) | faster matmuls | low-med | **needs A/B** |
| 3 | warm MIOpen cache via DOCTOR `--deep` | removes first-run conv compile stall | low | none |
| — | inference_mode | redundant w/ existing no_grad | — | — (drop) |
| — | true `[N,L]` GPU batching | real occupancy | **high (model surgery)** | high (alignment) | (defer) |

## 7. MEASURED results — Wave 2 benchmarked on the real RX 7800 XT (2026-06-18)

> Empirical run of the implemented levers on the actual GPU (rocm, torch 2.12.1+rocm7.2).
> Text: one ~85-char sentence; medians of 4 steady-state runs after a warm-up; timed with
> `torch.cuda.synchronize()`. **These measurements overturn the pre-implementation ranking.**

| Lever | First call | Steady median | vs fp32 | Waveform sim vs fp32 | Verdict |
|-------|-----------|---------------|---------|----------------------|---------|
| fp32 (baseline) | ~11.4s | **0.873s** | — | — | baseline |
| fp16 autocast | — | 0.863s | **1.01× (noise)** | 0.865 (audible loss) | ✗ not worth it |
| bf16 autocast | — | 0.868s | **1.01× (noise)** | **0.059 (broken)** | ✗✗ destroys audio |
| TunableOp (fp32) | 7.3s (tuning) | 0.870s | **1.00× (none)** | identical | ✗ no gain (plumbing OK) |
| **Warm MIOpen cache** | removes ~11s→0.87s | — | **~13× on first call** | identical | ★ the only real win |

### Why the GPU levers flopped (root cause)
Kokoro-82M is **tiny and memory-/launch-bound at batch-size-1**, not compute-bound:
- **Mixed precision** speeds up large GEMMs; here the GEMMs are small, so fp16/bf16 save
  nothing measurable AND the istftnet vocoder hits experimental **ComplexHalf** ops
  (`istftnet.py:98` warns) → quality loss (bf16's complex STFT path collapses to sim 0.06).
- **TunableOp** tunes GEMM kernel choice; with already-small GEMMs the default
  rocBLAS/hipBLASLt pick is already fine → 0.3% delta (within noise). The CSV writes
  correctly (23 lines, device-ordinal filename), so it'll help on *bigger* models/other HW.
- **The dominant real cost** is the one-time MIOpen kernel compilation (~11s) + per-segment
  `.cpu()` sync + Python-loop dispatch — none of which precision/GEMM-tuning touch.

### Revised conclusion (replaces §6 ranking)
1. **★ Phase 11 (warm MIOpen via `--doctor --deep`) is the only lever that pays off here** —
   it moves the ~11s first-chapter stall out of the timed run. Real felt-latency win.
2. **Phases 9 & 10 measured flat on THIS setup** but are correctly built, flag-gated, and
   default-off; kept for larger models / NVIDIA / future Kokoro versions. Both now warn /
   are documented as "no measured gain on Kokoro-82M; audition before use."
3. The genuine throughput ceiling is the **batch-1 architecture** (findings §6 #1-2) — only
   real `[N,L]` batching (model surgery, deferred as high-risk) would raise GPU occupancy.
4. **Even CPU vs GPU is nearly a wash for this model: GPU 0.872s vs CPU 0.970s = 1.11×**
   (measured; model weights confirmed on `cuda:0`, fp32). Kokoro-82M *runs* on the GPU but
   barely benefits — it's tiny, batch-1, has a sequential LSTM in the duration predictor
   (`model.py:106`), and is dominated by kernel-launch overhead + the per-segment `.cpu()`
   sync (`model.py:134`), not GEMM FLOPs. This is the root cause behind §7 #1-3: the
   workload simply isn't compute-bound, so no precision/GEMM lever can move it. The backend
   auto-select still rightly prefers GPU (frees the CPU, ~10% faster, scales better with
   future batching), but users should not expect a large GPU speedup on this model.

## 4. KEYSTONE spike — RESOLVED (Phase 8, slice 1 shipped)

The three decision gates from `task_plan.md` Phase 8, resolved and implemented as the
opt-in cache-only slice (`audiblez/cache.py`, behind `--cache`).

1. **Granularity vs batching.** Resolution: the cache unit is the **sentence**. When
   `--cache` is on, synthesis runs per sentence so each cached array maps 1:1 to a
   sentence and concatenation seams are identical to a per-sentence run. The Phase-4
   batching path is bypassed while caching (the two co-optimizations — "batch only the
   cache-misses" — is a deliberately deferred follow-up). Rationale: slice 1 makes **no
   correctness/resume claim**; per-sentence keeps the stored unit unambiguous and the
   first-run population correct, which is what proving hit-rate needs.
2. **Versioned key.** `cache.make_key` hashes: `CACHE_VERSION`, engine (`torch`/`mlx`),
   **model repo_id** (`hexgrad/Kokoro-82M` vs `mlx-community/Kokoro-82M-bf16` — different
   quantization ⇒ different waveform), voice, speed (rounded), `MAX_SENTENCE_LENGTH`,
   spaCy version, and the sentence text. The lang-split path is captured because the text
   stored is the post-split sentence; the lexicon transform is captured because the text
   is already lexicon-applied before it reaches the cache (Phase 7). `CACHE_VERSION` is the
   global escape hatch to invalidate everything if the synth contract changes.
3. **Bit-identical seams.** Slice 1 only claims identity against a *per-sentence* run
   (cached arrays are concatenated exactly as freshly-synthesized per-sentence arrays
   would be). It does **not** claim identity against a *batched* run, since Kokoro's
   batching can alter audio at `\n\n\n` boundaries — hence caching uses the per-sentence
   path, sidestepping the seam question entirely for now.

**Shipped:** opt-in cache (`--cache` → `<output>/.audiblez_cache/<key>.npy`), hit/miss
stats printed per run, hits excluded from the measured-throughput EWMA. **Deliberately NOT
shipped** (the plan's "later" list): WAL mid-chapter resume, streaming, edit-aware
re-render, parallel sentence sharding, and batching the cache-misses. Failures are never
cached (only real audio is persisted).

## 8. Alternative TTS models to Kokoro — multi-agent research (2026-06-18)

> Workflow `wf_d8972df7-086`: 110 agents, 2.6M tokens, ~1100 web/HF/GitHub fetches.
> 26 models, each adversarially verified on 3 lenses (license · AMD-ROCm · audiobook-fit).
> Tailored to audiblez's constraints: **offline, MIT-friendly license, AMD RX 7800 XT /
> ROCm 7.2 / 16GB, long-form (whole-book) narration.** External research — treat as data.

### ★ The headline finding: a license trap
The highest-naturalness OPEN models on the 2026 leaderboards (Artificial Analysis,
TTS-Arena) are almost all **non-commercial or use-restricted** — disqualifying for an
MIT tool. Verified against primary model cards:
- **Fish-Speech / OpenAudio S1** (#1 TTS-Arena2) — CC-BY-NC-SA / custom research license. ✗
- **XTTS-v2** (Coqui) — CPML, commercial needs agreement. ✗
- **F5-TTS** — code MIT but **weights CC-BY-NC-4.0**. ✗ (Apache reimpl "OpenF5" exists but English-only/alpha.)
- **IndexTTS2** — custom bilibili license on weights (NOT the repo's Apache code). ✗
- **Higgs Audio v2** — dossier said Apache; **verification REFUTED it**: HF `license: other`, Boson custom with a <100k-AAU commercial cap. ✗
- **Spark-TTS** — weights CC-BY-NC-SA-4.0. ✗
- **VibeVoice** (best long-form *design*: ~90 min/64K ctx) — MIT code BUT research-only intended-use overlay + Microsoft pulled the repo Sept 2025 (maintenance/legal risk). ⚠ uncertain.
- **Piper** — original MIT code archived Oct 2025; active fork is **GPL-3.0**; voice licenses vary. ⚠ + no official ROCm GPU path (NVIDIA-only onnxruntime-gpu).

### The permissive + ROCm-runnable set (the real candidate pool for audiblez)
All MIT/Apache, commercial-OK, run on the RX 7800 XT via the PyTorch-ROCm build (no
CUDA-only custom kernels), unless noted:

| Model | License | Params | Arch | Long-form | Quality vs Kokoro | ROCm | Note |
|-------|---------|--------|------|-----------|-------------------|------|------|
| **VoxCPM2** (OpenBMB) | Apache-2.0 | 2B | diffusion-AR, continuous latents | mixed (chunk) | **SOTA-tier 2026**, WER ~0.4% | ✓ base path confirmed | newest (Apr 2026); actually uses the GPU; voice-design variance |
| **CosyVoice2** | Apache-2.0 | 0.5B | LLM-AR + flow | mixed (chunk) | high, NMOS ~3.96 | ✓ community guide | very active (21k★), streaming |
| **StyleTTS2** | MIT | 148M | **non-AR** style-diffusion | **strong** (same stability class as Kokoro) | MOS 4.55 (> Kokoro) | ✓ pure PyTorch | Kokoro's bigger *predecessor* — cleanest quality bump that keeps non-AR stability; repo stale; GPL phonemizer = non-issue (audiblez already uses espeak-ng) |
| **GPT-SoVITS** | MIT | ~mid | AR GPT + VITS | mixed (v3 better) | high, great cloning | ✓ **official** `install.sh --device ROCM` | very active (59k★) |
| **Chatterbox** (Resemble) | MIT | 0.5B | AR Llama-codec | mixed (chunk) | high (beat ElevenLabs 63.75% blind) | ✓ confirmed | ROCm speed reportedly poor in one report; embeds watermark |
| **Orpheus-TTS** | Apache-2.0* | 3B | AR Llama-3.2 | mixed (chunk, 8K ctx) | very expressive | ✓ via vLLM/llama.cpp | *Llama-3.2 base license applies |
| **MeloTTS** | MIT | ~tens-M | **non-AR** VITS2 | **strong** | Kokoro-tier (not an upgrade) | ✓ | stable but stale (2024) |
| **OuteTTS-0.6B** | Apache-2.0 | 0.6B | AR codec | ok per-sentence | good multilingual+clone | ✓ llama.cpp HIPBLAS | 1B variant is NC |
| Sesame CSM-1B | Apache-2.0 | ~1.1B | AR | poor (conversational, ~10s) | high but slow | ✓ | wrong shape for books |
| Dia (Nari) | Apache-2.0 | 1.6B | AR dialogue | **poor** (~5-20s) | dialogue-focused | ✗ triton-CUDA | not for narration |
| Zonos-v0.1 | Apache-2.0 | 1.6B | AR codec | weak | good | ⚠ NVIDIA-first | partial community AMD |
| Parler-TTS | Apache-2.0 | 0.88-2.2B | AR | weak | good w/ prompt | ✓ (no flash-attn) | slow, stale |
| Bark / Tortoise | MIT / Apache | ~1B | AR | poor / poor | expressive / hi-fi | ✓ | unmaintained, slow |
| Kitten TTS | Apache-2.0 | 15-80M | ONNX | n/a | preview | ✗ GPU (CPU only) | tiny CPU toy |
| **Kokoro-82M** (baseline) | Apache-2.0 | 82M | **non-AR** StyleTTS2 | **strong** | top open ELO for size | ✓ | already excellent for this use case |

### Conclusions for audiblez
1. **The existing Kokoro choice is well-justified.** For an *offline, MIT, long-form* tool,
   its non-AR architecture (no drift over hours, no chunking babysitting), Apache license,
   and CPU+GPU speed are a near-ideal fit. Most "better" models are non-commercial or
   autoregressive (drift-prone on whole books).
2. **Switching is a QUALITY decision, not a GPU-utilization one.** The reason Kokoro barely
   uses the GPU (§7: tiny, non-AR) is the *same* reason it's stable and fast. Models that
   genuinely exercise the RX 7800 XT (VoxCPM2 2B, CosyVoice2, Orpheus 3B) use the GPU more
   but are slower wall-clock and need chunking — "more GPU" ≠ "faster" here.
3. **Top 3 permissive upgrades, ranked for audiblez:**
   - **StyleTTS2 (MIT)** — lowest-risk quality bump: same non-AR stability audiblez already
     relies on (it's literally Kokoro's 148M parent), higher MOS, runs on ROCm. Downsides: stale repo.
   - **VoxCPM2 (Apache)** — highest ceiling: 2026 SOTA-tier naturalness, actually uses the GPU,
     ROCm-confirmed. Downsides: 2B, newer/less battle-tested, needs chunking, some voice variance.
   - **CosyVoice2 (Apache)** — best maintained of the high-quality set; streaming; ROCm guide.
4. **Integration is clean:** audiblez's `build_synthesizer` already returns an engine-agnostic
   `synth(text, speed) -> list[np.ndarray]` closure (torch + mlx engines today). A new model is
   just another engine closure + a `BackendInfo`/voice mapping — the mlx path (core.py:322) is
   the template. No pipeline rewrite needed.
5. **Avoid** (license/fit traps for this project): Fish-Speech, XTTS-v2, F5-TTS weights,
   IndexTTS2, Higgs v2/v3, Spark-TTS, VibeVoice, Dia (no ROCm), Piper-GPL-fork.

### Verification corrections worth recording (adversarial pass earned its keep)
- Higgs Audio v2 license: REFUTED Apache → actually custom "other" (commercial cap). 
- Fish-Speech AMD: REFUTED "unsupported" → official ROCm merged June 2026 (PR #1292) — but still NC-licensed.
- IndexTTS2 AMD: confirmed NO ROCm (CUDA 12.8 + DeepSpeed/BigVGAN CUDA kernels).
- Zonos / Spark AMD: upgraded from "untested" to partial community success — still NVIDIA-first.

## 9. ROCm install recipes for StyleTTS2 / VoxCPM2 / CosyVoice2 (workflow `wf_c04fa83b-721`)

> Per-model install + minimal snippet + reference API, primary-source verified. Common rule:
> **install the ROCm torch wheel FIRST, then install the model under a pip constraints file**
> pinning `torch`/`torchaudio` (+`torchcodec` for VoxCPM) so no transitive dep swaps in a CUDA
> torch and breaks the venv. Order of attack (easy→hard): StyleTTS2 → VoxCPM2 → CosyVoice2.

### StyleTTS2 (easiest, pure-PyTorch, MIT) — Phase 14
- Install: `pip install torch torchaudio` (rocm index) → `pip install -c constraints.txt styletts2`
  (the maintained PyPI wrapper by sidharthrajaram). NO flash-attn/triton/deepspeed. py3.12 OK.
- Weights: auto-download ~1GB on first `tts.StyleTTS2()` (HF `yl4579/StyleTTS2-LibriTTS`).
- Synth: `model.inference(text, target_voice_path=ref.wav, output_sample_rate=24000, alpha=0.3, beta=0.7, diffusion_steps=10)` → **24kHz** float32 mono. **No ref transcript needed** (zero-shot from audio; uses gruut, espeak optional).

### VoxCPM2 (Apache, 2B diffusion-AR, SOTA-tier) — Phase 15
- Install: torch+torchaudio from rocm index FIRST → `pip install -c constraints.txt voxcpm`
  (v2.0.3, needs torch≥2.5 ✓). **DANGER: unpinned `torchcodec`** (no ROCm wheel, ABI-locked to torch)
  → pin `torchcodec==0.8.0` in constraints or `--no-deps`. py3.10-3.12 OK.
- Weights: auto `VoxCPM.from_pretrained("openbmb/VoxCPM2")` (~4-8GB, ~8GB VRAM).
- Synth: `VoxCPM.from_pretrained(..., device="cuda", optimize=False, load_denoiser=False)` then
  `model.generate(text, prompt_wav_path=ref, prompt_text=ref_transcript, reference_wav_path=ref, cfg_value=2.0, inference_timesteps=10)`.
  **`optimize=False` REQUIRED on ROCm** (torch.compile is CUDA-tuned). Output **48kHz** (`model.tts_model.sample_rate`). **Ref transcript REQUIRED.**

### CosyVoice2-0.5B (Apache, hardest install) — Phase 16
- Install from SOURCE: `git clone --recursive` (Matcha-TTS submodule). **requirements.txt is a trap**:
  has `--extra-index-url .../cu121` + `torch==2.3.1` + `tensorrt-cu12*` + `deepspeed` + `onnxruntime-gpu`
  → strip torch/torchaudio/tensorrt/deepspeed lines, change onnxruntime-gpu→onnxruntime, install rocm
  torch first. System dep: `sox`. **pynini==2.1.5 has NO py3.12 wheel** → prefer a **py3.10** venv +
  `conda install -c conda-forge pynini==2.1.5` + `WeTextProcessing`; on 3.12 expect OpenFst build pain.
- Weights: explicit `snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir=...)` (~2GB).
- Synth: `sys.path.append('third_party/Matcha-TTS')`; `CosyVoice2(dir, load_jit=False, load_trt=False, fp16=False)`;
  `load_wav(ref,16000)` (ref MUST be 16k) → `inference_zero_shot(text, prompt_text, prompt_speech_16k, stream=False)`
  yields `out['tts_speech']` **24kHz**. **Ref transcript REQUIRED.** No `device` kwarg (auto cuda).

### Baseline to beat (Phase 12, measured on RX 7800 XT)
Kokoro: RTF **0.155** (3.14s synth for 20.2s audio), load 3.3s, VRAM 1.49GB, 24kHz. `ab_out/kokoro.{wav,json}`.
Shared reference voice minted: `tools/ref_voice.wav` (7.0s @24kHz) + `tools/ref_voice.txt`.

## 10. A/B RESULTS — measured on the RX 7800 XT (ROCm 7.2)

> Same book passage (~430 chars → ~20-27s audio), `tools/tts_ab.py`, steady-state median of 4
> runs after warmup, `torch.cuda.synchronize()`. Cloning models clone the real human reference
> `tools/ref_human.wav`. WER via Whisper base.en round-trip (intelligibility, not naturalness).
> Listen to the wavs in `ab_out/` for naturalness — that's the axis only the user can judge.

| Model | RTF (↓) | synth_s | audio_s | vs Kokoro | WER (↓) | VRAM | install | sample |
|-------|---------|---------|---------|-----------|---------|------|---------|--------|
| **Kokoro-82M** (baseline) | **0.155** | 3.14 | 20.2 | 1× | 0.016 | 1.49GB | (already in) | `ab_out/kokoro.wav` |
| **StyleTTS2** (LibriTTS) | 1.40 | 38.3 | 27.3 | **~9× slower** | **0.000** | 1.76GB | medium* | `ab_out/styletts2.wav` |
| **VoxCPM2** (2B) | 4.87 | 144.0 | 29.6 | **~31× slower** | **0.000** | 6.60GB | medium** | `ab_out/voxcpm2.wav` (48kHz) |
| **CosyVoice2** (0.5B) | 2.40 | 66.6 | 27.7 | **~15× slower** | **0.000** | 3.00GB | hard*** | `ab_out/cosyvoice2.wav` |

\*\*\* CosyVoice2 (hardest): source repo (`git clone --recursive`, Matcha-TTS submodule), edit
requirements (strip cu121 extra-index + torch/torchaudio/tensorrt/deepspeed/grpcio/tensorboard);
install a MINIMAL inference subset (full reqs fail: openai-whisper/grpcio-tools lack cp312 wheels →
C-build fails). openai-whisper installed `--no-deps`. **torchaudio 2.11 forces the torchcodec
backend, whose only wheel is CUDA-only (libnvrtc) → dead on ROCm**; patched `torchaudio.load` to a
soundfile reader. `inference_zero_shot` takes the prompt as a FILE PATH (not a tensor, contra the
recipe). No pynini issue (uses `wetext`). 24kHz.

\*\* VoxCPM2: `pip install -c constraints voxcpm` after rocm torch held the line (torchcodec 0.14.0
coexisted, no CUDA torch pulled). `from_pretrained(..., device="cuda", optimize=False,
load_denoiser=False)` — **optimize=False REQUIRED on ROCm** (torch.compile is CUDA-tuned). 48kHz out
(needs resample to 24k for audiblez). 2B → 6.6GB VRAM, 50s load. Clones prompt_wav + prompt_text.

\* StyleTTS2 install gotchas (ROCm, py3.12): `pip install -c constraints styletts2` after rocm
torch; **torch≥2.6 `weights_only=True`** breaks its checkpoints → patch `torch.load` default to
False (trusted weights); NLTK needs `punkt_tab`. torchaudio only at 2.11.0 on the rocm7.2 index
(coexists with torch 2.12.1). Voice clone via `compute_style()` ONCE + reuse `ref_s=` (per-call
recompute was ~half the cost: RTF 2.9→1.4).

### VERDICT (all 3 validated on the RX 7800 XT)
- **All 3 install + run on ROCm and produce clean, fully-intelligible audio (WER 0.000)**; voice
  cloning of the real human reference works in every case. Main `.venv` (Kokoro) never touched.
- **But every alternative is multiples SLOWER than Kokoro on RDNA3**, and all are slower than
  real-time: StyleTTS2 ~9×, CosyVoice2 ~15×, VoxCPM2 ~31×. Kokoro RTF 0.155 vs 1.40 / 2.40 / 4.87.
  For a 10-hour book: **Kokoro ~1.5h, StyleTTS2 ~14h, CosyVoice2 ~24h, VoxCPM2 ~48h.**
- This re-confirms §7/§10: these are bigger AR/diffusion models; they exercise the GPU more but are
  far slower wall-clock. On ROCm/RDNA3 the penalty is amplified (StyleTTS2 is faster-than-real-time
  on NVIDIA). **Quality/naturalness is the ONLY axis where they might beat Kokoro — judge by ear**
  (`ab_out/*.wav`); intelligibility is already a wash.
- **Recommendation:** integration (Phase 18) only makes sense as an **opt-in "quality mode"** flag
  for users who will accept 9-31× slower renders. StyleTTS2 is the best speed/quality trade of the
  three (closest to real-time, smallest VRAM, MIT, drop-in 24kHz). VoxCPM2 needs a 48k→24k resample.
  Default stays Kokoro. Do NOT replace Kokoro; ADD a backend.
- Evaluation loop (install→synth→speed+WER+sample) PROVEN end-to-end for all three.

## 11. VibeVoice — status re-checked 2026-06-18 (updates §8)

Authoritative `gh` check of `microsoft/VibeVoice`: repo LIVE (not archived), **MIT license**,
49.4k stars, last push 2026-05-06. README is explicit:
- **2025-09-05:** Microsoft **removed the original VibeVoice-TTS code** (1.5B/7B) after misuse.
- **2025-12-03:** released **VibeVoice-Realtime-0.5B** (HF `microsoft/VibeVoice-Realtime-0.5B`) —
  streaming + "robust long-form"; this is the currently-available model (0.5B, smaller/faster than
  the pulled 1.5B).
- **Risks section (verbatim):** *"We do not recommend using VibeVoice in commercial or real-world
  applications without further testing and development. This model is intended for research and
  development purposes only."*

### Fit for audiblez (offline, MIT-shipping tool, AMD RX 7800 XT)
- **Architecture:** Qwen2.5 LLM backbone + diffusion head → LLM-class (installable like Orpheus/CosyVoice).
- **Long-form:** genuine strength (its design goal) — the most audiobook-relevant on paper.
- **License/intent:** MIT *code*, but Microsoft's explicit **"research/development only, not for
  real-world/commercial use"** disclaimer is a real adoption flag for a tool users run on real books.
  This is stronger wording than a generic RAI notice — treat as a yellow/red flag, not clean MIT.
- **AMD/ROCm:** §8 evidence (issue #185, RX 7900 XTX) = slower-than-real-time, 15-19% GPU util,
  **flash-attention AMD/Triton path fragile** (install risk on gfx1101). Expect VoxCPM2-band speed
  or worse on the RX 7800 XT; the 0.5B realtime variant would be faster than the pulled 1.5B but
  still in the multiples-slower-than-Kokoro range, with extra flash-attn install risk.
- **Verdict:** most interesting long-form candidate, but the research-only disclaimer + AMD speed/
  flash-attn fragility make it a poor practical fit. Testable (VibeVoice-Realtime-0.5B) in the
  harness if desired, with those caveats.

## 11b. devnen/Chatterbox-TTS-Server — honest look (2026-06-18)

`gh` + README check. MIT (server wrapper), 1.3k★, active (push 2026-05-26). A **FastAPI HTTP
server** (OpenAI-compatible API + Web UI) wrapping Resemble AI's **Chatterbox**; fully local/offline
after model download. Three models: Original (0.5B, emotion control, 10-step diffusion), Multilingual
(0.5B, 23 langs), **Turbo (350M, 1-step diffusion, `[laugh]/[cough]` tags, "significantly improved
throughput")**. Optional `TTS_BF16=on` (~40% throughput). Python **3.10 only**. AMD: `--rocm` →
`requirements-rocm.txt` with **PyTorch 2.5.1+rocm6.1**; Strix Halo / RDNA4 via Docker (ROCm 7.2). No
RDNA3/RX 7800 perf numbers.

### The honest catch (vs the user's "AMD-optimized" goal)
- **Its AMD acceleration is native PyTorch-ROCm EAGER mode** — the *exact slow path* that made Wave 3
  models 9-31× slower than Kokoro. "Runs accelerated on AMD (ROCm)" is true (uses the GPU) but it is
  **NOT an AMD-first/optimized runtime** (not llama.cpp-Vulkan, not ONNX-ROCm/MIGraphX). The server is
  polished convenience packaging over the same eager-PyTorch-ROCm path. Chatterbox is autoregressive →
  expect Wave-3-band slowness on the RX 7800 XT (Turbo+BF16 helps, maybe 2-3×, still likely > Kokoro).
- **Architecture fit:** it's a SERVER/daemon — against audiblez's offline-single-process, no-daemon
  principle. For audiblez you'd use the underlying `chatterbox` LIBRARY directly, not this server. The
  server's value here is just (a) a clean ROCm install recipe to copy, (b) standalone use / Web UI.
- **Quality:** GENUINELY better than Kokoro (Chatterbox: ~63.75% blind preference vs ElevenLabs; top
  open naturalness). So it nails the QUALITY goal — just not the AMD-SPEED goal.
- **Watermark:** upstream Resemble Chatterbox embeds a Perth audio watermark in outputs (not mentioned
  in this server's README; flagged from upstream knowledge — verify on the Turbo variant).

### Verdict
Best-in-class for the *quality* complaint, well-maintained, ROCm-documented — but it does **not** solve
"optimized for AMD": it's eager PyTorch-ROCm (slow path) wrapped in a server (wrong shape for audiblez).
Worth A/B-ing **Chatterbox-Turbo (350M) + BF16** via the underlying lib for the quality win, with the
expectation it's slower than Kokoro. The real answer to quality+AMD-speed depends on whether the running
AMD-first research (§12) finds a GGUF/llama.cpp or ONNX path to a Chatterbox-class model.

## 11c. lemonade-sdk/llamacpp-rocm — honest look (2026-06-18)

`gh`+README. MIT, 542★, VERY active (nightly, push 2026-06-17). **Prebuilt llama.cpp with ROCm 7
built-in** (based on AMD's "TheRock"), **self-contained — no separate ROCm install needed**. Linux +
Windows. Targets explicitly include **gfx110X = RDNA3, listing the RX 7800 XT by name** ✓ (one binary
covers gfx1100/1101/1102). This is a genuine AMD-first runtime and exactly the right *kind* of thing.

### The honest catch
- **It is LLM/GGUF-ONLY.** README focuses on `llama-server -m model.gguf`; **no TTS mentioned.** So by
  itself it does nothing for audiblez TTS. It's a RUNTIME, not a model.
- It only helps TTS if a model runs through **llama.cpp as GGUF**, which is a SHORT list: llama.cpp's
  TTS support is essentially **OuteTTS** (`llama-tts` + WavTokenizer) and, via the transformer-LLM
  backbone, **Orpheus** (Orpheus.cpp / community GGUF). The high-quality models the user wants
  (Chatterbox, F5, CosyVoice, VibeVoice, StyleTTS2) are **NOT llama.cpp-expressible** — their diffusion/
  flow-matching/StyleTTS2 graphs aren't GGUF transformer models. llama.cpp ≠ universal TTS runtime.
- Quality of what it CAN run: **OuteTTS ≈ Kokoro-tier** (not a clear upgrade); **Orpheus is more
  expressive** than Kokoro and IS a Llama-3B → the one promising combo = **Orpheus GGUF on this
  ROCm llama.cpp** (quality bump + real AMD-accelerated runtime). Whether the lemonade release zip
  bundles a `llama-tts` binary is UNVERIFIED (LLM-focused; may need the upstream tts tool).
- Usage = CLI/server binary → audiblez would shell out or hit the local server (adds process mgmt,
  but the binary is self-contained). Against the pure-in-process design, but tolerable behind a flag.

### Verdict
Keep it — it's the best AMD-first **runtime** for the RX 7800 XT and the right substrate for the
quality+speed combo. But it's LLM/GGUF-only, so the real question is the MODEL: **Orpheus (GGUF) on
llamacpp-rocm** is the concrete experiment worth running (the research §12 is checking exactly this:
Orpheus.cpp / OuteTTS-GGUF). Verifying `llama-tts` in the build + benchmarking Orpheus-GGUF on the
RX 7800 XT is the validation step (Phase 20).

## VERIFY: Llasa-3B GGUF naturalness > Kokoro-82M? (verdict: REFUTED)
- Kokoro-82M reached #1 on TTS Arena (blind crowdsourced vote); ~1056-1067 ELO, top open-weights; UTMOS 4.48 (near-human, ties commercial). Sources: siliconflow, offlinetts, codesota, texttolab.
- Llasa/Llasa-3B is ABSENT from TTS Arena leaderboard (no ELO, no blind votes). No MOS/blind test placing it above Kokoro found.
- Only Llasa-associated UTMOS is XCodec2 CODEC reconstruction = 4.13 (resynthesis, not TTS gen), BELOW Kokoro's 4.48 generation UTMOS; not directly comparable.
- Llasa paper (arXiv 2502.04128) reports SOTA WER + speaker-SIM (zero-shot cloning strength), NOT naturalness wins; no Kokoro comparison.
- Net: no leaderboard/MOS/blind-test evidence that Llasa beats Kokoro on naturalness; strong contrary evidence. Dossier "plausibly > Kokoro" not supported.

## 14. PyTorch on AMD / RDNA3 (gfx1101) — research + ON-BOX measurements (workflow `wf_9fd1a682-87f`)

> 6 levers researched; agents BENCHMARKED several on this exact box (torch 2.12.1+rocm7.2, gfx1101).
> Skeptic pass pressure-tested every "yes". Headline: the bottleneck we kept hitting (launch-overhead /
> batch-1 / many-small-ops) has a real fix — **torch.compile** — that fp16/TunableOp couldn't provide.

| Lever | Helps gfx1101 inference? | Measured / expected | Verdict |
|-------|------------------------|---------------------|---------|
| **★ torch.compile (Inductor, mode="default")** | **YES for launch-bound** | **~8-11× on many-small-op graphs** (measured here); **~1.0× on matmul-bound** | CONFIRMED by skeptic. THE lever for our symptom |
| AOTriton flash-attn SDPA | partial, fragile | ~1.2-1.5× attention-bound — but **off by default on gfx1101** (prints "Flash attention was not compiled for gfx1101", silently falls back to math) | OVERSTATED; needs `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, experimental |
| hipBLASLt / TunableOp GEMM | **NO** | measured: rocBLAS 54.4 TFLOPS default; hipBLASLt **17% SLOWER**; TunableOp ~1% (noise) | Don't touch — rocBLAS default is already the fast path (confirms Phase 9) |
| RDNA3 support tier / HSA_OVERRIDE | n/a (table-stakes) | gfx1101 natively supported in ROCm 7.2.x; HSA_OVERRIDE = non-lever, leave UNSET | confirms our `.envrc` decision |
| fp16/bf16 · channels_last · inference_mode | partial | fp16 ~1.5-2× over fp32 **only if GEMM hits hipBLASLt WMMA** (matmul-bound models); inference_mode free; bf16 slightly worse on gfx1101 | partial; why fp16 was flat on tiny Kokoro (Phase 10) but may help bigger models |
| Switch runtime off eager-PyTorch (llama.cpp Vulkan/HIP, ONNX/MIGraphX) | YES | llama.cpp Vulkan ~118-120 vs ROCm ~96-101 tok/s on 7800 XT (~18-25%, but skeptic: fragile/mis-attributed) | reliable for LLM-class; for TTS only OuteTTS/Orpheus (see §11c) |

### The key insight (corrects Wave-2 task_plan note "torch.compile — RDNA3 Triton uneven; skip")
- **torch.compile WORKS on gfx1101** with the stock wheel (ROCm Triton bundled) and gives **~8-11× on
  launch-overhead/many-small-op graphs** — which is EXACTLY Kokoro's profile (§6/§7: batch-1, tiny ops,
  per-segment sync). fp16/bf16/TunableOp were flat precisely because the bottleneck is launch overhead,
  NOT GEMM FLOPs — and torch.compile is the one lever that attacks launch overhead (kernel fusion).
- Use `mode="default"` (NOT "reduce-overhead"/cudagraphs — fragile on ROCm). Already installed. Cost:
  first-call compile latency (seconds-min), amortized over a book-length run.
- **The catch to TEST:** TTS inputs are VARIABLE-LENGTH (each sentence a different token count) →
  torch.compile may recompile per shape (killing the win) unless `dynamic=True` or length-bucketing is
  used. The measured ~11× was a FIXED shape. Must benchmark on real variable-length sentences. → Phase 20b.
- Don't bother: hipBLASLt force, HSA_OVERRIDE, AOTriton-SDPA (for small TTS). fp16 only helps bigger
  matmul-bound models (StyleTTS2/CosyVoice2), not tiny Kokoro.

### §14 follow-up — torch.compile MEASURED on REAL Kokoro (not a microbenchmark): FLAT
Tested `torch.compile(pipeline.model.forward_with_tokens, dynamic=True)` in the A/B harness on the real
passage: **RTF 0.152 vs 0.155 baseline = no change.** The research's ~11× was a synthetic 20-elementwise-op
microbenchmark; it did NOT transfer to the real model. Why: Kokoro's `forward_with_tokens` has
data-dependent ops (`repeat_interleave(pred_dur)`, `.item()`, `pred_aln_trg[indices,...]` index-assign)
+ a sequential LSTM → graph breaks / unfusable; and Kokoro is already launch-light/fast (RTF 0.155).
**Lesson: microbenchmark speedups ≠ real-model speedups when the real graph has data-dependent control
flow.** torch.compile is likely a dead end for Kokoro; UNTESTED (and not assumed) for the bigger
quality models (StyleTTS2/CosyVoice2), which are more op-heavy but also have dynamic shapes.
`ab_out/kokoro-compiled.wav` identical to baseline.

## MOSS-TTS-GGUF license + offline verification (verified 2026-06-18)
- License: Apache 2.0 (standard, unmodified text in repo LICENSE; copyright OpenMOSS Team, Fudan/SII/MOSI 2026). No NC/research-only/acceptable-use clauses.
- Commercial use: YES (Apache 2.0 permissive).
- Offline: YES. torch-free llama.cpp + ONNX Runtime audio tokenizer; runs fully local/edge/CPU. No mandatory cloud.
- GATING CAVEAT: Official OpenMOSS-Team/MOSS-TTS-GGUF repo IS gated on HF ("You need to agree to share your contact information to access this model"; must log in to review conditions). License itself is still Apache 2.0 — gating is an HF access wall, not a license restriction. Ungated community mirror exists: John9007/MOSS-TTS-Local-GGUF (apache-2.0, freely downloadable, llama.cpp).

## 12. AMD-first TTS research — SYNTHESIS (workflow `wf_84cfbaf9-d53`, 102 agents)

> Hunt: a TTS that is (a) more natural than Kokoro in BLIND tests, (b) genuinely AMD-accelerated on
> THIS Linux/ROCm box, (c) permissively licensed. Adversarial verify on all three. Result is sobering.

> **[OVERRULED 2026-06-18 — user listening test]** The user A/B'd `ab_out/moss.wav` vs
> `ab_out/kokoro.wav` and judged **Kokoro sounds bad, MOSS sounds good**. This resolves the
> open question the docs explicitly deferred to the user's ear (TTS_MODELS_COMPARISON.md:62,
> findings.md:410). The blind-leaderboard "#1-tier naturalness" claim below (point 1, and the
> "Honest reframe" section) is **superseded for this project**: MOSS is the quality default,
> Kokoro the fast fallback. See `docs/adr/0003-moss-default-engine.md`. The AMD-runtime
> conclusions (points 2-3, the Orpheus/MOSS llama.cpp-Vulkan path) still stand.

### The hard truth (verifiers refuted most optimism)
1. **Kokoro is actually #1-tier on BLIND naturalness.** TTS-Arena V2 / Artificial-Analysis 2026:
   Kokoro-82M v1.0 ELO ~1056-1067, ~54% blind win-rate — top of the OPEN-weight models. Models that
   "feel" better usually do NOT beat it blind: **Chatterbox HD ELO ~1050 < Kokoro** (quality claim
   REFUTED); Piper/Kitten/sherpa-onnx rank BELOW Kokoro (sherpa literally hosts Kokoro).
2. **ONNX-on-AMD-Linux is a DEAD END.** ONNX Runtime ROCm EP was REMOVED in onnxruntime 1.23 (user is
   on ROCm 7.2); DirectML EP is Windows-only; sherpa-onnx exposes only CUDA/CoreML providers (no AMD
   GPU); MIGraphX EP is unproven for TTS (silent CPU fallback). So Kokoro-ONNX/Piper/F5-ONNX/sherpa get
   ZERO AMD GPU accel here — all AMD-accel claims REFUTED on this platform.
3. **The genuinely-more-natural models are non-commercial and/or off-platform:** F5-TTS (plausibly
   better MOS) = CC-BY-NC weights + AMD path only via DirectML-on-Windows; Llasa-3B = CC-BY-NC + no
   clean AMD path. Both fail license AND platform.

### The ONE viable upgrade path
**Orpheus-3B (Canopy Labs), Apache-2.0** — FIT 4/5, the only candidate clearing AMD-accel + license:
- **AMD accel CONFIRMED:** its Llama-3.2-3B backbone runs as GGUF on **llama.cpp Vulkan/HIP** — real,
  fast, vendor-neutral on the RX 7800 XT (your card is a confirmed Vulkan coopmat device). The small
  SNAC audio decoder runs separately (CPU-fast / PyTorch — not the bottleneck). This ESCAPES the
  eager-PyTorch-ROCm slow path. (This is exactly what lemonade `llamacpp-rocm` §11c provides the runtime for.)
- **Quality vs Kokoro: NOT a proven MOS win** (verifier: MOS ~4.6 vs ~4.5 = within noise). BUT it offers
  what Kokoro lacks: **emotion/expressiveness** (emotion tags, prosody) — Kokoro has flat affect. So
  Orpheus is a *different character* (expressive), not a guaranteed naturalness bump.
- Caveat: needs realtime ~83-86 LM tok/s on the 7800 XT to keep up — plausible via Vulkan but UNMEASURED.

### Honest reframe of "Kokoro quality is bad"
Blind data says Kokoro is top-tier among open models — the complaint is most likely (a) **flat emotion**
(real; Orpheus addresses it) and/or (b) **voice choice** (Kokoro has 54 voices + supports voice-blending;
the default/voice tried may not be its best). Cheapest experiments before any big switch: try other
Kokoro voices / blends; if it's *expressiveness* you want, prototype Orpheus-3B-GGUF on llama.cpp-Vulkan.

## 13. torch.compile A/B on RDNA3 — MEASURED (dead end for TTS)
| model | baseline RTF | compiled RTF | result |
|-------|--------------|--------------|--------|
| Kokoro | 0.155 | 0.152 | no change |
| StyleTTS2 (diffusion+encoders compiled; decoder excluded) | 1.40 | 1.54 | no help / slightly worse |
- StyleTTS2 `decoder` (iSTFTNet vocoder) **CRASHES under Inductor on ROCm** (`upsample_linear1d` lowering
  bug, dynamic shapes) → had to exclude it. Compiling the rest gave nothing.
- Conclusion: torch.compile does NOT speed up these TTS models on gfx1101 — data-dependent control flow
  (durations/alignment) + LSTM cause graph breaks, and the heaviest compilable part (vocoder) breaks.
  The research's ~11× microbenchmark does not transfer to real TTS graphs. CosyVoice2-compiled not run
  (AR LLM, even less compile-friendly; pattern conclusive).

## 15. Orpheus-3B on llama.cpp-Vulkan — WORKING PROTOTYPE (RX 7800 XT, 2026-06-18)

Built the §12 "one viable AMD-first upgrade" end-to-end: orpheus-cpp (Orpheus-3B-Q4_K_M GGUF) with the
LLM on **llama.cpp-VULKAN** + SNAC-ONNX decoder. Isolated `.venv-orpheus`; main `.venv` untouched.
Toolchain: built `llama-cpp-python 0.3.30` with `-DGGML_VULKAN=on` (needed sudo `vulkan-headers shaderc
glslang spirv-headers spirv-tools`).

**CONFIRMED genuine AMD GPU acceleration** (not a silent CPU fallback): llama.cpp log shows
`using device Vulkan0 (AMD Radeon RX 7800 XT (RADV NAVI32))` + every layer `assigned to device Vulkan0`,
KHR_coopmat matrix cores. This is the ONLY path in the whole investigation that runs a better-than-Kokoro-
*class* model with real, non-eager-PyTorch AMD acceleration.

### Measured A/B (same passage, vs the others)
| model | RTF | synth_s | audio_s | WER | runtime | quality |
|-------|-----|---------|---------|-----|---------|---------|
| Kokoro (baseline) | **0.155** | 3.1 | 20.2 | 0.016 | pytorch-rocm eager | flat affect |
| **Orpheus-3B** | **1.35** | 31.3 | 23.2 | 0.016 | **llama.cpp-Vulkan (GPU ✓)** | expressive (judge by ear) |
| StyleTTS2 | 1.40 | 38 | 27 | 0.000 | pytorch-rocm | — |
| CosyVoice2 | 2.40 | 67 | 28 | 0.000 | pytorch-rocm | — |
| VoxCPM2 | 4.87 | 144 | 30 | 0.000 | pytorch-rocm | — |

### Honest read
- **Orpheus is the FASTEST of the quality models** (RTF 1.35) AND the only one on a true AMD-first runtime
  (Vulkan, confirmed GPU offload). Voice 'tara' (8 voices), Apache-2.0, 24kHz, perfectly intelligible.
- **BUT still slower than real-time and ~9× Kokoro.** 10h book: ~13.5h (Orpheus) vs ~1.5h (Kokoro).
  So: usable as an opt-in "expressive mode", not a default. Speed is intrinsic to a 3B AR model.
- **The decisive open question is QUALITY** — does `ab_out/orpheus.wav` sound more expressive/natural than
  `ab_out/kokoro.wav` to the USER? Orpheus's pitch is emotion/prosody (Kokoro's weakness). If yes → real
  (slow) upgrade, integratable behind a flag. If no → Kokoro stays. Only the user can call this.
- Possible speed tuning not yet tried: smaller quant (Q4 already), shorter `max_tokens`, or accept the
  RTF for a "quality pass" use case. Streaming (orpheus-cpp supports it) helps latency, not total time.

## 16. GPU-utilization reality (user-observed + rocm-smi, 2026-06-18)

The user noticed in `btop` that the GPU sat near-idle while one CPU core pegged 100% during the
PyTorch-ROCm model runs. CONFIRMED via rocm-smi during a fish run: **GPU use 1–11% (mostly ~3%)** while
the model occupied VRAM. The card is LOADED but STARVED.

**Root cause (ties together §6/§7/§13/§15):** these models (fish, StyleTTS2, CosyVoice2, VoxCPM2, even
Kokoro) are **batch-1 autoregressive/sequential** — they emit one token/segment at a time. Each GPU op
finishes in microseconds, then the GPU waits for the CPU (sampling, Python, next dispatch). So the GPU
is idle most of the time → low util, CPU-bound. This is THE reason:
- the models are slow (bottleneck = serial CPU loop, not GPU FLOPs);
- no GPU lever helped (fp16/TunableOp/torch.compile optimize compute that isn't the bottleneck);
- Kokoro is only ~1.1× faster on GPU than CPU.
**It is architectural, not a misconfiguration.** Fix = non-AR model (Kokoro is one) or true batching.
Exception: **Orpheus on llama.cpp-Vulkan** keeps the GPU fed better (llama.cpp is built for batch-1
token-gen) — the user observed real GPU use there and during fish's generation attempt.

## 17. Fish-Speech / OpenAudio S1-mini — ✅ RESOLVED & MEASURED (2026-06-18)
**WORKS end-to-end on the RX 7800 XT.** WER **0.032** (genuine, intelligible speech; voice-cloned the
shared human reference). Output: `ab_out/fish.wav` (19.9 s @ 44.1 kHz).

**The blocker (and the corrected diagnosis):** my first guess (repo HEAD too OLD / S2-pro era) was
**backwards**. Repo HEAD `e5e2926` is *too NEW*: its `fish_speech/tokenizer.py:57` rewrote `FishTokenizer`
to call `AutoTokenizer.from_pretrained` (expects an HF `tokenizer.json`), but s1-mini ships only the older
`tokenizer.tiktoken`. The "`dual_ar` ... Transformers out of date" message is a **red herring** — `dual_ar`
is fish-speech's custom model type, never in any transformers release; downgrading transformers cannot help.

**The fix (cheap, no reinstall):**
1. `git fetch --unshallow --tags` → s1-era commit = `781bf1c` "Finetune support of OpenAudio-S1" (2025-10-20);
   `AutoTokenizer` landed later in `b72bcb3`/`daa9b4f` "S2 beta" (2026-03-10).
2. Glue (content_sequence/conversation/inference) ALSO changed in S2-beta → one-file swap unsafe → checked
   out the whole **internally-consistent** s1-era source: `git checkout 781bf1c -- fish_speech tools`.
3. Install is PEP 660 editable (finder maps pkg-root→worktree) → **source checkout needs no reinstall**;
   ROCm torch 2.12.1 + tiktoken 0.13.0 already satisfy it.
4. Adapter (`tools/tts_ab.py:load_fish`) needed one more shim: stub `torchaudio.list_audio_backends`
   (removed in torchaudio 2.9; s1-era code predates it; HEAD fixed it in `75a9afe`). Only used to pick
   ffmpeg-vs-soundfile, and we already force soundfile → safe.

**Speed (the catch):** measured **RTF 19.4** (synth 386 s for 19.9 s audio). Breakdown: AR generation ran on
GPU at ~9.9 tok/s, ~43 s (gen-only RTF ≈ **2.2**, same band as Orpheus); the **DAC decode added ~340 s**
because MIOpen can't allocate GEMM workspace on this path (`GemmFwdRest ... provided ptr:0 size:0` → slow
immediate-mode fallback) + experimental mem-efficient SDPA on RDNA3. So the *decode*, not generation, is the
pathology here.

**Stability hazard (new, important):** the **first** run **crashed the desktop session** — a transient ROCm
fault during the cold-kernel DAC decode reset the GPU (which also drives the display). The **retry succeeded**
(warm MIOpen kernels). So eager-ROCm here is not just slow but *session-fragile* on a display GPU. Reinforces
§16 / [[audiblez-amd-tts-runtime-reality]].

**Bottom line:** s1-mini is reachable and intelligible on this box, but ~125× slower than Kokoro as-measured
(~14× even counting generation only) **and** it risked crashing the desktop. Quality/naturalness vs Kokoro is
the only remaining open question — decide by EAR (`ab_out/fish.wav` vs `ab_out/kokoro.wav`). Not viable as a
daily audiobook driver on this hardware regardless.

## 18. MOSS-TTS (OpenMOSS) — ✅ RESOLVED & MEASURED — the best quality-upgrade path (2026-06-18)

**RESULT: MOSS-TTS (MossTTSDelay 8B) on llama.cpp-Vulkan is the best Kokoro-upgrade path found — an expressive
8 B model that renders faster than real-time on this box, WER 0.000, no crash.** As-built RTF **0.29** (~10 h
book ≈ **2.9 h**) — i.e. **~2× Kokoro** (0.155), NOT a tie. ⚠️ The 0.29 is the HONEST as-built number: the
first-class path is a one-shot CLI that reloads ~7.7 GB **every call** (load 2.69 s), and Kokoro's 0.155 is
*resident*. Gen+decode-only RTF is **0.168** (≈ Kokoro) but that requires a **resident/server wrapper that does
not exist yet** (unbuilt/unmeasured). Either way it's ~4–8× faster than Orpheus and beats real-time, so it's
practically usable. Supersedes §12's "no quality model is viable-fast here" — MOSS is. Only open question =
subjective naturalness (`ab_out/moss.wav` vs `ab_out/kokoro.wav`; WER can't rank it).

**Measured (`tools/moss_bench.py`, Q5_K_M backbone + f16 decoder, -ngl -1):**
- warm passage runs: 6.36 / 6.36 / 6.34 s wall for **21.84 s** audio (rock-steady)
- load/init baseline (tiny text): 2.69 s → gen+decode = 3.67 s → **RTF_gen+decode 0.168**, RTF_total **0.29**
- VRAM: backbone 5.27 GB (37/37 layers) + decoder 1.69 GB (9/9) + KV 0.32 + compute 0.19 ≈ **8.25 GB**
- Vulkan device: RX 7800 XT RADV NAVI32, **fp16 + KHR_coopmat matrix cores** (the speed lever vs Orpheus eager)
- Q5_K_M quant of the 32 `output_audio.*` multi-codebook heads did **NOT** degrade: GPU-Q5 WER 0.000 == CPU-f16 WER 0.000
- CPU (-ngl 0) f16 also validated WER 0.000 (isolates conversion-correctness from quant & Vulkan)
- Why so much faster than Orpheus-3B (RTF 1.35) despite 8B: coopmat matrix cores + MOSS's low 12.5 Hz audio-token
  rate (≈275 steps for 22 s) + Q5. The earlier "slow" smoke run was just first-run RADV pipeline (shader) compilation.

**Repro / setup (all under `/home/caio/Projects/`):**
1. `git clone --depth 1 -b moss-tts-firstclass https://github.com/OpenMOSS/llama.cpp.git llama.cpp-moss`
2. `cmake -S llama.cpp-moss -B llama.cpp-moss/build-vulkan -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON`
   then `cmake --build … --target llama-moss-tts llama-quantize -j`
3. `hf download OpenMOSS-Team/MOSS-TTS` (8B, 16 GB) + `hf download OpenMOSS-Team/MOSS-Audio-Tokenizer` (6.7 GB)
   → `moss-work/MOSS-TTS-hf`, `moss-work/MOSS-Audio-Tokenizer-hf`. Base model NOT gated.
4. **Convert (needs `sentencepiece` — installed isolated to `moss-work/pydeps`, NOT main .venv; `gguf` via fork's
   `gguf-py` on PYTHONPATH):** `convert_hf_to_gguf.py MOSS-TTS-hf --outfile …f16.gguf --outtype f16` (→17 GB; falls
   back to BPE vocab — this model ships tokenizer.json, no spm tokenizer.model). The pre-quantized
   `OpenMOSS-Team/MOSS-TTS-GGUF` repo is **NOT** compatible with the first-class binary; you must convert.
   Audio tokenizer: `convert_moss_audio_tokenizer_split_to_gguf.py MOSS-Audio-Tokenizer-hf --encoder-outfile …
   --decoder-outfile … --outtype f16` (flags are `--encoder-outfile/--decoder-outfile`, NOT `--outdir`).
5. `llama-quantize …f16.gguf …Q5_K_M.gguf Q5_K_M` (→6.06 GB; f16 17 GB won't fit a 16 GB *display* GPU).
6. Run: `build-vulkan/bin/llama-moss-tts -m …Q5_K_M.gguf --audio-decoder-model …decoder_f16.gguf --text "…"
   --wav-out out.wav -ngl -1`. Voice clone: add `--audio-encoder-model …encoder_f16.gguf --reference-audio
   ref_24k_mono.wav` (ref MUST be 24 kHz mono — `tools/ref_human.wav` qualifies). The binary emits NO llama_perf
   timings, so `moss_bench.py` measures wall-clock and subtracts a tiny-text baseline for gen+decode RTF.

GGUFs live at `/home/caio/Projects/moss-work/gguf/` (`moss_delay_firstclass_{f16,Q5_K_M}.gguf`,
`moss_tts_audio_{encoder,decoder}_f16.gguf`). See [[audiblez-amd-tts-runtime-reality]].

---
### Original assessment (pre-test) — 2026-06-18

User asked to vet OpenMOSS/MOSS-TTS. It's the **2nd model family (after Orpheus) with a real llama.cpp path**
= the one proven fast AMD runtime here. Apache-2.0 (already noted line 586). Built for **expressiveness**
(token-level duration, phoneme control, zero-shot clone, 20 langs) — Kokoro's exact weakness.

**Runtime reality per-variant (the decisive axis), verified from HF cards + fork:**
- **MossTTSLocal-v1.5 (4B, Qwen3-4B, 48 kHz stereo)** — "recommended for evaluation", BUT **PyTorch/Transformers-only**
  (`AutoModel … trust_remote_code=True`); **no GGUF/llama.cpp/ONNX/mlx**. Wants Transformers 5.0+ & FlashAttn-2 (CUDA-only,
  absent on ROCm). → would run on the **eager-PyTorch-ROCm slow band** (StyleTTS2/CosyVoice2/VoxCPM2 territory). Docs also
  state Delay is *faster by design* than Local. **User chose NOT to test this** — quality-only, no viable runtime.
- **MossTTSDelay (8B, Qwen3-8B, 24 kHz)** — has a **first-class llama.cpp impl** (fork `OpenMOSS/llama.cpp`
  @ `moss-tts-firstclass`, sha `b785003`): GGUF backbone + GGUF/ONNX audio codec. **← THIS is what we're testing.**
- MOSS-TTS-Nano (~100M, MossTTSNano) — pure-CPU fallback, 48 kHz, streaming, untested.

**Critical gotcha:** the first-class binary needs a GGUF **converted from the full HF dir** via the fork's
`convert_hf_to_gguf.py` (→ `moss_delay_firstclass_f16.gguf`). The pre-quantized `OpenMOSS-Team/MOSS-TTS-GGUF` repo
is **NOT compatible** with `llama-moss-tts` (different path). So: download HF `OpenMOSS-Team/MOSS-TTS` (4 shards ≈16 GB)
→ convert f16 → quantize (Q5_K_M to fit 16 GB VRAM); + `OpenMOSS-Team/MOSS-Audio-Tokenizer` → encoder/decoder GGUFs.

**Open uncertainty:** the first-class llama.cpp e2e guide documents **CUDA only** — Vulkan support for the custom
`llama-moss-tts` binary (esp. the audio codec graph) is **unverified**. The Qwen3 backbone will run on Vulkan
(Orpheus proves it); the codec may need CPU fallback. Building with `-DGGML_VULKAN=ON` to find out.

**Plan/state:** parallel background jobs running — (1) clone fork + Vulkan build of `llama-moss-tts`
(`/home/caio/Projects/llama.cpp-moss`), (2) `hf download` model + tokenizer (`/home/caio/Projects/moss-work`).
Next: convert→quantize→smoke-test on GPU (watch for s1-mini-style desktop crash)→benchmark + WER + `ab_out/moss.wav`.
