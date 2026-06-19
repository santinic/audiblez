# TTS Model Research for audiblez Narration (mid-2026)

**Question:** best open, locally-runnable TTS for **long-form English audiobook narration** —
constraints: any open license OK, **single preset narrator** (not cloning), **balanced**
quality/speed, with **Apple Silicon** and **AMD ROCm** support.

**Method:** deep-research harness — 5 search angles → 22 sources fetched → 106 claims →
25 verified by 3-vote adversarial check (20 confirmed, 5 refuted). Findings below are the
confirmed ones; see *Caveats* for what's uncertain.

---

## TL;DR

- **No open model decisively beats Kokoro-82M for *this* use case.** On the Artificial Analysis
  Speech Arena (blind human-preference Elo), Kokoro is **4th among open-weights** (Elo ~1062),
  behind Fish Audio S2 Pro (1123), Step Audio EditX (1111), Voxtral TTS (1067) — but those three
  are **cloning/general systems, not preset narrators**, and the Kokoro↔Voxtral gap is ~5 Elo and
  drifts week to week.
- **Best English quality that actually fits the constraints → Chatterbox** (via *Chatterbox-TTS-Server*),
  the leading higher-quality challenger. Caveat: its "predefined voices" are curated reference clips
  run through a cloning pathway, and it has an open chunk-boundary distortion bug on long inputs.
- **Best on Apple Silicon → Kokoro** (native **MLX** path via `mlx-audio`). **Piper** is *faster* on
  an M4 (~33× vs ~14× RTF) but lower fidelity.
- **Best on AMD ROCm → Kokoro** (runs on the standard ROCm PyTorch wheel; real-world proof exists).
- **Recommendation for audiblez: keep Kokoro as default**, optionally offer Chatterbox as a
  higher-quality engine, and — highest leverage — **add MPS + ROCm device paths** (audiblez today
  only does CPU/CUDA).

---

## 1. Best English narration *quality* (leaderboard view)

Source: **Artificial Analysis Speech Arena** (the most relevant open benchmark) — blind pairwise
human preference, LMSYS-style Elo, ~500-char prompts. *Not* MOS/WER.

| Rank (open-weights) | Model | Elo | Notes |
|---|---|---|---|
| 1 | Fish Audio S2 Pro | ~1123 | cloning/general; audiblez-fit **unverified** |
| 2 | Step Audio EditX | ~1111 | cloning/general; **unverified** |
| 3 | Voxtral TTS | ~1067 | **unverified** |
| 4 | **Kokoro-82M v1.0** | ~1062 | true preset narrator; the baseline |
| 5 | Magpie-Multilingual | ~1056 | — |

⚠️ **Two big caveats:** (1) the top three are not single-preset-narrator models and their
long-form/Mac/ROCm fit was **not verified** — they're quality leaders only. (2) The arena tests
**500-char clips**, so it does **not** measure the thing that matters most for audiobooks:
**long-form stability** (hallucination/repetition/word-drop over hours). No strong open long-form
benchmark surfaced.

**Practical quality pick (fits preset-narrator + long-form): Chatterbox-TTS-Server.**
- Exposes curated **"Predefined Voices"** in a dropdown (no manual cloning), does **sentence-aware
  chunking + concatenation** explicitly marketed for audiobooks, runs on CUDA/ROCm/MPS/CPU.
- **Caveats:** the predefined voices are curated reference clips routed through Chatterbox's
  **zero-shot cloning** pathway (not model-native speaker IDs like `af_sky`), and **Issue #75**
  reports voice distortion at chunk boundaries on long inputs. So "consistent narrator for a whole
  book" is plausible but not proven.

---

## 2. Best on Apple Silicon

- **Kokoro — top quality pick.** Native **MLX** path via **`mlx-audio`** (MIT, ~7.2k★,
  `pip install mlx-audio`), loading `mlx-community/Kokoro-82M-bf16` with 54 voice presets + speed
  control — a genuinely M-series-optimized route beyond PyTorch MPS.
- **Piper — speed champ.** On M4/16GB: **~33× RTF**, 62 ms warm time-to-first-audio. Kokoro on the
  same M4 is **~13.8× RTF (MPS)** / 9× (CPU). Piper wins throughput but is lower naturalness.
- **Avoid XTTS-v2 on Mac:** `--device mps` **hangs indefinitely** (`aten::_fft_r2c` unimplemented on
  MPS); even with `PYTORCH_ENABLE_MPS_FALLBACK` it's **slower than CPU** (5.57 s vs 2.40 s). Coqui is
  effectively defunct, so no fix is coming. (Also cloning-only, non-commercial CPML.)

## 3. Best on AMD ROCm

- **Kokoro.** As a standard PyTorch model it runs via the official ROCm PyTorch wheel
  (`--index-url https://download.pytorch.org/whl/rocm6.x`). Real proof: **Kokoro-FastAPI** ships a
  ROCm Docker image, and its Issue #454 documents **ROCm 7.2 on AMD Strix Halo (gfx1151)** running
  Kokoro at **~10× over CPU**.
- **Caveat:** ROCm is **Linux-only** (Ubuntu 20.04+/RHEL 8+), official GPU coverage is limited
  (RX 7900-class+; others need `HSA_OVERRIDE_GFX_VERSION`), and AMD support is labeled experimental.
- **Chatterbox** also documents ROCm 6.1 → 7.2 (RDNA4 / Strix Halo), making it the only *high-quality
  challenger* that runs on both Mac and AMD — but those are claimed/documented, not independently
  benchmarked per config.

---

## 4. Comparison vs the Kokoro baseline

| Model | English quality | Long-form | Speed (RTF) | Size | Apple Silicon | AMD ROCm | Preset narrator? | License |
|---|---|---|---|---|---|---|---|---|
| **Kokoro-82M** *(baseline)* | Strong (4th open, Elo ~1062) | Good (in use by audiblez) | CUDA **~101×**; M4 MPS ~13.8×, CPU 9× | 82M (~0.7 GB) | ✅ native **MLX** + MPS | ✅ ROCm wheel (Linux) | ✅ **native** (`af_sky`, 54 voices) | **Apache-2.0** |
| **Chatterbox** (server) | Higher (challenger) | ⚠️ chunk-boundary artifacts | not benchmarked here | ~0.5B class | ✅ MPS (rough edges) | ✅ ROCm 6.1–7.2 | ⚠️ "predefined" = curated clips via cloning | MIT (server) |
| **Piper** | Lower fidelity | Stable | M4 **~33×** (champ) | tiny | ✅ | ✅ | ✅ presets | MIT |
| **XTTS-v2** | High | — | CPU-only on Mac | 0.5B+ | ❌ MPS hangs | partial | ❌ cloning-only | **CPML (non-commercial)** |
| **F5-TTS** | High | — | CUDA ~6.7× (3090); slow elsewhere | large | ❌ not clean on MPS | CUDA-centric | ❌ cloning-only (needs ref audio+text) | research |
| Fish S2 Pro / Step Audio EditX / Voxtral | **Top Elo** | unverified | unverified | unverified | unverified | unverified | **unverified** | unverified |

*RTF numbers from `tts-bench` (single-author repo, ~31★, but matches independent benchmarks) and
vendor/community sources; treat as approximate. Short-prompt warm RTFs scale ~linearly to long-form.*

---

## 5. Recommendation for audiblez

**Keep Kokoro-82M as the default.** It's the only candidate that is simultaneously: a true preset
English narrator (`af_sky`), Apache-licensed, the most efficient (82M / ~0.7 GB), already integrated
via `kokoro.KPipeline`, and **runs on all three targets** (CUDA, Apple Silicon via MLX/MPS, AMD ROCm).
Nothing verified here beats that *combination* for long-form narration.

**Highest-leverage improvement is hardware support, not a model swap.** audiblez today only has
CPU + CUDA paths (`cli.py` branches on `--cuda` → `torch.cuda`; the GUI only offers CPU/CUDA radios).
Two concrete additions:

1. **AMD ROCm — almost free.** PyTorch-ROCm exposes AMD GPUs through the **`torch.cuda` API**, so the
   existing `--cuda` flag *already works* on a ROCm PyTorch build — it's just undocumented. Action:
   document it (install the ROCm torch wheel + use `--cuda` on Linux) and consider auto-detecting.
2. **Apple Silicon (MPS) — a real gap.** Add a device path:
   `torch.backends.mps.is_available()` → `torch.set_default_device('mps')`, exposed as a CLI flag and
   a GUI radio button. (Watch for ops that fall back to CPU on MPS.) For a bigger speedup, an optional
   **MLX engine** via `mlx-audio` is the native-fast route on Macs.

**Optionally offer Chatterbox as a second, higher-quality engine** (opt-in), pending validation of its
long-form chunk-boundary behavior — not as the default, given the cloning-pathway and artifact caveats.

---

## Caveats & open questions

- **Live benchmark:** the Speech Arena re-runs ~4×/day; exact Elos/ordering drift. The 3rd/4th
  open-weights spot is volatile (one snapshot even put Voxtral *below* Kokoro). Treat as approximate.
- **No long-form benchmark:** the arena measures 500-char preference, not hours-long stability — the
  decisive axis for audiobooks. This is the biggest evidence gap.
- **Quality leaders unvalidated for fit:** Fish S2 Pro, Step Audio EditX, Voxtral top the Elo tier but
  their preset-narrator capability, long-form stability, Mac/ROCm support, and licenses were **not**
  verified. Worth a follow-up if you want to chase the top of the quality table.
- **Chatterbox "preset" ≠ native speaker:** it's curated reference clips through a cloning pathway;
  behaves preset-like but may affect long-form voice consistency.
- **Not assessed** (no surviving verified claims): Dia, Orpheus, Parler-TTS, StyleTTS2,
  Fish-Speech/OpenAudio, Sesame CSM, MeloTTS, Higgs Audio.
- **Refuted (excluded):** "F5-TTS runs on MPS with no setup" (0-3); a stale offlinetts.com leaderboard
  snapshot (0-3); "audiblez has no Apple Silicon support because no Kokoro MLX exists" (0-3 — MLX
  Kokoro *does* exist via `mlx-audio`).

## Key sources
- Artificial Analysis Speech Arena — https://artificialanalysis.ai/text-to-speech/leaderboard ·
  methodology: https://artificialanalysis.ai/text-to-speech/methodology
- Chatterbox-TTS-Server — https://github.com/devnen/Chatterbox-TTS-Server
- mlx-audio (Apple Silicon / MLX) — https://github.com/Blaizzy/mlx-audio
- Kokoro on AMD ROCm — https://github.com/pinguy/kokoro-tts-addon/blob/main/AMD_RADEON.md ·
  https://github.com/remsky/Kokoro-FastAPI (Issue #454: Strix Halo ROCm 7.2)
- RTF benchmarks — https://github.com/5uck1ess/tts-bench
- XTTS-v2 MPS hang — https://github.com/coqui-ai/TTS/issues/3649
- F5-TTS (cloning, RTF) — https://arxiv.org/abs/2410.06885 · https://github.com/SWivid/F5-TTS
