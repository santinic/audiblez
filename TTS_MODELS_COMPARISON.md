# TTS Models — Tested Comparison (audiblez)

**Date:** 2026-06-18
**Machine:** AMD Radeon RX 7800 XT (gfx1101 / RDNA3, 16 GB), ROCm 7.2, Linux (CachyOS), Python 3.12
**Goal:** find a TTS for audiblez (EPUB→audiobook) that beats Kokoro on naturalness while staying viable on AMD.

## Method

- All models synthesized the **same ~430-char book passage** via `tools/tts_ab.py`, each in its **own isolated venv** (the main `.venv` ROCm/Kokoro setup was never touched).
- **RTF** (real-time factor) = synth time ÷ audio duration; **lower is faster**, `<1` = faster than real-time. Median of 3–4 runs after a warm-up, timed with `torch.cuda.synchronize()` where applicable.
- Voice-cloning models cloned one shared **real human reference** (`tools/ref_human.wav`); Kokoro used its native `af_sky`, Orpheus its `tara` voice.
- **WER** = Whisper `base.en` round-trip word error rate (an **intelligibility** floor, *not* a naturalness ranking).
- Audio samples for listening: `ab_out/<model>.wav`.

## Measured results (real synthesis on the RX 7800 XT)

| Model | Params | RTF (↓) | Synth (s) | vs Kokoro | ~10 h book | VRAM | SR | WER | Runtime / AMD path |
|-------|--------|---------|-----------|-----------|------------|------|-----|-----|--------------------|
| **Kokoro-82M** *(baseline)* | 82 M | **0.155** | 3.1 | 1× | **~1.5 h** | 1.49 GB | 24 kHz | 0.016 | PyTorch-ROCm (eager) |
| **Orpheus-3B** | 3 B | **1.35** | 31.3 | ~8.7× slower | ~13.5 h | ~GPU¹ | 24 kHz | 0.016 | **llama.cpp-Vulkan (GPU offload ✓)** |
| **StyleTTS2** (LibriTTS) | 148 M | 1.40 | 38.3 | ~9.0× slower | ~14 h | 1.76 GB | 0.000 | PyTorch-ROCm (eager) |
| **CosyVoice2-0.5B** | 0.5 B | 2.40 | 66.6 | ~15.5× slower | ~24 h | 3.00 GB | 24 kHz | 0.000 | PyTorch-ROCm (eager) |
| **VoxCPM2** | 2 B | 4.87 | 144.0 | ~31.4× slower | ~49 h | 6.60 GB | 48 kHz | 0.000 | PyTorch-ROCm (eager) |
| **Fish-Speech / OpenAudio S1-mini** | 0.5 B | 19.4² | 386.3 | ~125× slower | ~194 h | 7.47 GB | 44.1 kHz | 0.032 | PyTorch-ROCm (eager) |
| **MOSS-TTS (MossTTSDelay 8B)** | 8 B | **0.29**³ | 6.4 | **~1.9× slower** | **~2.9 h** | 8.25 GB | 24 kHz | 0.000 | **llama.cpp-Vulkan (GPU offload ✓, coopmat)** |

¹ Orpheus runs via llama.cpp-Vulkan (not torch), so the torch-based VRAM probe reports `null`; GPU offload to "Vulkan0 (AMD Radeon RX 7800 XT)" was confirmed in the llama.cpp load log.

² s1-mini RTF is **decode-bound**, not generation-bound: the DualAR LLM generates on-GPU at ~9.9 tok/s (gen-only RTF ≈ **2.2**, Orpheus's band), but the DAC decode adds ~340 s because MIOpen can't allocate GEMM workspace on this path (`GemmFwdRest … provided ptr:0 size:0` → slow fallback) + experimental mem-efficient SDPA on RDNA3. **Also: the first attempt crashed the desktop session** (cold-kernel ROCm fault reset the display GPU); the retry succeeded on warm kernels. Session-fragile, not just slow. It voice-cloned a reference; WER 0.032 = fully intelligible. To run it: `git checkout 781bf1c -- fish_speech tools` in the fish-speech repo (the s1-era source; HEAD is too new for these weights).

³ MOSS RTF here is the **as-built / demonstrated** number: **0.29** = median 6.36 s wall for 21.84 s audio, **including** the per-call GGUF load. The first-class path is a one-shot CLI (`llama-moss-tts`) that reloads ~7.7 GB **every call** (load/init baseline 2.69 s), and audiblez's synth closure as designed is per-chunk — so 0.29 is what you actually get today. ⚠️ This is NOT apples-to-apples with Kokoro's 0.155, which IS resident (a live Python object). Honest as-built comparison ≈ **2× Kokoro** (both still faster than real-time). A **resident/server wrapper** (plausible but **unbuilt/unmeasured**) would drop MOSS to gen+decode RTF **0.168** (≈ Kokoro) → ~1.7 h/book. Measured via `tools/moss_bench.py`. Backbone **Q5_K_M** (6.06 GB); q5_K of the 32 multi-codebook audio output heads did **not** hurt (WER 0.000 == f16). **No GPU/desktop crash** — llama.cpp-Vulkan/RADV (`coopmat` matrix cores), not the eager-ROCm stack that crashed s1-mini. Setup: `findings.md` §18.

**All seven are perfectly intelligible** (WER ≈ 0–0.03). The differentiator is **naturalness (subjective) and speed**, not intelligibility.

## Optimization experiments (Kokoro speed — all flat)

| Experiment | RTF | vs baseline | Result |
|------------|-----|-------------|--------|
| Kokoro fp16 autocast | 0.86s* | ~1.01× | no gain; **bf16 broke audio** (sim 0.06) |
| Kokoro TunableOp (GEMM autotune) | 0.87s* | ~1.00× | no gain (rocBLAS default already fastest) |
| **Kokoro + torch.compile** | 0.152 | ~1.00× | **no gain** (data-dependent graph + LSTM don't fuse) |
| **StyleTTS2 + torch.compile** | 1.54 | ~0.9× (worse) | **no gain**; vocoder crashes under Inductor on ROCm |

\*per-sentence micro-benchmark earlier in the investigation. **Conclusion: no PyTorch-ROCm tuning lever (fp16/bf16/TunableOp/torch.compile) speeds these models on gfx1101** — the bottleneck is launch overhead + data-dependent control flow, not GEMM FLOPs.

## Per-model notes

| Model | License | Arch | Key caveats |
|-------|---------|------|-------------|
| Kokoro-82M | Apache-2.0 | StyleTTS2 (non-AR) | **#1-tier on blind naturalness** (TTS-Arena ELO ~1056) but **flat affect / no emotion**; already fast on CPU+GPU |
| Orpheus-3B | Apache-2.0 | Llama-3.2-3B + SNAC codec | **Expressive/emotional** (Kokoro's weakness); 8 voices; real AMD-Vulkan accel; slower than real-time |
| StyleTTS2 | MIT | Style-diffusion + iSTFTNet | More natural than Kokoro (its parent); install needs `weights_only` patch + NLTK `punkt_tab`; slow on ROCm |
| CosyVoice2-0.5B | Apache-2.0 | Qwen2.5-0.5B LLM + flow | High quality; hardest install (Matcha submodule, torchaudio→torchcodec patch); slow |
| VoxCPM2 | Apache-2.0 | 2 B diffusion-AR | SOTA-tier; 48 kHz (needs resample for audiblez); slowest + most VRAM; `optimize=False` required on ROCm |
| **MOSS-TTS (MossTTSDelay)** | Apache-2.0 | Qwen3-8B + 12.5 Hz RVQ codec | **Expressive** (duration/pronunciation control, zero-shot clone, 20 langs); first-class llama.cpp; Q5_K_M fits 8.25 GB; needs HF→GGUF convert (`findings.md` §18) |

## Verdict

- **MOSS-TTS (MossTTSDelay 8B) is the best quality-upgrade path found — and it's fast enough to matter.** As-built it renders ~10 h book in **~2.9 h** (RTF 0.29, faster than real-time, **~2× Kokoro** — NOT a tie; see ³). That's an **8 B expressive model** (duration/pronunciation control, zero-shot clone, 20 langs) on the one true AMD-first runtime (llama.cpp-Vulkan, coopmat), Apache-2.0, WER 0.000, **no crash** — at ~2× Kokoro's render time instead of Orpheus's ~9×.
- **Potential headroom:** a resident/server wrapper (unbuilt) would bring MOSS to ≈ Kokoro's speed (RTF ~0.168, ~1.7 h). Engineering, not a given.
- **Orpheus-3B** is now **dominated by MOSS** (~4–8× slower, smaller model). MOSS replaces it as the lead expressive option.
- **Only open question = subjective naturalness.** MOSS clears every objective bar Kokoro does (WER 0, faster-than-real-time) with a far bigger expressive model — so the decision is purely: **does MOSS *sound* better than Kokoro to you?** Compare `ab_out/moss.wav` vs `ab_out/kokoro.wav` by ear. (WER cannot rank naturalness.)

## Researched but NOT viable here (not run; see `findings.md` §8, §12)

| Model | Why excluded |
|-------|--------------|
| Chatterbox | Blind ELO ~1050 *below* Kokoro; PyTorch-ROCm (slow); output watermark |
| XTTS-v2 | CPML **non-commercial**; aging quality |
| F5-TTS | Better MOS but **weights CC-BY-NC** + AMD only via DirectML-on-Windows |
| Fish-Speech / OpenAudio | **Non-commercial** license |
| VibeVoice | MIT code but **"research-only" disclaimer**; full TTS code pulled by Microsoft; slow on AMD |
| Piper / Kitten | **Below** Kokoro on naturalness; no AMD-GPU path on Linux |
| sherpa-onnx / ONNX-Runtime | **No AMD GPU acceleration on Linux** (ROCm EP removed in onnxruntime 1.23; DirectML Windows-only) |

---
*Full methodology, install recipes, and analysis: `findings.md` §6–§15. Harness: `tools/tts_ab.py`, `tools/score_wer.py`.*
