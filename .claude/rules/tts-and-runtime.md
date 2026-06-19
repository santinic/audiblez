# TTS model selection & the AMD runtime

**This is a personal project — ignore software/model licenses when choosing models.** Non-commercial
(CC-BY-NC), research-only, and gated models are all fair game. Weight only the real constraints:
(1) offline/local, (2) runs accelerated on the user's **AMD Radeon RX 7800 XT (gfx1101/RDNA3, 16 GB,
ROCm)**, (3) better naturalness than Kokoro, (4) reasonable speed. Don't impose a commercial-license
filter the user never asked for.

**Runtime reality (measured on-box):**
- **eager-PyTorch-on-ROCm is the bottleneck.** Quality models run ~9–125× slower than Kokoro and
  slower than real-time (StyleTTS2, CosyVoice2, VoxCPM2, Fish-Speech).
- **The one fast AMD path is llama.cpp Vulkan (RADV)** with GGUF models. **MOSS-TTS-8B
  (`MossTTSDelay`) is the chosen quality DEFAULT** — integrated as a resident pipe co-process; see
  `BUILD_ORDER.md` + `docs/adr/0001-0003`. Orpheus-3B also runs this way.
- **Kokoro-82M** is fast and top-tier on blind *leaderboards*, but flat in affect — and the user
  A/B'd by ear (2026-06-18) and judged it **sounds bad vs MOSS**, so it is demoted to the **fast
  fallback** (`docs/adr/0003`). No PyTorch-ROCm lever speeds it: fp16/bf16 are flat (bf16 breaks the
  audio), TunableOp flat, torch.compile flat/crashes on gfx1101. Don't chase them.
- Dead ends on this Linux box: ONNX-Runtime ROCm EP (removed), DirectML (Windows-only), sherpa-onnx
  (no AMD provider). `HSA_OVERRIDE_GFX_VERSION` is not needed here — leave it unset.

**Never install into the main `.venv`** (it holds a working `torch 2.12.1+rocm7.2`). Alternative TTS
models go in isolated venvs (`.venv-<model>`). Benchmark with `tools/tts_ab.py` / `tools/moss_bench.py`
and score intelligibility with `tools/score_wer.py` (the reference is `tools/ref_human.wav`, a REAL
human clip — keep it distinct from the Kokoro-minted `tools/ref_voice.wav` or the A/B biases toward
Kokoro). Full detail: `findings.md` and `TTS_MODELS_COMPARISON.md`.
