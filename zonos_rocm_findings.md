# Zonos-v0.1 AMD/ROCm runnability verification

## Verdict: CONFIRMED (with material nuance / update)

### Confirmed from primary sources
- README (HF Zonos-v0.1-hybrid + GitHub): OS = Linux (Ubuntu 22.04/24.04)/macOS; GPU = "6GB+ VRAM, Hybrid additionally requires a 3000-series or newer Nvidia GPU". NVIDIA-only framing; no AMD/ROCm mention.
- pyproject.toml: flash-attn>=2.7.3, mamba-ssm>=2.2.4, causal-conv1d>=1.5.0.post8 are ONLY in the optional [compile] extra. "technically optional, but mamba-ssm is required to run hybrid models." => transformer variant does NOT need them.
- Issue #81 (18 comments): multiple users (brcisna RX Pro W6600, billbeans/RX6600, esanscoopsers RX6700XT) failed for hours on ROCm 6.2 to compile flash-attn/mamba-ssm/causal-conv1d. Concrete errors: HIP version-parsing ("nvcc not found", torch 2.6.0+cu124 build), symbol collision ("multiple definition of __low2float(__hip_bfloat162)"). Confirms dossier verbatim.

### UPDATE / mitigant strengthened (dossier said transformer "may run, unsupported, untested")
- YellowRoseCx confirmed (2025-02-24): "the Transformer backend works as is" on RX 6800XT. mamba-ssm + causal-conv1d got working; flash-attn not really needed (only rotary files).
- Working community fork: https://github.com/YellowRoseCx/Zonos-ROCm with full install guide. Requires ROCm 6.2+. Hybrid needs RDNA2 (RX6000)+; flash-attn works on RX 7000 series+ (older AMD need HipBLASLt-disable env vars).
- KEY for audiblez target (RX 7800 XT = RDNA3/gfx1100): this is the BEST-supported AMD tier per fork (flash-attn works on RX7000+). So even hybrid is plausible via the fork; transformer runs without custom kernels.

### Net for audiblez fitness
- Upstream Zonos = NOT AMD-supported (NVIDIA-only README/deps).
- Transformer variant runs on ROCm in eager PyTorch without the CUDA-custom kernels (DEMONSTRATED, not merely theoretical).
- A maintained community ROCm fork exists and explicitly targets RX 7000 series as best case.
- Caveat: requires the fork / manual build; not single-pip-install; not officially supported by Zyphra.
