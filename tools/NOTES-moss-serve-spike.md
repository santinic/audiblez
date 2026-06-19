# NOTES — Phase-0 MOSS resident `--serve` spike (verdict)

**Date:** 2026-06-18 · **Question:** can `llama-moss-tts` stay resident (load once) and synth
per-request correctly + fit 16 GiB VRAM? (gates from `docs/moss-coprocess-spec.md` + the handoff)
**Artifacts (THROWAWAY):** patch `moss_serve()` in
`/home/caio/Projects/llama.cpp-moss/tools/tts/run-moss-tts-delay.cpp` (`--serve` mode; duplicates the
gen+decode bodies so the one-shot path stays an untouched oracle); driver `tools/moss_serve_spike.py`;
raw JSON `ab_out/serve_spike/spike_results.json` (gitignored). Re-run:
`.venv/bin/python tools/moss_serve_spike.py`.

## Headline (honest)
**Resident co-process is FEASIBLE and unambiguously makes MOSS per-sentence VIABLE** — load is paid
once (2.0 s), so per-sentence drops from one-shot's unusable RTF > 1 (2.69 s reload *every* sentence)
to **~0.33** (≈3× faster, beats real-time). Correctness gates (determinism, KV isolation, variable-frame
decoder reuse, cloning, VRAM-fits) all **PASS**.
**BUT it does NOT reach Kokoro parity per-sentence.** Per-sentence RTF 0.30–0.44 is *above* the doc's
0.29 FAIL bar. The ~0.168 target is met only at **coarse (paragraph) granularity: RTF 0.206**. And the
~0.5 s/request floor that causes this is **in the backbone and is likely architecturally intrinsic**
(see below) — so it can't simply be optimized away. Per-sentence parity is therefore a **live design
trade**, not a settled green light.

## Measured per gate (MEASURED on-box, RX 7800 XT RADV, Q5_K_M backbone + f16 decoder)

| Gate | Question | Result | Verdict |
|---|---|---|---|
| 1 | Load once; marginal RTF → 0.168 | Load **2.04 s ONCE** (no per-request reload). Coarse-chunk (22 s passage) **RTF 0.206**, rock-steady. Per-sentence (2–5 s) **0.30–0.44**, median **0.33** — *above* the 0.29 bar. | ⚠ see trade below |
| 2 | KV-clear + RNG-reseed correct | A/B/A′ in one session: **wav(A) byte-identical to wav(A′)** (md5 `8cee7bf2…`); B differs. | ✅ PASS |
| 3 | Decoder ctx reuse, variable M | short 2.24 s / 53 760 samp **WER 0.0**; long 16.32 s / 391 680 samp **WER 0.0** through the same fixed non-causal decoder ctx. | ✅ PASS |
| 4a | Text-only VRAM ≤ 16 GiB | peak **13.14 GiB** (MOSS **10.84 GiB** over 2.30 GiB desktop idle). | ✅ PASS |
| 4b | Clone VRAM (3 co-resident) | peak **15.67 GiB** (MOSS **13.37 GiB**). Fits, but **~0.3 GiB headroom over desktop** → OOM-risk. | ✅ PASS ⚠ tight |
| 5 | Cloning resident | encoder kept loaded; `moss_clone_krupp.wav` (24 kHz mono) encoded once + reused; synth **WER 0.083**, ok. | ✅ PASS |

## Where the per-request cost lives (backbone/decoder split, measured via in-binary timers)
`wall ≈ 0.53 s + 0.181·audio_s`. Split:
- **Backbone ≈ 508 ms fixed + 14.2 ms/frame** (batch-1 autoregressive decode).
- **Decoder ≈ negligible** (42 ms short → 113 ms passage; scales cleanly, no fixed cost). Decoder ctx
  size is irrelevant: `--max-raw-frames` 256/768/2048 all give identical wall.

**The entire floor is the backbone, and it is almost certainly intrinsic:** 508 ms ≈ **n_vq(32) ×
14.2 ms = 454 ms**, i.e. the MossTTSDelay **de-delay flush** — the delay pattern must generate ~n_vq
extra frames to complete *any* utterance, a fixed tail independent of content length. The 14.2 ms/frame
marginal is batch-1 autoregressive generation; per `findings.md` §6/7 the torch-ROCm levers don't help
batch-1 (launch-bound), and this is llama.cpp-Vulkan anyway. **Conclusion: the ~0.5 s floor is not a
cheap implementation win to reclaim.**

## The live trade for Phase-2 (NOT a settled "PASS")
ADR 0002's motivation was **per-sentence parity** (keep audiblez's per-sentence cache + retry/dead-letter).
The spike shows you can't get both Kokoro-parity speed *and* per-sentence granularity from resident MOSS.
Three real options:

- **(a) Resident per-sentence @ ~0.33 RTF** — keeps the committed per-sentence cache/resilience/
  `sentence-level-edit-resynth` model unchanged; ~2× Kokoro but usable (beats real-time). Strictly 3×
  better than the one-shot fallback and adds load amortization the chapter-subprocess fallback lacks.
- **(b) Coarse-chunk (paragraph) @ ~0.206 RTF** — reaches near-target speed, but **re-opens the
  per-sentence premise**: coarsens the cache key, the retry/dead-letter granularity, AND the Phase-0
  edit-resynth work. This is the same coarse-cache downside as the documented chapter-subprocess
  fallback — resident just additionally amortizes the load.
- **(c) Attribute + kill the 508 ms floor** — the only path to per-sentence *parity*, but the evidence
  says it's the delay-pattern flush (intrinsic), so **likely not reachable**. Confirm before betting on it.

**Recommendation:** ship **(a)** as the default (preserves the committed architecture, clearly viable),
and offer **(b)** as an opt-in "fast/coarse" mode for users who want speed over edit-granularity. Don't
treat the chapter-subprocess fallback as needed — resident dominates it on every axis. For the clone
path, **load/free the encoder on demand** (don't keep it resident) given the thin 0.3 GiB headroom.

## Spike gate: resident co-process FEASIBLE + correct; per-sentence speed is a known trade, not a blocker.
Next: `/handoff` → Phase-2 `/to-prd` carrying the (a)/(b) decision.
