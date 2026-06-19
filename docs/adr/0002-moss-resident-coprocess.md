# 0002 — MOSS integrates as a resident pipe co-process (no-daemon overruled, scoped to a pipe)

Status: Accepted — 2026-06-18

## Context
- The brief locked: offline, single-process, **"no web / Gradio / daemon rewrite."**
- MOSS's fork binary `llama-moss-tts` is **one-shot**: load GGUF → synth one text →
  `--wav-out` → exit. GGUFs on disk at `/home/caio/Projects/moss-work/gguf/`: backbone Q5
  6.06GB + decoder 1.78GB (+ encoder 1.78GB for cloning).
- Measured (findings.md:736-746): **as-built RTF 0.29 INCLUDES the per-call GGUF reload**
  (~7.7GB, load/init 2.69s) — the one-shot CLI reloads every call. The **gen+decode-only RTF is
  0.168** (≈ Kokoro), obtained by *subtracting* that tiny-text load baseline — but reaching it
  requires keeping the model **resident** (load paid once, not per call). The cost to eliminate
  is the per-subprocess reload, not the marginal synth rate.
- Per-sentence subprocess invocation pays that load thousands of times per novel → unusable.
- The user requires **per-sentence parity**: keep audiblez's per-sentence cache + retry /
  dead-letter for MOSS too.

## Decision
- Keep the model **resident**: patch the fork's `llama-moss-tts` to loop without reloading
  (read a sentence on stdin → emit a wav, model stays in VRAM).
- audiblez spawns **one child per run**, owned over a stdin/stdout **pipe** — NOT a network
  /HTTP server.
- The user **explicitly overruled** the no-daemon constraint. The pipe shape is chosen because
  it honors the constraint's *intent* (no socket, no port, no security surface, dies with its
  parent) while delivering load-once parity. An HTTP/localhost server was rejected as the
  forbidden shape with the most machinery.
- **Gate on a C++ feasibility spike**: confirm the resident loop works with no per-sentence
  reload (and that cloning works) before committing the full XL.

## Consequences
- Requires a **C++ patch to the OpenMOSS llama.cpp fork** — real, unproven; the spike de-risks
  it. **Fallback if the spike fails:** chapter-granularity subprocess (chapter-level
  cache/resilience only).
- **VRAM:** lazy load on first sentence, then resident for the child's life. Co-resident estimate
  **~9.4GB text-only / ~11.5GB clone** (summed-from-logs, **NOT measured** — the bench's 8.25GB was a
  *sequential single-model* peak; the current one-shot destructs the backbone before the decoder loads,
  so all-three-co-resident is new, unexercised behavior). Fits 16GB on paper; measure on-box. Needs an
  idle-timeout / explicit stop to release VRAM between runs.
- Child **lifecycle** (spawn, detect-death/restart, guaranteed kill on parent exit so VRAM
  isn't leaked) becomes audiblez's responsibility.
- See [resident engine co-process](../../CONTEXT.md) in the glossary.

## Resolved sub-decisions (grounding 2026-06-18 — both feasibility claims adversarially confirmed)
- **Spike confirmed FEASIBLE** with exact anchors: the fork's 3 model loads are hoistable; the
  vocab-only load is *eliminable*; per-request needs only `llama_memory_clear` + RNG reseed (no
  reload); no hidden per-call state forces a reload.
- **Audio transport = 16-bit WAV file-handoff** (not float32/stdout — no audible gain after the m4b
  AAC pass; reuses `save_wav16` untouched). Control = newline-JSON on stdout with synth-path `LOG()`
  redirected to stderr.
- **Speed = ffmpeg `atempo` at chapter assembly** (MOSS has no speed knob). MOSS sentence cache stays
  speed-agnostic; speed enters the chapter `.sig`.
- **Failure model** lives inside the synth closure: error-line → dead-letter+silence (child lives);
  EOF/poll/timeout → restart+re-dispatch; **K=2** consecutive deaths per sentence → dead-letter (as
  `_PERMANENT_SYNTH_ERRORS`); global circuit-breaker aborts the run on excessive restart rate.
- **Cache:** MOSS key adds seed + sampling params; `engine` already separates MOSS from Kokoro so **no
  `CACHE_VERSION` bump for engine separation**; `repo_id` hashes all 3 GGUF identities.
- Full spec: [`docs/moss-coprocess-spec.md`](../moss-coprocess-spec.md).
