# 0005 — MOSS ships per-sentence @ ~0.33 RTF; coarse-chunk is opt-in; the floor is intrinsic

Status: Accepted — 2026-06-18 · Refines ADR 0002 (its "parity" premise meant *granularity*, not *speed*).

## Context
ADR 0002 committed MOSS as a resident pipe co-process motivated by **per-sentence parity**
(keep audiblez's per-sentence cache + retry/dead-letter). The Phase-0 `--serve` spike
(`tools/NOTES-moss-serve-spike.md`) came back **feasible + correct**, but measured a cost
the parity premise hid: `wall ≈ 0.53 s + 0.181·audio_s`. The ~0.5 s/request floor is the
**MossTTSDelay de-delay flush** (≈ n_vq(32) × 14.2 ms) — a fixed tail to complete *any*
utterance, **architecturally intrinsic**, not a cheap implementation win. So per-sentence
MOSS lands at **RTF ~0.33** (above the doc's 0.29 Kokoro-parity bar); parity speed (~0.206)
is reachable **only** at coarse (paragraph) granularity. Parity-speed and per-sentence
granularity are mutually exclusive — the choice ADR 0002 didn't know it was making.

Aggregate RTF is governed entirely by mean sentence audio-duration `ā`:
`RTF_agg = 0.181 + 0.53/ā` → `ā=4s → 0.31`, `ā=2s` (heavy dialogue) `→ 0.45`. **Always beats
real-time**, worst realistic case ~0.45 (~2× today's Kokoro wall-clock; ~3.3 h synth for a
10 h book).

The code makes the choice nearly structural: with the sentence cache on (the committed
default), `synth_chapter` (core.py:662–684) calls `synth()` **one sentence at a time**;
batching exists only on the uncached path and relies on the engine re-splitting on `\n\n\n`
and returning one array per sentence. So the MOSS closure **must** issue one request per
sentence to honor the per-sentence cache + list contract — per-sentence granularity *forces*
per-request floor cost.

## Decision
- **(a) Resident per-sentence @ ~0.33 RTF is the accepted shipping default.** It preserves
  everything ADR 0002 committed (per-sentence cache, retry/dead-letter, `sentence-level-edit-
  resynth`) unchanged. Usable; beats real-time.
- **(b) Coarse-chunk mode is built in Phase-2 as an opt-in** "fast/coarse" path (not the
  default): a **paragraph synthesized as one utterance**, **capped at the spike's
  correctness-validated length, 16.32 s** (WER 0.0, gate 3) — `MAX_CHUNK_FRAMES`. The 22 s
  figure was **speed-only** (gate 1, no WER), so extending the cap there requires a WER check
  first. Oversize paragraphs split at sentence boundaries; resident backbone ctx sized to the
  cap. Caches per chunk (keyed by chunk text → never collides with the per-sentence cache).
  Failure unit is the whole chunk; an edit re-synths the whole chunk.
- **(c) Killing the 508 ms floor is deferred and de-prioritized** — the evidence says it is
  the intrinsic de-delay flush. **Do not re-run the spike to chase it** without new evidence.
- **Coarse-chunk + clone can COEXIST** (the earlier mutual-exclusion is DROPPED). Resolved by
  on-box measurement (T1-r3, see sub-decisions): **encode-once-then-free-encoder** holds
  **13.55 GiB** at synth time (vs encoder-resident 15.20 GiB tight) with **byte-identical**
  output, leaving comfortable headroom for a coarse-inflated context. No need to forbid the
  combination.

## Resolved sub-decisions (from the 2026-06-18 grill)
- **No automatic engine swap.** When the resident child won't spawn or the global
  circuit-breaker trips mid-book, the run **aborts loud** (non-zero, points at `-b cpu`),
  preserving partial chapters + dead-letters for resume. A silent fall-to-Kokoro would change
  the **voice mid-book** — a new silent-wrongness class. The chapter-subprocess fallback of
  ADR 0002 is dropped (resident dominates it; the spike passed, so it's moot).
- **Clone-VRAM lifecycle is RESOLVED (T1-r3, on-box, JSON `--serve`):** **encode the reference
  → codes once at startup, then free the encoder** is the **shipping clone default**. Measured
  peak **13.55 GiB** vs encoder-resident **15.20 GiB** (the spike's 15.67 "tight" confirmed) —
  a **1.65 GiB saving**, output **byte-identical** so it is not a quality trade. Low-risk: the
  encoder was never on the per-request path (`moss_encode_audio_llama` already loads+frees it
  transiently; synth reuses the cached codes). A `--clone-keep-encoder` flag retains the
  resident-encoder path as an escape hatch for any future **per-request / per-sentence
  reference voice** (Phase-2 is **one clone voice per book**, so default-free is safe now).
  The integration binary is branch `moss-serve-clone-encoder-free` (strict superset of
  `moss-serve-json-v1`: JSON protocol + the flag, default-free).
- **Cache stability: pin seed AND all six sampling params** (`text_temperature/top_k`,
  `audio_temperature/top_p/top_k/repetition_penalty`) to fixed defaults folded into the MOSS
  cache key — any drifted param silently breaks every cache hit.
- **GUI preview/audition uses MOSS in Phase-2** via **one resident child per GUI session**,
  reused across preview/audition/full-run (cold-start ~2.7 s paid once). Previewing a
  different engine than the render is the same silent-wrongness we reject above.

## Consequences
- The MOSS-default ships a ~2× Kokoro wall-clock; acceptable (beats real-time), but the PRD's
  acceptance criteria should include **one real full-book wall-clock check** (the spike
  measured 2–5 s sentences; `RTF_agg` is what a real book pays).
- (b) is genuine added scope — a separate non-splitting code path with its own per-chunk
  cache + whole-chunk failure gap — not a config flag. It earns its slot only as the explicit
  speed-over-edit-granularity escape hatch; revisit dropping it if it goes unused.
- **Confidence:** the (a)-default came in as the session's input; the other branches
  (coarse-mode kept in scope, abort-loud, GUI-warm-child) were **agent-recommended and
  confirmed without pushback** — treat them as **provisional** in `/to-prd`, not
  user-hardened. The **16.32 → 22 s cap extension** is now the one remaining **open,
  measure-on-box** item; the **clone-VRAM lifecycle is resolved** (above).
- See [Synthesis granularity] and [Coarse-chunk mode] in `CONTEXT.md`; the impl spec is
  `docs/moss-coprocess-spec.md`. VRAM is now **measured on-box** (T1-r3): text-only 13.14,
  clone encoder-free **13.55** (shipping default), clone keep-encoder 15.20 GiB (tight, escape
  hatch only).
