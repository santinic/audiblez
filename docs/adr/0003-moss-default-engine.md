# 0003 — MOSS is the quality default engine; Kokoro is the fast fallback

Status: Accepted — 2026-06-18

## Context
- findings.md / TTS_MODELS_COMPARISON.md concluded Kokoro is **#1-tier on BLIND naturalness**,
  but **explicitly deferred the subjective verdict to the user's ear**
  (TTS_MODELS_COMPARISON.md:62, findings.md:410 — "the axis only the user can judge").
- The user A/B'd `ab_out/moss.wav` vs `ab_out/kokoro.wav` and judged: **Kokoro sounds bad,
  MOSS sounds good.**
- This contradicts the blind-leaderboard claim. For a personal project, the user's ear is the
  named tiebreaker — so this resolves the open question rather than discarding rigorous data.

## Decision
- **MOSS = the quality default** engine; **Kokoro = the fast / zero-setup fallback.**
- Implement by extending the existing *auto-select best working backend*: prefer MOSS when its
  binary + models + resident pipe are present and working; else fall back to Kokoro. `-b`
  forces a specific engine.
- The MOSS spike's quality bar therefore reduces to **technical** (resident loop + cloning
  works) — plain-narration quality is already settled by the listening test.
- findings.md's "Kokoro #1-tier naturalness" conclusion is marked **OVERRULED for this
  project** (banner added at findings.md §"The hard truth"). The AMD-runtime conclusions there
  still stand.

## Consequences
- Until MOSS lands, the default keeps shipping the engine the user finds bad → **reinforces
  engine-first urgency** (does not reorder the plan).
- MOSS is slower (as-built RTF 0.29 ≈ 2× Kokoro's 0.155; only ~parity if the resident loop
  reaches the 0.168 gen+decode rate) and heavier (co-resident VRAM est. ~9.4–11.5GB, fork patch, ~8GB
  model download) than Kokoro; the default now pays that whenever MOSS is available.
- The `tts-and-runtime` project rule ("Kokoro… top-tier on blind naturalness") is now
  partially stale — flagged for update.
