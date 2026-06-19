# 0004 — Phase-2 multi-speaker: exact (recorded-boundary) alignment + MOSS-default hybrid cast

Status: Accepted — 2026-06-18 (grounded by the `ground-open-branches` workflow; both load-bearing
claims adversarially confirmed, alignment with an empirical booknlp run)

## Context
- ADR 0001 demoted multi-speaker to phase-2, riding MOSS cloning. Two things were unresolved: the
  **chapter-alignment** problem (the doc called it "unsolved") and the **cast-map shape**.
- **Alignment:** booknlp emits a flat whole-book speaker script (no chapter boundaries); audiblez
  synthesizes per chapter. The fear was that mapping back is heuristic/fragile.
- Grounding found `.tokens` carries per-token **character** offsets and joins `.book.txt` 1:1 by
  `sentence_id` — so the mapping is **exact**, *if* audiblez owns booknlp input construction.

## Decision
- **Alignment = strategy (d), recorded-boundary bucketing.** audiblez concatenates selected chapters'
  pre-lexicon text, records each chapter's char-range, feeds that to booknlp, and buckets each booknlp
  sentence into its chapter by char-offset. **Zero fuzzy matching.** Reject fuzzy-match, per-chapter
  booknlp (loses cross-chapter character identity), and user-sidecar.
- **Cast = `book.cast.json` discriminated-union sidecar, MOSS-default / Kokoro-ids-optional.** Default
  per character is a MOSS clone; a Kokoro voice id is an explicit override. `engine` is the explicit
  discriminator. Hybrid is effectively free (MOSS-default + Kokoro-fallback already multiplexes engines
  per sentence). v1 is CLI-only.
- Full impl detail (seam threading, the two alignment constraints, cache/`.sig` changes):
  [`docs/phase2-multispeaker-spec.md`](../phase2-multispeaker-spec.md).

## Consequences
- Phase-2 re-rates from "XL with an unsolved blocker" to "XL, alignment solved" — but it gains real
  plumbing: carry spaCy `start_char` through `_split_into_sentences`/`split_long_sentence` (layer-2),
  `synth(text,speed,voice_spec)` threaded through the batch seam, `pack_sentences` speaker-aware,
  per-sentence cast into both `make_key` (no `CACHE_VERSION` bump — per-sentence voice/engine is
  already in the key, so a re-cast misses naturally; same logic as engine separation) and
  `_render_signature`/`.sig`.
- **Two silent-wrongness landmines** to test, not assume: char-vs-byte offsets (drift proven at the
  first smart-quote), and per-sentence voice must reach `.sig` (else re-cast serves stale chapter wav).
- **The real residual risk is attribution** (span → character), which is upstream of this design — the
  cast/alignment are correct but inert if attribution is wrong. English-only (booknlp).
