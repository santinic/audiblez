# Cache & resilience invariants

These are correctness landmines: get them wrong and the audiobook is *silently* wrong (no crash).

- **The synth cache key must capture EVERYTHING that changes the waveform.** `cache.make_key` /
  `core._cache_key_fields` hash: engine, model `repo_id`, voice, speed, precision,
  `MAX_SENTENCE_LENGTH`, spaCy version, and the (already lexicon-transformed) sentence text. If you
  add any knob that alters audio, add it to the key (or bump `CACHE_VERSION`). Miss one → a re-run
  serves stale/wrong audio.
- **Never cache a failure or an empty result** (`np.zeros(0)`). A cached "no audio" would be served
  as a silent hit on every future run.
- **A degraded run must be visible.** A failed sentence retries (deterministic errors skip retry),
  splices silence, and is recorded to `<chapter>.failed.jsonl`; `core.main()` returns the failure
  count and the CLI exits non-zero. Don't swallow failures into a reported success.
- **Chapter resume:** an existing chapter `.wav` is reused only if it's valid AND its sibling `.sig`
  (lexicon fingerprint + speed + precision) matches AND it has no non-empty `.failed.jsonl`. Keep
  the `.sig` write in sync with whatever affects the audio.
- **Chapter markers:** measure durations with `_robust_duration` (ffprobe → soundfile header
  fallback), never `probe_duration(...) or 0.0` — that zeroes every marker when ffprobe is absent.
- **Untrusted input:** epub-derived chapter names/titles flow into ffmpeg (the concat list) and into
  filenames. Sanitize control characters; keep concat-list paths absolute and quote-escaped.
