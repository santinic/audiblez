# CONTEXT — audiblez domain glossary

Ubiquitous language for audiblez. Glossary only — no implementation details, no
roadmap. When a term here conflicts with how code or a spec uses a word, this file wins
(or gets updated deliberately). Born during the 2026-06-18 digest of
`CROSS_PROJECT_ANALYSIS.md`.

---

## Engine
The synthesis technology that turns text into audio: **Kokoro** (PyTorch / MLX) and
**MOSS** (llama.cpp-Vulkan GGUF). One engine implements one **synth closure** contract.
Not the same as a **Backend**.

## Backend
A runtime *placement/config* of an engine: `cpu`, `cuda`, `rocm`, `mps`, `mlx`, and the
new `moss`. In code, `BackendInfo.id` is the backend; `BackendInfo.engine` is the engine.
The two words are blurred in upstream prose — keep them distinct here: many backends can
share one engine (cpu/cuda/rocm all run the Kokoro torch engine).

## Synth closure (the synth seam)
The single function every engine must satisfy: `synth(text, speed) -> list[np.ndarray]`
at 24 kHz, returned by `core.build_synthesizer`. Adding an engine = one dispatch branch +
one closure. The seam every feature and every entry point ultimately routes through.

## Entry points (the three)
`cli.py`, `ui.py` (the wxPython `CoreThread`), and `core.py` itself — the three callers
that reach synthesis. A change to a `core` signature must update all three. The GUI is the
one historically missed; this recurring miss is its own named regression class.

## Render signature (`.sig`)
The fingerprint that decides whether an existing chapter render can be safely *reused* on
the resume-by-skip path, instead of re-synthesized. Must capture everything that changes a
chapter's audio. (Known gap as of this digest: it omits the chapter text itself.)

## Sentence cache
The content-addressed, per-sentence audio store. Its key must capture **everything that
changes the waveform** (engine, model identity, voice, speed, precision, transformed
text, …). A failure or empty result is **never** cached.

## Degraded run / silent-wrongness
The core correctness stance: a degraded result must stay **visible**. A failed sentence
retries, then splices silence, records a dead-letter, and the run exits non-zero — never
swallowed into a reported success. "Silent-wrongness" = any change that produces wrong
audio with no crash and no warning (stale reuse, misaligned titles, uncached knobs). The
class of bug this project most guards against.

## Cast map
A character → voice assignment, used by multi-speaker casting (phase-2). Built
deterministically from character attributes, with manual override and single-narrator
fallback.

## Speaker script
The flat, ordered `[Character] "quote" [/]` dialogue text that booknlp emits (`.book.txt`).
audiblez *ingests* it; booknlp itself stays an external, English-only preprocessor and
never becomes an audiblez dependency.

## Voice cloning / clone reference
Reproducing a target voice from a short reference WAV (MOSS zero-shot). The **clone
reference** is that WAV. Cloning is the lever that lifts multi-speaker from coarse
gender/age buckets to genuinely distinct per-character voices.

## Resident engine co-process
A long-lived child process, owned by the audiblez process and fed over a pipe, that loads
a heavy engine **once** and synthesizes sentence-by-sentence **without reloading**. The
chosen integration shape for MOSS: it keeps per-sentence cache + resilience parity while
amortizing the one-time model load. Distinct from a network/HTTP daemon — no socket, no
port, dies with its parent.

## Attribution
Deciding *which character speaks* a given span of text. The upstream input to multi-speaker
casting (supplied by booknlp). Distinct from **casting**, which maps an already-attributed
character to a voice. Attribution quality — not the cast map — is the real product risk of
multi-speaker: a perfect cast map is inert if attribution is wrong.

## Synthesis granularity
The size of the text span treated as **one utterance**, which is simultaneously the unit
of the **sentence cache**, the retry/dead-letter gap, and edit-resynth. **Per-sentence**
(the default — what the cache + resilience model assume) vs **coarse-chunk** (a whole
paragraph). Not interchangeable: coarsening granularity coarsens the cache unit, the
failure gap, and the edit unit together. The axis the MOSS speed-vs-edit-granularity trade
turns on (per ADR 0005).

## Coarse-chunk mode
The opt-in path that treats a **paragraph as one utterance**, trading per-sentence
granularity for speed. Distinct from the default per-sentence path. Its failure unit is the
whole chunk — one bad sentence silences the paragraph (no per-sentence fallback inside a
single utterance) — and an edit re-synths the whole chunk.

## Recorded-boundary alignment
Mapping a flat, chapter-less whole-book speaker script back onto per-chapter synthesis **exactly**
(not by fuzzy matching) — by having audiblez itself build the preprocessor's input and *record*
each chapter's character-offset range, then bucketing each emitted sentence into its chapter by
offset. The technique that makes phase-2 multi-speaker preserve per-chapter `.wav`/`.sig`/markers.
