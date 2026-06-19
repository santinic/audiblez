"""Coarse-chunk packing for the MOSS opt-in fast path.

``pack_chunks`` is the paragraph-aware analogue of ``core.pack_sentences``:
instead of batching sentences for Kokoro's re-splitter, it groups consecutive
sentences into *utterance chunks* that will be sent to MOSS as one request.

Each chunk is capped at ``max_seconds`` of estimated audio.  The duration is
estimated from character count — the constant below is calibrated for English
prose at Kokoro's measured ~0.168 RTF (≈ 150 wpm, ~5 chars/word, 24 kHz):

    audio_seconds ≈ n_chars × EST_SECS_PER_CHAR

Measured on the Phase-0 spike sentences (2–5 s, 30–120 chars):
    n_chars ≈ audio_s / 0.040   →  EST_SECS_PER_CHAR ≈ 0.040

The ADR 0005 correctness-validated cap is 16.32 s; ``MAX_CHUNK_FRAMES`` records
the matching sample-count so the caller can cross-check when needed.

Design constraints (all must hold — they mirror core.pack_sentences):
- Pure function, no I/O.
- stdlib-only (no torch / kokoro / spaCy at module scope — must stay import-light
  so ``--doctor`` and any caller that imports this module cold are fast).
- Order is ALWAYS preserved: ``[s for c in chunks for s in c] == sentences``.
- A single sentence that exceeds ``max_seconds`` alone becomes its own (oversized)
  chunk rather than being dropped — the caller must handle it.
- An empty input returns an empty list.
"""

__all__ = ["pack_chunks", "MAX_CHUNK_FRAMES", "EST_SECS_PER_CHAR"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Correctness-validated audio cap from the Phase-0 spike (ADR 0005, gate 3,
#: WER 0.0).  Do NOT raise to 22 s without a WER check on-box first.
MAX_CHUNK_SECONDS: float = 16.32

#: Equivalent frame count at MOSS's 24 kHz sample rate.
MAX_CHUNK_FRAMES: int = int(MAX_CHUNK_SECONDS * 24_000)  # 391 680

#: Estimated seconds of audio per character of English prose text.
#: Derived from Phase-0 spike: mean sentence ~60 chars → ~2.4 s → 0.040 s/char.
#: Deliberately conservative (slightly high) so the cap is not breached in
#: practice; the real waveform may be slightly shorter.
EST_SECS_PER_CHAR: float = 0.040


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pack_chunks(
    sentences: list[str],
    max_seconds: float = MAX_CHUNK_SECONDS,
    est_seconds_per_char: float = EST_SECS_PER_CHAR,
) -> list[list[str]]:
    """Group consecutive sentences into utterance chunks capped at ``max_seconds``.

    Parameters
    ----------
    sentences:
        Ordered list of sentence strings (already passed through the lexicon).
    max_seconds:
        Maximum estimated audio duration per chunk.  Defaults to the ADR 0005
        correctness-validated cap of 16.32 s.
    est_seconds_per_char:
        Duration estimate per character.  Tune this if the prose style or
        language deviates significantly from English narration.

    Returns
    -------
    list[list[str]]
        Each inner list is one chunk (one MOSS request).  Flattening recovers
        ``sentences`` in the original order.

    Notes
    -----
    - A single sentence longer than ``max_seconds`` becomes its own oversized
      chunk (not dropped, not split at sub-sentence boundaries).
    - The estimate is deliberately conservative; real audio may be shorter.
    - This function is stdlib-only and import-light; keep it that way.
    """
    if not sentences:
        return []

    max_chars = max_seconds / est_seconds_per_char

    chunks: list[list[str]] = []
    current: list[str] = []
    current_chars: float = 0.0

    for sentence in sentences:
        s_chars = len(sentence)
        # Would adding this sentence exceed the cap AND the current chunk is
        # non-empty?  Flush first.
        if current and current_chars + s_chars > max_chars:
            chunks.append(current)
            current = []
            current_chars = 0.0
        current.append(sentence)
        current_chars += s_chars

    if current:
        chunks.append(current)

    return chunks
