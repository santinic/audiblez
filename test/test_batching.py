"""Tests for sentence batching (core.pack_sentences + gen_audio_segments)
and coarse-chunk packing (chunking.pack_chunks).

Guarded imports so tests skip without the heavy stack. The synth and spaCy
pipeline are faked, so no model/audio is needed: a "segment" is just the chunk
string, which lets us assert ordering and call count.
"""
import unittest
from unittest import mock
from types import SimpleNamespace

try:
    import audiblez.core as core
    _ERR = None
except Exception as e:
    _ERR = e

try:
    from audiblez.chunking import pack_chunks, MAX_CHUNK_SECONDS, EST_SECS_PER_CHAR
    _CHUNKING_ERR = None
except Exception as e:
    _CHUNKING_ERR = e


@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class PackSentencesTest(unittest.TestCase):
    def test_packs_under_cap_into_fewer_batches(self):
        sents = ['a' * 100] * 10  # 1000 chars total
        batches = core.pack_sentences(sents, batch_max_chars=300)
        self.assertTrue(all(sum(len(s) for s in b) <= 300 for b in batches))
        self.assertLess(len(batches), len(sents))  # actually batched
        self.assertEqual([s for b in batches for s in b], sents)  # order preserved

    def test_oversized_sentence_is_its_own_batch(self):
        sents = ['short', 'x' * 5000, 'also short']
        batches = core.pack_sentences(sents, batch_max_chars=1000)
        self.assertIn(['x' * 5000], batches)
        self.assertEqual([s for b in batches for s in b], sents)

    def test_tiny_cap_is_one_sentence_per_batch(self):
        sents = ['one', 'two', 'three']
        batches = core.pack_sentences(sents, batch_max_chars=1)
        self.assertEqual(batches, [['one'], ['two'], ['three']])

    def test_empty(self):
        self.assertEqual(core.pack_sentences([], 1000), [])


@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class GenAudioSegmentsBatchingTest(unittest.TestCase):
    def _fake_nlp(self):
        # Sentences are pipe-separated in the input text.
        return lambda text: SimpleNamespace(sents=[SimpleNamespace(text=s) for s in text.split('|')])

    def test_batched_call_preserves_order_and_count(self):
        # fake synth: Kokoro re-splits on \n\n\n -> one "segment" (the chunk str) each.
        synth = mock.MagicMock(side_effect=lambda t, sp: t.split('\n\n\n'))
        with mock.patch.object(core, 'load_spacy', return_value=self._fake_nlp()):
            out = core.gen_audio_segments(synth, 'one|two|three', voice='af_sky', speed=1.0,
                                          batch_max_chars=1000)
        self.assertEqual(out, ['one', 'two', 'three'])   # order preserved
        self.assertEqual(synth.call_count, 1)            # one batched call

    def test_tiny_cap_falls_back_to_per_sentence_calls(self):
        synth = mock.MagicMock(side_effect=lambda t, sp: t.split('\n\n\n'))
        with mock.patch.object(core, 'load_spacy', return_value=self._fake_nlp()):
            out = core.gen_audio_segments(synth, 'one|two|three', voice='af_sky', speed=1.0,
                                          batch_max_chars=1)
        self.assertEqual(out, ['one', 'two', 'three'])
        self.assertEqual(synth.call_count, 3)            # per-sentence

    def test_max_sentences_caps_output(self):
        synth = mock.MagicMock(side_effect=lambda t, sp: t.split('\n\n\n'))
        with mock.patch.object(core, 'load_spacy', return_value=self._fake_nlp()):
            out = core.gen_audio_segments(synth, 'one|two|three|four', voice='af_sky', speed=1.0,
                                          max_sentences=2, batch_max_chars=1000)
        self.assertEqual(out, ['one', 'two'])


@unittest.skipIf(_CHUNKING_ERR is not None, f"audiblez.chunking unavailable: {_CHUNKING_ERR}")
class PackChunksTest(unittest.TestCase):
    """Tests for pack_chunks — the coarse-chunk packing function.

    Pack_chunks groups consecutive sentences into utterance chunks capped at
    MAX_CHUNK_SECONDS (16.32 s, ADR 0005 correctness-validated cap).  Duration
    is estimated via ``len(sentence) * EST_SECS_PER_CHAR``.
    """

    def test_cap_respected_for_multi_sentence_chunks(self):
        """No multi-sentence chunk may exceed the estimated audio cap."""
        sents = ["x " * 30] * 20   # ~1.2 s each; many fit under 16.32 s
        chunks = pack_chunks(sents, max_seconds=MAX_CHUNK_SECONDS)
        for chunk in chunks:
            if len(chunk) > 1:
                est = sum(len(s) for s in chunk) * EST_SECS_PER_CHAR
                self.assertLessEqual(
                    est, MAX_CHUNK_SECONDS + 1e-9,
                    f"multi-sentence chunk exceeded cap: {est:.2f}s",
                )

    def test_oversize_single_sentence_is_own_chunk_not_dropped(self):
        """A sentence that alone exceeds the cap becomes its own (oversized)
        chunk — it must NOT be dropped, truncated, or split mid-word."""
        giant = "w " * 1000   # ~80 s of estimated audio
        sents = ["Before.", giant, "After."]
        chunks = pack_chunks(sents, max_seconds=MAX_CHUNK_SECONDS)
        flat = [s for c in chunks for s in c]
        self.assertEqual(flat, sents)
        # The giant sentence must appear alone in its own chunk
        giant_chunks = [c for c in chunks if giant in c]
        self.assertEqual(len(giant_chunks), 1)
        self.assertEqual(giant_chunks[0], [giant])

    def test_order_preserved(self):
        """Flattening chunks always recovers the original sentence list."""
        import random
        random.seed(0)
        sents = [f"Sentence {i}." for i in range(30)]
        chunks = pack_chunks(sents, max_seconds=2.0)
        flat = [s for c in chunks for s in c]
        self.assertEqual(flat, sents)

    def test_tiny_cap_one_sentence_per_chunk(self):
        """A cap so small that no two sentences can share a chunk."""
        sents = ["one", "two", "three"]
        chunks = pack_chunks(sents, max_seconds=0.001)
        self.assertEqual(chunks, [["one"], ["two"], ["three"]])

    def test_short_sentences_merged_into_fewer_chunks(self):
        """Short sentences must be grouped; result has fewer chunks than sentences."""
        sents = ["Hi."] * 10   # ~0.12 s each → well under 16.32 s combined
        chunks = pack_chunks(sents, max_seconds=MAX_CHUNK_SECONDS)
        self.assertLess(len(chunks), len(sents))

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(pack_chunks([]), [])

    def test_single_sentence_returns_one_chunk(self):
        self.assertEqual(pack_chunks(["Only one."]), [["Only one."]])

    def test_all_sentences_fit_in_one_chunk(self):
        """All short sentences that fit under the cap are returned as one chunk."""
        sents = ["a", "b", "c"]
        chunks = pack_chunks(sents, max_seconds=MAX_CHUNK_SECONDS)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], sents)

    def test_custom_cap_honoured(self):
        """A custom max_seconds overrides the default."""
        # Two 30-char sentences; at 0.04 s/char each is 1.2 s.
        # With a 3.0 s cap (> 2.4 s combined) they fit in one chunk.
        sents = ["A" * 30, "B" * 30]
        chunks_combined = pack_chunks(sents, max_seconds=3.0)
        self.assertEqual(len(chunks_combined), 1)
        # With a 1.0 s cap they don't fit together (1.2 s each > 1.0 s each).
        chunks_split = pack_chunks(sents, max_seconds=1.0)
        self.assertEqual(len(chunks_split), 2)

    def test_boundary_sentence_not_double_counted(self):
        """A sentence that exactly fills the remaining cap starts a new chunk."""
        # Each sentence is 100 chars → 4.0 s; cap = 8.0 s → 2 per chunk
        sents = ["x" * 100] * 6
        chunks = pack_chunks(sents, max_seconds=8.0)
        # Each chunk holds exactly 2 sentences (2×4.0 s = 8.0 s ≤ cap)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 2)
        flat = [s for c in chunks for s in c]
        self.assertEqual(flat, sents)


@unittest.skipIf(_CHUNKING_ERR is not None, f"audiblez.chunking unavailable: {_CHUNKING_ERR}")
class PackChunksPropertyTest(unittest.TestCase):
    """Randomised property invariants for pack_chunks.

    Uses a seeded ``random.Random`` (not the global ``random`` module) so the
    test is deterministic — same seed gives same inputs every run.
    """

    RNG = __import__("random").Random(1234)

    def _random_sentences(self, n: int, min_len: int = 3, max_len: int = 200) -> list[str]:
        """Generate n random-length strings (deterministic via self.RNG)."""
        return [
            "x" * self.RNG.randint(min_len, max_len)
            for _ in range(n)
        ]

    def _check_invariants(self, sents: list[str], max_seconds: float) -> list[list[str]]:
        """Run pack_chunks and assert all three invariants; return chunks."""
        chunks = pack_chunks(sents, max_seconds=max_seconds)

        # --- invariant 1: no text dropped or duplicated (flatten == input) ---
        flat = [s for c in chunks for s in c]
        self.assertEqual(flat, sents,
                         "flatten(chunks) must equal the original sentence list exactly")

        # --- invariant 2: order preserved (already implied by flat==sents, belt+suspenders) ---
        self.assertEqual(flat, sents, "order must be preserved")

        # --- invariant 3: cap never exceeded for multi-sentence chunks ---
        for chunk in chunks:
            est = sum(len(s) for s in chunk) * EST_SECS_PER_CHAR
            if len(chunk) > 1:
                self.assertLessEqual(
                    est, max_seconds + 1e-9,
                    f"multi-sentence chunk estimated {est:.3f}s exceeds cap {max_seconds}s. "
                    f"chunk sizes={[len(s) for s in chunk]}",
                )

        return chunks

    def test_property_short_sentences_default_cap(self):
        """50 short sentences under the default 16.32 s cap."""
        sents = self._random_sentences(50, min_len=3, max_len=30)
        self._check_invariants(sents, MAX_CHUNK_SECONDS)

    def test_property_medium_sentences_default_cap(self):
        """30 medium sentences (up to 200 chars) under the default cap."""
        sents = self._random_sentences(30, min_len=50, max_len=200)
        self._check_invariants(sents, MAX_CHUNK_SECONDS)

    def test_property_mixed_sizes_tight_cap(self):
        """Mixed sentence lengths under a tight 2.0 s cap forces many single-sentence chunks."""
        sents = self._random_sentences(40, min_len=1, max_len=300)
        self._check_invariants(sents, 2.0)

    def test_property_single_giant_sentence_never_dropped(self):
        """A single sentence much larger than the cap must appear exactly once in output."""
        giant = "w " * 2000    # >> 16.32 s
        sents = self._random_sentences(10, min_len=5, max_len=50) + [giant]
        self.RNG.shuffle(sents)
        chunks = self._check_invariants(sents, MAX_CHUNK_SECONDS)
        giant_appearances = sum(1 for c in chunks for s in c if s == giant)
        self.assertEqual(giant_appearances, 1, "giant sentence must appear exactly once")

    def test_property_many_sentences_varied_caps(self):
        """100 sentences under three different caps — invariants must hold for each."""
        sents = self._random_sentences(100, min_len=1, max_len=150)
        for cap in (1.0, 5.0, MAX_CHUNK_SECONDS):
            with self.subTest(cap=cap):
                self._check_invariants(sents, cap)

    def test_property_all_same_length_exact_boundary(self):
        """Sentences whose combined estimate lands exactly on the cap boundary."""
        # 25 chars each → 1.0 s each; cap = 3.0 s → 3 per chunk (3.0 s ≤ cap)
        sents = ["A" * 25] * 30
        chunks = self._check_invariants(sents, 3.0)
        # Each chunk must have at most 3 sentences
        for chunk in chunks:
            est = sum(len(s) for s in chunk) * EST_SECS_PER_CHAR
            if len(chunk) > 1:
                self.assertLessEqual(est, 3.0 + 1e-9)

    def test_property_empty_input_always_empty(self):
        """Empty input always gives empty output regardless of cap."""
        for cap in (0.001, 1.0, MAX_CHUNK_SECONDS, 1000.0):
            with self.subTest(cap=cap):
                self.assertEqual(pack_chunks([], max_seconds=cap), [])


@unittest.skipIf(_ERR is not None or _CHUNKING_ERR is not None,
                 "audiblez.core/chunking unavailable")
class GenAudioSegmentsCoarseTest(unittest.TestCase):
    """The coarse-chunk synth path in gen_audio_segments (issue #2 stories 22-25 / ADR 0005).

    Coarse mode sends each packed chunk to the engine as ONE request (vs per-sentence), caches /
    fails / edits at chunk granularity. The synth + spaCy are faked, so no model/audio is needed.
    """
    import numpy as np  # core is importable here (guarded), so numpy is present

    def _fake_nlp(self):
        return lambda text: SimpleNamespace(sents=[SimpleNamespace(text=s) for s in text.split('|')])

    def _recording_synth(self, calls):
        # One MOSS request per call; returns a 1-frame float32 array so np.concatenate works.
        def synth(text, sp):
            calls.append(text)
            return [self.np.zeros(1, dtype=self.np.float32)]
        return synth

    def test_short_sentences_become_one_chunk_one_request(self):
        # All under the cap -> a single chunk -> ONE request, joined (NOT \n\n\n-batched).
        calls = []
        with mock.patch.object(core, 'load_spacy', return_value=self._fake_nlp()):
            out = core.gen_audio_segments(self._recording_synth(calls), 'one|two|three',
                                          voice='af_sky', speed=1.0, coarse=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], 'one two three')
        self.assertNotIn('\n\n\n', calls[0])
        self.assertEqual(len(out), 1)

    def test_oversize_sentence_is_its_own_chunk(self):
        # A sentence over the cap forces chunk boundaries; pack_chunks never drops/sub-splits it.
        big = 'x' * (int(MAX_CHUNK_SECONDS / EST_SECS_PER_CHAR) + 50)
        calls = []
        with mock.patch.object(core, 'load_spacy', return_value=self._fake_nlp()):
            core.gen_audio_segments(self._recording_synth(calls), f'small|{big}|tail',
                                    voice='af_sky', speed=1.0, coarse=True)
        self.assertEqual(calls, ['small', big, 'tail'])  # 3 chunks, text intact

    def test_failed_chunk_splices_silence_and_dead_letters(self):
        import os
        import tempfile
        def boom(text, sp):
            raise RuntimeError('synth boom')
        with tempfile.TemporaryDirectory() as tmp:
            dl = os.path.join(tmp, 'c.failed.jsonl')
            with mock.patch.object(core, 'load_spacy', return_value=self._fake_nlp()):
                out = core.gen_audio_segments(boom, 'one|two', voice='af_sky', speed=1.0,
                                              coarse=True, synth_retries=0, dead_letter_path=dl)
            self.assertEqual(len(out), 1)          # one chunk -> one (silence) segment
            self.assertGreater(out[0].size, 0)     # silence spliced, not empty
            with open(dl) as f:
                self.assertEqual(sum(1 for _ in f), 1)  # exactly one dead-letter for the chunk

    def test_rerun_serves_chunk_from_cache(self):
        from audiblez import cache as cache_mod
        import tempfile
        fields = core._cache_key_fields('moss', 'af_sky', 1.0)  # llamacpp cache fields
        with tempfile.TemporaryDirectory() as tmp:
            c = cache_mod.SynthCache(tmp)
            first = []
            with mock.patch.object(core, 'load_spacy', return_value=self._fake_nlp()):
                core.gen_audio_segments(self._recording_synth(first), 'one|two', voice='af_sky',
                                        speed=1.0, coarse=True, cache=c, cache_key_fields=fields)
            self.assertEqual(len(first), 1)  # one chunk synthesized + cached
            second = []
            with mock.patch.object(core, 'load_spacy', return_value=self._fake_nlp()):
                core.gen_audio_segments(self._recording_synth(second), 'one|two', voice='af_sky',
                                        speed=1.0, coarse=True, cache=c, cache_key_fields=fields)
            self.assertEqual(len(second), 0)  # re-run fully served from cache (0 new requests)


if __name__ == '__main__':
    unittest.main()
