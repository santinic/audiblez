"""Tests for the opt-in synth cache (audiblez/cache.py).

Pure numpy/hashlib/json module, so these run anywhere (no skip guard).
"""
import unittest
from unittest import mock
from types import SimpleNamespace
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from audiblez import cache

try:
    import audiblez.core as core
    _CORE_ERR = None
except Exception as e:
    _CORE_ERR = e


_FIELDS = dict(engine='torch', repo_id='hexgrad/Kokoro-82M', voice='af_sky',
               speed=1.0, max_sentence_length=400, spacy_version='3.8.3')


class MakeKeyTest(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(cache.make_key(text='hello', **_FIELDS),
                         cache.make_key(text='hello', **_FIELDS))

    def test_sensitive_to_every_field(self):
        base = cache.make_key(text='hello', **_FIELDS)
        self.assertNotEqual(base, cache.make_key(text='world', **_FIELDS))
        self.assertNotEqual(base, cache.make_key(text='hello', **{**_FIELDS, 'voice': 'am_adam'}))
        self.assertNotEqual(base, cache.make_key(text='hello', **{**_FIELDS, 'speed': 1.5}))
        self.assertNotEqual(base, cache.make_key(text='hello', **{**_FIELDS, 'engine': 'mlx'}))
        # repo_id differs by quantization between backends -> must change the key
        self.assertNotEqual(base, cache.make_key(text='hello',
                                                 **{**_FIELDS, 'repo_id': 'mlx-community/Kokoro-82M-bf16'}))
        self.assertNotEqual(base, cache.make_key(text='hello', **{**_FIELDS, 'max_sentence_length': 200}))

    def test_version_changes_key(self):
        k1 = cache.make_key(text='hello', **_FIELDS)
        old = cache.CACHE_VERSION
        try:
            cache.CACHE_VERSION = old + 1
            self.assertNotEqual(k1, cache.make_key(text='hello', **_FIELDS))
        finally:
            cache.CACHE_VERSION = old

    def test_precision_changes_key(self):
        base = cache.make_key(text='hello', **_FIELDS)  # precision defaults to fp32
        self.assertNotEqual(base, cache.make_key(text='hello', precision='bf16', **_FIELDS))
        self.assertNotEqual(base, cache.make_key(text='hello', precision='fp16', **_FIELDS))
        self.assertEqual(base, cache.make_key(text='hello', precision='fp32', **_FIELDS))

    def test_kokoro_key_is_byte_stable(self):
        # GOLDEN REGRESSION GUARD: the existing Kokoro call path must hash byte-identically
        # across the seed/sampling_sig additions, or every cached .npy on disk goes cold.
        # Captured on main BEFORE the MOSS changes; assumes CACHE_VERSION == 1. The relative
        # !=/= tests above can't catch an accidental payload change — this literal can.
        if cache.CACHE_VERSION != 1:
            self.skipTest('golden hash pinned at CACHE_VERSION 1')
        self.assertEqual(
            cache.make_key(text='hello', **_FIELDS),
            '8cc3269e8a76befbb3ce262ee367000a362c05c596bbb54cfb20a9d42844e7d8')
        self.assertEqual(
            cache.make_key(text='hello', precision='bf16', **_FIELDS),
            'e969ebcab0fa4978d9a5fedcebaf58ec4d9c493cd64a0f38964336061e167a39')


_MOSS_FIELDS = dict(engine='llamacpp', repo_id='moss-gguf:deadbeef', voice='af_sky',
                    speed=1.0, seed=12345, sampling_sig='t1.5_k50_at1.7_ap0.8_ak25_rp1.0')


class MossKeyTest(unittest.TestCase):
    """MOSS adds seed + sampling_sig + a GGUF-hashing repo_id; engine separates it from Kokoro."""

    def test_moss_engine_never_collides_with_kokoro(self):
        # Same text/voice/speed but engine differs -> distinct keys, no CACHE_VERSION bump needed.
        kok = cache.make_key(text='hello', engine='torch', repo_id='hexgrad/Kokoro-82M',
                             voice='af_sky', speed=1.0, max_sentence_length=400)
        moss = cache.make_key(text='hello', **_MOSS_FIELDS)
        self.assertNotEqual(kok, moss)

    def test_seed_changes_key(self):
        base = cache.make_key(text='hi', **_MOSS_FIELDS)
        self.assertNotEqual(base, cache.make_key(text='hi', **{**_MOSS_FIELDS, 'seed': 999}))

    def test_sampling_sig_changes_key(self):
        base = cache.make_key(text='hi', **_MOSS_FIELDS)
        self.assertNotEqual(base, cache.make_key(text='hi', **{**_MOSS_FIELDS, 'sampling_sig': 'other'}))

    def test_repo_id_changes_key(self):
        # repo_id hashes all three GGUF identities -> a swapped GGUF changes the key.
        base = cache.make_key(text='hi', **_MOSS_FIELDS)
        self.assertNotEqual(base, cache.make_key(text='hi', **{**_MOSS_FIELDS, 'repo_id': 'moss-gguf:0000'}))

    def test_two_clone_refs_do_not_collide_via_voice(self):
        # End-to-end clone-correctness: core passes backends.clone_voice_id(wav) as `voice`, so
        # two different reference WAVs synthesizing the SAME sentence must yield different keys
        # (else a re-run serves the wrong cloned voice — the banned silent-wrongness class).
        from audiblez import backends
        import os
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as d:
            ra, rb = os.path.join(d, 'a.wav'), os.path.join(d, 'b.wav')
            with open(ra, 'wb') as f:
                f.write(b'voiceA-bytes')
            with open(rb, 'wb') as f:
                f.write(b'voiceB-bytes')
            ka = cache.make_key(text='hi', **{**_MOSS_FIELDS, 'voice': backends.clone_voice_id(ra)})
            kb = cache.make_key(text='hi', **{**_MOSS_FIELDS, 'voice': backends.clone_voice_id(rb)})
            self.assertNotEqual(ka, kb)
            # same reference -> same key (a re-run of the same clone hits the cache)
            self.assertEqual(ka, cache.make_key(text='hi',
                             **{**_MOSS_FIELDS, 'voice': backends.clone_voice_id(ra)}))

    def test_seed_and_sampling_are_omitted_when_none(self):
        # When seed/sampling_sig are None (the Kokoro path), they must not appear in the payload,
        # so a torch key with no MOSS knobs equals the same call with explicit None.
        a = cache.make_key(text='hi', engine='torch', repo_id='r', voice='v', speed=1.0,
                           max_sentence_length=400)
        b = cache.make_key(text='hi', engine='torch', repo_id='r', voice='v', speed=1.0,
                           max_sentence_length=400, seed=None, sampling_sig=None)
        self.assertEqual(a, b)

    def test_msl_omitted_for_moss_does_not_equal_msl_present(self):
        # MOSS omits max_sentence_length (it doesn't split on MSL); that omission is itself part
        # of the key, so adding an msl yields a different hash (no accidental cross-contamination).
        no_msl = cache.make_key(text='hi', **_MOSS_FIELDS)
        with_msl = cache.make_key(text='hi', **{**_MOSS_FIELDS, 'max_sentence_length': 400})
        self.assertNotEqual(no_msl, with_msl)


@unittest.skipIf(_CORE_ERR is not None, f"audiblez.core unavailable: {_CORE_ERR}")
class MossVoiceSentinelTest(unittest.TestCase):
    """core._cache_key_fields under MOSS with no clone ref (slice 2 / #8).

    The GUI greys the Kokoro voice dropdown under MOSS but it still holds *some* voice string;
    MOSS ignores it, so the sentence-cache key must be the fixed MOSS_DEFAULT_VOICE sentinel —
    independent of the dropdown — else the cache fragments by an inert voice and re-runs miss.
    A clone reference still takes precedence on the voice axis (re-confirmed here).
    """

    def test_no_clone_uses_sentinel_independent_of_kokoro_voice(self):
        f1 = core._cache_key_fields('moss', 'af_sky', 1.0)
        f2 = core._cache_key_fields('moss', 'am_adam', 1.0)
        self.assertEqual(f1['voice'], core.MOSS_DEFAULT_VOICE)
        self.assertEqual(f2['voice'], core.MOSS_DEFAULT_VOICE)
        # whole key (text + fields) is identical across two different Kokoro voices -> a re-run hits
        self.assertEqual(cache.make_key(text='hi', **f1), cache.make_key(text='hi', **f2))

    def test_clone_ref_overrides_the_sentinel_and_two_refs_differ(self):
        with TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a.wav', Path(tmp) / 'b.wav'
            a.write_bytes(b'AAAA'); b.write_bytes(b'BBBB')
            fa = core._cache_key_fields('moss', 'af_sky', 1.0, clone_ref=str(a))
            fb = core._cache_key_fields('moss', 'af_sky', 1.0, clone_ref=str(b))
            self.assertNotEqual(fa['voice'], core.MOSS_DEFAULT_VOICE)  # clone wins over the sentinel
            self.assertNotEqual(fa['voice'], fb['voice'])              # two refs never collide
            # same ref -> same voice id (a re-run of the same clone hits)
            self.assertEqual(fa['voice'], core._cache_key_fields('moss', 'af_sky', 1.0,
                                                                 clone_ref=str(a))['voice'])

    def test_kokoro_voice_is_not_sentinelled(self):
        # The sentinel is MOSS-only; a torch backend must keep passing the real voice through.
        self.assertEqual(core._cache_key_fields('cpu', 'af_sky', 1.0)['voice'], 'af_sky')


@unittest.skipIf(_CORE_ERR is not None, f"audiblez.core unavailable: {_CORE_ERR}")
class RenderSignatureTest(unittest.TestCase):
    """The chapter `.sig` resume gate (core._render_signature) — runs BEFORE the sentence cache,
    so it must capture every MOSS waveform axis too, or a model/sampling/clone swap reuses a stale
    chapter wav (silent-wrongness). Kokoro must keep emitting the pre-MOSS string (no cache bust).
    """

    def test_kokoro_signature_has_no_moss_axes(self):
        # A torch backend (and the backend=None default) must emit exactly lex;speed;precision —
        # unchanged from before the MOSS axes existed, so old .sig files stay valid.
        sig = core._render_signature({}, 1.0, 'fp32', backend='cpu')
        self.assertNotIn('engine=llamacpp', sig)
        self.assertEqual(sig, core._render_signature({}, 1.0, 'fp32'))

    def test_moss_signature_includes_model_seed_and_sampling(self):
        sig = core._render_signature({}, 1.0, 'fp32', backend='moss')
        self.assertIn('engine=llamacpp', sig)
        self.assertIn('repo=', sig)
        self.assertIn(f'seed={core.MOSS_SEED}', sig)
        self.assertIn('sampling=', sig)

    def test_clone_ref_changes_moss_signature(self):
        # Two different clone refs (with a constant voice) must NOT share a .sig — else clone-ref-B
        # re-run silently reuses clone-ref-A's chapter wav (the filename collision cache-rev flagged).
        with TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a.wav', Path(tmp) / 'b.wav'
            a.write_bytes(b'AAAA'); b.write_bytes(b'BBBB')
            none = core._render_signature({}, 1.0, 'fp32', backend='moss')
            sig_a = core._render_signature({}, 1.0, 'fp32', backend='moss', clone_ref=str(a))
            sig_b = core._render_signature({}, 1.0, 'fp32', backend='moss', clone_ref=str(b))
            self.assertNotEqual(sig_a, none)
            self.assertNotEqual(sig_a, sig_b)

    def test_coarse_changes_moss_signature(self):
        # Coarse synthesizes a chunk as one utterance -> materially different audio, so a
        # coarse/per-sentence toggle MUST bust the chapter wav on resume (story 11). The default
        # (per-sentence) MOSS sig carries no 'coarse' marker, so old .sig files stay valid.
        plain = core._render_signature({}, 1.0, 'fp32', backend='moss')
        coarse = core._render_signature({}, 1.0, 'fp32', backend='moss', coarse=True)
        self.assertNotIn('coarse', plain)
        self.assertIn('coarse=1', coarse)
        self.assertNotEqual(plain, coarse)

    def test_coarse_flag_does_not_affect_kokoro_signature(self):
        # Coarse is MOSS-only; a torch backend's sig is unchanged by the flag (backward compat).
        self.assertEqual(core._render_signature({}, 1.0, 'fp32', backend='cpu'),
                         core._render_signature({}, 1.0, 'fp32', backend='cpu', coarse=True))


class SynthCacheTest(unittest.TestCase):
    def test_miss_then_put_then_hit(self):
        with TemporaryDirectory() as tmp:
            c = cache.SynthCache(tmp)
            key = cache.make_key(text='hello', **_FIELDS)
            self.assertIsNone(c.get(key))      # miss
            self.assertEqual(c.misses, 1)
            audio = np.arange(10, dtype=np.float32)
            c.put(key, audio)
            got = c.get(key)                   # hit
            self.assertIsNotNone(got)
            np.testing.assert_array_equal(got, audio)
            self.assertEqual(c.hits, 1)

    def test_corrupt_entry_is_a_miss(self):
        with TemporaryDirectory() as tmp:
            c = cache.SynthCache(tmp)
            key = cache.make_key(text='x', **_FIELDS)
            (c._path(key)).write_bytes(b'not a real npy')
            self.assertIsNone(c.get(key))
            self.assertEqual(c.misses, 1)

    def test_stats_string(self):
        with TemporaryDirectory() as tmp:
            c = cache.SynthCache(tmp)
            c.hits, c.misses = 3, 1
            self.assertIn('75% hit rate', c.stats())

    def test_clear_removes_entries(self):
        with TemporaryDirectory() as tmp:
            c = cache.SynthCache(tmp)
            for t in ('a', 'b', 'c'):
                c.put(cache.make_key(text=t, **_FIELDS), np.arange(4, dtype=np.float32))
            self.assertEqual(c.clear(), 3)
            self.assertEqual(c.clear(), 0)  # idempotent

    def test_put_is_atomic_and_leaves_no_tmp(self):
        with TemporaryDirectory() as tmp:
            c = cache.SynthCache(tmp)
            c.put(cache.make_key(text='a', **_FIELDS), np.arange(4, dtype=np.float32))
            leftovers = [p.name for p in Path(tmp).glob('*.tmp.npy')]
            self.assertEqual(leftovers, [])

    def test_corrupt_entry_is_deleted_on_read(self):
        with TemporaryDirectory() as tmp:
            c = cache.SynthCache(tmp)
            key = cache.make_key(text='x', **_FIELDS)
            c._path(key).write_bytes(b'not a real npy')
            self.assertIsNone(c.get(key))            # miss
            self.assertFalse(c._path(key).exists())  # corrupt file cleaned up


@unittest.skipIf(_CORE_ERR is not None, f"audiblez.core unavailable: {_CORE_ERR}")
class GenAudioSegmentsCacheTest(unittest.TestCase):
    def _fake_nlp(self):
        return lambda text: SimpleNamespace(sents=[SimpleNamespace(text=s) for s in text.split('|')])

    def test_second_run_hits_cache_and_skips_synth(self):
        fields = dict(_FIELDS)
        synth = mock.MagicMock(side_effect=lambda t, sp: [np.ones(3, dtype=np.float32)])
        with TemporaryDirectory() as tmp:
            c = cache.SynthCache(tmp)
            with mock.patch.object(core, 'load_spacy', return_value=self._fake_nlp()):
                out1 = core.gen_audio_segments(synth, 'one|two', voice='af_sky', speed=1.0,
                                               cache=c, cache_key_fields=fields)
            self.assertEqual(synth.call_count, 2)   # two sentences synthesized (misses)
            self.assertEqual((c.hits, c.misses), (0, 2))

            synth.reset_mock()
            c2 = cache.SynthCache(tmp)              # fresh cache obj, same dir
            with mock.patch.object(core, 'load_spacy', return_value=self._fake_nlp()):
                out2 = core.gen_audio_segments(synth, 'one|two', voice='af_sky', speed=1.0,
                                               cache=c2, cache_key_fields=fields)
            self.assertEqual(synth.call_count, 0)   # all served from cache
            self.assertEqual((c2.hits, c2.misses), (2, 0))
            for a, b in zip(out1, out2):
                np.testing.assert_array_equal(a, b)

    def test_empty_synth_result_is_not_cached(self):
        # synth returns [] (no audio, no exception) -> must NOT persist np.zeros(0), else
        # every later run serves silent "no audio" for that sentence as a cache hit.
        synth = mock.MagicMock(side_effect=lambda t, sp: [])
        with TemporaryDirectory() as tmp:
            c = cache.SynthCache(tmp)
            with mock.patch.object(core, 'load_spacy', return_value=self._fake_nlp()):
                core.gen_audio_segments(synth, 'one', voice='af_sky', speed=1.0,
                                        cache=c, cache_key_fields=dict(_FIELDS))
            self.assertEqual(list(Path(tmp).glob('*.npy')), [])  # nothing cached
            # a second run is therefore a miss again (not a spurious hit)
            c2 = cache.SynthCache(tmp)
            with mock.patch.object(core, 'load_spacy', return_value=self._fake_nlp()):
                core.gen_audio_segments(synth, 'one', voice='af_sky', speed=1.0,
                                        cache=c2, cache_key_fields=dict(_FIELDS))
            self.assertEqual(c2.hits, 0)


if __name__ == '__main__':
    unittest.main()
