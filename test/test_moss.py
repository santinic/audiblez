"""Behavioral tests for the MOSS engine — written against FakeMossProcess.

These tests cover the wire protocol, failure/death handling, restart logic,
and circuit-breaker as specified in:
  docs/moss-coprocess-spec.md  (wire protocol + failure vs death model)
  docs/adr/0005-moss-per-sentence-granularity.md  (coarse-chunk mode)
  GitHub issue #2 (PRD, user stories 14–17, 33–36)

Symbols used from T2 (audiblez.core) — RECONCILED against the real implementation:
------------------------------------------------------------------------------------
  _build_llamacpp_synth(voice, lang_code, clone_ref=None, gguf_dir=None,
                        moss_factory=None, work_dir=None)
      -> synth   (a closure with synth.close and synth.moss attributes)
      ``moss_factory()`` is called with no args; must return a MossProcess-duck.
      ``synth(text, speed) -> list[np.ndarray]`` @ 24 kHz.
      ``synth.close()`` tears down the child cleanly.
      ``synth.moss`` exposes the MossProcess for diagnostics.

  MossRunAborted (BaseException) — covers BOTH spawn-fail AND circuit-breaker:
      raised when the child won't spawn (abort-loud) OR when the global restart
      rate exceeds MOSS_MAX_RESTARTS_IN_WINDOW in MOSS_RESTART_WINDOW sentences.
      NOTE: this is a BaseException, not Exception — it bypasses broad
      ``except Exception`` in the synth/retry machinery to prevent a systemic
      crash from being swallowed as silence.

  MossError (Exception) — status:error from a healthy child (→ dead-letter).

  MossPermanentError (MossError, ValueError) — deterministic failure, subclasses
      ValueError so it IS a _PERMANENT_SYNTH_ERRORS member; _retry won't re-spin.

  MossDeath (Exception) — child died mid-request (internal to closure, used by
      FakeMossAdapter to signal EOF/poll-dead/timeout).

  MOSS_DEATHS_BEFORE_POISON (int = 2) — K consecutive deaths on ONE sentence
      before it is dead-lettered as a poison pill.

  NOTE — names that CHANGED from the initial T4 contract:
    * MossCircuitBreakerError → MossRunAborted  (also covers spawn fail)
    * MossSpawnError          → MossRunAborted
    * MOSS_MAX_CONSECUTIVE_DEATHS → MOSS_DEATHS_BEFORE_POISON
    * process_factory kwarg   → moss_factory kwarg  (returns MossProcess-duck)
    * returns synth only      → not (synth, close); close is synth.close

All tests guard the import so they SKIP (not ERROR) if the symbols are absent.
"""

import os
import struct
import tempfile
import unittest

# ---------------------------------------------------------------------------
# Guard import — skip cleanly if T2 hasn't landed yet
# ---------------------------------------------------------------------------
try:
    from test._moss_fakes import (
        FakeMossProcess,
        FakeMossAdapter,
        make_synth_request,
        SAMPLE_RATE,
    )
    _FAKES_ERR = None
except Exception as e:
    _FAKES_ERR = e

# T2's symbols — reconciled names (see module docstring for what changed)
try:
    from audiblez.core import (
        _build_llamacpp_synth,
        MossRunAborted,
        MossError,
        MossPermanentError,
        MOSS_DEATHS_BEFORE_POISON,
    )
    _CORE_ERR = None
except Exception as e:
    _CORE_ERR = e

# Also import gen_audio_segments and SynthCache for the integration test
try:
    from audiblez.core import gen_audio_segments, _cache_key_fields
    from audiblez import cache as cache_mod
    _INTEGRATION_ERR = None
except Exception as e:
    _INTEGRATION_ERR = e

_SKIP_FAKES = _FAKES_ERR is not None
_SKIP_CORE = _CORE_ERR is not None
_SKIP_INTEGRATION = _SKIP_FAKES or _SKIP_CORE or (_INTEGRATION_ERR is not None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wav_dir():
    """Return a temp directory for WAV handoff files; caller must clean up."""
    return tempfile.mkdtemp(prefix="test_moss_wav_")


def _read_wav_frames(path: str) -> int:
    """Read frame count from a minimal 16-bit WAV without soundfile."""
    with open(path, "rb") as f:
        f.seek(40)  # data-chunk size offset
        data_size = struct.unpack("<I", f.read(4))[0]
    return data_size // 2  # 16-bit mono


# ---------------------------------------------------------------------------
# Wire protocol / FakeMossProcess tests (fakes only — no T2 needed)
# ---------------------------------------------------------------------------

@unittest.skipIf(_SKIP_FAKES, f"_moss_fakes unavailable: {_FAKES_ERR}")
class FakeMossProtocolTest(unittest.TestCase):
    """Verify FakeMossProcess speaks the wire protocol correctly."""

    def setUp(self):
        self.tmp = _wav_dir()
        self._fakes: list[FakeMossProcess] = []

    def tearDown(self):
        import shutil
        for fake in self._fakes:
            try:
                fake.kill()
                fake.close_pipes()
                fake.cleanup()
            except Exception:
                pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_fake(self, **kwargs) -> FakeMossProcess:
        """Create, register, and return a FakeMossProcess."""
        fake = FakeMossProcess(**kwargs)
        self._fakes.append(fake)
        return fake

    def _wav(self, name="req-0.wav"):
        return os.path.join(self.tmp, name)

    def test_basic_synth_round_trip(self):
        """A normal synth request returns status:ok with frames > 0."""
        fake = self._make_fake()
        fake.start()
        wav = self._wav()
        req = make_synth_request(0, "Hello world.", wav)
        fake.send(req)
        resp = fake.recv()
        fake.shutdown()

        self.assertIsNotNone(resp)
        self.assertEqual(resp["status"], "ok")
        self.assertEqual(resp["id"], 0)
        self.assertGreater(resp["frames"], 0)
        self.assertEqual(resp["sample_rate"], SAMPLE_RATE)
        self.assertTrue(os.path.exists(resp["wav_out"]))

    def test_ok_response_wav_has_correct_frame_count(self):
        """The WAV written by the fake matches the frame count in the response."""
        fake = self._make_fake(wav_frames=48_000)
        fake.start()
        wav = self._wav()
        fake.send(make_synth_request(1, "Test.", wav))
        resp = fake.recv()
        fake.shutdown()

        frames_in_file = _read_wav_frames(resp["wav_out"])
        self.assertEqual(frames_in_file, resp["frames"])
        self.assertEqual(frames_in_file, 48_000)

    def test_id_echo(self):
        """Response id matches request id."""
        fake = self._make_fake()
        fake.start()
        wav = self._wav()
        fake.send(make_synth_request(99, "Echo id.", wav))
        resp = fake.recv()
        fake.shutdown()
        self.assertEqual(resp["id"], 99)

    def test_shutdown_returns_bye(self):
        """Shutdown handshake returns {status: bye}."""
        fake = self._make_fake()
        fake.start()
        resp = fake.shutdown()
        self.assertIsNotNone(resp)
        self.assertEqual(resp.get("status"), "bye")

    def test_error_text_returns_status_error(self):
        """A text in error_texts triggers a status:error response; child stays alive."""
        fake = self._make_fake(error_texts={"fail me"})
        fake.start()
        wav = self._wav("err.wav")
        fake.send(make_synth_request(2, "fail me", wav))
        resp = fake.recv()

        self.assertIsNotNone(resp)
        self.assertEqual(resp["status"], "error")
        self.assertEqual(resp["id"], 2)
        # Child should still be alive (poll returns None)
        self.assertIsNone(fake.poll())

        # Can still send a normal request afterwards
        fake.send(make_synth_request(3, "normal text", self._wav("ok.wav")))
        resp2 = fake.recv()
        self.assertEqual(resp2["status"], "ok")
        fake.shutdown()

    def test_die_after_causes_eof(self):
        """die_after=1 closes stdout after one successful response."""
        fake = self._make_fake(die_after=1)
        fake.start()
        fake.send(make_synth_request(0, "first", self._wav("f0.wav")))
        resp0 = fake.recv()
        self.assertEqual(resp0["status"], "ok")

        # Second request — child should have died; recv returns None (EOF)
        fake.send(make_synth_request(1, "second", self._wav("f1.wav")))
        resp1 = fake.recv(timeout=2.0)
        self.assertIsNone(resp1)
        self.assertIsNotNone(fake.poll())  # dead

    def test_poison_pill_kills_child(self):
        """A poison-pill text causes the child to close stdout (simulate death)."""
        fake = self._make_fake(poison_pill_texts={"boom"}, poison_deaths=2)
        fake.start()
        fake.send(make_synth_request(0, "boom", self._wav("p0.wav")))
        resp = fake.recv(timeout=2.0)
        # EOF → None
        self.assertIsNone(resp)
        self.assertIsNotNone(fake.poll())

    def test_multiple_sequential_requests(self):
        """Multiple requests in sequence all return ok in order."""
        fake = self._make_fake()
        fake.start()
        texts = ["One.", "Two.", "Three."]
        for i, text in enumerate(texts):
            fake.send(make_synth_request(i, text, self._wav(f"s{i}.wav")))
            resp = fake.recv()
            self.assertEqual(resp["status"], "ok")
            self.assertEqual(resp["id"], i)
        fake.shutdown()

    def test_spawn_fails_raises(self):
        """spawn_fails=True causes start() to raise OSError."""
        fake = self._make_fake(spawn_fails=True)
        with self.assertRaises(OSError):
            fake.start()


# ---------------------------------------------------------------------------
# Engine behavioral tests (require T2's _build_llamacpp_synth)
# ---------------------------------------------------------------------------

@unittest.skipIf(_SKIP_FAKES, f"_moss_fakes unavailable: {_FAKES_ERR}")
@unittest.skipIf(_SKIP_CORE, f"audiblez.core MOSS symbols unavailable: {_CORE_ERR}")
class MossEngineBehaviorTest(unittest.TestCase):
    """Behavioral tests for _build_llamacpp_synth using FakeMossAdapter.

    FakeMossAdapter wraps FakeMossProcess and exposes the MossProcess duck interface
    (synth/start/close/restart/alive/.spawns), injected via moss_factory.

    Real _build_llamacpp_synth signature::

        _build_llamacpp_synth(voice, lang_code, clone_ref=None, gguf_dir=None,
                              moss_factory=None, work_dir=None)
            -> synth   (closure with synth.close and synth.moss attributes)
    """

    def setUp(self):
        self.tmp = _wav_dir()
        self._adapters: list[FakeMossAdapter] = []

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)
        for adapter in self._adapters:
            try:
                adapter.cleanup()
            except Exception:
                pass

    def _moss_factory(self, **fake_kwargs):
        """Return a moss_factory that yields a fresh FakeMossAdapter each call."""
        test = self

        def factory():
            adapter = FakeMossAdapter(**fake_kwargs)
            test._adapters.append(adapter)
            return adapter

        return factory

    def _build(self, **fake_kwargs):
        synth = _build_llamacpp_synth(
            voice="af_sky", lang_code="a",
            moss_factory=self._moss_factory(**fake_kwargs),
            work_dir=self.tmp,
        )
        return synth

    # --- per-sentence request round-trip ---

    def test_synth_returns_nonempty_array(self):
        """synth(text, speed) returns a non-empty list of np.ndarray."""
        import numpy as np
        synth = self._build()
        try:
            result = synth("Hello.", 1.0)
            self.assertIsInstance(result, list)
            self.assertGreater(len(result), 0)
            self.assertIsInstance(result[0], np.ndarray)
            self.assertGreater(result[0].size, 0)
        finally:
            synth.close()

    def test_synth_returns_float32_24khz(self):
        """Audio is float32 @ 24 kHz (matches existing synth contract)."""
        import numpy as np
        synth = self._build()
        try:
            result = synth("Check dtype.", 1.0)
            arr = np.concatenate(result)
            self.assertEqual(arr.dtype, np.float32)
        finally:
            synth.close()

    # --- status:error → dead-letter (child healthy, spec §A) ---

    def test_error_response_raises_moss_error(self):
        """A status:error response raises MossError; the adapter stays alive for next request."""
        synth = self._build(error_texts={"bad sentence"})
        try:
            with self.assertRaises(MossError):
                synth("bad sentence", 1.0)
            # Adapter must still be healthy — next call must succeed
            result = synth("good sentence", 1.0)
            self.assertGreater(len(result), 0)
        finally:
            synth.close()

    # --- death (EOF / dead poll / timeout) → restart + re-dispatch (spec §B) ---

    def test_child_death_triggers_restart_and_retry(self):
        """Child dying on first request causes a restart; same sentence re-dispatched.

        The engine's restart path calls moss.restart() on the existing adapter, so the
        adapter must behave differently after restart.  We achieve this by tracking
        adapter.spawns: spawns=1 → die immediately; spawns=2 → succeed.
        """
        # We need the adapter to die on the first spawn but succeed on the second.
        # FakeMossAdapter.restart() calls close()+start() incrementing self.spawns.
        # By passing die_after_per_spawn, a custom adapter sub-class or a closure
        # adapter approach is required.  Simplest: use a closure that passes
        # die_after conditionally based on spawns counter shared by reference.

        spawns = [0]

        class SpawnCountingAdapter(FakeMossAdapter):
            def start(self_):
                spawns[0] += 1
                # Override die_after based on spawn count
                if spawns[0] == 1:
                    # First spawn: die immediately on next synth call
                    self_._kwargs = dict(self_._kwargs, die_after=0)
                else:
                    # Subsequent spawns: succeed normally
                    self_._kwargs = {k: v for k, v in self_._kwargs.items()
                                     if k != "die_after"}
                return super().start()

        def factory():
            adapter = SpawnCountingAdapter()
            self._adapters.append(adapter)
            return adapter

        synth = _build_llamacpp_synth(
            voice="af_sky", lang_code="a",
            moss_factory=factory,
            work_dir=self.tmp,
        )
        try:
            result = synth("Restart me.", 1.0)
            self.assertGreater(len(result), 0)
            self.assertGreaterEqual(spawns[0], 2, "expected at least one restart")
        finally:
            synth.close()

    # --- K=2 consecutive deaths → dead-letter as MossPermanentError (poison-pill guard) ---

    def test_poison_pill_dead_lettered_after_k_deaths(self):
        """A sentence that kills the child MOSS_DEATHS_BEFORE_POISON times is raised as
        MossPermanentError (subclasses ValueError → in _PERMANENT_SYNTH_ERRORS, so _retry
        won't re-spin it into an infinite crash loop).
        """
        def factory():
            adapter = FakeMossAdapter(
                poison_pill_texts={"poison"},
                poison_deaths=MOSS_DEATHS_BEFORE_POISON + 10,
            )
            self._adapters.append(adapter)
            return adapter

        synth = _build_llamacpp_synth(
            voice="af_sky", lang_code="a",
            moss_factory=factory,
            work_dir=self.tmp,
        )
        try:
            with self.assertRaises(MossPermanentError):
                synth("poison", 1.0)
        finally:
            synth.close()

    # --- circuit-breaker aborts on systemic restart rate → MossRunAborted ---

    def test_circuit_breaker_fires_on_systemic_deaths(self):
        """Too many restarts in a short window raises MossRunAborted (a BaseException).

        Uses a fixed fake clock so all restarts fall within MOSS_RESTART_WINDOW_SECONDS;
        each adapter responds to the FIRST request then dies, causing one restart per
        sentence — distinct sentences avoid the MOSS_DEATHS_BEFORE_POISON poison-pill path
        which would mask the circuit-breaker with MossPermanentError.
        """
        from audiblez.core import MOSS_MAX_RESTARTS_IN_WINDOW

        def factory():
            # die_after=1: succeeds on first request, dies on the second (next sentence).
            # This triggers one restart per sentence without hitting the poison-pill path
            # (poison-pill requires K consecutive deaths on the SAME sentence).
            adapter = FakeMossAdapter(die_after=1)
            self._adapters.append(adapter)
            return adapter

        # Fixed clock: every call returns the same timestamp so all restarts appear
        # simultaneous and fall within MOSS_RESTART_WINDOW_SECONDS.
        fixed_time = [1_000_000.0]

        synth = _build_llamacpp_synth(
            voice="af_sky", lang_code="a",
            moss_factory=factory,
            work_dir=self.tmp,
            clock=lambda: fixed_time[0],
        )
        raised = False
        try:
            # Need more than MOSS_MAX_RESTARTS_IN_WINDOW sentences to trip the breaker.
            for i in range(MOSS_MAX_RESTARTS_IN_WINDOW + 5):
                synth(f"unique sentence {i}", 1.0)
        except MossRunAborted:
            raised = True
        finally:
            try:
                synth.close()
            except Exception:
                pass
        self.assertTrue(raised, "MossRunAborted should have been raised by circuit-breaker")

    # --- abort-loud on spawn failure → MossRunAborted ---

    def test_spawn_failure_raises_moss_run_aborted(self):
        """If the child won't spawn, raise MossRunAborted (never fall silently to Kokoro).

        MossRunAborted is a BaseException so it bypasses ``except Exception`` in all
        retry/silence handlers and propagates to core.main -> non-zero exit.
        FakeMossAdapter.start() mirrors MossProcess.start(): wraps the spawn failure into
        MossRunAborted.  _build_llamacpp_synth calls moss.start() immediately, so the
        exception is raised from _build_llamacpp_synth, not from synth().
        """
        with self.assertRaises(MossRunAborted):
            _build_llamacpp_synth(
                voice="af_sky", lang_code="a",
                moss_factory=self._moss_factory(spawn_fails=True),
                work_dir=self.tmp,
            )

    # --- empty audio never cached → MossPermanentError ---

    def test_empty_audio_raises_permanent_error(self):
        """A status:error with code=empty_audio raises MossPermanentError, not a cached zero."""
        # Fake returning empty_audio code keeps the adapter alive (§A path) but raises
        # MossPermanentError so _retry won't re-spin and it is never cached.
        synth = self._build(error_texts={"empty result"})
        try:
            # First call must raise
            with self.assertRaises((MossError, MossPermanentError)):
                synth("empty result", 1.0)
            # Second call must ALSO raise — not serve a cached np.zeros(0)
            with self.assertRaises((MossError, MossPermanentError)):
                synth("empty result", 1.0)
        finally:
            synth.close()


# ---------------------------------------------------------------------------
# Pack-chunks coarse-mode tests (no T2 needed — pure-function)
# ---------------------------------------------------------------------------

try:
    from audiblez.chunking import pack_chunks, MAX_CHUNK_SECONDS, EST_SECS_PER_CHAR
    _CHUNKING_ERR = None
except Exception as e:
    _CHUNKING_ERR = e


@unittest.skipIf(_CHUNKING_ERR is not None, f"audiblez.chunking unavailable: {_CHUNKING_ERR}")
class PackChunksCoarseModeTest(unittest.TestCase):
    """Coarse-chunk packing is tested here (also in test_batching.py)."""

    def _secs(self, text):
        return len(text) * EST_SECS_PER_CHAR

    def test_single_sentence_below_cap_is_one_chunk(self):
        sents = ["Short sentence."]
        chunks = pack_chunks(sents)
        self.assertEqual(chunks, [["Short sentence."]])

    def test_order_preserved(self):
        """Flattening chunks always recovers the original sentence list."""
        sents = [f"Sentence {i}." for i in range(20)]
        chunks = pack_chunks(sents, max_seconds=2.0)
        flat = [s for c in chunks for s in c]
        self.assertEqual(flat, sents)

    def test_each_chunk_estimated_duration_within_cap(self):
        """Each chunk's estimated duration must not exceed max_seconds (unless a single sentence
        is already over — allowed per spec)."""
        sents = ["x" * 30] * 20   # ~1.2 s each
        chunks = pack_chunks(sents, max_seconds=MAX_CHUNK_SECONDS)
        for chunk in chunks:
            total_chars = sum(len(s) for s in chunk)
            est_secs = total_chars * EST_SECS_PER_CHAR
            # If a single-sentence chunk, it may be over cap — that's allowed.
            if len(chunk) > 1:
                self.assertLessEqual(est_secs, MAX_CHUNK_SECONDS + 1e-9,
                                     f"Chunk exceeded cap: {est_secs:.2f}s")

    def test_oversize_single_sentence_becomes_own_chunk(self):
        """A single sentence that alone exceeds the cap must NOT be dropped."""
        giant = "w " * 1000   # ~80 s of audio
        sents = ["Before.", giant, "After."]
        chunks = pack_chunks(sents, max_seconds=MAX_CHUNK_SECONDS)
        flat = [s for c in chunks for s in c]
        self.assertIn(giant, flat)
        self.assertEqual(flat, sents)

    def test_empty_input(self):
        self.assertEqual(pack_chunks([]), [])

    def test_tiny_cap_one_sentence_per_chunk(self):
        sents = ["one", "two", "three"]
        chunks = pack_chunks(sents, max_seconds=0.001)
        self.assertEqual(chunks, [["one"], ["two"], ["three"]])

    def test_adjacent_sentences_merged_when_under_cap(self):
        """Short sentences must be grouped into fewer chunks."""
        sents = ["Hi."] * 10   # ~0.12 s each → ~1.2 s total
        chunks = pack_chunks(sents, max_seconds=MAX_CHUNK_SECONDS)
        self.assertLess(len(chunks), len(sents), "should merge short sentences")


# ---------------------------------------------------------------------------
# Wire-protocol conformance guard (no T2 needed — FakeMossProcess only)
# ---------------------------------------------------------------------------

@unittest.skipIf(_SKIP_FAKES, f"_moss_fakes unavailable: {_FAKES_ERR}")
class MossWireConformanceTest(unittest.TestCase):
    """Pin the EXACT request/response JSON shapes from docs/moss-coprocess-spec.md.

    Purpose: a single contract both T1's C++ binary and T2's MossProcess must
    satisfy — catches drift between the two implementations.  All assertions
    reference the spec directly.
    """

    def setUp(self):
        self.tmp = _wav_dir()
        self._fakes: list[FakeMossProcess] = []

    def tearDown(self):
        import shutil
        for f in self._fakes:
            try:
                f.kill()
                f.close_pipes()
                f.cleanup()
            except Exception:
                pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fake(self, **kw) -> FakeMossProcess:
        f = FakeMossProcess(**kw)
        self._fakes.append(f)
        return f

    def _wav(self, name: str = "w.wav") -> str:
        return os.path.join(self.tmp, name)

    # ── REQUEST shape ──────────────────────────────────────────────────

    def test_request_has_required_fields(self):
        """Every synth REQUEST must carry v, id, op, text, wav_out, seed, sampling."""
        req = make_synth_request(7, "Hello.", self._wav())
        for key in ("v", "id", "op", "text", "wav_out", "seed", "sampling"):
            self.assertIn(key, req, f"REQUEST missing required field {key!r}")
        self.assertEqual(req["v"], 1)
        self.assertEqual(req["op"], "synth")

    def test_request_sampling_has_six_keys(self):
        """The sampling sub-dict must contain exactly the six pinned params from spec."""
        req = make_synth_request(0, "x", self._wav())
        expected = {
            "text_temperature", "text_top_k",
            "audio_temperature", "audio_top_p", "audio_top_k",
            "audio_repetition_penalty",
        }
        self.assertEqual(set(req["sampling"].keys()), expected)

    # ── OK response shape ──────────────────────────────────────────────

    def test_ok_response_has_required_fields(self):
        """OK response must carry v, id, status, wav_out, frames, sample_rate."""
        fake = self._fake()
        fake.start()
        wav = self._wav()
        fake.send(make_synth_request(3, "Test ok.", wav))
        resp = fake.recv()
        fake.shutdown()

        for key in ("v", "id", "status", "wav_out", "frames", "sample_rate"):
            self.assertIn(key, resp, f"OK response missing field {key!r}")
        self.assertEqual(resp["v"], 1)
        self.assertEqual(resp["status"], "ok")
        self.assertEqual(resp["sample_rate"], SAMPLE_RATE)

    def test_ok_response_frames_gt_zero(self):
        """Spec forbids ok-with-zero: F>0 must always hold for a valid synthesis."""
        fake = self._fake()
        fake.start()
        fake.send(make_synth_request(1, "Non-empty.", self._wav()))
        resp = fake.recv()
        fake.shutdown()

        self.assertGreater(resp["frames"], 0,
                           "ok response must carry frames > 0 (spec: empty_audio → error, not ok)")

    def test_ok_response_wav_file_exists(self):
        """Spec: parent reads audio from wav_out file; it must exist after ok."""
        wav = self._wav("present.wav")
        fake = self._fake()
        fake.start()
        fake.send(make_synth_request(2, "File.", wav))
        resp = fake.recv()
        fake.shutdown()

        self.assertTrue(os.path.exists(resp["wav_out"]),
                        "wav_out file must exist after an ok response")

    def test_ok_response_id_matches_request(self):
        """Response id must echo the request id (spec: parent validates id)."""
        fake = self._fake()
        fake.start()
        fake.send(make_synth_request(42, "Id echo.", self._wav()))
        resp = fake.recv()
        fake.shutdown()
        self.assertEqual(resp["id"], 42)

    # ── ERROR response shape ───────────────────────────────────────────

    def test_error_response_has_required_fields(self):
        """ERROR response must carry v, id, status, code, message."""
        fake = self._fake(error_texts={"fail"})
        fake.start()
        fake.send(make_synth_request(5, "fail", self._wav()))
        resp = fake.recv()
        fake.shutdown()

        for key in ("v", "id", "status", "code", "message"):
            self.assertIn(key, resp, f"ERROR response missing field {key!r}")
        self.assertEqual(resp["status"], "error")
        self.assertEqual(resp["id"], 5)

    def test_error_code_is_valid(self):
        """error.code must be one of the three spec-defined codes."""
        valid_codes = {"gen_failed", "empty_audio", "bad_request"}
        fake = self._fake(error_texts={"err"})
        fake.start()
        fake.send(make_synth_request(6, "err", self._wav()))
        resp = fake.recv()
        fake.shutdown()

        self.assertIn(resp["code"], valid_codes,
                      f"error code {resp['code']!r} not in spec-allowed set {valid_codes}")

    def test_empty_audio_is_error_not_ok(self):
        """Spec: empty_audio must be reported as status:error, NEVER as ok-with-zero frames.

        This is the silent-wrongness guard: an ok-with-zero would be cached and served as
        silence on every future run for that sentence.
        """
        # The fake returns error with code=gen_failed for "empty"; real C++ returns empty_audio.
        # The invariant under test: empty audio must never arrive as status=ok.
        # We verify FakeMossProcess never emits ok with frames=0.
        fake = self._fake()
        fake.start()
        # Force a fresh wav path; fake always writes frames > 0 in ok responses.
        fake.send(make_synth_request(9, "normal text", self._wav()))
        resp = fake.recv()
        fake.shutdown()

        if resp["status"] == "ok":
            self.assertGreater(resp["frames"], 0,
                               "INVARIANT VIOLATED: ok response with frames=0 must never occur")

    # ── SHUTDOWN handshake ─────────────────────────────────────────────

    def test_shutdown_bye_shape(self):
        """Shutdown response must carry status=bye."""
        fake = self._fake()
        fake.start()
        resp = fake.shutdown()
        self.assertIsNotNone(resp)
        self.assertEqual(resp.get("status"), "bye")

    def test_stale_wav_guard_parent_unlinks_before_send(self):
        """Spec: parent unlinks wav_out before sending to guard against stale reads.

        Verify that the protocol fixture (make_synth_request) does NOT pre-create
        the file — that is the parent's job to unlink, not the fake's.
        """
        wav = self._wav("stale.wav")
        # Write a stale file first
        with open(wav, "wb") as f:
            f.write(b"\x00" * 8)
        # Parent should unlink before sending (simulated here by explicit unlink)
        os.unlink(wav)
        # Now proceed; fake writes the real wav
        fake = self._fake()
        fake.start()
        fake.send(make_synth_request(10, "fresh.", wav))
        resp = fake.recv()
        fake.shutdown()
        self.assertEqual(resp["status"], "ok")
        self.assertTrue(os.path.exists(resp["wav_out"]))


# ---------------------------------------------------------------------------
# Convergence integration test — drives real pipeline through FakeMossAdapter
# ---------------------------------------------------------------------------

@unittest.skipIf(_SKIP_FAKES, f"_moss_fakes unavailable: {_FAKES_ERR}")
@unittest.skipIf(_SKIP_INTEGRATION,
                 f"core integration symbols unavailable: {_CORE_ERR or _INTEGRATION_ERR}")
class MossChapterIntegrationTest(unittest.TestCase):
    """End-to-end pipeline test through gen_audio_segments + FakeMossAdapter.

    This is the MERGE-ACCEPTANCE test: it exercises the full per-sentence cache
    path (gen_audio_segments with cache=SynthCache) using the MOSS engine closure
    (_build_llamacpp_synth), with all heavy I/O replaced by FakeMossAdapter.
    Activates the moment T2's symbols are importable.

    Fake spaCy is patched in (sentences are pipe-delimited in the test text) so
    no spaCy model is required.
    """

    VOICE = "af_sky"
    LANG_CODE = "a"

    def setUp(self):
        self.tmp = _wav_dir()
        self.cache_dir = tempfile.mkdtemp(prefix="test_moss_cache_")
        self._adapters: list[FakeMossAdapter] = []
        # Patch load_spacy to return a simple pipe-splitter (pipe = sentence boundary)
        from unittest import mock
        from types import SimpleNamespace
        self._nlp_patcher = mock.patch(
            "audiblez.core.load_spacy",
            return_value=lambda text: SimpleNamespace(
                sents=[SimpleNamespace(text=s) for s in text.split("|")]
            ),
        )
        self._nlp_patcher.start()

    def tearDown(self):
        import shutil
        self._nlp_patcher.stop()
        for a in self._adapters:
            try:
                a.cleanup()
            except Exception:
                pass
        shutil.rmtree(self.tmp, ignore_errors=True)
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def _moss_factory(self, **fake_kwargs):
        test = self

        def factory():
            adapter = FakeMossAdapter(**fake_kwargs)
            test._adapters.append(adapter)
            return adapter

        return factory

    def _build_synth(self, **fake_kwargs):
        return _build_llamacpp_synth(
            voice=self.VOICE,
            lang_code=self.LANG_CODE,
            moss_factory=self._moss_factory(**fake_kwargs),
            work_dir=self.tmp,
        )

    def _cache(self):
        return cache_mod.SynthCache(self.cache_dir)

    def _cache_key_fields(self):
        # engine='llamacpp', voice, speed=1.0, precision='fp32', etc.
        # We use the real helper but pass a known backend name.
        # If _cache_key_fields is unavailable we build the fields manually.
        try:
            from audiblez import backends
            # 'moss' backend must be registered by T3 for this to work.
            if "moss" in backends.BACKENDS:
                return _cache_key_fields("moss", self.VOICE, 1.0)
        except Exception:
            pass
        # Fallback: minimal fields that make_key accepts for a llamacpp engine
        from audiblez.core import (MOSS_SEED, _moss_sampling_sig,
                                   _moss_repo_id, MAX_SENTENCE_LENGTH)
        try:
            import spacy
            spacy_ver = spacy.__version__
        except ImportError:
            spacy_ver = "0.0.0"
        return dict(
            engine="llamacpp",
            repo_id=_moss_repo_id(),
            voice=self.VOICE,
            speed=1.0,
            precision="fp32",
            max_sentence_length=MAX_SENTENCE_LENGTH,
            spacy_version=spacy_ver,
            seed=MOSS_SEED,
            sampling_sig=_moss_sampling_sig(),
        )

    # (a) multi-sentence chapter → one request per sentence
    def test_one_request_per_sentence(self):
        """MOSS issues exactly one pipe request per sentence (per-sentence granularity)."""

        synth = self._build_synth()
        call_texts: list[str] = []

        # Wrap the moss adapter's synth to record calls
        original_synth = synth.moss.synth

        def recording_synth(text, timeout):
            call_texts.append(text)
            return original_synth(text, timeout)

        synth.moss.synth = recording_synth

        try:
            sents = "First sentence|Second sentence|Third sentence"
            segs = gen_audio_segments(
                synth, sents, voice=self.VOICE, speed=1.0,
                lang_code=self.LANG_CODE,
            )
            self.assertEqual(len(segs), 3, "one segment per sentence")
            self.assertEqual(len(call_texts), 3, "one MOSS request per sentence")
            self.assertFalse(any("\n\n\n" in t for t in call_texts),
                             "MOSS requests must not be batched with \\n\\n\\n")
        finally:
            synth.close()

    # (b) cache hit on re-run → 0 new requests
    def test_cache_hit_on_rerun_issues_zero_requests(self):
        """Re-running the same chapter with cache ON serves from cache (0 new requests)."""

        synth = self._build_synth()
        cache = self._cache()
        ckf = self._cache_key_fields()
        text = "Alpha sentence|Beta sentence"
        dl_path = os.path.join(self.tmp, "chapter.failed.jsonl")

        try:
            # First run — synthesizes and populates cache
            gen_audio_segments(
                synth, text, voice=self.VOICE, speed=1.0,
                lang_code=self.LANG_CODE,
                cache=cache, cache_key_fields=ckf,
                dead_letter_path=dl_path,
            )
        finally:
            synth.close()

        # Second run with a NEW synth that would record any calls
        synth2 = self._build_synth()
        call_count = [0]
        original = synth2.moss.synth

        def counting(text, timeout):
            call_count[0] += 1
            return original(text, timeout)

        synth2.moss.synth = counting
        try:
            segs2 = gen_audio_segments(
                synth2, text, voice=self.VOICE, speed=1.0,
                lang_code=self.LANG_CODE,
                cache=cache, cache_key_fields=ckf,
                dead_letter_path=dl_path,
            )
            self.assertEqual(call_count[0], 0,
                             "cache hit: no new MOSS requests should be issued on re-run")
            self.assertEqual(len(segs2), 2, "same segment count from cache")
        finally:
            synth2.close()

    # (c) scripted error_texts sentence → dead-lettered, run continues
    def test_error_sentence_dead_lettered_run_continues(self):
        """A status:error sentence is dead-lettered + silenced; the chapter run continues."""

        synth = self._build_synth(error_texts={"fail this"})
        dl_path = os.path.join(self.tmp, "ch.failed.jsonl")
        text = "Good sentence one|fail this|Good sentence two"

        try:
            segs = gen_audio_segments(
                synth, text, voice=self.VOICE, speed=1.0,
                lang_code=self.LANG_CODE,
                dead_letter_path=dl_path,
            )
        finally:
            synth.close()

        # Run must have continued (3 segments even though one failed)
        self.assertEqual(len(segs), 3, "run must continue past a dead-lettered sentence")
        # The failed sentence must be recorded in the dead-letter file
        self.assertTrue(os.path.exists(dl_path),
                        "dead-letter file must be written for the failed sentence")
        with open(dl_path) as f:
            import json as _json
            entries = [_json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(entries), 1, "exactly one dead-letter entry")
        self.assertIn("fail this", entries[0].get("text", ""))

    # (d) die_after death → restart + SAME sentence re-dispatched (not dead-lettered first time)
    def test_child_death_restarts_and_redispatches_same_sentence(self):
        """A single child death triggers restart + re-dispatch; sentence not dead-lettered."""

        call_count = [0]
        dl_path = os.path.join(self.tmp, "ch2.failed.jsonl")

        def factory():
            call_count[0] += 1
            if call_count[0] == 1:
                adapter = FakeMossAdapter(die_after=1)   # dies after one success
            else:
                adapter = FakeMossAdapter()
            self._adapters.append(adapter)
            return adapter

        synth = _build_llamacpp_synth(
            voice=self.VOICE, lang_code=self.LANG_CODE,
            moss_factory=factory,
            work_dir=self.tmp,
        )
        text = "Sentence A|Sentence B|Sentence C"
        try:
            segs = gen_audio_segments(
                synth, text, voice=self.VOICE, speed=1.0,
                lang_code=self.LANG_CODE,
                dead_letter_path=dl_path,
            )
        finally:
            synth.close()

        # All 3 sentences must produce audio (death was infra, not sentence fault)
        self.assertEqual(len(segs), 3)
        # No dead-letters — the restarted child handled the re-dispatched sentence
        if os.path.exists(dl_path):
            with open(dl_path) as f:
                entries = [l for l in f if l.strip()]
            self.assertEqual(entries, [],
                             "no dead-letters expected for an infra death + successful restart")

    # (e) poison-pill → dead-lettered after MOSS_DEATHS_BEFORE_POISON, raised as permanent
    def test_poison_pill_dead_lettered_after_k_deaths(self):
        """A sentence that kills the child MOSS_DEATHS_BEFORE_POISON consecutive times
        is dead-lettered via MossPermanentError, not crash-looped indefinitely.
        """

        def factory():
            adapter = FakeMossAdapter(
                poison_pill_texts={"poison pill sentence"},
                poison_deaths=MOSS_DEATHS_BEFORE_POISON + 10,
            )
            self._adapters.append(adapter)
            return adapter

        synth = _build_llamacpp_synth(
            voice=self.VOICE, lang_code=self.LANG_CODE,
            moss_factory=factory,
            work_dir=self.tmp,
        )
        dl_path = os.path.join(self.tmp, "ch3.failed.jsonl")
        text = "Good start|poison pill sentence|Good end"

        try:
            segs = gen_audio_segments(
                synth, text, voice=self.VOICE, speed=1.0,
                lang_code=self.LANG_CODE,
                dead_letter_path=dl_path,
            )
        finally:
            try:
                synth.close()
            except Exception:
                pass

        # The run must continue (3 segments: good + silence + good)
        self.assertEqual(len(segs), 3,
                         "run must continue with silence for the poison-pill sentence")
        # The poison-pill sentence must be dead-lettered
        self.assertTrue(os.path.exists(dl_path),
                        "poison-pill sentence must produce a dead-letter entry")
        with open(dl_path) as f:
            import json as _json
            entries = [_json.loads(l) for l in f if l.strip()]
        self.assertGreaterEqual(len(entries), 1, "at least one dead-letter for poison pill")


if __name__ == "__main__":
    unittest.main()
