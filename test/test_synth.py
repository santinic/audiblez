"""Guarded tests for core.build_synthesizer.

Importing audiblez.core needs the TTS stack, so these skip cleanly when it is
absent (and run in CI where deps are installed). No model/network is used — the
engines are faked.
"""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import numpy as np
    import soundfile
    import audiblez.core as core
    from audiblez import backends
    _ERR = None
except Exception as e:  # heavy deps (torch/kokoro/...) may be absent
    _ERR = e


@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class BuildSynthesizerTest(unittest.TestCase):
    def test_torch_path_passes_device_and_returns_numpy(self):
        fake_audio = np.zeros(4, dtype=np.float32)
        fake_pipeline = mock.MagicMock(return_value=[('g', 'p', fake_audio)])
        with mock.patch.object(core, 'KPipeline', return_value=fake_pipeline) as kp:
            synth = core.build_synthesizer('af_sky', 'mps')
            self.assertEqual(kp.call_args.kwargs.get('device'), 'mps')  # device threaded through
            out = synth('hello world', 1.0)
            self.assertIsInstance(out, list)
            self.assertTrue(out and all(isinstance(x, np.ndarray) for x in out))

    def test_rocm_maps_to_cuda_device(self):
        fake_pipeline = mock.MagicMock(return_value=[])
        with mock.patch.object(core, 'KPipeline', return_value=fake_pipeline) as kp:
            core.build_synthesizer('af_sky', 'rocm')
            self.assertEqual(kp.call_args.kwargs.get('device'), 'cuda')

    def test_preset_blend_resolves_lang_and_blend_string(self):
        fake_audio = np.zeros(4, dtype=np.float32)
        fake_pipeline = mock.MagicMock(return_value=[('g', 'p', fake_audio)])
        with mock.patch.object(core, 'KPipeline', return_value=fake_pipeline) as kp:
            synth = core.build_synthesizer('af_warm', 'cpu')  # Heart + Bella
            self.assertEqual(kp.call_args.kwargs.get('lang_code'), 'a')
            synth('hello world', 1.0)
            # The comma blend string is what both Kokoro engines average.
            self.assertEqual(fake_pipeline.call_args.kwargs.get('voice'), 'af_heart,af_bella')

    def test_weighted_blend_passes_repetition_string(self):
        fake_audio = np.zeros(4, dtype=np.float32)
        fake_pipeline = mock.MagicMock(return_value=[('g', 'p', fake_audio)])
        with mock.patch.object(core, 'KPipeline', return_value=fake_pipeline):
            synth = core.build_synthesizer('af_bella:60,af_heart:40', 'cpu')
            synth('hello world', 1.0)
            self.assertEqual(fake_pipeline.call_args.kwargs.get('voice'),
                             'af_bella,af_bella,af_bella,af_heart,af_heart')

    def test_unknown_voice_raises_valueerror(self):
        with self.assertRaises(ValueError):
            core.build_synthesizer('af_nonexistent', 'cpu')

    def test_unknown_backend_raises_valueerror(self):
        with self.assertRaises(ValueError):
            core.build_synthesizer('af_sky', 'bogus')

    def test_mlx_unavailable_raises_clear_runtimeerror(self):
        with mock.patch.object(backends, '_mlx_importable', return_value=False):
            with self.assertRaises(RuntimeError):
                core.build_synthesizer('af_sky', 'mlx')

    def test_torch_synth_coerces_tensor_segment_to_numpy(self):
        # Regression: kokoro may yield a torch.Tensor (possibly on GPU). build_synthesizer's closure must
        # to_numpy() each segment so np.concatenate/soundfile.write downstream work. Fake the pipeline to
        # emit a torch tensor and assert the closure returns a concatenatable numpy array.
        import torch
        fake_audio = torch.zeros(4, dtype=torch.float32)
        fake_pipeline = mock.MagicMock(return_value=[('g', 'p', fake_audio)])
        with mock.patch.object(core, 'KPipeline', return_value=fake_pipeline):
            synth = core.build_synthesizer('af_sky', 'cpu')
            out = synth('hello world', 1.0)
        self.assertTrue(out and all(isinstance(x, np.ndarray) for x in out))
        # np.concatenate is what core actually does with the segments; it must not raise on the coerced output.
        self.assertEqual(np.concatenate(out).shape, (4,))


def _drive_main(d, chapters, voice='af_heart', shutil_which=None, post_event=None,
                gen_segments=None, document_chapters=None, selected_chapters=None):
    """Run core.main() over `chapters` with all heavy I/O mocked (no epub/spaCy/ffmpeg/disk).

    `shutil_which` is the return of shutil.which (None => ffmpeg absent). `gen_segments`, if given,
    is used as gen_audio_segments' side_effect/return so callers can inspect the text it receives.
    `document_chapters` (defaults to `chapters`) is what find_document_chapters_and_extract_texts
    returns — pass a SUPERSET when exercising the headless --chapters index path so the resolved
    selection is observably a subset. `selected_chapters` is forwarded to main() verbatim.
    """
    if gen_segments is None:
        gen_segments = mock.Mock(return_value=[np.zeros(4, dtype=np.float32)])
    if document_chapters is None:
        document_chapters = chapters
    with mock.patch.object(core, 'load_spacy'), \
         mock.patch.object(core, 'epub') as ep, \
         mock.patch.object(core, 'extract_book_metadata', return_value=('Title', 'Author')), \
         mock.patch.object(core, 'find_cover', return_value=None), \
         mock.patch.object(core, 'find_document_chapters_and_extract_texts', return_value=document_chapters), \
         mock.patch.object(core, 'find_good_chapters', return_value=chapters), \
         mock.patch.object(core, 'set_espeak_library'), \
         mock.patch.object(core, 'build_synthesizer', return_value=lambda *a, **k: []), \
         mock.patch.object(core, 'gen_audio_segments', gen_segments), \
         mock.patch.object(core, 'shutil') as sh, \
         mock.patch.object(core, 'soundfile') as sf:
        ep.read_epub.return_value = object()
        sh.which.return_value = shutil_which
        sf.write.return_value = None
        core.main('book.epub', voice, False, 1.0, output_folder=d, post_event=post_event,
                  selected_chapters=selected_chapters)
    return gen_segments


def _chapter(text, index=0, name='Text/Chap01.xhtml'):
    return SimpleNamespace(extracted_text=text, chapter_index=index, get_name=lambda: name)


def _bare_synth():
    """A Kokoro-shaped synth: a bare callable carrying a no-op .close (the contract main relies on)."""
    s = lambda *a, **k: [np.zeros(4, dtype=np.float32)]
    s.close = lambda: None
    return s


@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class MainTerminalEventTest(unittest.TestCase):
    def test_core_finished_emitted_when_ffmpeg_absent(self):
        # GUI-hang regression: with ffmpeg missing (shutil.which -> None) main() still posts CORE_FINISHED,
        # even for a short/empty chapter set that produces no wavs — otherwise the GUI waits on it forever.
        events = []
        with tempfile.TemporaryDirectory() as d:
            _drive_main(d, [_chapter('short')],  # < 10 chars -> skipped, no wavs synthesized
                        shutil_which=None, post_event=lambda name, **kw: events.append(name))
        self.assertIn('CORE_FINISHED', events)


@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class IntroOnResumeTest(unittest.TestCase):
    def test_intro_not_reprepended_when_chapter1_already_exists(self):
        # Resume regression: an already-written chapter-1 wav (text len >= 10) already contains the prepended
        # 'Title – Author' intro, so the resumed synth of chapter 2 must NOT prepend it again.
        ch1 = _chapter('Chapter one body, definitely longer than ten characters.', index=0,
                       name='Text/Chap01.xhtml')
        ch2 = _chapter('Chapter two body, also clearly longer than ten characters.', index=1,
                       name='Text/Chap02.xhtml')
        seen = []
        gen = mock.Mock(side_effect=lambda synth, text, *a, **k: (seen.append(text),
                                                                  [np.zeros(4, dtype=np.float32)])[1])
        with tempfile.TemporaryDirectory() as d:
            voice_tag = 'af_heart'
            # Pre-create chapter 1's wav at the exact path main() computes, so it is skipped (resume).
            (Path(d) / f'book_chapter_1_{voice_tag}_Text_Chap01.xhtml.wav').write_bytes(b'RIFF')
            # The stub wav can't pass the real validity/render-signature gate (no audio, no .sig),
            # so force it 'valid & complete' to isolate the intro-on-resume behavior under test.
            with mock.patch.object(core, 'is_valid_chapter_wav', return_value=True), \
                 mock.patch.object(core, '_chapter_is_complete', return_value=True):
                _drive_main(d, [ch1, ch2], gen_segments=gen)

        # Only chapter 2 was synthesized (chapter 1 skipped as already-existing).
        self.assertEqual(len(seen), 1)
        synthesized = seen[0]
        self.assertNotIn('Title – Author', synthesized)  # intro was consumed by the skipped chapter 1
        self.assertTrue(synthesized.startswith('Chapter two body'))


@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class HeadlessChapterIndexTest(unittest.TestCase):
    """Headless --chapters passes 1-based indices into document_chapters; main() must resolve
    them to the right chapter objects (the GUI passes objects instead, which bypass this)."""

    def _run(self, selected, n_doc=3):
        docs = [_chapter(f'Body of chapter {k}, definitely longer than ten characters.',
                         index=k - 1, name=f'Text/Chap{k:02d}.xhtml') for k in range(1, n_doc + 1)]
        seen = []
        gen = mock.Mock(side_effect=lambda synth, text, *a, **k: (seen.append(text),
                                                                  [np.zeros(4, dtype=np.float32)])[1])
        with tempfile.TemporaryDirectory() as d:
            _drive_main(d, docs, gen_segments=gen, document_chapters=docs, selected_chapters=selected)
        return seen

    def test_indices_resolve_to_matching_chapters(self):
        seen = self._run([1, 3])
        self.assertEqual(len(seen), 2)
        # Order preserved; chapter 1 carries the prepended intro, chapter 3 does not.
        self.assertIn('Body of chapter 1', seen[0])
        self.assertIn('Body of chapter 3', seen[1])
        self.assertNotIn('Body of chapter 2', ' '.join(seen))

    def test_out_of_range_index_is_warned_and_dropped(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            seen = self._run([1, 99])
        self.assertEqual(len(seen), 1)
        self.assertIn('Body of chapter 1', seen[0])
        self.assertIn('out-of-range', buf.getvalue())
        self.assertIn('99', buf.getvalue())

    def test_all_indices_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            self._run([99])


# ─── MOSS resident-pipe engine (engine='llamacpp') ───────────────────────────────────────
# Hermetic: a FakePopen scripts the child's stdout JSON + writes a tiny WAV per request, so
# the protocol / failure-vs-death / circuit-breaker logic runs with no binary, GPU, or model.
# T4's test/test_moss.py is the comprehensive suite; these prove the closure + transport.

def _write_tiny_wav(path, n=4):
    soundfile.write(str(path), np.zeros(n, dtype=np.float32), core.MOSS_SAMPLE_RATE if _ERR is None else 24000)


if _ERR is None:
    import queue as _queue

    class _FakePipe:
        """Minimal stdin/stdout/stderr stand-in for a Popen child.

        Requests written to stdin are parsed by the owning FakePopen; each queued response
        line is delivered to readline(). A blocking Queue backs stdout so the closure's
        timeout-death path (which polls a deadline while a reader thread blocks) is exercised.
        """
        def __init__(self, popen, kind):
            self._popen = popen
            self._kind = kind
            self._q = _queue.Queue()
            self._closed = False

        # stdin
        def write(self, data):
            if self._kind == 'stdin':
                self._popen._on_stdin(data)
            return len(data)

        def flush(self):
            pass

        # stdout/stderr
        def readline(self):
            if self._kind != 'stdout':
                return b''
            item = self._q.get()  # blocks until the FakePopen enqueues a line (or EOF sentinel)
            return b'' if item is None else item

        def feed(self, line):
            self._q.put(line)

        def eof(self):
            self._q.put(None)

        def close(self):
            self._closed = True

    class FakePopen:
        """Scripted fake of subprocess.Popen for MossProcess.

        ``script`` is a callable(request_dict, wav_out_path) -> list[response_dict|None] (None
        => EOF/death). It is invoked per synth request; ``ready`` is emitted on spawn. ``die_after``
        kills the child (poll() != None) after N requests to simulate an infra death.
        """
        def __init__(self, *, script=None, ready=True, die_after=None, emit_ready=True):
            self._script = script or (lambda req, wav: [{'v': 1, 'id': req['id'], 'status': 'ok',
                                                         'wav_out': req['wav_out'],
                                                         'frames': 4, 'sample_rate': 24000}])
            self._die_after = die_after
            self._returncode = None
            self.requests = 0
            self.stdin = _FakePipe(self, 'stdin')
            self.stdout = _FakePipe(self, 'stdout')
            self.stderr = _FakePipe(self, 'stderr')
            if emit_ready and ready:
                self.stdout.feed(b'{"v":1,"status":"ready"}\n')

        def _on_stdin(self, data):
            text = data.decode('utf-8') if isinstance(data, bytes) else data
            for raw in text.splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                msg = json.loads(raw)
                if msg.get('op') == 'shutdown':
                    self._returncode = 0
                    self.stdout.eof()
                    continue
                self.requests += 1
                wav_out = msg.get('wav_out')
                responses = self._script(msg, wav_out)
                for resp in responses:
                    if resp is None:
                        self._returncode = 1
                        self.stdout.eof()
                        return
                    # An 'ok' response means the child wrote the WAV; do it here.
                    if resp.get('status') == 'ok' and wav_out:
                        _write_tiny_wav(wav_out, resp.get('frames', 4))
                    self.stdout.feed((json.dumps(resp) + '\n').encode('utf-8'))
                if self._die_after is not None and self.requests >= self._die_after:
                    self._returncode = 1
                    self.stdout.eof()

        def poll(self):
            return self._returncode

        def wait(self, timeout=None):
            return self._returncode or 0

        def kill(self):
            self._returncode = -9
            self.stdout.eof()


@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class MossProcessTransportTest(unittest.TestCase):
    def _proc(self, **kw):
        self._tmp = tempfile.TemporaryDirectory()  # MossProcess makes its own subdir under this
        return core.MossProcess(backbone='bb.gguf', decoder='dec.gguf',
                                work_dir=self._tmp.name,
                                popen_factory=lambda *a, **k: FakePopen(**kw),
                                ready_timeout=5.0)

    def test_ok_response_returns_audio_and_sends_per_sentence_request(self):
        captured = {}

        def script(req, wav):
            captured['req'] = req
            return [{'v': 1, 'id': req['id'], 'status': 'ok', 'wav_out': wav,
                     'frames': 4, 'sample_rate': 24000}]
        proc = self._proc(script=script)
        proc.start()
        out = proc.synth('hello world', timeout=5.0)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.dtype, np.float32)
        # Request honors the spec shape: pinned seed + sampling, no speed field.
        self.assertEqual(captured['req']['op'], 'synth')
        self.assertEqual(captured['req']['seed'], core.MOSS_SEED)
        self.assertNotIn('speed', captured['req'])
        self.assertEqual(set(captured['req']['sampling']), set(core.MOSS_SAMPLING))
        proc.close()

    def test_command_has_t1_serve_invocation_flags(self):
        # T1's --serve binary (moss-serve-json-v1) requires these exact flags at spawn;
        # the fixed frame caps size the resident contexts. Regression-guard the invocation.
        proc = core.MossProcess(backbone='bb.gguf', decoder='dec.gguf', binary='llama-moss-tts',
                                work_dir=tempfile.mkdtemp())
        cmd = proc._command()
        self.assertEqual(cmd[:2], ['llama-moss-tts', '--serve'])
        for flag in ('--language', 'en', '-ngl', '-1', '--max-prompt-frames', '512',
                     '--max-raw-frames', '768', '--audio-decoder-model'):
            self.assertIn(flag, cmd)
        self.assertNotIn('--reference-audio', cmd)  # no clone ref -> no encoder/reference flags

    def test_command_adds_clone_flags_when_cloning(self):
        proc = core.MossProcess(backbone='bb', decoder='dec', encoder='enc.gguf',
                                clone_ref='/ref.wav', work_dir=tempfile.mkdtemp())
        cmd = proc._command()
        self.assertIn('--audio-encoder-model', cmd)
        self.assertIn('--reference-audio', cmd)
        self.assertIn('/ref.wav', cmd)

    def test_error_status_raises_mosserror(self):
        proc = self._proc(script=lambda req, wav: [{'v': 1, 'id': req['id'], 'status': 'error',
                                                    'code': 'gen_failed', 'message': 'boom'}])
        proc.start()
        with self.assertRaises(core.MossError):
            proc.synth('x', timeout=5.0)
        self.assertTrue(proc.alive())  # error keeps the child HEALTHY
        proc.close()

    def test_bad_request_and_empty_audio_are_permanent(self):
        for code in ('bad_request', 'empty_audio'):
            proc = self._proc(script=lambda req, wav, c=code: [{'v': 1, 'id': req['id'],
                                                               'status': 'error', 'code': c}])
            proc.start()
            with self.assertRaises(core.MossPermanentError):
                proc.synth('x', timeout=5.0)
            proc.close()

    def test_ok_with_zero_frames_is_permanent(self):
        # Spec forbids ok-with-zero; treat as permanent so np.zeros(0) is never cached.
        proc = self._proc(script=lambda req, wav: [{'v': 1, 'id': req['id'], 'status': 'ok',
                                                    'wav_out': wav, 'frames': 0}])
        proc.start()
        with self.assertRaises(core.MossPermanentError):
            proc.synth('x', timeout=5.0)
        proc.close()

    def test_eof_raises_mossdeath(self):
        proc = self._proc(script=lambda req, wav: [None])  # EOF instead of a response
        proc.start()
        with self.assertRaises(core.MossDeath):
            proc.synth('x', timeout=5.0)
        proc.close()

    def test_spawn_failure_aborts_loud(self):
        def boom(*a, **k):
            raise FileNotFoundError('no such binary')
        proc = core.MossProcess(backbone='bb', decoder='dec',
                                work_dir=tempfile.gettempdir(), popen_factory=boom)
        with self.assertRaises(core.MossRunAborted):
            proc.start()

    # ── wedged-but-alive child: the unbounded-stdin-write deadlock (engine review HIGH) ──
    # bufsize=0 means a write to a full stdin pipe blocks at the OS level with no timeout, so a
    # child that is alive (poll() is None) but not draining its stdin must NOT hang synth/close
    # forever — _write bounds the write on a watchdog thread and the caller treats a stuck write
    # as a death. A never-set Event makes the fake child's write block indefinitely.

    @staticmethod
    def _wedged_proc(killed_flag):
        blocked = threading.Event()  # never set -> write() blocks forever

        class _Stdin:
            def write(self, data): blocked.wait()
            def flush(self): pass

        class _Proc:
            def __init__(self): self.stdin = _Stdin(); self.stdout = None
            def poll(self): return None          # alive
            def kill(self): killed_flag.append(True)
            def wait(self, timeout=None): return 0
        return _Proc()

    def _moss(self):
        return core.MossProcess(backbone='bb', decoder='dec', work_dir=tempfile.mkdtemp(),
                                popen_factory=lambda *a, **k: None)

    def test_write_returns_false_when_child_wedged(self):
        proc = self._moss()
        proc.proc = self._wedged_proc([])
        self.assertFalse(proc._write(b'x\n', timeout=0.2))  # bounded, not a hard hang

    def test_synth_raises_mossdeath_when_write_wedged(self):
        proc = self._moss()
        proc.proc = self._wedged_proc([])
        with self.assertRaises(core.MossDeath):
            proc.synth('hello', timeout=0.2)

    def test_close_kills_wedged_child_instead_of_hanging(self):
        killed = []
        proc = self._moss()
        proc.proc = self._wedged_proc(killed)
        with mock.patch.object(core, 'MOSS_SHUTDOWN_WRITE_TIMEOUT', 0.2):
            proc.close()  # must escalate to kill, not block on a graceful exit it can't perform
        self.assertEqual(killed, [True])


@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class MossClosureTest(unittest.TestCase):
    """The _build_llamacpp_synth closure: failure/death/poison/circuit-breaker policy."""

    def _build(self, factory, clock=None):
        kw = dict(moss_factory=factory, work_dir=tempfile.mkdtemp())
        if clock is not None:
            kw['clock'] = clock
        return core._build_llamacpp_synth('af_voice', None, **kw)

    def _ok_proc(self):
        return core.MossProcess(backbone='bb', decoder='dec', work_dir=tempfile.mkdtemp(),
                                popen_factory=lambda *a, **k: FakePopen(), ready_timeout=5.0)

    def test_batch_marker_splits_into_one_request_per_sentence(self):
        synth = self._build(self._ok_proc)
        out = synth('one\n\n\ntwo\n\n\nthree', 1.0)
        self.assertEqual(len(out), 3)  # one aligned array per sentence
        self.assertTrue(all(isinstance(a, np.ndarray) for a in out))
        synth.close()

    def test_death_then_restart_redispatches_same_sentence(self):
        # restart() re-invokes the SAME MossProcess._popen_factory, so make that factory
        # stateful: spawn #1 dies on its first request, spawn #2 (the restart) succeeds.
        spawns = {'n': 0}

        def popen_factory(*a, **k):
            spawns['n'] += 1
            dying = spawns['n'] == 1
            return FakePopen(script=(lambda req, wav: [None]) if dying else None)

        def factory():
            return core.MossProcess(backbone='bb', decoder='dec', work_dir=tempfile.mkdtemp(),
                                    popen_factory=popen_factory, ready_timeout=5.0)
        synth = self._build(factory)
        out = synth('hello', 1.0)  # 1st death -> restart -> 2nd child returns audio
        self.assertEqual(len(out), 1)
        self.assertGreaterEqual(spawns['n'], 2)  # the child was restarted
        synth.close()

    def test_two_consecutive_deaths_poison_pill_permanent(self):
        # Every child dies on its first request -> K=2 -> MossPermanentError (dead-letter).
        def factory():
            return core.MossProcess(backbone='bb', decoder='dec', work_dir=tempfile.mkdtemp(),
                                    popen_factory=lambda *a, **k: FakePopen(
                                        script=lambda req, wav: [None]),
                                    ready_timeout=5.0)
        synth = self._build(factory)
        with self.assertRaises(core.MossPermanentError):
            synth('poison', 1.0)
        synth.close()

    def test_circuit_breaker_burst_aborts_run(self):
        # Many deaths bunched in time -> > MOSS_MAX_RESTARTS_IN_WINDOW within the window -> abort.
        # Each sentence dies once (1 restart) then the restarted child also dies on the SAME
        # sentence -> poison; but the restart marks accumulate fast under a frozen clock.
        clock_val = [1000.0]

        def factory():
            return core.MossProcess(backbone='bb', decoder='dec', work_dir=tempfile.mkdtemp(),
                                    popen_factory=lambda *a, **k: FakePopen(
                                        script=lambda req, wav: [None]),
                                    ready_timeout=5.0)
        synth = self._build(factory, clock=lambda: clock_val[0])  # frozen clock: all in-window
        with self.assertRaises(core.MossRunAborted):
            # Each poison sentence contributes a restart mark; the frozen clock keeps them all
            # inside the window, so after enough sentences the breaker trips.
            for i in range(12):
                try:
                    synth(f'sentence {i}', 1.0)
                except core.MossPermanentError:
                    continue  # poison dead-letter; keep going until the breaker trips
        synth.close()

    def test_sparse_deaths_do_not_abort(self):
        # The SAME number of restarts spread far apart in time must NOT trip the breaker.
        clock_val = [1000.0]

        def factory():
            # Child dies once then the restart succeeds: exactly one restart mark per call.
            first = {'flag': True}

            def script(req, wav):
                if first['flag']:
                    first['flag'] = False
                    return [None]
                return [{'v': 1, 'id': req['id'], 'status': 'ok', 'wav_out': wav, 'frames': 4}]
            return core.MossProcess(backbone='bb', decoder='dec', work_dir=tempfile.mkdtemp(),
                                    popen_factory=lambda *a, **k: FakePopen(script=script),
                                    ready_timeout=5.0)
        synth = self._build(factory, clock=lambda: clock_val[0])
        for i in range(10):
            clock_val[0] += 1000.0  # advance >> window between sentences: never bunched
            out = synth(f'sentence {i}', 1.0)
            self.assertEqual(len(out), 1)  # each recovers; no abort
        synth.close()


@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class MossAbortNotSwallowedTest(unittest.TestCase):
    """The circuit-breaker abort must propagate THROUGH the broad excepts in
    _synth_one_or_silence / _synth_batch / gen_audio_segments (it's a BaseException), not be
    caught and degraded to silence — that's the banned all-silence-reported-as-success class."""

    def _aborting_synth(self):
        def synth(text, speed):
            raise core.MossRunAborted('circuit breaker tripped')
        synth.close = lambda: None
        return synth

    def test_synth_one_or_silence_does_not_swallow_abort(self):
        synth = self._aborting_synth()
        with self.assertRaises(core.MossRunAborted):
            core._synth_one_or_silence(synth, 'x', 1.0, 2, 'chapter 1', None)

    def test_gen_audio_segments_propagates_abort(self):
        synth = self._aborting_synth()
        with mock.patch.object(core, 'load_spacy'), \
             mock.patch.object(core, '_split_into_sentences', return_value=['a', 'b']):
            with self.assertRaises(core.MossRunAborted):
                core.gen_audio_segments(synth, 'a b', voice='af_heart', speed=1.0, lang_code='a')


@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class MossCacheFieldsTest(unittest.TestCase):
    def test_cache_fields_for_moss_include_seed_sampling_and_repo(self):
        info = backends.BackendInfo('moss', 'MOSS', 'llamacpp', None)
        with mock.patch.dict(core.backends.BACKENDS, {'moss': info}):
            fields = core._cache_key_fields('moss', 'af_voice', 1.5, precision='fp16')
        self.assertEqual(fields['engine'], 'llamacpp')
        self.assertEqual(fields['seed'], core.MOSS_SEED)
        self.assertIn('sampling_sig', fields)
        # repo_id comes from the SINGLE source of truth (backends.moss_repo_id() -> 'moss-gguf:').
        self.assertTrue(fields['repo_id'].startswith('moss-gguf:'))
        # MOSS cache is speed-agnostic (speed handled by atempo at assembly) -> pinned to 1.0.
        self.assertEqual(fields['speed'], 1.0)
        # precision is irrelevant to MOSS -> normalised, never the fp16 the caller passed.
        self.assertEqual(fields['precision'], 'fp32')

    def test_sampling_sig_changes_when_a_param_changes(self):
        base = core._moss_sampling_sig()
        drifted = dict(core.MOSS_SAMPLING)
        drifted['audio_top_k'] = 99
        self.assertNotEqual(base, core._moss_sampling_sig(drifted))

    def test_clone_ref_changes_the_cache_key_via_voice_axis(self):
        # The clone reference IS the voice (backends.clone_voice_id); two different refs must
        # produce different cache fields, else a re-run serves the prior clone's audio.
        info = backends.BackendInfo('moss', 'MOSS', 'llamacpp', None)
        with mock.patch.dict(core.backends.BACKENDS, {'moss': info}):
            with tempfile.NamedTemporaryFile(suffix='.wav') as a, \
                 tempfile.NamedTemporaryFile(suffix='.wav') as b:
                a.write(b'AAAA'); a.flush()
                b.write(b'BBBB'); b.flush()
                fa = core._cache_key_fields('moss', 'af_voice', 1.0, clone_ref=a.name)
                fb = core._cache_key_fields('moss', 'af_voice', 1.0, clone_ref=b.name)
                ftext = core._cache_key_fields('moss', 'af_voice', 1.0)
        self.assertNotEqual(fa['voice'], fb['voice'])   # different clip -> different voice id
        # No clone ref -> the fixed MOSS_DEFAULT_VOICE sentinel, NOT the inert Kokoro voice spec
        # (slice 2 / #8): MOSS ignores the Kokoro voice, so the key must not fragment on it.
        self.assertEqual(ftext['voice'], core.MOSS_DEFAULT_VOICE)


@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class MossSpeedAtempoTest(unittest.TestCase):
    """Speed must NOT be silently dropped for MOSS: it is applied via ffmpeg atempo at chapter
    assembly (MOSS synthesizes at 1.0), and must NOT double-stretch Kokoro (which baked it in)."""

    def test_atempo_filter_chains_for_out_of_range_speed(self):
        # atempo handles 0.5-2.0 per stage; 3.0 must chain (2.0 * 1.5).
        f = core._atempo_filter(3.0)
        self.assertEqual(f.count('atempo='), 2)

    def test_apply_atempo_noop_at_speed_one(self):
        with mock.patch.object(core, 'shutil'), mock.patch.object(core, 'subprocess') as sp:
            core._apply_atempo('x.wav', 1.0)
        sp.run.assert_not_called()  # 1.0 is a no-op; no ffmpeg invoked

    def test_main_applies_atempo_for_moss_not_for_torch(self):
        info_moss = backends.BackendInfo('moss', 'MOSS', 'llamacpp', None)
        ch = _chapter('A real chapter body, definitely much longer than ten characters.')
        for backend, expect_atempo in (('moss', True), ('cpu', False)):
            with self.subTest(backend=backend):
                calls = self._drive_main_backend(backend, ch, info_moss)
                if expect_atempo:
                    self.assertTrue(calls, 'MOSS run must invoke atempo to honor --speed')
                    self.assertEqual(calls[0][1], 1.5)  # the requested speed
                else:
                    self.assertEqual(calls, [])  # torch baked speed in; never double-stretch

    def _drive_main_backend(self, backend, ch, info_moss):
        """Run core.main once with `backend` wired in; return the (path, speed) atempo calls."""
        calls = []
        gen = mock.Mock(return_value=[np.zeros(4, dtype=np.float32)])
        from audiblez import doctor as _doctor
        with mock.patch.object(_doctor, 'run_checks', return_value=[]), \
             mock.patch.dict(core.backends.BACKENDS, {'moss': info_moss}), \
             mock.patch.object(core, '_apply_atempo', side_effect=lambda p, s: calls.append((p, s))), \
             mock.patch.object(core, 'load_spacy'), \
             mock.patch.object(core, 'epub') as ep, \
             mock.patch.object(core, 'extract_book_metadata', return_value=('T', 'A')), \
             mock.patch.object(core, 'find_cover', return_value=None), \
             mock.patch.object(core, 'find_document_chapters_and_extract_texts', return_value=[ch]), \
             mock.patch.object(core, 'find_good_chapters', return_value=[ch]), \
             mock.patch.object(core, 'set_espeak_library'), \
             mock.patch.object(core, 'build_synthesizer', return_value=_bare_synth()), \
             mock.patch.object(core, 'gen_audio_segments', gen), \
             mock.patch.object(core, 'gpu'), \
             mock.patch.object(core, 'shutil') as sh, \
             mock.patch.object(core, 'soundfile'):
            ep.read_epub.return_value = object()
            sh.which.return_value = None  # no ffmpeg -> skip m4b; isolate the atempo call
            with tempfile.TemporaryDirectory() as d:
                core.main('book.epub', 'af_heart', False, 1.5, output_folder=d, backend=backend)
        return calls


@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class SynthParamsBuildTest(unittest.TestCase):
    def test_build_synthesizer_accepts_params_object(self):
        fake_pipeline = mock.MagicMock(return_value=[('g', 'p', np.zeros(4, dtype=np.float32))])
        with mock.patch.object(core, 'KPipeline', return_value=fake_pipeline), \
             mock.patch.object(core, 'set_espeak_library'):
            synth = core.build_synthesizer(params=core.SynthParams(voice='af_sky', backend='cpu'))
            self.assertTrue(callable(synth))
            self.assertTrue(hasattr(synth, 'close'))  # bare-callable contract + teardown hook
            synth.close()  # no-op for Kokoro, must not raise

    def test_legacy_kwargs_still_work(self):
        fake_pipeline = mock.MagicMock(return_value=[])
        with mock.patch.object(core, 'KPipeline', return_value=fake_pipeline), \
             mock.patch.object(core, 'set_espeak_library'):
            synth = core.build_synthesizer('af_sky', 'cpu', precision='fp32')
            self.assertTrue(hasattr(synth, 'close'))

    def test_dispatches_to_moss_for_llamacpp_engine(self):
        info = backends.BackendInfo('moss', 'MOSS', 'llamacpp', None)
        sentinel = _bare_synth()
        with mock.patch.dict(core.backends.BACKENDS, {'moss': info}), \
             mock.patch.object(core, '_build_llamacpp_synth', return_value=sentinel) as bl:
            out = core.build_synthesizer(params=core.SynthParams(voice='af_voice', backend='moss'))
        self.assertIs(out, sentinel)
        self.assertTrue(bl.called)  # routed to the MOSS builder, not Kokoro/espeak


if __name__ == '__main__':
    unittest.main()
