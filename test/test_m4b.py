"""Hermetic tests for the ffmpeg/m4b assembly path (create_m4b, create_index_file).

No real ffmpeg/ffprobe runs: subprocess.run and probe_duration are mocked, so these assert
the FFMETADATA bytes, gap-free 1..N chapter numbering, ABSOLUTE concat paths (regression for
the relative -o bug), and the error/cleanup contract. Importing audiblez.core needs the TTS
stack, so the suite skips cleanly when it is absent (and runs in CI where deps are installed).
"""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import numpy as np
    import audiblez.core as core
    _ERR = None
except Exception as e:  # heavy deps (torch/kokoro/spacy) may be absent
    _ERR = e


@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class CreateIndexFileTest(unittest.TestCase):
    def test_sequential_numbering_and_ffmetadata_escaping(self):
        with tempfile.TemporaryDirectory() as d:
            files = [Path(d) / 'a.wav', Path(d) / 'b.wav']
            with mock.patch.object(core, 'probe_duration', return_value=2.0):
                path = core.create_index_file('Ti=tle', 'Cre;ator', files, d)
            text = Path(path).read_text(encoding='utf-8')
            self.assertIn('title=Ti\\=tle', text)      # title/artist run through _escape_ffmetadata
            self.assertIn('artist=Cre\\;ator', text)
            self.assertIn('title=Chapter 1', text)     # numbering starts at 1, gap-free
            self.assertIn('title=Chapter 2', text)
            self.assertIn('START=0', text)
            self.assertIn('END=2000', text)            # 2.0s -> 2000ms
            self.assertIn('START=2000', text)          # cumulative offset
            self.assertIn('END=4000', text)


@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class CreateM4bTest(unittest.TestCase):
    def test_concat_list_uses_absolute_paths_with_relative_output_folder(self):
        # Regression for the relative -o bug: with a RELATIVE output_folder the concat entries must still
        # be absolute, because ffmpeg resolves relative entries against the list-file's dir (not the cwd).
        captured = {}

        def fake_run(args, **kwargs):
            wav_list = args[args.index('-i') + 1]  # ffmpeg ... -f concat ... -i <wav_list> ...
            captured['list'] = Path(wav_list).read_text()
            return mock.Mock(returncode=0, stderr='')

        prev = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            try:
                os.chdir(d)
                rel = 'out'
                os.mkdir(rel)
                wavs = [Path(rel) / 'c1.wav', Path(rel) / 'c2.wav']  # relative paths trigger the bug
                for w in wavs:
                    w.write_bytes(b'RIFF')
                with mock.patch.object(core, 'probe_duration', return_value=1.0), \
                     mock.patch('audiblez.core.subprocess.run', side_effect=fake_run):
                    core.create_m4b(wavs, 'book.epub', b'', rel)
            finally:
                os.chdir(prev)

        lines = [ln for ln in captured['list'].splitlines() if ln.startswith("file '")]
        self.assertEqual(len(lines), 2)
        for ln in lines:
            p = ln[len("file '"):-1]
            self.assertTrue(os.path.isabs(p), f'concat path is not absolute: {p}')

    def test_raises_runtimeerror_and_cleans_temp_on_ffmpeg_failure(self):
        with tempfile.TemporaryDirectory() as d:
            wavs = [Path(d) / 'c1.wav']
            wavs[0].write_bytes(b'RIFF')
            with mock.patch.object(core, 'probe_duration', return_value=1.0), \
                 mock.patch('audiblez.core.subprocess.run',
                            return_value=mock.Mock(returncode=1, stderr='boom')):
                with self.assertRaises(RuntimeError):
                    core.create_m4b(wavs, 'book.epub', b'', d)
            # the wav-list temp file is removed in the finally block even on failure
            self.assertFalse((Path(d) / 'book_wav_list.txt').exists())

    def test_concat_list_escapes_apostrophe_in_wav_filename(self):
        # Regression: a wav filename containing an apostrophe must be written to the concat list using
        # ffmpeg's close/escape/reopen form ('\'') so the single-quoted `file '<path>'` entry parses.
        captured = {}

        def fake_run(args, **kwargs):
            wav_list = args[args.index('-i') + 1]
            captured['list'] = Path(wav_list).read_text()
            return mock.Mock(returncode=0, stderr='')

        with tempfile.TemporaryDirectory() as d:
            wav = Path(d) / "ch1 o'brien.wav"  # apostrophe in the chapter name
            wav.write_bytes(b'RIFF')
            with mock.patch.object(core, 'probe_duration', return_value=1.0), \
                 mock.patch('audiblez.core.subprocess.run', side_effect=fake_run):
                core.create_m4b([wav], 'book.epub', b'', d)

        # The escape form is exactly what _escape_concat_path produces for the resolved path.
        expected = core._escape_concat_path(wav.resolve())
        self.assertIn("'\\''", expected)                       # apostrophe became close/escape/reopen
        self.assertIn(f"file '{expected}'\n", captured['list'])  # and that is the concat entry verbatim


@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class ChapterFilenameSanitizationTest(unittest.TestCase):
    def test_malicious_get_name_yields_safe_wav_path(self):
        # Regression: get_name() comes from the attacker-controllable epub manifest href. A raw newline or
        # slash would break out of the ffmpeg concat list, so main() whitelist-sanitizes it into the wav name.
        # Drive main() with everything heavy mocked; capture the chapter_wav_path passed to soundfile.write.
        written = []
        chapter = SimpleNamespace(
            extracted_text='This is a long enough chapter body to be synthesized.',
            chapter_index=0,
            get_name=lambda: 'chapter/1\ninjected')

        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(core, 'load_spacy'), \
                 mock.patch.object(core, 'epub') as ep, \
                 mock.patch.object(core, 'extract_book_metadata', return_value=('T', 'A')), \
                 mock.patch.object(core, 'find_cover', return_value=None), \
                 mock.patch.object(core, 'find_document_chapters_and_extract_texts', return_value=[chapter]), \
                 mock.patch.object(core, 'find_good_chapters', return_value=[chapter]), \
                 mock.patch.object(core, 'set_espeak_library'), \
                 mock.patch.object(core, 'build_synthesizer', return_value=lambda *a, **k: []), \
                 mock.patch.object(core, 'gen_audio_segments',
                                   return_value=[np.zeros(4, dtype=np.float32)]), \
                 mock.patch.object(core, 'shutil') as sh, \
                 mock.patch.object(core, 'soundfile') as sf:
                ep.read_epub.return_value = object()
                sh.which.return_value = None  # no ffmpeg -> no m4b assembly, isolate the wav-path logic
                sf.write.side_effect = lambda path, *a, **k: written.append(Path(path))
                core.main('book.epub', 'af_heart', False, 1.0, output_folder=d)

        self.assertEqual(len(written), 1)
        name = written[0].name
        self.assertNotIn('\n', name)   # newline stripped by the re.sub whitelist
        self.assertNotIn('/', name)    # slash stripped too (would escape the concat list / path)
        self.assertIn('chapter_1_injected', name)  # 'chapter/1\ninjected' -> 'chapter_1_injected'


if __name__ == '__main__':
    unittest.main()
