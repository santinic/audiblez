"""CLI tests.

`audiblez --help` and the no-arg path do not import the heavy TTS stack (core is
imported lazily inside cli_main after argument parsing), so these run anywhere the
package is importable — no models, ffmpeg, or network required. End-to-end epub
-> m4b conversion is covered by the CI smoke job and test_main.py.
"""
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_cli(*args):
    return subprocess.run(
        [sys.executable, '-m', 'audiblez.cli', *args],
        cwd=REPO_ROOT, capture_output=True, text=True)


# Bootstrap that runs cli_main() with available_backends() stubbed and the heavy
# audiblez.core module faked out, so the backend-resolution path can be exercised
# hermetically (no torch/kokoro/ffmpeg). It prints the backend that cli_main passed
# down to core.main on the last line as "BACKEND=<id>".
_BACKEND_BOOTSTRAP = """
import sys, types
fake_core = types.ModuleType('audiblez.core')
_chosen = {}
fake_core.main = lambda *a, **k: _chosen.update(k)
sys.modules['audiblez.core'] = fake_core
from audiblez import backends
backends.available_backends = lambda: %(avail)r
from audiblez.cli import cli_main
sys.argv = ['audiblez', '--backend', %(backend)r, 'dummy.epub']
# cli_main() ends in sys.exit(code); the backend kwarg is captured before that, so swallow
# SystemExit and report what was passed down.
try:
    cli_main()
except SystemExit:
    pass
print('BACKEND=' + _chosen.get('backend', '<unset>'))
"""


def run_cli_with_backends(backend, avail):
    """Resolve --backend `backend` against a stubbed available_backends() list `avail`."""
    script = _BACKEND_BOOTSTRAP % {'backend': backend, 'avail': avail}
    return subprocess.run(
        [sys.executable, '-c', script],
        cwd=REPO_ROOT, capture_output=True, text=True)


class CliHelpTest(unittest.TestCase):
    def test_help_lists_voices_and_usage(self):
        proc = run_cli('--help')
        out = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0)
        self.assertIn('usage:', out)
        self.assertIn('af_sky', out)

    def test_no_args_exits_nonzero_with_usage(self):
        proc = run_cli()
        out = proc.stdout + proc.stderr
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('usage:', out)

    def test_help_lists_backends(self):
        proc = run_cli('--help')
        out = proc.stdout + proc.stderr
        self.assertIn('--backend', out)
        for b in ('cpu', 'cuda', 'rocm', 'mps', 'mlx'):
            self.assertIn(b, out)

    def test_invalid_backend_rejected(self):
        proc = run_cli('--backend', 'bogus', 'dummy.epub')
        out = proc.stdout + proc.stderr
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('invalid choice', out)

    def test_help_mentions_recommended_default_and_presets(self):
        out = run_cli('--help').stdout
        self.assertIn('af_heart', out)        # the new default / recommended voice
        self.assertIn('(A)', out)             # quality grades are surfaced
        self.assertIn('af_warm', out)         # a curated preset blend

    def test_invalid_voice_rejected_cleanly(self):
        # A bad voice fails at the CLI (clean message), not deep in synthesis.
        proc = run_cli('--voice', 'af_nope', 'dummy.epub')
        out = proc.stdout + proc.stderr
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('Unknown voice', out)


class BackendSiblingSwapTest(unittest.TestCase):
    """A torch wheel exposes the GPU as EITHER 'cuda' or 'rocm', never both. Asking
    for the absent sibling should map to the present one — NOT silently fall to cpu."""

    def test_cuda_requested_maps_to_rocm_not_cpu(self):
        proc = run_cli_with_backends('cuda', ['cpu', 'rocm'])
        out = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, out)
        self.assertIn('BACKEND=rocm', out)          # used the sibling GPU
        self.assertNotIn('BACKEND=cpu', out)        # did NOT fall back to cpu
        self.assertIn("using rocm", out)            # announced the swap

    def test_rocm_requested_maps_to_cuda_not_cpu(self):
        proc = run_cli_with_backends('rocm', ['cpu', 'cuda'])
        out = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, out)
        self.assertIn('BACKEND=cuda', out)          # used the sibling GPU
        self.assertNotIn('BACKEND=cpu', out)        # did NOT fall back to cpu
        self.assertIn("using cuda", out)            # announced the swap


class ChapterSpecTest(unittest.TestCase):
    """The headless --chapters parser (parse_chapter_spec) and its CLI wiring."""

    def test_parses_comma_list(self):
        from audiblez.cli import parse_chapter_spec
        self.assertEqual(parse_chapter_spec('1,3,5'), [1, 3, 5])

    def test_parses_ranges_and_dedupes_sorted(self):
        from audiblez.cli import parse_chapter_spec
        self.assertEqual(parse_chapter_spec('1-4,7'), [1, 2, 3, 4, 7])

    def test_invalid_spec_raises_valueerror(self):
        from audiblez.cli import parse_chapter_spec
        with self.assertRaises(ValueError):
            parse_chapter_spec('abc')

    def test_invalid_chapters_arg_exits_nonzero(self):
        # A malformed --chapters spec is turned into parser.error() (clean message,
        # non-zero exit) rather than blowing up deep in synthesis.
        proc = run_cli('--chapters', 'abc', 'dummy.epub')
        out = proc.stdout + proc.stderr
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('chapter', out)


if __name__ == '__main__':
    unittest.main()
