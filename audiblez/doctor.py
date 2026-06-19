# -*- coding: utf-8 -*-
"""Preflight checks — fail fast and LOUD before an hour of synthesis.

Pure and import-light by design: this module imports only stdlib plus
:mod:`audiblez.backends` (which itself imports nothing heavy). It never imports
``torch``/``kokoro``/``spacy``/``phonemizer`` at module scope, so ``audiblez
--doctor`` stays fast and works even when the heavy TTS stack is half-installed —
which is exactly the situation it exists to diagnose. The optional ``--deep``
check loads the real model and is the only path that imports :mod:`audiblez.core`.
"""
import os
import platform
import shutil
import subprocess
import importlib.util
from glob import glob
from pathlib import Path
from dataclasses import dataclass

from audiblez import backends

# ANSI colours, matching the red used elsewhere in core.py.
_COLORS = {'ok': '\033[92m', 'warn': '\033[93m', 'fail': '\033[91m'}
_MARKS = {'ok': '[ OK ]', 'warn': '[WARN]', 'fail': '[FAIL]'}
_RESET = '\033[0m'

SPACY_MODEL = 'xx_ent_wiki_sm'


@dataclass
class CheckResult:
    name: str
    status: str   # 'ok' | 'warn' | 'fail'
    detail: str


def find_espeak_library() -> str:
    """Resolve the espeak-ng shared-library path, or raise RuntimeError.

    Pure path resolution (``ESPEAK_LIBRARY`` env override / platform glob /
    Homebrew Cellar). Does NOT import phonemizer or register anything — it is both
    the preflight espeak check and the resolver used by
    :func:`audiblez.core.set_espeak_library`. Raises with an actionable, OS-specific
    install hint instead of the old swallow-and-continue.
    """
    override = os.environ.get('ESPEAK_LIBRARY')
    if override:
        if not Path(override).exists():
            raise RuntimeError(f'ESPEAK_LIBRARY={override!r} is set but the file does not exist')
        return override

    system = platform.system()
    if system == 'Darwin':
        try:
            cellar = Path(subprocess.check_output(['brew', '--cellar'], text=True).strip())
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "Cannot locate the Homebrew Cellar (is 'brew' installed and on PATH?). "
                'Install espeak-ng with: brew install espeak-ng') from e
        lib = next(iter(glob(str(cellar / 'espeak-ng' / '*' / 'lib' / '*.dylib'))), None)
        if not lib:
            raise RuntimeError('No espeak-ng library found. Install it with: brew install espeak-ng')
        return lib
    if system == 'Linux':
        libs = (glob('/usr/lib/*/libespeak-ng*')
                + glob('/usr/lib/libespeak-ng*')
                + glob('/usr/local/lib/libespeak-ng*'))
        if not libs:
            raise RuntimeError('No espeak-ng library found. Install it with: sudo apt install espeak-ng')
        return libs[0]
    if system == 'Windows':
        lib = next(iter(glob('C:\\Program Files*\\eSpeak NG\\libespeak-ng.dll')), None)
        if not lib:
            raise RuntimeError(
                'No espeak-ng library found. Install eSpeak NG from '
                'https://github.com/espeak-ng/espeak-ng/releases')
        return lib
    raise RuntimeError(
        f'Unsupported OS {system!r}; set ESPEAK_LIBRARY to the espeak-ng library path manually')


def check_ffmpeg() -> CheckResult:
    path = shutil.which('ffmpeg')
    return CheckResult('ffmpeg', 'ok' if path else 'fail',
                       path or 'not found on PATH — install ffmpeg to write .m4b/.wav output')


def check_ffprobe() -> CheckResult:
    # ffprobe only sharpens chapter-marker timing; its absence degrades, never fails.
    path = shutil.which('ffprobe')
    return CheckResult('ffprobe', 'ok' if path else 'warn',
                       path or 'not found — chapter markers will fall back to file size')


def check_espeak() -> CheckResult:
    try:
        return CheckResult('espeak-ng', 'ok', find_espeak_library())
    except RuntimeError as e:
        return CheckResult('espeak-ng', 'fail', str(e))


def check_kokoro() -> CheckResult:
    if importlib.util.find_spec('kokoro') is None:
        return CheckResult('kokoro', 'fail', 'not installed (pip install kokoro==0.9.4)')
    return CheckResult('kokoro', 'ok', 'installed')


def check_spacy_model(model: str = SPACY_MODEL) -> CheckResult:
    if importlib.util.find_spec('spacy') is None:
        return CheckResult('spaCy', 'fail', 'not installed (pip install spacy)')
    if importlib.util.find_spec(model) is None:
        return CheckResult(f'spaCy model {model}', 'warn',
                           'not installed — audiblez will auto-download it on first run')
    return CheckResult(f'spaCy model {model}', 'ok', 'installed')


def check_moss() -> CheckResult:
    """Report MOSS (llamacpp engine) availability — binary + the three GGUF weights.

    Always runs in :func:`run_checks` (not only under ``-b moss``) so a user preflighting a
    long run sees *which* piece is missing even when MOSS won't auto-select (PRD story 18).
    Import-light: resolves paths via :func:`audiblez.backends.moss_paths` and only stats
    files — never spawns the binary or loads a model (that is ``--doctor --deep``).

    - binary (:data:`audiblez.backends._MOSS_BIN_NAME`, NOT ``llama-cli``) + every required
      GGUF present → ``ok``.
    - encoder absent is a ``warn`` (needed only for voice cloning), not a fail.
    - anything required missing → ``warn`` (MOSS is optional; Kokoro is the fallback), with
      the exact missing pieces named.
    """
    paths = backends.moss_paths()
    missing = []
    if paths['binary'] is None:
        missing.append(f"binary '{backends._MOSS_BIN_NAME}' not on PATH (set AUDIBLEZ_MOSS_BIN)")
    for kind in backends.MOSS_REQUIRED:
        if paths[kind] is None:
            missing.append(f'{kind} GGUF missing (set AUDIBLEZ_MOSS_{kind.upper()})')
    if missing:
        return CheckResult('MOSS (llamacpp)', 'warn',
                           'not usable — ' + '; '.join(missing) + ' — Kokoro will be used instead')
    detail = f"binary {paths['binary']}"
    if paths['encoder'] is None:
        detail += ' — encoder GGUF absent (voice cloning unavailable; set AUDIBLEZ_MOSS_ENCODER)'
    return CheckResult('MOSS (llamacpp)', 'ok', detail)


def check_backend(backend: str) -> CheckResult:
    if backend not in backends.BACKENDS:
        return CheckResult(f'backend {backend}', 'fail',
                           f'unknown; choose from {", ".join(backends.BACKEND_IDS)}')
    info = backends.BACKENDS[backend]
    avail = backends.available_backends()
    if info.engine == 'torch':
        if importlib.util.find_spec('torch') is None:
            return CheckResult(f'backend {backend}', 'fail', 'torch not installed (pip install torch)')
        if backend not in avail:
            return CheckResult(f'backend {backend}', 'fail',
                               f'device not available here (usable: {", ".join(avail)})')
        return CheckResult(f'backend {backend}', 'ok', info.label)
    if info.engine == 'llamacpp':
        # Forced -b moss: a missing piece is a FAIL (the user explicitly asked for MOSS), and
        # names exactly what to install. Encoder absence only warns (cloning-only).
        paths = backends.moss_paths()
        miss = []
        if paths['binary'] is None:
            miss.append(f"binary '{backends._MOSS_BIN_NAME}' not on PATH (set AUDIBLEZ_MOSS_BIN)")
        for kind in backends.MOSS_REQUIRED:
            if paths[kind] is None:
                miss.append(f'{kind} GGUF missing (set AUDIBLEZ_MOSS_{kind.upper()})')
        if miss:
            return CheckResult(f'backend {backend}', 'fail', '; '.join(miss))
        if paths['encoder'] is None:
            return CheckResult(f'backend {backend}', 'warn',
                               f'{info.label} — encoder GGUF absent (voice cloning unavailable)')
        return CheckResult(f'backend {backend}', 'ok', info.label)
    # mlx engine
    if backend not in avail:
        return CheckResult(f'backend {backend}', 'fail',
                           'requires Apple Silicon + mlx-audio (pip install "audiblez[mlx]")')
    return CheckResult(f'backend {backend}', 'ok', info.label)


def run_checks(backend: str = 'cpu') -> list[CheckResult]:
    """Fast preflight: tools + deps + MOSS availability + the selected backend. No model load."""
    return [check_ffmpeg(), check_ffprobe(), check_espeak(),
            check_kokoro(), check_spacy_model(), check_moss(), check_backend(backend)]


def deep_check(backend: str = 'cpu', voice: str = 'af_sky', tune: bool = False,
               output_folder: str = '.', precision: str = 'fp32') -> CheckResult:
    """Slow check: actually build the synthesizer and synth one word.

    Loads the full model (seconds), so it is opt-in (``--doctor --deep``) and the
    only check that imports :mod:`audiblez.core`. Doubles as a **cache warm-up**: the
    synth compiles & caches GPU conv kernels (MIOpen, ``~/.cache/miopen``) so the
    first real chapter doesn't pay that stall; with ``tune`` it also seeds the
    TunableOp results CSV **under output_folder** (the same place the real run reads it,
    so the warm-up isn't wasted) and exercises the requested ``precision``. Run
    ``audiblez --doctor --deep --tune -b rocm -o out`` once before a long conversion.
    """
    try:
        import audiblez.core as core
        from audiblez import gpu
        core.set_espeak_library()
        gpu.configure_tunableop(tune, backend, results_dir=output_folder, out=lambda *_: None)
        synth = core.build_synthesizer(voice, backend, precision=precision)
        out = synth('Hello, this warms the kernel caches.', 1.0)
        produced = bool(out) and sum(len(seg) for seg in out) > 0
        warmed = 'MIOpen' + (' + TunableOp CSV' if gpu._is_torch_gpu(backend) and tune else '')
        detail = f'produced audio; warmed {warmed} cache' if produced else 'synth returned no audio'
        return CheckResult('deep: synth + warm caches', 'ok' if produced else 'fail', detail)
    except Exception as e:  # any failure in the heavy path is a diagnostic, not a crash
        return CheckResult('deep: synth + warm caches', 'fail', f'{type(e).__name__}: {e}')


def format_report(results: list[CheckResult], color: bool = True) -> str:
    lines = []
    for r in results:
        line = f'{_MARKS[r.status]} {r.name}: {r.detail}'
        if color:
            line = f'{_COLORS[r.status]}{line}{_RESET}'
        lines.append(line)
    return '\n'.join(lines)


def run_doctor(backend: str = 'cpu', voice: str = 'af_sky', deep: bool = False,
               color: bool = True, out=print, tune: bool = False,
               output_folder: str = '.', precision: str = 'fp32') -> bool:
    """Run the preflight, print a red/green report, and return True iff nothing failed."""
    results = run_checks(backend)
    if deep:
        results.append(deep_check(backend, voice, tune=tune, output_folder=output_folder,
                                  precision=precision))
    out(format_report(results, color=color))
    ok = not any(r.status == 'fail' for r in results)
    summary = 'All required checks passed.' if ok else 'Some required checks FAILED.'
    out('')
    out(f'{_COLORS["ok" if ok else "fail"]}{summary}{_RESET}' if color else summary)
    return ok
