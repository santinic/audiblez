# -*- coding: utf-8 -*-
import argparse
import os
import sys
from pathlib import Path

from audiblez import voices as voicelib
from audiblez import backends


class _UnreachableAbort(BaseException):
    """Sentinel `except` target used only when a (stubbed) core lacks ``MossRunAborted`` —
    it is never raised, so the handler simply never matches."""


def parse_chapter_spec(spec: str) -> list[int]:
    """Parse a --chapters spec like '1,3,5' or '1-4,7' into a sorted, unique list of 1-based ints.

    Pure helper (no argparse dependency) so it is easy to test. Raises ValueError on any
    malformed token (non-integer, empty, inverted range, or non-positive index); the caller
    turns that into a clean parser.error().
    """
    result: set[int] = set()
    for token in spec.split(','):
        token = token.strip()
        if not token:
            raise ValueError(f'empty chapter token in {spec!r}')
        if '-' in token:
            start_str, _, end_str = token.partition('-')
            try:
                start, end = int(start_str), int(end_str)
            except ValueError:
                raise ValueError(f'invalid chapter range {token!r}; expected integers like 1-4') from None
            if start > end:
                raise ValueError(f'inverted chapter range {token!r}; start must be <= end')
            if start < 1:
                raise ValueError(f'chapter numbers are 1-based; {start} is out of range in {token!r}')
            result.update(range(start, end + 1))
        else:
            try:
                n = int(token)
            except ValueError:
                raise ValueError(f'invalid chapter number {token!r}; expected an integer') from None
            if n < 1:
                raise ValueError(f'chapter numbers are 1-based; {n} is out of range')
            result.add(n)
    return sorted(result)


def cli_main():
    epilog = ('examples:\n' +
              '  audiblez book.epub -v af_heart           # best-quality default voice\n' +
              '  audiblez book.epub -v af_warm -b mlx     # a curated "house narrator" blend\n' +
              "  audiblez book.epub -v 'af_bella:60,af_heart:40'  # your own weighted blend\n\n" +
              'to run GUI just run:\n'
              '  audiblez-ui\n\n' +
              'available voices:\n' +
              voicelib.available_voices_str)
    parser = argparse.ArgumentParser(epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('epub_file_path', nargs='?', help='Path to the epub file')
    parser.add_argument('-v', '--voice', default=voicelib.DEFAULT_VOICE,
                        help=(f'Narrating voice (default: {voicelib.DEFAULT_VOICE}). A voice id '
                              f'(e.g. af_heart), a preset blend ({", ".join(voicelib.PRESET_BLENDS)}), '
                              "or a custom blend like 'af_bella:60,af_heart:40'. See the list below."))
    parser.add_argument('-p', '--pick', default=False, help='Interactively select which chapters to read in the audiobook', action='store_true')
    parser.add_argument('--chapters', default=None, metavar='SPEC',
                        help="Select chapters by 1-based index without the interactive picker, "
                             "e.g. '1,3,5' or '1-4,7'. Headless alternative to --pick.")
    parser.add_argument('--chapter-text-dir', default=None, metavar='DIR',
                        help='Directory of per-chapter text overrides to read instead of the extracted epub text.')
    parser.add_argument('-s', '--speed', default=1.0, help='Set speed from 0.5 to 2.0', type=float)
    parser.add_argument('-b', '--backend', choices=backends.BACKEND_IDS, default=None,
                        help='Narration backend: cpu, cuda (NVIDIA), rocm (AMD), mps (Apple Silicon), '
                             'mlx (Apple Silicon native). Default: cpu.')
    parser.add_argument('-c', '--cuda', default=False, action='store_true',
                        help=argparse.SUPPRESS)  # deprecated alias for --backend cuda
    parser.add_argument('-o', '--output', default='.', help='Output folder for the audiobook and temporary files', metavar='FOLDER')
    parser.add_argument('--doctor', default=False, action='store_true',
                        help='Run preflight checks (ffmpeg, espeak-ng, spaCy, selected backend) and exit')
    parser.add_argument('--deep', default=False, action='store_true',
                        help='With --doctor: also load the model and synthesize one word (slow)')
    parser.add_argument('--merge', default=False, action='store_true',
                        help='Assemble already-synthesized chapter wavs into an m4b and exit '
                             '(recover a playable audiobook from an interrupted run)')
    parser.add_argument('--trailer', default=False, action='store_true',
                        help='Render a short trailer.wav sampling the opening of each chapter '
                             'and exit (audition voice + chapter detection before a full run)')
    parser.add_argument('--seed-lexicon', dest='seed_lexicon', default=False, action='store_true',
                        help='Write a <book>.lexicon.json of candidate names/acronyms to edit, '
                             'then exit; applied as pronunciation overrides on the next run')
    parser.add_argument('--cache', default=False, action='store_true',
                        help='Cache synthesized sentences under <output>/.audiblez_cache and '
                             'reuse them on re-runs (opt-in; cache only, no resume)')
    parser.add_argument('--cache-clear', dest='cache_clear', default=False, action='store_true',
                        help='Delete the sentence cache under <output>/.audiblez_cache and exit')
    parser.add_argument('--tune', default=False, action='store_true',
                        help='Enable GPU GEMM autotuning (PyTorch TunableOp) for cuda/rocm. '
                             'First run is slower while it tunes; results are cached under '
                             '<output> and reused. No-op on cpu/mlx.')
    parser.add_argument('--precision', choices=('fp32', 'fp16', 'bf16'), default='fp32',
                        help='GPU compute precision via autocast (cuda/rocm only). fp16/bf16 are '
                             'faster but change the waveform — audition before a full run. '
                             'Default: fp32. No-op on cpu/mlx.')
    parser.add_argument('--clone-ref', dest='clone_ref', default=None, metavar='WAV',
                        help='Reference WAV for zero-shot voice cloning (MOSS engine only). The '
                             'audiobook is narrated in the reference voice; encoded once at start.')
    parser.add_argument('--coarse', default=False, action='store_true',
                        help='(EXPERIMENTAL, MOSS engine only) Opt-in coarse-chunk mode: pack '
                             'consecutive sentences into one utterance (capped at ~16 s) for more '
                             'speed, at the cost of sentence-level edit/cache granularity. Ignored '
                             'for non-MOSS backends. Audition the output — verify quality by ear.')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    # Validate the voice/blend spec up front for a clean error (instead of a deep
    # traceback at synthesis time). Pure-Python, keeps the --help path torch-free.
    try:
        voicelib.parse_voice_spec(args.voice)
    except ValueError as e:
        parser.error(str(e))

    # --doctor checks the backend the real run will actually use — for a bare invocation
    # that is now the auto-selected backend, not a silent CPU default — and runs before
    # importing the heavy TTS stack so it works on a broken install.
    if args.doctor:
        from audiblez import doctor
        requested = (args.backend if args.backend is not None
                     else ('cuda' if args.cuda else backends.default_backend()))
        sys.exit(0 if doctor.run_doctor(backend=requested, voice=args.voice, deep=args.deep,
                                        tune=args.tune, output_folder=args.output,
                                        precision=args.precision) else 1)

    # --cache-clear wipes the sentence cache; needs no epub/backend/model.
    if args.cache_clear:
        from audiblez.cache import SynthCache
        cache_dir = os.path.join(args.output, '.audiblez_cache')
        print(f'Cleared {SynthCache(cache_dir).clear()} cached sentence(s) from {cache_dir}.')
        sys.exit(0)

    if not args.epub_file_path:
        parser.error('the following arguments are required: epub_file_path (or use --doctor)')

    # Ensure the output folder exists before any subcommand writes into it (--merge,
    # --seed-lexicon, --trailer all run before main(), which is what otherwise creates it).
    if args.output != '.':
        os.makedirs(args.output, exist_ok=True)

    # --merge needs no backend/model: just stitch existing chapter wavs into an m4b.
    if args.merge:
        from audiblez.core import merge_chapters
        result = merge_chapters(args.epub_file_path, args.voice, args.output)
        sys.exit(0 if result else 1)

    # --seed-lexicon writes editable pronunciation candidates; no backend/model needed.
    if args.seed_lexicon:
        from ebooklib import epub
        from audiblez import lexicon
        from audiblez.core import find_document_chapters_and_extract_texts
        chapters = find_document_chapters_and_extract_texts(epub.read_epub(args.epub_file_path))
        text = '\n'.join(c.extracted_text for c in chapters)
        path = lexicon.lexicon_path(args.epub_file_path, args.output)
        merged = lexicon.build_seed_lexicon(text)
        merged.update(lexicon.load_lexicon(path))  # never clobber existing user edits
        lexicon.save_lexicon(path, merged)
        print(f'Wrote {len(merged)} candidate term(s) to {path}.')
        print('Edit the JSON values to set pronunciations (respellings); they apply on the next run.')
        sys.exit(0)

    # Headless chapter selection (--chapters). Parsed after the early-exit subcommands so a
    # bad spec only matters for a real run; fails fast and cleanly.
    selected_chapters = None
    if args.chapters is not None:
        try:
            selected_chapters = parse_chapter_spec(args.chapters)
        except ValueError as e:
            parser.error(str(e))

    avail = backends.available_backends()
    if args.backend is not None:
        backend = args.backend
        if args.cuda and backend not in ('cuda', 'rocm'):
            parser.error(f'--cuda conflicts with --backend {backend}')
    elif args.cuda:
        print('Note: --cuda is deprecated; use --backend cuda')
        # Legacy intent = "use the torch GPU". On a ROCm build that GPU is exposed as 'rocm'.
        backend = 'cuda' if 'cuda' in avail else ('rocm' if 'rocm' in avail else 'cpu')
        if backend == 'cpu':
            print('CUDA GPU not available. Defaulting to CPU')
    else:
        # A PARTIAL MOSS install (binary present but a GGUF missing) means MOSS was clearly
        # intended; auto-select silently uses Kokoro instead (the user's disliked engine), so
        # warn LOUDLY and name what's missing (ADR 0005 / PRD stories 3-vs-17). Forced -b moss
        # while incomplete aborts in main()'s preflight; this is only the auto-select path.
        if getattr(backends, 'moss_status', lambda: 'absent')() == 'partial':
            missing = ', '.join(k for k, v in backends.moss_paths().items() if v is None) or 'some pieces'
            print(f'\033[93mWarning: MOSS appears installed but is incomplete (missing: {missing}); '
                  f'auto-selecting Kokoro instead. Fix the install or pass -b moss to see the '
                  f'exact preflight error.\033[0m')
        # Auto-select the best available backend (GPU when present), falling back to CPU.
        # Probe the GPU with a real kernel: torch.cuda.is_available() can be True while the
        # device faults on first use (e.g. RDNA3 without HSA_OVERRIDE_GFX_VERSION), which
        # used to abort the run or emit silent audio. We only auto-pick a GPU that works.
        backend = backends.default_backend()
        if backend != 'cpu':
            if backends.gpu_works(backend):
                print(f'Auto-selected {backend} backend (pass -b cpu to force CPU)')
            else:
                print(f'\033[93mAuto-selected {backend} GPU failed a test kernel; falling back '
                      f'to CPU. Pass -b {backend} to force it (e.g. after setting '
                      'HSA_OVERRIDE_GFX_VERSION).\033[0m')
                backend = 'cpu'

    if backend not in avail:
        # A torch wheel exposes the GPU as EITHER 'cuda' (NVIDIA) or 'rocm' (AMD/HIP), never both. A user
        # who asks for one when the build provides the other still wants their GPU, so map across first.
        sibling = {'cuda': 'rocm', 'rocm': 'cuda'}.get(backend)
        if sibling in avail:
            print(f'{backend!r} requested, but this torch build exposes the GPU as {sibling!r}; using {sibling}.')
            backend = sibling
        else:
            print(f'Backend {backend!r} not available on this machine (have: {", ".join(avail)}). '
                  'Falling back to cpu.')
            backend = 'cpu'

    print(f'Using {backends.BACKENDS[backend].label} backend')

    # --trailer auditions voice + chapter detection cheaply before a full hour-long run.
    if args.trailer:
        from audiblez import doctor
        checks = doctor.run_checks(backend)
        if any(c.status == 'fail' for c in checks):
            print(doctor.format_report(checks))
            print('\033[91mPreflight failed; fix the above before auditioning a trailer.\033[0m')
            sys.exit(1)
        from audiblez.core import make_trailer
        out = os.path.join(args.output, 'trailer.wav')
        result = make_trailer(args.epub_file_path, args.voice, out, speed=args.speed, backend=backend,
                              precision=args.precision, output_folder=args.output,
                              clone_ref=args.clone_ref)
        sys.exit(0 if result else 1)

    if args.precision != 'fp32':
        # Measured on Kokoro-82M / RX 7800 XT: fp16/bf16 give ~1% (noise) speedup while
        # degrading the waveform (fp16 ~0.87 similarity, bf16 ~0.06 = broken). The model
        # is small + memory-bound, so autocast does not help here. Kept opt-in for other
        # hardware/models, but warn loudly and audition the output first.
        print(f'\033[93mWarning: --precision {args.precision} changes the audio and showed no '
              'measured speedup on Kokoro-82M; audition the output (bf16 is known-broken here).\033[0m')

    cache_dir = os.path.join(args.output, '.audiblez_cache') if args.cache else None
    chapter_text_dir = Path(args.chapter_text_dir) if args.chapter_text_dir is not None else None
    from audiblez import core as _core
    main = _core.main
    # Resolve the abort type defensively: a stubbed core (used by some hermetic CLI tests)
    # may not define it, and a missing symbol must not turn into an ImportError at startup.
    MossRunAborted = getattr(_core, 'MossRunAborted', _UnreachableAbort)
    try:
        failures = main(args.epub_file_path, args.voice, args.pick, args.speed, args.output, backend=backend,
                        selected_chapters=selected_chapters, chapter_text_dir=chapter_text_dir,
                        cache_dir=cache_dir, tune=args.tune, precision=args.precision,
                        clone_ref=args.clone_ref, coarse=args.coarse)
    except MossRunAborted as e:
        # The MOSS engine couldn't spawn or the circuit-breaker tripped. Abort LOUD (never a
        # silent swap to Kokoro — that would change the voice mid-book). Partial chapters +
        # dead-letters are preserved on disk for a resume.
        print(f'\033[91mRun aborted: {e}\033[0m')
        sys.exit(2)
    # Non-zero exit on a degraded run (sentences dead-lettered -> gaps), so scripts/CI notice.
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    cli_main()
