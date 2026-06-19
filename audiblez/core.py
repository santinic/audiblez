#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# audiblez - A program to convert e-books into audiobooks using
# Kokoro-82M model for high-quality text-to-speech synthesis.
# by Claudio Santini 2025 - https://claudio.uk
import spacy
import ebooklib
import soundfile
import numpy as np
import os
import json
import time
import shutil
import subprocess
import platform
import re
import hashlib
import tempfile
import threading
from glob import glob
from dataclasses import dataclass
from types import SimpleNamespace
from tabulate import tabulate
from pathlib import Path
from string import Formatter
from bs4 import BeautifulSoup
from kokoro import KPipeline
from ebooklib import epub
from pick import pick

from audiblez import backends
from audiblez import gpu
from audiblez import lexicon
from audiblez import voices as voicelib

sample_rate = 24000
_nlp = None  # cached spaCy pipeline (loaded once, reused across chapters/previews)
_espeak_registered = False  # set once; every synth entry point registers espeak idempotently

# Model weights, defined once so the synth builders and the cache key can never drift
# (a one-sided edit would otherwise make --cache serve audio from the wrong model).
TORCH_REPO_ID = 'hexgrad/Kokoro-82M'           # torch backends (cpu/cuda/rocm/mps)
MLX_REPO_ID = 'mlx-community/Kokoro-82M-bf16'   # Apple-Silicon-native (quantized) weights

MAX_SENTENCE_LENGTH = 400  # chars; Kokoro truncates long non-English sentences, so we split them
GPU_CHARS_PER_SEC = 500    # rough synthesis throughput, used only for the ETA display
CPU_CHARS_PER_SEC = 50

# Truncation floor for validate-before-skip: no real narration packs more than this
# many characters into a second of audio, so anything shorter than text/this is too
# short to be a complete chapter. Deliberately generous to avoid deleting valid wavs.
VALIDATION_MAX_CHARS_PER_SEC = 30

EWMA_ALPHA = 0.3       # weight of the newest measurement in the rolling chars/sec ETA
HEARTBEAT_EVERY = 10   # emit a progress heartbeat line every N synthesized sentences

# Batch multiple sentences into one synth() call (Kokoro re-splits on \n\n\n internally),
# trading per-call Python/model overhead for throughput. Set to a tiny value to disable
# batching and fall back to one sentence per call.
BATCH_MAX_CHARS = 1000

SYNTH_RETRIES = 2                     # extra attempts after the first before giving up on a unit
SYNTH_RETRY_BACKOFF = 0.5             # seconds between retries, so a transient fault (e.g. OOM) can clear
# Deterministic errors that retrying cannot fix (bad input / code bug): fail fast to silence
# instead of re-running the same losing call N times (and amplifying it across a batch).
_PERMANENT_SYNTH_ERRORS = (ValueError, TypeError, KeyError, AttributeError, IndexError)
FALLBACK_SILENCE_CHARS_PER_SEC = 15   # gap length (proportional to text) for a dead-lettered sentence

TRAILER_SENTENCES_PER_CHAPTER = 2     # opening sentences sampled per chapter in --trailer
TRAILER_GAP_SECONDS = 1.0             # silent gap between trailer segments

# ─── MOSS resident-pipe engine (engine='llamacpp') ──────────────────────────────────
# Binary + GGUF identities for the OpenMOSS llama.cpp fork. The resident child loads these
# once and synthesizes sentence-by-sentence over a stdin/stdout pipe (ADR 0002, the impl
# spec docs/moss-coprocess-spec.md). All knobs that change the waveform are PINNED here so
# the sentence cache hits across runs (a drifted sampling param silently breaks every hit).
MOSS_BINARY = 'llama-moss-tts'        # default name on PATH; the real path comes from backends.moss_paths()
# GGUF basenames are kept only for the test/odd-layout `gguf_dir=` override in
# _build_llamacpp_synth; the real run resolves all paths via backends.moss_paths() (T3, the
# single source of truth that --doctor preflights), NOT these.
MOSS_BACKBONE_GGUF = 'moss_delay_firstclass_Q5_K_M.gguf'
MOSS_DECODER_GGUF = 'moss_tts_audio_decoder_f16.gguf'
MOSS_ENCODER_GGUF = 'moss_tts_audio_encoder_f16.gguf'   # only for voice cloning (Phase-2)

MOSS_SEED = 42                        # pinned: the cache key must be stable across runs
# Cache-key sentinel for the MOSS *voice* axis when there is NO clone reference. MOSS ignores the
# Kokoro voice, but the GUI's greyed voice dropdown still holds *some* Kokoro voice string; folding
# that inert string into the cache key would fragment the sentence cache by whatever voice happened
# to be selected (slice 2 / #8). Substituting this fixed constant keeps the MOSS-built-in-voice key
# stable regardless of the dropdown. A clone reference still takes precedence (folded via
# backends.clone_voice_id), so two different references never collide. NOT a Kokoro voice id and not
# a 'clone:' id, so it can never alias either.
MOSS_DEFAULT_VOICE = 'moss-builtin'
# The six sampling params, pinned to fixed defaults (ADR 0005). SINGLE source: alias the
# import-light backends copy so the synth request, the cache key (_moss_sampling_sig), and
# doctor can never drift to two different dicts. Order is irrelevant — the cache fingerprints
# the dict by sorted keys (see _moss_sampling_sig).
MOSS_SAMPLING = backends.MOSS_SAMPLING_DEFAULTS
SAMPLING_DEFAULTS = MOSS_SAMPLING     # canonical alias (the name T3/T4 reference)
MOSS_MAX_NEW_TOKENS = 2048            # per-request cap sent to the child
MOSS_SAMPLE_RATE = 24000              # the child writes 24 kHz wavs; must match `sample_rate`
# Fixed `--serve` invocation flags T1's binary (branch moss-serve-json-v1) requires at spawn.
MOSS_LANGUAGE = 'en'                  # --language
MOSS_NGL = '-1'                       # -ngl: offload all layers to the GPU
MOSS_MAX_PROMPT_FRAMES = '512'       # --max-prompt-frames (backbone ctx sizing)
MOSS_MAX_RAW_FRAMES = '768'           # --max-raw-frames (audio-decoder ctx sizing)

# Failure-vs-death tuning (the two-axis model; see _build_llamacpp_synth).
MOSS_DEATHS_BEFORE_POISON = 2         # K: consecutive deaths on ONE sentence -> dead-letter it
MOSS_MAX_CONSECUTIVE_DEATHS = MOSS_DEATHS_BEFORE_POISON  # canonical alias for the same K
MOSS_RESPONSE_TIMEOUT_FLOOR = 30.0    # seconds; a no-response past this == death
MOSS_TIMEOUT_SECONDS_PER_CHAR = 0.25  # deadline scales with sentence length above the floor
MOSS_RESTART_WINDOW_SECONDS = 120.0   # circuit-breaker: look at restarts in this trailing window
MOSS_MAX_RESTARTS_IN_WINDOW = 5       # ...abort the run if more than this many fall inside it
MOSS_SPAWN_READY_TIMEOUT = 180.0      # cold-start model load can take a while (RADV compile)
MOSS_SHUTDOWN_WRITE_TIMEOUT = 5.0     # bound the graceful-shutdown write; a wedged child -> kill


class MossError(Exception):
    """A `status:"error"` response from a HEALTHY child (real synth failure, not a death).

    Raised by the synth closure so the existing `_synth_one_or_silence` dead-letters the
    sentence and splices silence — the child stays alive for the next sentence.
    """


class MossPermanentError(MossError, ValueError):
    """A deterministic MOSS failure that retrying cannot fix (bad_request/empty_audio, or a
    poison-pill sentence that keeps killing the child). Subclasses ``ValueError`` so it is a
    member of :data:`_PERMANENT_SYNTH_ERRORS`: ``_retry`` re-raises it immediately instead of
    re-running the same losing call (or re-spinning a child-killing sentence) N times."""


class MossDeath(Exception):
    """The child DIED mid-request (EOF on stdout / ``poll() is not None`` / response-timeout).

    Internal to :class:`MossProcess` / the closure's restart loop — never reaches
    ``_synth_one_or_silence``. The closure restarts the child and re-dispatches the SAME
    sentence (infra fault, not the sentence's fault — don't dead-letter the first death).
    """


class MossRunAborted(BaseException):
    """Abort the whole run, LOUD: the child can't spawn, or the circuit-breaker tripped.

    Deliberately a ``BaseException`` (like ``KeyboardInterrupt``), NOT ``Exception``: the
    unchanged ``_synth_one_or_silence`` / ``_synth_batch`` / ``_retry`` all ``except
    Exception`` broadly, so a plain exception here would be swallowed into silence + a
    dead-letter and the run would CONTINUE — turning a crash storm into a silently all-silence
    book reported as success-with-gaps (the banned ADR-0005 outcome). As a ``BaseException`` it
    propagates THROUGH those handlers up to ``core.main`` -> a non-zero exit, never a swap to
    Kokoro (changing the voice mid-book is a banned silent-wrongness class).

    Two named subclasses distinguish the cause (both still caught by ``except MossRunAborted``):
    :class:`MossSpawnError` (child won't spawn) and :class:`MossCircuitBreakerError` (restart
    rate exceeded).
    """


class MossSpawnError(MossRunAborted):
    """The resident child could not be spawned (missing/broken binary). Abort loud, never swap
    to Kokoro. A subclass of :class:`MossRunAborted`, so callers catching the base type catch it."""


class MossCircuitBreakerError(MossRunAborted):
    """The restart rate exceeded the circuit-breaker threshold (systemic failure, e.g. VRAM
    exhaustion mid-book). A subclass of :class:`MossRunAborted` (caught by the base handler)."""


def to_numpy(audio):
    """Normalise a kokoro audio segment to a 1-D numpy array.

    Depending on version/device kokoro may yield a torch.Tensor (possibly on GPU)
    or a numpy array; this handles both so np.concatenate / soundfile.write work.
    """
    if hasattr(audio, 'detach'):  # torch.Tensor
        return audio.detach().cpu().numpy()
    return np.asarray(audio)


def load_spacy():
    """Load (once) and cache the multilingual spaCy model used for sentence splitting.

    xx_ent_wiki_sm has no sentence boundaries on its own, so a 'sentencizer' is
    added the first time the model is loaded (guarded in case a future model ships
    one). Caching at module level avoids reloading the model from disk for every
    chapter and preview, which was a major performance drain.
    """
    global _nlp
    if _nlp is None:
        if not spacy.util.is_package("xx_ent_wiki_sm"):
            print("Downloading Spacy model xx_ent_wiki_sm...")
            spacy.cli.download("xx_ent_wiki_sm")
        _nlp = spacy.load("xx_ent_wiki_sm")
        if "sentencizer" not in _nlp.pipe_names:
            _nlp.add_pipe("sentencizer")
    return _nlp


def set_espeak_library(library=None):
    """Locate the espeak-ng library and register it with phonemizer.

    Fails LOUD: the path resolution (delegated to :func:`audiblez.doctor.find_espeak_library`)
    raises a ``RuntimeError`` with an actionable, OS-specific install hint instead of the
    old swallow-and-continue, which used to let a run proceed for an hour and emit nothing.
    Run ``audiblez --doctor`` to check this ahead of a long synthesis.

    Idempotent: the first call registers; later calls (e.g. from :func:`build_synthesizer`
    on the trailer / audition paths) are no-ops, so registration happens exactly once per
    process. ``library`` lets a caller pass an already-resolved path (e.g. the one the
    preflight just found) so the espeak library isn't globbed for twice.
    """
    global _espeak_registered
    if _espeak_registered:
        return
    if library is None:
        from audiblez.doctor import find_espeak_library
        library = find_espeak_library()
    print('Using espeak library:', library)
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    EspeakWrapper.set_library(library)
    _espeak_registered = True


def extract_book_metadata(book):
    """Return (title, creator) from an ebooklib book's Dublin Core metadata."""
    meta_title = book.get_metadata('DC', 'title')
    title = meta_title[0][0] if meta_title else ''
    meta_creator = book.get_metadata('DC', 'creator')
    creator = meta_creator[0][0] if meta_creator else ''
    return title, creator


def main(file_path: str, voice: str, pick_manually: bool, speed: float, output_folder: str = '.',
         max_chapters: int | None = None, max_sentences: int | None = None,
         selected_chapters: list | None = None, backend: str = 'cpu', post_event=None,
         chapter_text_dir=None, cache_dir: str | None = None, tune: bool = False,
         precision: str = 'fp32', clone_ref: str | None = None, coarse: bool = False) -> int:
    if post_event: post_event('CORE_STARTED')
    # Fast preflight: abort in seconds with an actionable message rather than dying
    # 40 minutes in on a missing dep. See `audiblez --doctor` for the same checks.
    from audiblez import doctor
    checks = doctor.run_checks(backend)
    if any(c.status == 'fail' for c in checks):
        print(doctor.format_report(checks))
        failed = '; '.join(f'{c.name}: {c.detail}' for c in checks if c.status == 'fail')
        raise RuntimeError(f'Preflight failed — fix these before synthesis: {failed}')
    load_spacy()
    if output_folder != '.':
        Path(output_folder).mkdir(parents=True, exist_ok=True)

    filename = Path(file_path).name

    book = epub.read_epub(file_path)
    title, creator = extract_book_metadata(book)

    cover_maybe = find_cover(book)
    cover_image = cover_maybe.get_content() if cover_maybe else b""
    if cover_maybe:
        print(f'Found cover image {cover_maybe.file_name} in {cover_maybe.media_type} format')

    # Book-scoped pronunciation overrides (no sidecar -> empty dict -> no-op).
    book_lexicon = lexicon.load_lexicon(lexicon.lexicon_path(file_path, output_folder))
    if lexicon.fingerprint(book_lexicon):
        print(f'Loaded pronunciation lexicon with {len(lexicon._active(book_lexicon))} override(s).')

    # Opt-in per-sentence synth cache (KEYSTONE slice 1): reuse audio on re-runs/Preview.
    synth_cache, cache_fields = None, None
    if cache_dir:
        from audiblez import cache as cache_mod
        synth_cache = cache_mod.SynthCache(cache_dir)
        cache_fields = _cache_key_fields(backend, voice, speed, precision, clone_ref=clone_ref)
        print(f'Sentence cache enabled at {cache_dir}')

    document_chapters = find_document_chapters_and_extract_texts(book)

    if not selected_chapters:
        if pick_manually is True:
            selected_chapters = pick_chapters(document_chapters)
        else:
            selected_chapters = find_good_chapters(document_chapters)
    elif all(isinstance(c, int) for c in selected_chapters):
        # Headless --chapters passes 1-based indices into document_chapters (the GUI passes
        # chapter objects). Resolve indices to objects; an out-of-range index is a degraded
        # run, so surface it (don't silently narrate fewer chapters than the user asked for).
        n = len(document_chapters)
        requested = selected_chapters
        out_of_range = [i for i in requested if not (1 <= i <= n)]
        if out_of_range:
            print(f'\033[93mWarning: ignoring out-of-range chapter index/indices '
                  f'{out_of_range} (book has {n} chapters).\033[0m')
        selected_chapters = [document_chapters[i - 1] for i in requested if 1 <= i <= n]
        if not selected_chapters:
            raise ValueError(f'No valid chapters selected from --chapters {requested}; '
                             f'book has {n} chapters (use 1-based indices).')
    print_selected_chapters(document_chapters, selected_chapters)

    if chapter_text_dir is not None:
        # Optional override: replace a selected chapter's extracted text with the contents
        # of '{chapter_text_dir}/chapter_{i}.txt' when that file exists (i is the 1-based
        # index into selected_chapters, matching the synthesis loop below). Lets callers
        # hand-edit/pre-process chapter text before narration. Done before `texts`/stats so
        # totals and the ETA reflect the overridden content.
        for i, chapter in enumerate(selected_chapters, start=1):
            override_path = Path(chapter_text_dir) / f'chapter_{i}.txt'
            if override_path.exists():
                chapter.extracted_text = override_path.read_text(encoding='utf-8')
                print(f'Overriding chapter {i} text from {override_path}')

    texts = [c.extracted_text for c in selected_chapters]

    has_ffmpeg = shutil.which('ffmpeg') is not None
    if not has_ffmpeg:
        print('\033[91m' + 'ffmpeg not found. Please install ffmpeg to create mp3 and m4b audiobook files.' + '\033[0m')

    stats = SimpleNamespace(
        total_chars=sum(map(len, texts)),
        processed_chars=0,
        sentences_done=0,
        # Flat constant is only the initial prior; the real rate is measured per
        # sentence and folded into an EWMA as synthesis proceeds (see _update_eta).
        chars_per_sec=GPU_CHARS_PER_SEC if backends.is_gpu(backend) else CPU_CHARS_PER_SEC)
    print('Started at:', time.strftime('%H:%M:%S'))
    print(f'Total characters: {stats.total_chars:,}')
    print('Total words:', len(' '.join(texts).split()))
    eta = strfdelta((stats.total_chars - stats.processed_chars) / stats.chars_per_sec)
    print(f'Estimated time remaining (assuming {stats.chars_per_sec} chars/sec): {eta}')
    # Reuse the path the preflight already resolved so espeak isn't globbed for twice.
    espeak_lib = next((c.detail for c in checks if c.name == 'espeak-ng' and c.status == 'ok'), None)
    set_espeak_library(espeak_lib)
    # GPU GEMM autotuning must be configured before the first synth (TunableOp caches
    # its env on first read); persist the results CSV under the output folder.
    gpu.configure_tunableop(tune, backend, results_dir=output_folder)
    chapter_wav_files = []
    intro_added = False
    total_failures = 0  # dead-lettered sentences across the whole book (degraded-run signal)
    synth = None  # bound inside the try so a raise at/after engine-build can't skip the finally
    try:
      # Slice-0: one SynthParams threads voice/backend/precision/clone_ref/coarse through the
      # engine seam (so a new knob can't be wired into cli.py and missed in ui.py). Spawn the
      # resident MOSS child (engine='llamacpp') INSIDE the try so a raise here OR in the setup
      # below (voice parse, signature) still reaches the finally — else an 8 GB-VRAM child leaks.
      synth = build_synthesizer(params=SynthParams(
          voice=voice, backend=backend, precision=precision, clone_ref=clone_ref, coarse=coarse),
          work_dir=output_folder)
      # Resolve the spec's language code once for the whole run and thread it into each
      # gen_audio_segments call, instead of re-parsing the spec per chapter.
      lang_code = voicelib.voice_lang_code(voice)
      # Coarse-chunk mode is MOSS-only (Kokoro re-splits internally, so it would be a no-op there).
      # Gate it on the engine HERE so gen_audio_segments just honors a clean bool (build_synthesizer
      # already warned if --coarse was passed for a non-MOSS backend).
      coarse_effective = coarse and backends.BACKENDS[backend].engine == 'llamacpp'
      render_signature = _render_signature(book_lexicon, speed, precision,
                                           backend=backend, clone_ref=clone_ref,
                                           coarse=coarse_effective)
      for i, chapter in enumerate(selected_chapters, start=1):
        if max_chapters and i > max_chapters: break
        text = chapter.extracted_text
        chapter_wav_path = Path(output_folder) / _chapter_wav_name(
            Path(filename).stem, i, voice, chapter.get_name())
        chapter_wav_files.append(chapter_wav_path)
        if Path(chapter_wav_path).exists():
            # Only skip if the existing wav is actually usable — a truncated/zero-byte
            # file from a prior crash must be regenerated, not reused into the m4b.
            expected_len = None if max_sentences else len(text)
            if (is_valid_chapter_wav(chapter_wav_path, expected_len)
                    and _chapter_is_complete(chapter_wav_path, render_signature)):
                print(f'File for chapter {i} already exists. Skipping')
                stats.processed_chars += len(text)
                # On a resumed run this existing wav already contains the prepended intro, so
                # mark the intro consumed — otherwise it gets re-attached (and re-spoken) on
                # the next synthesized chapter.
                if len(text.strip()) >= 10:
                    intro_added = True
                if post_event:
                    post_event('CORE_CHAPTER_FINISHED', chapter_index=chapter.chapter_index)
                continue
            print(f'Existing file for chapter {i} is invalid, incomplete, or stale; regenerating.')
            Path(chapter_wav_path).unlink(missing_ok=True)
            Path(chapter_wav_path).with_suffix('.sig').unlink(missing_ok=True)
        if len(text.strip()) < 10:
            print(f'Skipping empty chapter {i}')
            chapter_wav_files.remove(chapter_wav_path)
            continue
        if not intro_added:
            # Prepend the book intro to the first chapter actually synthesized
            # (not necessarily i == 1, which may have been skipped or empty).
            text = f'{title} – {creator}.\n\n' + text
            intro_added = True
        # Apply pronunciation overrides before synthesis (pre-phonemization).
        text = lexicon.apply_lexicon(text, book_lexicon)
        start_time = time.time()
        if post_event: post_event('CORE_CHAPTER_STARTED', chapter_index=chapter.chapter_index)
        # Fresh dead-letter per (re)generated chapter; failed sentences are appended here.
        dead_letter_path = Path(chapter_wav_path).with_suffix('.failed.jsonl')
        Path(dead_letter_path).unlink(missing_ok=True)
        audio_segments = gen_audio_segments(
            synth, text, voice, speed, stats, post_event=post_event, max_sentences=max_sentences,
            chapter_label=f'chapter {i}', dead_letter_path=dead_letter_path,
            cache=synth_cache, cache_key_fields=cache_fields, lang_code=lang_code,
            coarse=coarse_effective)
        if audio_segments:
            final_audio = np.concatenate(audio_segments)
            soundfile.write(chapter_wav_path, final_audio, sample_rate)
            # MOSS synthesizes at 1.0 (no speed knob); apply playback speed here, once per
            # chapter, via ffmpeg atempo (ADR 0002) so the speed-agnostic MOSS sentence cache
            # is re-stretched rather than re-synthesized. Kokoro already baked speed into the
            # samples, so this is MOSS-only (engine='llamacpp'); never double-stretch.
            if backends.BACKENDS[backend].engine == 'llamacpp':
                _apply_atempo(chapter_wav_path, speed)
            # Record the render signature so a later run regenerates this chapter if the
            # lexicon/speed/precision changed (see _chapter_is_complete).
            Path(chapter_wav_path).with_suffix('.sig').write_text(render_signature, encoding='utf-8')
            end_time = time.time()
            delta_seconds = end_time - start_time
            chars_per_sec = len(text) / delta_seconds
            print('Chapter written to', chapter_wav_path)
            chapter_failures = _count_dead_letters(dead_letter_path)
            total_failures += chapter_failures
            if chapter_failures:
                print(f'\033[91m⚠ Chapter {i}: {chapter_failures} sentence(s) failed and were '
                      f'replaced with silence (see {dead_letter_path}). Re-run to retry them.\033[0m')
            if post_event: post_event('CORE_CHAPTER_FINISHED', chapter_index=chapter.chapter_index)
            print(f'Chapter {i} read in {delta_seconds:.2f} seconds ({chars_per_sec:.0f} characters per second)')
        else:
            print(f'Warning: No audio generated for chapter {i}')
            chapter_wav_files.remove(chapter_wav_path)
    finally:
        # Tear the engine down whatever happens — a normal finish, a MossRunAborted (which
        # propagates as a BaseException through gen_audio_segments' broad excepts), or any
        # other exception — so the resident MOSS child never leaks its VRAM (no-op for Kokoro).
        getattr(synth, 'close', lambda: None)()

    if synth_cache is not None:
        print(f'Sentence cache: {synth_cache.stats()}')

    if not chapter_wav_files:
        print('No chapters were synthesized — nothing to assemble into an audiobook.')
    elif has_ffmpeg:
        # Assemble only valid chapter audio, so one corrupt/short wav can't break the
        # whole m4b; completed chapters always persist as wavs and can also be assembled
        # later with `audiblez --merge` if the run is interrupted before this point.
        valid_wavs = [w for w in chapter_wav_files if is_valid_chapter_wav(w, None)]
        if valid_wavs:
            create_m4b(valid_wavs, filename, cover_image, output_folder, title=title, creator=creator)
        else:
            print('No valid chapter audio to assemble into an m4b.')

    if total_failures:
        print(f'\033[91m⚠ {total_failures} sentence(s) failed across the book and were replaced '
              f'with silence; the audiobook has gaps. See the .failed.jsonl files, and re-run to '
              f'retry them (affected chapters regenerate automatically).\033[0m')
    # Always signal completion so the GUI re-enables its controls even when nothing was
    # assembled (all chapters empty/invalid, or ffmpeg missing) — otherwise the UI locks.
    if post_event: post_event('CORE_FINISHED')
    return total_failures


def find_cover(book):
    def is_image(item):
        return item is not None and item.media_type.startswith('image/')

    for item in book.get_items_of_type(ebooklib.ITEM_COVER):
        if is_image(item):
            return item

    # https://idpf.org/forum/topic-715
    for meta in book.get_metadata('OPF', 'cover'):
        if is_image(item := book.get_item_with_id(meta[1]['content'])):
            return item

    if is_image(item := book.get_item_with_id('cover')):
        return item

    for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
        if 'cover' in item.get_name().lower() and is_image(item):
            return item

    return None


def print_selected_chapters(document_chapters, chapters):
    ok = 'X' if platform.system() == 'Windows' else '✅'
    selected_ids = {id(c) for c in chapters}
    print(tabulate([
        [i, c.get_name(), len(c.extracted_text), ok if id(c) in selected_ids else '', chapter_beginning_one_liner(c)]
        for i, c in enumerate(document_chapters, start=1)
    ], headers=['#', 'Chapter', 'Text Length', 'Selected', 'First words']))

def split_long_sentence(text, max_length=MAX_SENTENCE_LENGTH):
    """Split a long sentence near max_length chars, breaking at the last whitespace before it."""
    if len(text) <= max_length:
        return [text]
    parts = []
    while len(text) > max_length:
        split_index = text.rfind(' ', 0, max_length)
        if split_index == -1:
            split_index = max_length
        parts.append(text[:split_index].strip())
        text = text[split_index:].strip()
    if text:
        parts.append(text)
    return parts


def _kokoro_voice_string_from_comps(comps):
    """Engine-ready voice string from already-parsed (voice_id, weight) components.

    Same comma-repetition encoding as :func:`audiblez.voices.kokoro_voice_string`, but
    reuses a parse the caller already did instead of re-parsing the spec.
    """
    if len(comps) == 1 and comps[0][1] == 1:
        return comps[0][0]
    parts = []
    for vid, weight in comps:
        parts.extend([vid] * weight)
    return ','.join(parts)


@dataclass(frozen=True)
class SynthParams:
    """The synth-path parameters that ALL three entry points (cli/ui/core) must agree on.

    Slice-0 unification: collapsing voice/backend/precision (and the Phase-2 ``clone_ref`` /
    ``coarse`` knobs) into one object means a new synth knob is added in ONE place and threaded
    through ``build_synthesizer`` / the cache key uniformly — it can't be wired into cli.py and
    silently missed in ui.py (the historically-missed entry point; see architecture.md). Run
    orchestration (file_path, output_folder, selected_chapters, ...) is deliberately NOT here —
    only what shapes the waveform / the engine.

    ``clone_ref`` (a reference WAV for zero-shot voice cloning) and ``coarse`` (opt-in
    coarse-chunk mode) are accepted now so later slices don't re-thread three entry points.
    """
    voice: str
    backend: str = 'cpu'
    precision: str = 'fp32'
    clone_ref: str | None = None
    coarse: bool = False


def build_synthesizer(voice=None, backend: str = 'cpu', precision: str = 'fp32',
                      clone_ref=None, *, params: 'SynthParams | None' = None,
                      moss_factory=None, work_dir=None):
    """Return ``synth(text, speed) -> list[np.ndarray]`` (float32 @ 24000 Hz).

    Torch backends (cpu/cuda/rocm/mps) set the process-global default device and build
    a Kokoro KPipeline; the mlx backend loads the Apple-Silicon-native Kokoro model; the
    'llamacpp' engine spawns the resident MOSS co-process. All engines emit identical-format
    audio, so everything downstream is engine-agnostic. The returned ``synth`` carries a
    ``.close`` attribute (a no-op for Kokoro/mlx; the MOSS child teardown for 'llamacpp') so
    ``core.main`` can release the engine in a ``finally`` without unpacking a tuple — that
    keeps the bare-callable contract every existing caller and test mock relies on.

    Accepts EITHER a :class:`SynthParams` (slice-0 unification — the preferred call) OR the
    legacy ``voice``/``backend``/``precision`` kwargs, so the three entry points and the
    existing tests stay green during the migration.

    ``precision`` ('fp32' | 'fp16' | 'bf16') wraps the torch synth call in autocast for
    GPU throughput; it is a no-op on cpu/mlx/mps (see :mod:`audiblez.gpu`). Lower
    precision changes the waveform, so it is opt-in and should be auditioned.
    """
    if params is None:
        if voice is None:
            raise ValueError('build_synthesizer requires a voice (or a SynthParams)')
        params = SynthParams(voice=voice, backend=backend, precision=precision, clone_ref=clone_ref)
    voice, backend, precision, clone_ref = (
        params.voice, params.backend, params.precision, params.clone_ref)
    if backend not in backends.BACKENDS:
        raise ValueError(f'Unknown backend {backend!r}; choose from {backends.BACKEND_IDS}')
    info = backends.BACKENDS[backend]
    if getattr(params, 'coarse', False) and info.engine != 'llamacpp':
        # Coarse-chunk mode is a MOSS (llamacpp) fast path: paragraphs go to the engine as one
        # utterance. Kokoro re-splits internally, so coarse is a no-op there — warn instead of
        # silently implying a speedup. For MOSS the flag IS wired (see gen_audio_segments).
        print('\033[93mWarning: --coarse only applies to the MOSS engine; '
              'ignored for this backend (synthesizing at normal sentence granularity).\033[0m')
    if info.engine == 'llamacpp':
        # MOSS resident co-process: no espeak / Kokoro voice parsing — it owns its own g2p.
        # (`voice`/clone_ref select the narration; speed is applied downstream via atempo.)
        return _build_llamacpp_synth(voice, None, clone_ref=clone_ref,
                                     moss_factory=moss_factory, work_dir=work_dir)
    # Register espeak here too (idempotent), so every synth entry point — main, the trailer,
    # and the GUI audition/preview — has phonemization wired before the first synth call.
    set_espeak_library()
    # Resolve the voice spec (single id, preset blend, or 'a:60,b:40' custom blend) ONCE
    # into its (voice_id, weight) components, then derive both the language code and the
    # engine-ready voice string from that single parse — avoids re-parsing the same spec
    # in voice_lang_code() and kokoro_voice_string() (and again in gen_audio_segments).
    comps = voicelib.parse_voice_spec(voice)
    # Kokoro runs ONE G2P language code for the whole pipeline: the first component's lang.
    # For a cross-dialect blend like the 'ab_storyteller' preset (US af_heart + UK bf_emma)
    # this means the British voicepack is phonemised under the American 'a' G2P. That is a
    # known single-lang_code limitation of comma-blending in Kokoro, not a parsing bug here;
    # a proper fix would require per-voicepack G2P which the engine does not expose.
    lang_code = comps[0][0][0]
    kokoro_voice = _kokoro_voice_string_from_comps(comps)
    if info.engine == 'mlx':
        if not backends._mlx_importable():
            raise RuntimeError(
                "Backend 'mlx' requires mlx-audio on Apple Silicon. "
                'Install it with: pip install "audiblez[mlx]"')
        synth = _build_mlx_synth(kokoro_voice, lang_code)
        synth.close = lambda: None
        return synth
    # torch path (cpu/cuda/rocm/mps): KPipeline's device= does a proper model .to(device),
    # which (unlike a global torch.set_default_device) actually works on MPS and avoids
    # mutating process-wide torch state.
    pipeline = KPipeline(lang_code=lang_code, repo_id=TORCH_REPO_ID, device=info.torch_device)

    def synth(text, speed):
        # autocast must stay active while the generator runs the model forwards, so the
        # comprehension is consumed inside the context (nullcontext when precision=fp32).
        with gpu.autocast_context(backend, precision):
            return [to_numpy(audio)
                    for _gs, _ps, audio in pipeline(text, voice=kokoro_voice, speed=speed, split_pattern=r'\n\n\n')]
    synth.close = lambda: None  # Kokoro holds no resident child; teardown is a no-op
    return synth


def _build_mlx_synth(voice: str, lang_code: str):
    """Engine closure for the Apple-Silicon-native MLX backend (optional dependency).

    mlx-audio is imported here, never at module scope, so importing audiblez.core does
    not require it on non-Apple platforms.
    """
    from mlx_audio.tts.utils import load_model  # optional dep; only when mlx is selected
    model = load_model(MLX_REPO_ID)

    def synth(text, speed):
        return [np.asarray(seg.audio).reshape(-1)
                for seg in model.generate(text, voice=voice, speed=speed,
                                          lang_code=lang_code, split_pattern=r'\n\n\n')]
    return synth


def _moss_sampling_sig(sampling=None):
    """Stable short fingerprint of the six pinned sampling params (order-independent)."""
    payload = json.dumps(sampling if sampling is not None else MOSS_SAMPLING,
                         sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


class MossProcess:
    """Resident `llama-moss-tts --serve` child, owned over a stdin/stdout pipe.

    Dumb transport: spawn, speak the newline-JSON protocol from ``docs/moss-coprocess-spec.md``
    (request/OK/error/shutdown), hand audio off via 16-bit WAV files (read back with soundfile
    as float32), and drain stderr continuously so a full stderr pipe can't deadlock the child.
    Policy (retry / restart / dead-letter / circuit-breaker) lives in the synth closure; this
    class only RAISES :class:`MossDeath` (EOF / dead poll / response-timeout) and
    :class:`MossError` (a ``status:"error"`` line), keeping both testable against a fake process.

    ``popen_factory`` / ``wav_reader`` are injectable so tests drive a scripted fake child with
    no binary, GPU, or real WAVs.
    """

    def __init__(self, backbone, decoder, encoder=None, clone_ref=None, binary=None,
                 work_dir=None, popen_factory=None, wav_reader=None,
                 ready_timeout=MOSS_SPAWN_READY_TIMEOUT):
        self.binary = str(binary) if binary else MOSS_BINARY  # resolved path (or bare name on PATH)
        self.backbone = str(backbone)
        self.decoder = str(decoder)
        self.encoder = str(encoder) if encoder else None
        self.clone_ref = str(clone_ref) if clone_ref else None
        # Per-instance handoff dir for the request WAVs, kept OUT of the user's output folder
        # (those wavs are transient — one per sentence — and would otherwise litter it). Cleaned
        # up in close(). A unique name per process avoids two runs colliding on req-N.wav.
        base = Path(work_dir) if work_dir else Path(tempfile.gettempdir())
        self.work_dir = base / f'.moss_handoff_{os.getpid()}_{id(self)}'
        self._popen_factory = popen_factory or subprocess.Popen
        # soundfile.read by default; injectable so a fake child's tiny WAVs are read in tests.
        self._wav_reader = wav_reader or (lambda path: soundfile.read(str(path), dtype='float32'))
        self._ready_timeout = ready_timeout
        self.proc = None
        self._req_id = 0
        self._stderr_thread = None
        self._stderr_log = None
        self.spawns = 0  # how many times the child has been (re)spawned this run

    # ── lifecycle ────────────────────────────────────────────────────────────────────
    def _command(self):
        # Exact invocation T1's --serve binary (moss-serve-json-v1) expects; the fixed
        # frame caps size the resident backbone/audio contexts at startup (worst-case).
        cmd = [self.binary, '--serve', '-m', self.backbone,
               '--audio-decoder-model', self.decoder,
               '--language', MOSS_LANGUAGE, '-ngl', MOSS_NGL,
               '--max-prompt-frames', MOSS_MAX_PROMPT_FRAMES,
               '--max-raw-frames', MOSS_MAX_RAW_FRAMES]
        if self.clone_ref and self.encoder:
            cmd += ['--audio-encoder-model', self.encoder, '--reference-audio', self.clone_ref]
        return cmd

    def start(self):
        """Spawn the child and wait for its ``{"status":"ready"}`` line. Raises
        :class:`MossRunAborted` if the binary can't spawn or never reports ready (abort-loud:
        a missing/broken MOSS must NOT silently fall back to Kokoro)."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        try:
            try:
                self.proc = self._popen_factory(
                    self._command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, bufsize=0)
            except (OSError, FileNotFoundError) as e:
                raise MossSpawnError(
                    f"Could not spawn the MOSS engine ({self.binary!r}): {e}. "
                    f"Install/patch the OpenMOSS llama.cpp fork, or run with '-b cpu' to use "
                    f"Kokoro instead.") from e
            self.spawns += 1
            self._start_stderr_drain()
            self._wait_ready()
        except BaseException:
            # Spawn or ready-handshake failed: close() will never run (no synth was built), so
            # remove the handoff dir we just created rather than leak an empty .moss_handoff_*.
            shutil.rmtree(self.work_dir, ignore_errors=True)
            raise
        return self

    def _start_stderr_drain(self):
        """Drain the child's stderr (llama logs) on a daemon thread so a full pipe can't
        deadlock it. Bounded in-memory tail kept only for diagnostics on death."""
        self._stderr_log = []
        stderr = getattr(self.proc, 'stderr', None)
        if stderr is None:
            return

        def drain():
            try:
                for raw in iter(stderr.readline, b''):
                    line = raw.decode('utf-8', 'replace').rstrip('\n') if isinstance(raw, bytes) else str(raw).rstrip('\n')
                    self._stderr_log.append(line)
                    if len(self._stderr_log) > 200:
                        del self._stderr_log[:100]  # keep a bounded tail
            except (ValueError, OSError):
                pass  # stream closed on child exit

        self._stderr_thread = threading.Thread(target=drain, daemon=True)
        self._stderr_thread.start()

    def _wait_ready(self):
        deadline = time.time() + self._ready_timeout
        while True:
            line = self._read_line(deadline)
            if line is None:
                self._kill()
                raise MossSpawnError(
                    f"The MOSS engine ({self.binary!r}) spawned but never became ready "
                    f"(timeout/EOF). {self._stderr_tail()} Run with '-b cpu' to use Kokoro.")
            msg = self._parse_json(line)
            if msg is None:
                continue  # non-JSON stdout noise; the control channel should be clean, ignore
            if msg.get('status') == 'ready':
                return

    def _read_line(self, deadline):
        """Read one stdout line, returning None on EOF or if ``deadline`` passes / child dies.

        Uses a blocking readline on a daemon thread so a wedged child can't block forever; the
        thread is abandoned (daemon) if it never returns — its line is discarded.
        """
        stdout = self.proc.stdout
        result = {}
        done = threading.Event()

        def rd():
            try:
                result['line'] = stdout.readline()
            except (ValueError, OSError):
                result['line'] = b''
            finally:
                done.set()

        t = threading.Thread(target=rd, daemon=True)
        t.start()
        while not done.wait(timeout=0.1):
            if time.time() > deadline:
                return None
            if self.proc.poll() is not None:
                # Child exited; give the drain a beat to flush a final line, else EOF.
                if done.wait(timeout=0.5):
                    break
                return None
        raw = result.get('line', b'')
        if raw in (b'', ''):
            return None  # EOF
        return raw.decode('utf-8', 'replace').rstrip('\n') if isinstance(raw, bytes) else str(raw).rstrip('\n')

    def _write(self, data, timeout):
        """Write+flush ``data`` to the child's stdin under a deadline. Returns True on success,
        False if it didn't complete within ``timeout`` (or the pipe broke).

        ``bufsize=0`` means a write to a FULL stdin pipe blocks at the OS level with no timeout,
        so a wedged-but-alive child (the exact failure the death-timeout design exists to survive)
        would hang the synth hot path / teardown forever. Doing the write on a daemon thread turns
        a stuck write into a recoverable signal (caller treats False as a death -> restart/kill)
        instead of a hard hang. Mirrors :meth:`_read_line`'s watchdog pattern for the read side.
        """
        proc = self.proc
        if proc is None or proc.stdin is None:
            return False
        stdin = proc.stdin
        result = {}
        done = threading.Event()

        def wr():
            try:
                stdin.write(data)
                stdin.flush()
                result['ok'] = True
            except (BrokenPipeError, ValueError, OSError):
                result['ok'] = False
            finally:
                done.set()

        threading.Thread(target=wr, daemon=True).start()
        if not done.wait(timeout=timeout):
            return False  # write wedged: child alive but not draining its stdin
        return result.get('ok', False)

    @staticmethod
    def _parse_json(line):
        try:
            msg = json.loads(line)
            return msg if isinstance(msg, dict) else None
        except (ValueError, TypeError):
            return None

    def _stderr_tail(self, n=8):
        tail = (self._stderr_log or [])[-n:]
        return f"Last child stderr:\n{chr(10).join(tail)}" if tail else "(no child stderr captured)"

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    # ── request/response ─────────────────────────────────────────────────────────────
    def synth(self, text, timeout):
        """Issue ONE synth request for ``text`` and block for the response.

        Returns a float32 1-D numpy array @ 24 kHz. Raises :class:`MossError` on a
        ``status:"error"`` line (child healthy), :class:`MossPermanentError` for deterministic
        codes (bad_request/empty_audio), and :class:`MossDeath` on EOF / dead poll / timeout.
        """
        if not self.alive():
            raise MossDeath('child not running')
        self._req_id += 1
        req_id = self._req_id
        wav_out = self.work_dir / f'moss-req-{req_id}.wav'
        wav_out.unlink(missing_ok=True)  # stale-WAV guard: never read a previous request's audio
        request = {
            'v': 1, 'id': req_id, 'op': 'synth', 'text': text,
            'wav_out': str(wav_out), 'seed': MOSS_SEED,
            'sampling': dict(MOSS_SAMPLING), 'max_new_tokens': MOSS_MAX_NEW_TOKENS,
        }
        payload = (json.dumps(request, ensure_ascii=False) + '\n').encode('utf-8')
        # Bound the write: a wedged-but-alive child with a full stdin pipe would otherwise block
        # this write forever (before the deadline read-loop is ever reached). A non-completing
        # write is a death -> the closure restarts the child and re-dispatches the sentence.
        if not self._write(payload, timeout=timeout):
            raise MossDeath(f'request {req_id} write did not complete (wedged child or broken pipe)')

        deadline = time.time() + timeout
        while True:
            line = self._read_line(deadline)
            if line is None:
                raise MossDeath(
                    f'no response to request {req_id} (EOF/timeout/poll). {self._stderr_tail()}')
            msg = self._parse_json(line)
            if msg is None or msg.get('id') != req_id:
                continue  # stray/log line, or a response to an earlier id; keep reading
            try:
                return self._handle_response(msg, wav_out)
            finally:
                # The handoff WAV is transient (one per sentence) — drop it once read so a
                # book doesn't leave thousands of req-N.wavs behind. (No-op on an error path
                # where the child never wrote it.)
                Path(wav_out).unlink(missing_ok=True)

    def _handle_response(self, msg, wav_out):
        status = msg.get('status')
        if status == 'error':
            code = msg.get('code', 'gen_failed')
            detail = msg.get('message', '')
            # Deterministic codes can't be fixed by retrying the same pinned-seed request.
            if code in ('bad_request', 'empty_audio'):
                raise MossPermanentError(f'MOSS {code}: {detail}')
            raise MossError(f'MOSS {code}: {detail}')
        if status != 'ok':
            raise MossError(f'unexpected MOSS response status {status!r}')
        frames = msg.get('frames', 0)
        if not frames or frames <= 0:
            # Spec forbids ok-with-zero; treat as permanent so np.zeros(0) is never cached.
            raise MossPermanentError('MOSS reported ok with no frames')
        if not Path(wav_out).exists():
            raise MossError(f'MOSS reported ok but {wav_out} is missing')
        audio, sr = self._wav_reader(wav_out)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise MossPermanentError('MOSS wav decoded to empty audio')
        return audio

    # ── teardown ─────────────────────────────────────────────────────────────────────
    def close(self):
        """Graceful shutdown ({"op":"shutdown"} then wait), escalating to kill. Idempotent;
        safe to call from a ``finally`` even if the child already died or never spawned."""
        proc = self.proc
        try:
            if proc is not None and proc.poll() is None and proc.stdin is not None:
                # Bound the graceful-shutdown write too: a wedged-but-alive child would hang
                # core.main's finally forever (reachable on the breaker path, where the child
                # is still alive at teardown). If the write doesn't complete, kill rather than
                # wait on a graceful exit the child can't perform.
                if self._write(b'{"op":"shutdown"}\n', timeout=MOSS_SHUTDOWN_WRITE_TIMEOUT):
                    try:
                        proc.wait(timeout=10)
                    except Exception:
                        self._kill()
                else:
                    self._kill()
        finally:
            self.proc = None
            # Remove the per-instance handoff dir (and any straggler WAVs) so nothing leaks —
            # even when the child never spawned (start() already created the dir).
            shutil.rmtree(self.work_dir, ignore_errors=True)

    def _kill(self):
        proc = self.proc
        if proc is None:
            return
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass

    def restart(self):
        """Kill (if needed) and re-spawn the child — used by the closure on a death."""
        self._kill()
        self.proc = None
        return self.start()


def _build_llamacpp_synth(voice, lang_code, clone_ref=None, gguf_dir=None,
                          moss_factory=None, work_dir=None, clock=time.time):
    """Engine closure for the MOSS resident-pipe co-process (engine='llamacpp').

    Returns ``synth(text, speed) -> list[np.ndarray] @ 24 kHz``. MOSS issues ONE pipe request
    per sentence. ``synth`` honors the batch contract: a ``\\n\\n\\n``-joined ``text`` is split
    and one request is issued per piece, returning a list aligned per sentence. ``speed`` is
    IGNORED here — MOSS has no speed knob; speed is applied downstream via ffmpeg atempo at
    chapter assembly, and the MOSS sentence cache stays speed-agnostic (ADR 0002).

    The closure owns the failure-vs-death policy:
      * a ``status:"error"`` line -> ``MossError`` propagates to ``_synth_one_or_silence``
        (dead-letter + silence; child healthy);
      * a death (EOF / dead poll / timeout) -> restart the child and re-dispatch the SAME
        sentence; after K=:data:`MOSS_DEATHS_BEFORE_POISON` consecutive deaths raise
        ``MossPermanentError`` (poison-pill -> dead-letter, ``_retry`` won't re-spin it);
      * a systemic restart rate (circuit-breaker) raises ``MossRunAborted`` -> whole run aborts.
    ``moss_factory`` is injectable so tests supply a fake MossProcess (no binary/GPU).
    """
    if moss_factory is None:
        # Resolve the binary + GGUF paths through the SINGLE source of truth (T3's
        # backends.moss_paths — the same resolution doctor preflights), so the run spawns
        # exactly what --doctor checked. `gguf_dir` (if given) overrides for tests/odd layouts.
        if gguf_dir is not None:
            gd = Path(gguf_dir)
            binary, backbone = MOSS_BINARY, gd / MOSS_BACKBONE_GGUF
            decoder, encoder = gd / MOSS_DECODER_GGUF, (gd / MOSS_ENCODER_GGUF if clone_ref else None)
        else:
            p = backends.moss_paths()
            binary = str(p['binary']) if p.get('binary') else MOSS_BINARY
            backbone, decoder, encoder = p.get('backbone'), p.get('decoder'), p.get('encoder')

        def moss_factory():
            return MossProcess(binary=binary, backbone=backbone, decoder=decoder, encoder=encoder,
                               clone_ref=clone_ref, work_dir=work_dir)
    moss = moss_factory()
    moss.start()

    # Circuit-breaker: timestamps of recent RESTARTS (an infra death we recovered from). It must
    # distinguish a systemic STORM (e.g. VRAM exhaustion: many restarts bunched in time) from a
    # few sparse transient hiccups spread across a long book — so it is RATE-windowed: it trips
    # only when > MOSS_MAX_RESTARTS_IN_WINDOW restarts fall inside a trailing time window. Poison-
    # pill deaths are NOT counted here: they're handled locally (dead-letter the sentence), and
    # counting them would let one bad sentence trip the global breaker and abort the whole book.
    restart_times = []

    def _note_restart():
        now = clock()
        restart_times.append(now)
        cutoff = now - MOSS_RESTART_WINDOW_SECONDS
        restart_times[:] = [t for t in restart_times if t >= cutoff]
        if len(restart_times) > MOSS_MAX_RESTARTS_IN_WINDOW:
            raise MossCircuitBreakerError(
                f'MOSS engine restarted {len(restart_times)} times within '
                f'{MOSS_RESTART_WINDOW_SECONDS:.0f}s — a systemic failure (e.g. VRAM '
                f'exhaustion mid-book). Aborting to preserve partial chapters + dead-letters; '
                f"re-run to resume, or use '-b cpu' for Kokoro.")

    def _synth_one(text):
        """One sentence -> one array, with the death/restart/poison-pill loop.

        ``deaths`` is per-call (reset on success/return); it counts CONSECUTIVE deaths on THIS
        sentence. K=MOSS_DEATHS_BEFORE_POISON deaths -> poison-pill -> dead-letter (a permanent
        error _retry won't re-spin). An infra death below K restarts the child and re-dispatches
        the SAME sentence; only those restarts feed the (rate-windowed) global circuit-breaker.
        """
        deadline_s = max(MOSS_RESPONSE_TIMEOUT_FLOOR, len(text) * MOSS_TIMEOUT_SECONDS_PER_CHAR)
        deaths = 0
        while True:
            try:
                return moss.synth(text, timeout=deadline_s)
            except MossDeath as death:
                deaths += 1
                if deaths >= MOSS_DEATHS_BEFORE_POISON:
                    # Poison pill: this sentence keeps killing the child. Dead-letter it (raised
                    # as a permanent error so _retry won't re-spin) instead of crash-looping. The
                    # child stays dead; restart it so the NEXT sentence has a live engine.
                    try:
                        moss.restart()
                    except MossRunAborted:
                        raise
                    except Exception:
                        pass
                    raise MossPermanentError(
                        f'MOSS child died {deaths}x on the same sentence (poison pill); '
                        f'dead-lettering it.') from death
                # Infra death (first one): restart and re-dispatch the SAME sentence. The
                # rate-windowed breaker (in _note_restart) aborts only on a systemic storm.
                _note_restart()
                moss.restart()

    def synth(text, speed):
        pieces = text.split('\n\n\n') if '\n\n\n' in text else [text]
        return [_synth_one(piece) for piece in pieces]

    synth.close = moss.close   # core.main's finally + the GUI teardown path call this
    synth.moss = moss          # exposed for tests/diagnostics
    return synth


def ewma(prev, sample, alpha=EWMA_ALPHA):
    """Exponentially-weighted moving average of a throughput sample.

    ``prev`` is the previous estimate (e.g. the flat GPU/CPU_CHARS_PER_SEC prior on the
    first call); ``None`` seeds the average with the first real sample.
    """
    if prev is None:
        return sample
    return alpha * sample + (1 - alpha) * prev


def _update_eta(stats, measured_chars, elapsed):
    """Fold one measurement into stats: rolling chars/sec, progress %, and ETA string.

    The FIRST real measurement REPLACES the flat GPU/CPU_CHARS_PER_SEC prior outright
    (the prior is only a pre-run guess); subsequent measurements blend into the EWMA.
    Otherwise the prior — often ~10x off the true rate — would dominate the ETA for the
    first ~15 batches.
    """
    stats.processed_chars += measured_chars
    if elapsed and elapsed > 0:
        sample = measured_chars / elapsed
        stats.chars_per_sec = ewma(stats.chars_per_sec, sample) if getattr(stats, 'measured', False) else sample
        stats.measured = True
    stats.progress = min(100, stats.processed_chars * 100 // stats.total_chars) if stats.total_chars else 100
    remaining_chars = max(0, stats.total_chars - stats.processed_chars)  # intro chars can overshoot
    stats.eta = strfdelta(remaining_chars / (stats.chars_per_sec or 1))


def pack_sentences(sentences, batch_max_chars=BATCH_MAX_CHARS):
    """Group consecutive sentences into batches of at most ``batch_max_chars`` chars.

    A batch is joined with ``\\n\\n\\n`` and handed to synth in one call. A single
    sentence longer than the cap becomes its own (oversized) batch rather than being
    dropped or merged — preserving the per-sentence fallback for long sentences.
    Order is always preserved, so chapter audio concatenates identically.
    """
    batches, current, current_len = [], [], 0
    for s in sentences:
        if current and current_len + len(s) > batch_max_chars:
            batches.append(current)
            current, current_len = [], 0
        current.append(s)
        current_len += len(s)
    if current:
        batches.append(current)
    return batches


def _silence_for(text):
    """A silent placeholder segment, length roughly proportional to the failed text."""
    seconds = max(0.3, len(text) / FALLBACK_SILENCE_CHARS_PER_SEC)
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


def _retry(fn, retries, backoff=SYNTH_RETRY_BACKOFF):
    """Call ``fn`` up to ``retries + 1`` times (at least once); return it or re-raise.

    A deterministic error (:data:`_PERMANENT_SYNTH_ERRORS` — bad input / code bug) is
    re-raised immediately, since retrying the identical call cannot help and only wastes
    work (badly so inside a batch). Transient errors back off between attempts so a fault
    that needs a moment to clear (e.g. a GPU OOM still holding memory) gets one.
    """
    last = RuntimeError('no attempts made')
    attempts = max(1, retries + 1)
    for i in range(attempts):
        try:
            return fn()
        except _PERMANENT_SYNTH_ERRORS:
            raise
        except Exception as e:
            last = e
            if backoff and i < attempts - 1:
                time.sleep(backoff)
    raise last


def _record_dead_letter(path, chapter_label, text, error):
    """Append one failed sentence to a chapter's ``.failed.jsonl`` dead-letter file."""
    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'chapter': chapter_label, 'text': text, 'error': error},
                               ensure_ascii=False) + '\n')
    except OSError as e:
        print(f'Warning: could not write dead-letter entry to {path}: {e}')


def _count_dead_letters(path):
    """Number of failed-sentence records in a chapter's dead-letter file (0 if absent)."""
    p = Path(path)
    if not p.exists():
        return 0
    try:
        return sum(1 for line in p.read_text(encoding='utf-8').splitlines() if line.strip())
    except OSError:
        return 0


def _synth_one_or_silence(synth, sentence, speed, retries, chapter_label, dead_letter_path):
    """Synth one sentence with bounded retries; on persistent failure splice silence.

    Returns ``(segments, failed)``: the synthesized segment list (or ``[silence]`` on
    failure, with the sentence recorded to the dead-letter file). Shared by the batched
    and cached paths so their retry/silence/dead-letter behavior can never diverge.
    """
    try:
        return _retry(lambda: synth(sentence, speed), retries), False
    except Exception as e:
        print(f'\033[91mSentence failed after {retries + 1} attempts; inserting silence.\033[0m')
        if dead_letter_path:
            _record_dead_letter(dead_letter_path, chapter_label, sentence, str(e))
        return [_silence_for(sentence)], True


def _synth_batch(synth, batch, speed, retries=SYNTH_RETRIES, chapter_label=None,
                 dead_letter_path=None, on_fallback=None):
    """Synthesize a batch of sentences, never raising for a single bad sentence.

    Tries the whole batch in one call (with bounded retries). On persistent failure it
    falls back to per-sentence synthesis; any sentence that still fails after its retries
    is replaced with silence and recorded to a dead-letter file, so one bad sentence (or
    a failed batched call) degrades to a gap instead of nuking the entire chapter.
    ``on_fallback`` (if given) is called once when the batch path fails — the caller uses
    it to keep that batch's inflated wall time out of the throughput EWMA.
    """
    try:
        return _retry(lambda: synth('\n\n\n'.join(batch), speed), retries)
    except Exception as e:
        print(f'\033[91mBatch synth failed ({e}); retrying sentence-by-sentence.\033[0m')
        if on_fallback:
            on_fallback()
    segments = []
    for s in batch:
        segs, _ = _synth_one_or_silence(synth, s, speed, retries, chapter_label, dead_letter_path)
        segments.extend(segs)
    return segments


def _account(stats, n_sentences, chars, elapsed, post_event, chapter_label):
    """Fold one synthesized unit (batch or sentence) into progress/ETA + heartbeat output.

    ``elapsed`` of 0 (e.g. a cache hit) advances progress without polluting the measured
    chars/sec EWMA, since no real synthesis happened.
    """
    if not stats:
        return
    before = getattr(stats, 'sentences_done', 0)
    _update_eta(stats, chars, elapsed)
    stats.sentences_done = before + n_sentences
    # Snapshot the scalar fields: the worker keeps mutating `stats`, so passing it live
    # lets the GUI thread read a torn mix of fields from different updates.
    if post_event: post_event('CORE_PROGRESS', stats=SimpleNamespace(**vars(stats)))
    if stats.sentences_done // HEARTBEAT_EVERY > before // HEARTBEAT_EVERY:
        where = f' [{chapter_label}]' if chapter_label else ''
        print(f'♥ heartbeat{where}: {stats.sentences_done} sentences, '
              f'{stats.chars_per_sec:.0f} chars/sec (measured), ETA {stats.eta}')
    print(f'Estimated time remaining: {stats.eta}')
    print('Progress:', f'{stats.progress}%\n')


def _moss_repo_id():
    """MOSS ``repo_id`` for the cache key — delegates to the SINGLE source of truth.

    The GGUF-identity derivation lives in :func:`audiblez.backends.moss_repo_id` (T3 owns it;
    `doctor` consumes it too), so core must NOT re-derive "which GGUFs" or the two definitions
    drift and the cache serves stale audio after a model swap. The clone reference is NOT in the
    repo_id here — it enters the cache key on the *voice* axis via
    :func:`audiblez.backends.clone_voice_id` (see ``_cache_key_fields``).
    """
    return backends.moss_repo_id()


def _cache_key_fields(backend, voice, speed, precision='fp32', clone_ref=None):
    """Versioned key components that, with the sentence text, address a synth result.

    ``precision`` changes the waveform (fp16/bf16 autocast) but ONLY on GPU torch
    backends; it is a no-op on cpu/mlx/mps, so it is normalised to 'fp32' there to
    avoid fragmenting the cache pointlessly. Including it stops a cache populated under
    one precision from serving the wrong waveform on a re-run with another.

    For the MOSS engine ('llamacpp') the key hashes the PINNED seed + the six sampling params
    (via ``sampling_sig``) and the three GGUF identities (via ``backends.moss_repo_id()`` — the
    shared source of truth). A voice CLONE reference is folded onto the *voice* axis via
    ``backends.clone_voice_id`` (the reference WAV *is* the voice) so two different clips never
    collide on a constant — dropping it would serve the prior clone's audio on a re-run (a banned
    silent-wrongness). ``engine`` already separates MOSS from Kokoro (no CACHE_VERSION bump for
    engine separation — ADR 0002). Speed is pinned to 1.0 (MOSS synth@1.0; speed = atempo later),
    so the sentence cache stays speed-agnostic.
    """
    info = backends.BACKENDS[backend]
    if info.engine == 'llamacpp':
        moss_voice = backends.clone_voice_id(clone_ref) if clone_ref else MOSS_DEFAULT_VOICE
        return dict(engine='llamacpp', repo_id=backends.moss_repo_id(), voice=moss_voice,
                    speed=1.0,  # MOSS sentence cache is speed-agnostic (atempo applied later)
                    precision='fp32', max_sentence_length=MAX_SENTENCE_LENGTH,
                    spacy_version=spacy.__version__,
                    seed=MOSS_SEED, sampling_sig=_moss_sampling_sig())
    repo_id = MLX_REPO_ID if info.engine == 'mlx' else TORCH_REPO_ID
    eff_precision = precision if gpu._is_torch_gpu(backend) else 'fp32'
    return dict(engine=info.engine, repo_id=repo_id, voice=voice, speed=speed,
                precision=eff_precision, max_sentence_length=MAX_SENTENCE_LENGTH,
                spacy_version=spacy.__version__)


def _split_into_sentences(doc, lang_code):
    if lang_code in 'ab':
        return [s.text for s in doc.sents]
    # For non-english languages, Kokoro truncates long sentences, so we split them manually.
    sentences = []
    for sent in list(doc.sents):
        if len(sent.text) > MAX_SENTENCE_LENGTH:
            print(f'Warning: Sentence too long ({len(sent.text)} chars), splitting into smaller sentences.')
            sentences.extend(split_long_sentence(sent.text, MAX_SENTENCE_LENGTH))
        else:
            sentences.append(sent.text)
    return sentences


def gen_audio_segments(synth, text, voice, speed, stats=None, max_sentences=None, post_event=None,
                       chapter_label=None, batch_max_chars=BATCH_MAX_CHARS,
                       synth_retries=SYNTH_RETRIES, dead_letter_path=None,
                       cache=None, cache_key_fields=None, lang_code=None, coarse=False):
    nlp = load_spacy()
    audio_segments = []
    # build_synthesizer already parsed the spec; let it pass the resolved lang_code in to
    # avoid re-parsing per chapter. Fall back to deriving it when called standalone.
    if lang_code is None:
        lang_code = voicelib.voice_lang_code(voice)
    sentences = _split_into_sentences(nlp(text), lang_code)
    if max_sentences:
        sentences = sentences[:max_sentences]

    if coarse:
        # Opt-in coarse-chunk mode (ADR 0005, MOSS only — the caller gates this on the engine).
        # Consecutive sentences are packed into utterance chunks capped at the correctness-
        # validated 16.32 s (chunking.MAX_CHUNK_SECONDS), and each chunk is ONE MOSS request.
        # The cache/failure/edit unit is the whole chunk: a chunk's joined text is its cache key
        # (a multi-sentence chunk can't collide with a per-sentence entry — different text), and a
        # failed chunk degrades to silence for the whole paragraph + one dead-letter. An oversize
        # single sentence becomes its own chunk (pack_chunks never drops or sub-splits text).
        from audiblez import chunking
        from audiblez import cache as cache_mod
        for chunk in chunking.pack_chunks(sentences):
            chunk_text = ' '.join(chunk)
            key = cache_mod.make_key(text=chunk_text, **(cache_key_fields or {})) if cache is not None else None
            if cache is not None:
                hit = cache.get(key)
                if hit is not None:
                    audio_segments.append(hit)
                    _account(stats, len(chunk), len(chunk_text), 0.0, post_event, chapter_label)
                    continue
            t0 = time.time()
            segs, failed = _synth_one_or_silence(synth, chunk_text, speed, synth_retries,
                                                 chapter_label, dead_letter_path)
            audio = np.concatenate(segs) if segs else np.zeros(0, dtype=np.float32)
            # Never cache a failure or empty audio (would serve silence as a hit forever).
            if cache is not None and not failed and audio.size > 0:
                cache.put(key, audio)
            audio_segments.append(audio)
            _account(stats, len(chunk), len(chunk_text), time.time() - t0, post_event, chapter_label)
        return audio_segments

    if cache is not None:
        # KEYSTONE slice 1: cache per sentence (the addressable unit). Misses are synthesized
        # one sentence at a time so each stored array maps 1:1 to a sentence; batching the
        # misses is a documented later optimization.
        from audiblez import cache as cache_mod
        for sent in sentences:
            key = cache_mod.make_key(text=sent, **(cache_key_fields or {}))
            hit = cache.get(key)
            if hit is not None:
                audio_segments.append(hit)
                _account(stats, 1, len(sent), 0.0, post_event, chapter_label)
                continue
            t0 = time.time()
            segs, failed = _synth_one_or_silence(synth, sent, speed, synth_retries,
                                                 chapter_label, dead_letter_path)
            audio = np.concatenate(segs) if segs else np.zeros(0, dtype=np.float32)
            # Never persist a failure OR an empty result: caching np.zeros(0) would serve
            # silent "no audio" for that sentence on every future run, hiding the failure.
            if not failed and audio.size > 0:
                cache.put(key, audio)
            audio_segments.append(audio)
            _account(stats, 1, len(sent), time.time() - t0, post_event, chapter_label)
        return audio_segments

    for batch in pack_sentences(sentences, batch_max_chars):
        t0 = time.time()
        # Kokoro re-splits on \n\n\n, yielding one audio segment per sentence, in order.
        # Bounded retries + per-sentence fallback so a bad sentence becomes a gap, not a crash.
        fell_back = []
        audio_segments.extend(_synth_batch(synth, batch, speed, retries=synth_retries,
                                           chapter_label=chapter_label,
                                           dead_letter_path=dead_letter_path,
                                           on_fallback=lambda fb=fell_back: fb.append(1)))
        # A fallback batch spends wall time on retries while producing little real audio;
        # don't let that pollute the measured chars/sec (elapsed=0 advances progress only).
        elapsed = 0.0 if fell_back else (time.time() - t0)
        _account(stats, len(batch), sum(len(s) for s in batch), elapsed, post_event, chapter_label)
    return audio_segments


def gen_text(text, voice='af_heart', output_file='text.wav', speed=1, play=False, backend='cpu',
             precision='fp32'):
    load_spacy()
    synth = build_synthesizer(voice, backend, precision=precision)
    try:
        audio_segments = gen_audio_segments(synth, text, voice=voice, speed=speed)
    finally:
        getattr(synth, 'close', lambda: None)()  # close the resident MOSS child (no-op for Kokoro)
    if not audio_segments:
        print('Warning: no audio generated for the given text.')
        return
    final_audio = np.concatenate(audio_segments)
    soundfile.write(output_file, final_audio, sample_rate)
    # MOSS has no speed knob (synth@1.0); re-stretch to `speed` via atempo so a one-off
    # gen_text render honors --speed like the chapter path does.
    if backends.BACKENDS[backend].engine == 'llamacpp':
        _apply_atempo(output_file, speed)
    if play:
        subprocess.run(['ffplay', '-autoexit', '-nodisp', output_file])


def make_trailer(file_path, voice, output_file='trailer.wav', speed=1.0, backend='cpu',
                 sentences_per_chapter=TRAILER_SENTENCES_PER_CHAPTER, selected_chapters=None,
                 max_chapters=None, precision='fp32', output_folder='.', clone_ref=None):
    """Render a short audio sampler of a book: the opening sentences of each chapter.

    For each detected chapter, speaks a "Chapter N" label then its first
    ``sentences_per_chapter`` sentences, separated by short silent gaps. Lets a user
    hear in a couple of minutes whether chapter detection, voice, and pronunciation are
    right BEFORE committing to a full hour-long synthesis — and audibly reveals a
    misdetected chapter. Modeled on :func:`gen_text`; returns the output path (or None).
    """
    load_spacy()
    # Only parse the epub when chapters weren't supplied (the GUI already has them) —
    # re-reading + extracting a whole book just to discard it is a multi-second waste.
    if selected_chapters:
        chapters = selected_chapters
    else:
        chapters = find_good_chapters(find_document_chapters_and_extract_texts(epub.read_epub(file_path)))
    gap = np.zeros(int(TRAILER_GAP_SECONDS * sample_rate), dtype=np.float32)
    preview_chars = MAX_SENTENCE_LENGTH * (sentences_per_chapter + 1)

    pieces, sampled = [], 0
    synth = None  # built inside the try so load_lexicon raising can't orphan the MOSS child
    try:
        synth = build_synthesizer(voice, backend, precision=precision, clone_ref=clone_ref)
        # Same scope as seeding and the real run, so the trailer auditions the SAME pronunciations.
        book_lexicon = lexicon.load_lexicon(lexicon.lexicon_path(file_path, output_folder))
        for chapter in chapters:
            if max_chapters and sampled >= max_chapters:
                break
            # Only the opening sentences are sampled, so feed a prefix: gen_audio_segments
            # spaCy-parses ALL of its text before slicing to max_sentences.
            raw = chapter.extracted_text.strip()[:preview_chars]
            text = lexicon.apply_lexicon(raw, book_lexicon)
            if len(text) < 10:
                continue
            # Number by the sampled sequence (1..N over included chapters) so spoken labels are
            # consecutive and match the final m4b's chapter numbering, not the raw list index.
            pieces.extend(gen_audio_segments(synth, f'Chapter {sampled + 1}.', voice=voice, speed=speed))
            pieces.append(gap)
            pieces.extend(gen_audio_segments(synth, text, voice=voice, speed=speed,
                                             max_sentences=sentences_per_chapter))
            pieces.append(gap)
            sampled += 1
    finally:
        getattr(synth, 'close', lambda: None)()  # close the resident MOSS child (no-op for Kokoro)

    if not pieces:
        print('No chapters to put in the trailer.')
        return None
    soundfile.write(output_file, np.concatenate(pieces), sample_rate)
    # MOSS synthesizes at 1.0 (no speed knob); re-stretch the trailer to `speed` once so the
    # audition matches what the real render will produce (US-26). Kokoro already baked it in.
    if backends.BACKENDS[backend].engine == 'llamacpp':
        _apply_atempo(output_file, speed)
    print(f'Trailer written to {output_file} ({sampled} chapter(s) sampled).')
    return output_file


def find_document_chapters_and_extract_texts(book):
    """Returns every chapter that is an ITEM_DOCUMENT and enriches each chapter with extracted_text."""
    document_chapters = []
    for chapter in book.get_items():
        if chapter.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        xml = chapter.get_body_content()
        soup = BeautifulSoup(xml, features='lxml')
        chapter.extracted_text = ''
        html_content_tags = ['title', 'p', 'h1', 'h2', 'h3', 'h4', 'li']
        for text in [c.text.strip() for c in soup.find_all(html_content_tags) if c.text]:
            if not text.endswith('.'):
                text += '.'
            chapter.extracted_text += text + '\n'
        document_chapters.append(chapter)
    for i, c in enumerate(document_chapters):
        c.chapter_index = i  # this is used in the UI to identify chapters
    return document_chapters


def is_chapter(c):
    name = c.get_name().lower()
    has_min_len = len(c.extracted_text) > 100
    title_looks_like_chapter = bool(
        'chapter' in name.lower()
        or re.search(r'part_?\d{1,3}', name)
        or re.search(r'split_?\d{1,3}', name)
        or re.search(r'ch_?\d{1,3}', name)
        or re.search(r'chap_?\d{1,3}', name)
    )
    return has_min_len and title_looks_like_chapter


def chapter_beginning_one_liner(c, chars=20):
    s = c.extracted_text[:chars].strip().replace('\n', ' ').replace('\r', ' ')
    return s + '…' if len(s) > 0 else ''


def find_good_chapters(document_chapters):
    chapters = [c for c in document_chapters if c.get_type() == ebooklib.ITEM_DOCUMENT and is_chapter(c)]
    if len(chapters) == 0:
        print('Not easy to recognize the chapters, defaulting to all non-empty documents.')
        chapters = [c for c in document_chapters if c.get_type() == ebooklib.ITEM_DOCUMENT and len(c.extracted_text) > 10]
    return chapters


def pick_chapters(chapters):
    # Display the document name, the length and first 50 characters of the text
    chapters_by_names = {
        f'{c.get_name()}\t({len(c.extracted_text)} chars)\t[{chapter_beginning_one_liner(c, 50)}]': c
        for c in chapters}
    title = 'Select which chapters to read in the audiobook'
    ret = pick(list(chapters_by_names.keys()), title, multiselect=True, min_selection_count=1)
    selected_chapters_out_of_order = [chapters_by_names[r[0]] for r in ret]
    selected_ids = {id(c) for c in selected_chapters_out_of_order}
    selected_chapters = [c for c in chapters if id(c) in selected_ids]
    return selected_chapters


def strfdelta(tdelta, fmt='{D:02}d {H:02}h {M:02}m {S:02}s'):
    remainder = int(tdelta)
    f = Formatter()
    desired_fields = [field_tuple[1] for field_tuple in f.parse(fmt)]
    possible_fields = ('W', 'D', 'H', 'M', 'S')
    constants = {'W': 604800, 'D': 86400, 'H': 3600, 'M': 60, 'S': 1}
    values = {}
    for field in possible_fields:
        if field in desired_fields and field in constants:
            values[field], remainder = divmod(remainder, constants[field])
    return f.format(fmt, **values)


def _escape_concat_path(path):
    """Escape a path for ffmpeg's concat demuxer list file (single-quote syntax).

    ffmpeg's concat demuxer wraps each entry as ``file '<path>'``; a literal single
    quote inside the path must be written as ``'\\''`` (close quote, escaped quote,
    reopen quote), otherwise paths containing apostrophes break the parse. The demuxer is
    line-oriented, so any CR/LF is stripped too: a newline would otherwise split one
    ``file '...'`` entry across lines and let a crafted epub inject a concat directive.
    """
    return str(path).replace('\r', '').replace('\n', '').replace("'", "'\\''")


def _escape_ffmetadata(value):
    """Escape a value for the FFMETADATA1 format.

    The format is INI-like: ``=``, ``;``, ``#`` and ``\\`` are special and newlines
    delimit fields. Escaping these prevents untrusted epub title/author metadata from
    injecting extra fields.
    """
    text = str(value)
    for ch in ('\\', '=', ';', '#'):
        text = text.replace(ch, '\\' + ch)
    return text.replace('\r', ' ').replace('\n', ' ')


def probe_duration(file_name):
    """Return an audio file's duration in seconds, or None if it can't be probed."""
    args = ['ffprobe', '-i', str(file_name), '-show_entries', 'format=duration',
            '-v', 'quiet', '-of', 'default=noprint_wrappers=1:nokey=1']
    try:
        proc = subprocess.run(args, capture_output=True, text=True, check=True)
        return float(proc.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        # FileNotFoundError = ffprobe not on PATH; callers must degrade gracefully.
        print(f'Warning: could not probe duration of {file_name}: {e}')
        return None


def _render_signature(book_lexicon, speed, precision, backend=None, clone_ref=None, coarse=False):
    """Compact fingerprint of render-affecting inputs NOT already in the chapter filename.

    The wav filename encodes book stem, chapter index, and voice; this captures the rest
    (active lexicon, speed, precision). Stored next to each chapter wav as a ``.sig`` so
    that editing the lexicon (or changing speed/precision) invalidates the existing wav on
    the resume-by-skip path instead of silently reusing audio from the old settings.

    For the MOSS engine ('llamacpp') the chapter filename encodes only book/index/voice and
    speed is applied post-hoc via atempo, so the GGUF identity, pinned seed, sampling params,
    and any voice-clone reference are otherwise UNcaptured. Fold them in (the same axes as
    ``_cache_key_fields``) — else the resume-by-skip gate reuses a stale chapter wav after a
    model/sampling/clone swap, which also masks the clone-ref filename collision (two clone
    refs share a wav name): both are the banned silent-wrongness class. Kokoro backends emit
    the exact pre-MOSS string (no suffix), so existing ``.sig`` files stay valid.
    """
    sig = (f'lex={lexicon.fingerprint(book_lexicon)}'
           f';speed={round(float(speed), 4)};precision={precision}')
    info = backends.BACKENDS.get(backend) if backend is not None else None
    if info is not None and info.engine == 'llamacpp':
        clone = f';clone={backends.clone_voice_id(clone_ref)}' if clone_ref else ''
        # Coarse mode synthesizes a chunk as ONE utterance — materially different audio from the
        # per-sentence concat — so it IS a render axis: a coarse/per-sentence toggle MUST bust the
        # chapter wav on resume (story 11; the resume-by-skip landmine). Appended ONLY when coarse
        # so existing per-sentence .sig files stay valid (no needless regeneration).
        coarse_sig = ';coarse=1' if coarse else ''
        sig += (f';engine=llamacpp;repo={backends.moss_repo_id()}'
                f';seed={MOSS_SEED};sampling={_moss_sampling_sig()}{clone}{coarse_sig}')
    return sig


def _chapter_is_complete(wav_path, render_signature):
    """Whether an existing chapter wav is a complete, current render safe to skip.

    False when (a) sentences were dead-lettered (a non-empty ``.failed.jsonl`` sibling) —
    so a re-run retries them instead of shipping the silence forever — or (b) a recorded
    ``.sig`` disagrees with the current render signature. A missing ``.sig`` (a wav from
    before this guard) falls back to the validity check alone for backward compatibility.
    """
    if _count_dead_letters(Path(wav_path).with_suffix('.failed.jsonl')) > 0:
        return False
    sig_path = Path(wav_path).with_suffix('.sig')
    if sig_path.exists():
        try:
            return sig_path.read_text(encoding='utf-8').strip() == render_signature
        except OSError:
            return False
    return True


def _robust_duration(path):
    """Audio duration in seconds via ffprobe, falling back to the wav header (soundfile).

    soundfile reads the duration from the file header with no external process, so this
    still returns a real value when ffprobe is absent — which is why chapter markers must
    use it rather than ``probe_duration(...) or 0.0`` (that zeroes every marker)."""
    duration = probe_duration(path)
    if duration is None:
        try:
            duration = soundfile.info(str(path)).duration
        except Exception:
            duration = None
    return duration


def _atempo_filter(speed):
    """ffmpeg ``atempo`` filter chain for ``speed`` (atempo accepts 0.5–2.0 per stage, so a
    speed outside that is split into a chain whose product is ``speed``)."""
    remaining = float(speed)
    stages = []
    while remaining > 2.0:
        stages.append(2.0); remaining /= 2.0
    while remaining < 0.5:
        stages.append(0.5); remaining /= 0.5
    stages.append(remaining)
    return ','.join(f'atempo={s:.6f}' for s in stages)


def _apply_atempo(wav_path, speed):
    """Re-stretch an already-written chapter wav to ``speed`` in place (pitch-preserving).

    The MOSS engine has no speed knob: it always synthesizes at 1.0, so playback speed is
    applied here once per chapter via ffmpeg ``atempo`` (ADR 0002) — and the MOSS sentence
    cache stays speed-agnostic (re-stretch instead of re-synthesize on a speed change). A
    1.0 speed is a no-op. Requires ffmpeg; without it speed would silently no-op, which is a
    banned wrongness class, so we raise to surface it (caller decides; cli aborts/​warns).
    """
    if abs(float(speed) - 1.0) < 1e-6:
        return
    if not shutil.which('ffmpeg'):
        raise RuntimeError(
            f'--speed {speed} needs ffmpeg to time-stretch MOSS audio (MOSS has no speed '
            f'knob), but ffmpeg is not on PATH. Install ffmpeg, or run at speed 1.0.')
    src = Path(wav_path)
    tmp = src.with_suffix('.atempo.wav')
    args = ['ffmpeg', '-y', '-i', str(src), '-filter:a', _atempo_filter(speed), str(tmp)]
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        Path(tmp).unlink(missing_ok=True)
        raise RuntimeError(f'ffmpeg atempo failed (exit {proc.returncode}).\n{(proc.stderr or "")[-1000:]}')
    Path(tmp).replace(src)


def is_valid_chapter_wav(path, expected_text_len, max_chars_per_sec=VALIDATION_MAX_CHARS_PER_SEC):
    """Whether an existing chapter wav is safe to reuse instead of regenerating.

    Guards the resume-by-skip path against truncated or zero-byte wavs left behind by
    a prior crash, which would otherwise be silently piped into the final m4b. Checks,
    in order of preference and degrading gracefully:
      1. exists and is non-empty;
      2. duration is plausible for the text — via ffprobe, falling back to soundfile's
         header (when ffprobe is absent), then to a raw file-size floor;
    ``expected_text_len`` of None means the caller capped output (``max_sentences``),
    so the length is not comparable and only readability is required.
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return False

    duration = _robust_duration(p)  # ffprobe, then the wav header
    if duration is None:
        # Can't measure duration at all: accept any file too big to be a bare header.
        return p.stat().st_size > 1024
    if duration <= 0:
        return False
    if expected_text_len is None:
        return True  # length not comparable; readable + non-empty is enough
    # max_chars_per_sec is the fastest plausible narration; text/that is the MINIMUM
    # duration a complete chapter could have, halved for generous slack.
    expected_min_sec = expected_text_len / max_chars_per_sec
    return duration >= expected_min_sec * 0.5


def create_index_file(title, creator, chapter_files, output_folder):
    """Write an FFMETADATA1 file with sequential, gap-free chapter markers.

    Numbering runs 1..N over the chapters that actually made it into the audiobook,
    so skipped/empty chapters no longer produce 'Chapter 0' or off-by-one labels.
    Returns the path to the written file.
    """
    chapters_txt_path = Path(output_folder) / "chapters.txt"
    # Use a robust duration (ffprobe -> wav header) so markers aren't all collapsed to
    # START=END=0 when ffprobe is absent. If any chapter is genuinely unmeasurable, omit
    # the markers entirely (a book with no chapter nav beats one where every marker is 0:00).
    durations = [_robust_duration(c) for c in chapter_files]
    with open(chapters_txt_path, "w", encoding="utf-8") as f:
        f.write(f";FFMETADATA1\ntitle={_escape_ffmetadata(title)}\n"
                f"artist={_escape_ffmetadata(creator)}\n\n")
        if any(d is None for d in durations):
            print('\033[93mWarning: could not measure some chapter durations '
                  '(install ffprobe); writing the m4b without chapter markers.\033[0m')
        else:
            start = 0
            for i, duration in enumerate(durations, start=1):
                end = start + int((duration or 0.0) * 1000)  # all non-None here (guarded above)
                f.write(f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={start}\nEND={end}\ntitle=Chapter {i}\n\n")
                start = end
    return chapters_txt_path


def create_m4b(chapter_files, filename, cover_image, output_folder, title='', creator=''):
    """Create the final .m4b in a SINGLE ffmpeg pass: concat + chapter metadata + cover.

    Fixes several issues with the previous two-pass implementation:
      - Uses the native 'aac' encoder (present in every ffmpeg build) instead of
        'libfdk_aac' (absent from standard apt/brew builds), which previously made
        m4b creation fail for typical users.
      - Encodes only once (the old code transcoded WAV->192k AAC->64k AAC).
      - Escapes concat paths so chapter filenames with apostrophes work.
      - Raises RuntimeError on ffmpeg failure so callers never report a false success
        over a missing/corrupt output file.
      - Cleans up temp files via try/finally even when ffmpeg fails.
    """
    output_folder = Path(output_folder)
    final_filename = output_folder / (Path(filename).stem + '.m4b')
    wav_list_txt = output_folder / (Path(filename).stem + '_wav_list.txt')
    cover_file_path = None
    print('Creating M4B file...')
    try:
        with open(wav_list_txt, 'w') as f:
            for wav_file in chapter_files:
                # Absolute paths: ffmpeg's concat demuxer resolves relative entries
                # against the LIST FILE's directory, which would double a relative
                # output-folder prefix (e.g. out/out/chapter.wav) and fail to open.
                f.write(f"file '{_escape_concat_path(Path(wav_file).resolve())}'\n")

        chapters_txt_path = create_index_file(title, creator, chapter_files, output_folder)

        ffmpeg_args = ['ffmpeg', '-y',
                       '-f', 'concat', '-safe', '0', '-i', str(wav_list_txt),
                       '-i', str(chapters_txt_path)]
        if cover_image:
            cover_file_path = output_folder / 'cover'
            with open(cover_file_path, 'wb') as f:
                f.write(cover_image)
            ffmpeg_args += ['-i', str(cover_file_path)]

        ffmpeg_args += ['-map', '0:a', '-map_metadata', '1']
        if cover_file_path:
            ffmpeg_args += ['-map', '2:v', '-disposition:v', 'attached_pic', '-c:v', 'copy']
        ffmpeg_args += ['-c:a', 'aac', '-b:a', '64k', '-f', 'mp4', str(final_filename)]

        proc = subprocess.run(ffmpeg_args, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f'ffmpeg failed to create the m4b (exit {proc.returncode}).\n'
                f'{(proc.stderr or "")[-2000:]}')
        print(f'{final_filename} created. Enjoy your audiobook.')
        print('Feel free to delete the intermediary .wav chapter files, the .m4b is all you need.')
        return final_filename
    finally:
        for tmp in (wav_list_txt, cover_file_path):
            if tmp is not None:
                Path(tmp).unlink(missing_ok=True)


def _chapter_wav_name(stem, index, voice, xhtml_name):
    """Build a chapter wav filename: ``<stem>_chapter_<i>_<voice>_<xhtml>.wav``.

    Control chars, path separators, and spaces in the epub-derived name are replaced with
    ``_`` — a newline would otherwise survive into the ffmpeg concat list and let a crafted
    epub inject a directive. The voice is run through :func:`audiblez.voices.voice_label` so
    a blend spec (``af_bella:60,af_heart:40`` — illegal ``:`` on Windows) becomes a safe tag.
    The reader :func:`find_chapter_wavs` matches this exact scheme, so the two are
    intentionally co-located and must change together.
    """
    safe = re.sub(r'[\x00-\x1f/\\ ]', '_', xhtml_name)
    return f'{stem}_chapter_{index}_{voicelib.voice_label(voice)}_{safe}.wav'


def find_chapter_wavs(file_path, voice, output_folder='.'):
    """Return existing chapter wavs for a book/voice, ordered by chapter index.

    Matches the naming scheme built by :func:`_chapter_wav_name`
    (``<stem>_chapter_<i>_<voice>_<xhtml>.wav``) and sorts on the integer ``<i>`` so
    the merge order is the synthesis order, not lexicographic (chapter_10 after 9).
    """
    stem = Path(file_path).stem
    matches = glob(str(Path(output_folder) / f'{stem}_chapter_*_{voicelib.voice_label(voice)}_*.wav'))

    def chapter_index(path):
        m = re.search(rf'{re.escape(stem)}_chapter_(\d+)_', Path(path).name)
        return int(m.group(1)) if m else 0

    return [Path(p) for p in sorted(matches, key=chapter_index)]


def merge_chapters(file_path, voice, output_folder='.'):
    """Assemble already-synthesized chapter wavs into a playable m4b.

    The recovery path for an interrupted run: completed chapters persist as wavs, so
    even if synthesis (or the final mux) died partway, ``audiblez --merge`` stitches the
    valid ones into an m4b. Pulls title/author/cover from the epub for metadata. Returns
    the m4b path, or None if there is nothing valid to merge / ffmpeg is missing.
    """
    if not shutil.which('ffmpeg'):
        print('\033[91mffmpeg not found; cannot merge chapters into an m4b.\033[0m')
        return None
    book = epub.read_epub(file_path)
    title, creator = extract_book_metadata(book)
    cover = find_cover(book)
    cover_image = cover.get_content() if cover else b''
    wavs = [w for w in find_chapter_wavs(file_path, voice, output_folder)
            if is_valid_chapter_wav(w, None)]
    if not wavs:
        print('No completed/valid chapter wavs found to merge.')
        return None
    print(f'Merging {len(wavs)} completed chapter(s) into an m4b...')
    return create_m4b(wavs, Path(file_path).name, cover_image, output_folder,
                      title=title, creator=creator)
