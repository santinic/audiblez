#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# audiblez - A program to convert e-books into audiobooks using
# Kokoro-82M model for high-quality text-to-speech synthesis.
# by Claudio Santini 2025 - https://claudio.uk
import os
import traceback
from glob import glob

import torch
import spacy
import ebooklib
import soundfile
import numpy as np
import time
import shutil
import subprocess
import platform
import re
from io import StringIO
from types import SimpleNamespace
from tabulate import tabulate
from pathlib import Path
from string import Formatter
from bs4 import BeautifulSoup
from kokoro import KPipeline
from kokoro.model import KModel
from ebooklib import epub
from pick import pick

sample_rate = 24000


def pick_device(preference='auto'):
    """Return the best available torch device name for the given preference.

    preference: 'auto' | 'cpu' | 'cuda' | 'mps'
    """
    pref = (preference or 'auto').lower()
    if pref == 'cpu':
        return 'cpu'
    if pref == 'cuda':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if pref == 'mps':
        return 'mps' if torch.backends.mps.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def set_device(preference='auto'):
    """Resolve device preference, return device name.

    Note: we deliberately do NOT call torch.set_default_device() — doing so
    forces unrelated tensors (tokenizer, numpy bridges, etc.) onto the GPU and
    triggers constant CPU<->GPU ping-pong that made MPS ~2x slower than CPU.
    The model is moved with .to(device) when the pipeline is built instead.
    """
    device = pick_device(preference)
    if device == 'mps':
        # Defensive: if any op lacks an MPS kernel, fall back to CPU per-op
        # rather than failing. With disable_complex=True we don't expect any
        # fallbacks, but the env var costs nothing if unused.
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    return device


def chars_per_sec_estimate(device):
    if device == 'cuda':
        return 500
    if device == 'mps':
        return 180  # measured on M-series with disable_complex CustomSTFT
    return 50


_PRECISION_DTYPES = {
    'fp32': torch.float32,
    'bf16': torch.bfloat16,
    'fp16': torch.float16,
}


def build_pipeline(lang_code, device, precision='fp32'):
    """Construct a Kokoro KPipeline configured for the given device + precision.

    On MPS we build KModel with disable_complex=True so the vocoder uses
    Kokoro's conv1d-based CustomSTFT instead of torch.stft. The complex-tensor
    STFT path on MPS is significantly slower (and emits resize warnings on
    every call); CustomSTFT avoids both.

    precision: 'fp32' (default, safest), 'bf16', or 'fp16'.
      - bf16 keeps fp32's dynamic range with half the bandwidth — generally
        the safest 16-bit option on MPS.
      - fp16 has historically hit a Metal driver assertion
        (MPSNDArrayMatrixMultiplication accumulator dtype). Use at your own
        risk; may crash the process on some torch versions.
      - On CPU, half-precision math is usually slower than fp32 — precision
        is only applied when device is 'mps' or 'cuda'.
    """
    repo_id = 'hexgrad/Kokoro-82M'
    if precision not in _PRECISION_DTYPES:
        raise ValueError(f"Unknown precision {precision!r}; expected one of {list(_PRECISION_DTYPES)}")

    if device == 'mps':
        model = KModel(repo_id=repo_id, disable_complex=True).to(device).eval()
        pipeline = KPipeline(lang_code=lang_code, repo_id=repo_id, model=model)
    else:
        pipeline = KPipeline(lang_code=lang_code, repo_id=repo_id, device=device)

    if precision != 'fp32' and device in ('mps', 'cuda'):
        dtype = _PRECISION_DTYPES[precision]
        pipeline.model = pipeline.model.to(dtype=dtype)
        # Voice tensors are loaded from .pt files in fp32; cast them at load
        # time so they match the model's dtype during forward().
        _orig_load_voice = pipeline.load_voice
        def _load_voice_cast(voice, *args, **kwargs):
            return _orig_load_voice(voice, *args, **kwargs).to(dtype=dtype)
        pipeline.load_voice = _load_voice_cast

    return pipeline


def load_spacy():
    if not spacy.util.is_package("xx_ent_wiki_sm"):
        print("Downloading Spacy model xx_ent_wiki_sm...")
        spacy.cli.download("xx_ent_wiki_sm")


def set_espeak_library():
    """Find the espeak library path"""
    try:

        if os.environ.get('ESPEAK_LIBRARY'):
            library = os.environ['ESPEAK_LIBRARY']
        elif platform.system() == 'Darwin':
            from subprocess import check_output
            try:
                cellar = Path(check_output(["brew", "--cellar"], text=True).strip())
                pattern = cellar / "espeak-ng" / "*" / "lib" / "*.dylib"
                if not (library := next(iter(glob(str(pattern))), None)):
                    raise RuntimeError("No espeak-ng library found; please set the path manually")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                raise RuntimeError("Cannot locate Homebrew Cellar. Is 'brew' installed and in PATH?") from e
        elif platform.system() == 'Linux':
            library = glob('/usr/lib/*/libespeak-ng*')[0]
        elif platform.system() == 'Windows':
            library = 'C:\\Program Files*\\eSpeak NG\\libespeak-ng.dll'
        else:
            print('Unsupported OS, please set the espeak library path manually')
            return
        print('Using espeak library:', library)
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        EspeakWrapper.set_library(library)
    except Exception:
        traceback.print_exc()
        print("Error finding espeak-ng library:")
        print("Probably you haven't installed espeak-ng.")
        print("On Mac: brew install espeak-ng")
        print("On Linux: sudo apt install espeak-ng")


def main(file_path, voice, pick_manually, speed, output_folder='.',
         max_chapters=None, max_sentences=None, selected_chapters=None, post_event=None,
         device=None, precision='fp32'):
    if post_event: post_event('CORE_STARTED')
    # If caller didn't pre-select a device (CLI does), resolve it here.
    if device is None:
        device = pick_device('auto')
    print(f'Using device: {device} (precision: {precision})')
    load_spacy()
    if output_folder != '.':
        Path(output_folder).mkdir(parents=True, exist_ok=True)

    filename = Path(file_path).name

    extension = '.epub'
    book = epub.read_epub(file_path)
    meta_title = book.get_metadata('DC', 'title')
    title = meta_title[0][0] if meta_title else ''
    meta_creator = book.get_metadata('DC', 'creator')
    creator = meta_creator[0][0] if meta_creator else ''

    cover_maybe = find_cover(book)
    cover_image = cover_maybe.get_content() if cover_maybe else b""
    if cover_maybe:
        print(f'Found cover image {cover_maybe.file_name} in {cover_maybe.media_type} format')

    document_chapters = find_document_chapters_and_extract_texts(book)

    if not selected_chapters:
        if pick_manually is True:
            selected_chapters = pick_chapters(document_chapters)
        else:
            selected_chapters = find_good_chapters(document_chapters)
    print_selected_chapters(document_chapters, selected_chapters)
    texts = [c.extracted_text for c in selected_chapters]

    has_ffmpeg = shutil.which('ffmpeg') is not None
    if not has_ffmpeg:
        print('\033[91m' + 'ffmpeg not found. Please install ffmpeg to create mp3 and m4b audiobook files.' + '\033[0m')

    stats = SimpleNamespace(
        total_chars=sum(map(len, texts)),
        processed_chars=0,
        chars_per_sec=chars_per_sec_estimate(device))
    print('Started at:', time.strftime('%H:%M:%S'))
    print(f'Total characters: {stats.total_chars:,}')
    print('Total words:', len(' '.join(texts).split()))
    eta = strfdelta((stats.total_chars - stats.processed_chars) / stats.chars_per_sec)
    print(f'Estimated time remaining (assuming {stats.chars_per_sec} chars/sec): {eta}')
    set_espeak_library()
    pipeline = build_pipeline(voice[0], device, precision=precision)

    chapter_wav_files = []
    for i, chapter in enumerate(selected_chapters, start=1):
        if max_chapters and i > max_chapters: break
        text = chapter.extracted_text
        xhtml_file_name = chapter.get_name().replace(' ', '_').replace('/', '_').replace('\\', '_')
        chapter_wav_path = Path(output_folder) / filename.replace(extension, f'_chapter_{i}_{voice}_{xhtml_file_name}.wav')
        chapter_wav_files.append(chapter_wav_path)
        if Path(chapter_wav_path).exists():
            print(f'File for chapter {i} already exists. Skipping')
            stats.processed_chars += len(text)
            if post_event:
                post_event('CORE_CHAPTER_FINISHED', chapter_index=chapter.chapter_index)
            continue
        if len(text.strip()) < 10:
            print(f'Skipping empty chapter {i}')
            chapter_wav_files.remove(chapter_wav_path)
            continue
        if i == 1:
            # add intro text
            text = f'{title} – {creator}.\n\n' + text
        start_time = time.time()
        if post_event: post_event('CORE_CHAPTER_STARTED', chapter_index=chapter.chapter_index)
        audio_segments = gen_audio_segments(
            pipeline, text, voice, speed, stats, post_event=post_event, max_sentences=max_sentences)
        if audio_segments:
            final_audio = np.concatenate(audio_segments)
            soundfile.write(chapter_wav_path, final_audio, sample_rate)
            end_time = time.time()
            delta_seconds = end_time - start_time
            chars_per_sec = len(text) / delta_seconds
            print('Chapter written to', chapter_wav_path)
            if post_event: post_event('CORE_CHAPTER_FINISHED', chapter_index=chapter.chapter_index)
            print(f'Chapter {i} read in {delta_seconds:.2f} seconds ({chars_per_sec:.0f} characters per second)')
        else:
            print(f'Warning: No audio generated for chapter {i}')
            chapter_wav_files.remove(chapter_wav_path)

    if has_ffmpeg:
        create_index_file(title, creator, chapter_wav_files, output_folder)
        create_m4b(chapter_wav_files, filename, cover_image, output_folder)
        if post_event: post_event('CORE_FINISHED')


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
    print(tabulate([
        [i, c.get_name(), len(c.extracted_text), ok if c in chapters else '', chapter_beginning_one_liner(c)]
        for i, c in enumerate(document_chapters, start=1)
    ], headers=['#', 'Chapter', 'Text Length', 'Selected', 'First words']))

def split_long_sentence(text, max_length=400):
    """Split a long sentence around the 500 chars, picking the first whitespace after the 500th character. """
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


def batch_sentences(sentences, max_length=400):
    """Pack consecutive short sentences into chunks of at most max_length chars.

    Why: each call to Kokoro carries fixed per-call overhead (kernel launch on
    GPU, tensor allocation, Python interop). When the source text is full of
    short fragments — titles, list items, one-line paragraphs from an EPUB
    extractor — that overhead dominates and the GPU can end up slower than
    the CPU. Packing adjacent sentences into longer chunks amortizes the
    overhead. Kokoro is called once per chunk and produces natural prosody
    across the contained sentences.

    Sentences already longer than max_length are split first via
    split_long_sentence (same behavior as the non-English path).
    """
    chunks = []
    cur, cur_len = [], 0
    for s in sentences:
        s = (s or "").strip()
        if not s:
            continue
        if len(s) > max_length:
            if cur:
                chunks.append(' '.join(cur))
                cur, cur_len = [], 0
            chunks.extend(split_long_sentence(s, max_length))
            continue
        joiner = 1 if cur else 0
        if cur_len + joiner + len(s) > max_length:
            chunks.append(' '.join(cur))
            cur, cur_len = [s], len(s)
        else:
            cur.append(s)
            cur_len += joiner + len(s)
    if cur:
        chunks.append(' '.join(cur))
    return chunks


def gen_audio_segments(pipeline, text, voice, speed, stats=None, max_sentences=None, post_event=None):
    nlp = spacy.load('xx_ent_wiki_sm')
    nlp.add_pipe('sentencizer')
    audio_segments = []
    doc = nlp(text)
    lang_code = voice[0]

    if lang_code in 'ab':
        # Batch sentences into ~1200-char chunks (well past Kokoro's 510-phoneme
        # internal split, so Kokoro handles further splitting at optimal boundaries).
        # See batch_sentences() for rationale; 1200 picked from a max_length sweep.
        sentences = batch_sentences([s.text for s in doc.sents], max_length=1200)
    else:
        # For non-english languages, Kokoro truncates long sentences, so we split them manually
        sentences = []
        for sent in list(doc.sents):
            if len(sent.text) > 400:
                print(f'Warning: Sentence too long ({len(sent.text)} chars), splitting into smaller sentences.')
                sents = split_long_sentence(sent.text, 400)
                sentences.extend(sents)
            else:
                sentences.append(sent.text)

    for i, sent_text in enumerate(sentences):
        if max_sentences and i > max_sentences: break
        for gs, ps, audio in pipeline(sent_text, voice=voice, speed=speed, split_pattern=r'\n\n\n'):
            audio_segments.append(audio)
        if stats:
            stats.processed_chars += len(sent_text)
            stats.progress = stats.processed_chars * 100 // stats.total_chars
            stats.eta = strfdelta((stats.total_chars - stats.processed_chars) / stats.chars_per_sec)
            if post_event: post_event('CORE_PROGRESS', stats=stats)
            print(f'Estimated time remaining: {stats.eta}')
            print('Progress:', f'{stats.progress}%\n')
    return audio_segments


def gen_text(text, voice='af_heart', output_file='text.wav', speed=1, play=False, device=None, precision='fp32'):
    lang_code = voice[:1]
    if device is None:
        device = pick_device('auto')
    pipeline = build_pipeline(lang_code, device, precision=precision)
    load_spacy()
    audio_segments = gen_audio_segments(pipeline, text, voice=voice, speed=speed);
    final_audio = np.concatenate(audio_segments)
    soundfile.write(output_file, final_audio, sample_rate)
    if play:
        subprocess.run(['ffplay', '-autoexit', '-nodisp', output_file])


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
    selected_chapters = [c for c in chapters if c in selected_chapters_out_of_order]
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


def concat_wavs_with_ffmpeg(chapter_files, output_folder, filename):
    wav_list_txt = Path(output_folder) / filename.replace('.epub', '_wav_list.txt')
    with open(wav_list_txt, 'w') as f:
        for wav_file in chapter_files:
            f.write(f"file '{wav_file}'\n")
    concat_file_path = Path(output_folder) / filename.replace('.epub', '.tmp.mp4')
    subprocess.run([
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', wav_list_txt,
        # '-c', 'copy',
        '-c:a',  'libfdk_aac',
        '-b:a',  '192k',
        concat_file_path])
    Path(wav_list_txt).unlink()
    return concat_file_path


def create_m4b(chapter_files, filename, cover_image, output_folder):
    concat_file_path = concat_wavs_with_ffmpeg(chapter_files, output_folder, filename)
    final_filename = Path(output_folder) / filename.replace('.epub', '.m4b')
    chapters_txt_path = Path(output_folder) / "chapters.txt"
    print('Creating M4B file...')

    if cover_image:
        cover_file_path = Path(output_folder) / 'cover'
        with open(cover_file_path, 'wb') as f:
            f.write(cover_image)
        cover_image_args = [
            '-i', f'{cover_file_path}',
            '-map', '2:v',  # Map cover image
            '-disposition:v', 'attached_pic',  # Ensure cover is embedded
            '-c:v', 'copy',  # Keep cover unchanged
        ]
    else:
        cover_image_args = []

    proc = subprocess.run([
        'ffmpeg',
        '-y',  # Overwrite output
        
        '-i', f'{concat_file_path}',  # Input audio
        '-i', f'{chapters_txt_path}',  # Input chapters
        *cover_image_args,  # Cover image (if provided)

        '-map', '0:a',  # Map audio
        '-c:a', 'aac',  # Convert to AAC
        '-b:a', '64k',  # Reduce bitrate for smaller size

        '-map_metadata', '1', # Map metadata

        '-f', 'mp4',  # Output as M4B
        f'{final_filename}'  # Output file
    ])

    Path(concat_file_path).unlink()
    if proc.returncode == 0:
        print(f'{final_filename} created. Enjoy your audiobook.')
        print('Feel free to delete the intermediary .wav chapter files, the .m4b is all you need.')


def probe_duration(file_name):
    args = ['ffprobe', '-i', file_name, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'default=noprint_wrappers=1:nokey=1']
    proc = subprocess.run(args, capture_output=True, text=True, check=True)
    return float(proc.stdout.strip())


def create_index_file(title, creator, chapter_mp3_files, output_folder):
    with open(Path(output_folder) / "chapters.txt", "w", encoding="utf-8") as f:
        f.write(f";FFMETADATA1\ntitle={title}\nartist={creator}\n\n")
        start = 0
        i = 0
        for c in chapter_mp3_files:
            duration = probe_duration(c)
            end = start + (int)(duration * 1000)
            f.write(f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={start}\nEND={end}\ntitle=Chapter {i}\n\n")
            i += 1
            start = end


def unmark_element(element, stream=None):
    """auxiliarry function to unmark markdown text"""
    if stream is None:
        stream = StringIO()
    if element.text:
        stream.write(element.text)
    for sub in element:
        unmark_element(sub, stream)
    if element.tail:
        stream.write(element.tail)
    return stream.getvalue()


def unmark(text):
    """Unmark markdown text"""
    Markdown.output_formats["plain"] = unmark_element  # patching Markdown
    __md = Markdown(output_format="plain")
    __md.stripTopLevelTags = False
    return __md.convert(text)
