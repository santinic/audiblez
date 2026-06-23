#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# audiblez - A program to convert e-books into audiobooks using
# Kokoro-82M model for high-quality text-to-speech synthesis.
# by Claudio Santini 2025 - https://claudio.uk
import os
import traceback
from glob import glob

import torch.cuda
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
from ebooklib import epub
from pick import pick

sample_rate = 24000


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
         workers=1):
    if post_event: post_event('CORE_STARTED')
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
        chars_per_sec=500 if torch.cuda.is_available() else 50)
    print('Started at:', time.strftime('%H:%M:%S'))
    print(f'Total characters: {stats.total_chars:,}')
    print('Total words:', len(' '.join(texts).split()))
    eta = strfdelta((stats.total_chars - stats.processed_chars) / stats.chars_per_sec)
    print(f'Estimated time remaining (assuming {stats.chars_per_sec} chars/sec): {eta}')
    # Build the per-chapter plan: keep wav paths in chapter order for the m4b,
    # and collect the subset that still needs synthesis.
    chapter_wav_files = []
    todo = []  # list of (chapter_index, i, text, wav_path)
    for i, chapter in enumerate(selected_chapters, start=1):
        if max_chapters and i > max_chapters: break
        text = chapter.extracted_text
        xhtml_file_name = chapter.get_name().replace(' ', '_').replace('/', '_').replace('\\', '_')
        chapter_wav_path = Path(output_folder) / filename.replace(extension, f'_chapter_{i}_{voice}_{xhtml_file_name}.wav')
        if Path(chapter_wav_path).exists():
            print(f'File for chapter {i} already exists. Skipping')
            chapter_wav_files.append(chapter_wav_path)
            stats.processed_chars += len(text)
            if post_event:
                post_event('CORE_CHAPTER_FINISHED', chapter_index=chapter.chapter_index)
            continue
        if len(text.strip()) < 10:
            print(f'Skipping empty chapter {i}')
            continue
        if i == 1:
            # add intro text
            text = f'{title} – {creator}.\n\n' + text
        chapter_wav_files.append(chapter_wav_path)
        todo.append((chapter.chapter_index, i, text, chapter_wav_path))

    stats.chapters_total = len(todo)
    stats.chapters_done = 0
    stats.note = None
    if workers and workers > 1 and len(todo) > 1:
        synth_chapters_parallel(todo, voice, speed, max_sentences, stats, workers, post_event)
    else:
        synth_chapters_serial(todo, voice, speed, max_sentences, stats, post_event)

    if has_ffmpeg:
        chapter_wav_files = [p for p in chapter_wav_files if Path(p).exists()]
        create_index_file(title, creator, chapter_wav_files, output_folder)
        create_m4b(chapter_wav_files, filename, cover_image, output_folder)
        if post_event: post_event('CORE_FINISHED')


def synth_chapters_serial(todo, voice, speed, max_sentences, stats, post_event):
    """Render chapters one at a time in this process (fine-grained progress)."""
    set_espeak_library()
    pipeline = KPipeline(lang_code=voice[0])
    for chapter_index, i, text, wav_path in todo:
        start_time = time.time()
        if post_event: post_event('CORE_CHAPTER_STARTED', chapter_index=chapter_index)
        audio_segments = gen_audio_segments(
            pipeline, text, voice, speed, stats, post_event=post_event, max_sentences=max_sentences)
        if audio_segments:
            soundfile.write(wav_path, np.concatenate(audio_segments), sample_rate)
            print('Chapter written to', wav_path)
            stats.chapters_done += 1
            if post_event: post_event('CORE_CHAPTER_FINISHED', chapter_index=chapter_index)
            print(f'Chapter {i} read in {time.time() - start_time:.2f} seconds')
        else:
            print(f'Warning: No audio generated for chapter {i}')


def _synth_chapter_worker(args):
    """Top-level (picklable) worker for ProcessPoolExecutor: render ONE chapter
    to a wav file in its own process. Reports running char counts to a shared
    queue so the parent can show smooth overall progress. Returns True if audio
    was written."""
    import torch
    text, voice, speed, wav_path, threads, max_sentences, chapter_index, progress_queue = args
    torch.set_num_threads(threads)
    set_espeak_library()
    pipeline = KPipeline(lang_code=voice[0])
    local_stats = SimpleNamespace(total_chars=max(1, len(text)), processed_chars=0, chars_per_sec=50)

    def report(event, **kw):
        if event == 'CORE_PROGRESS' and progress_queue is not None:
            try:
                progress_queue.put((chapter_index, kw['stats'].processed_chars))
            except Exception:
                pass

    audio_segments = gen_audio_segments(
        pipeline, text, voice, speed, local_stats, max_sentences=max_sentences, post_event=report)
    if not audio_segments:
        return False
    soundfile.write(wav_path, np.concatenate(audio_segments), sample_rate)
    return True


def synth_chapters_parallel(todo, voice, speed, max_sentences, stats, workers, post_event):
    """Render chapters across worker processes. Chapters are independent, so this
    gives a real speedup on multi-core CPUs; the audio is identical to serial.
    Workers stream per-chapter char counts back through a queue so the overall
    progress bar advances smoothly, not only when a whole chapter finishes."""
    import multiprocessing
    import queue as queue_mod
    from concurrent.futures import ProcessPoolExecutor
    threads = max(1, ((os.cpu_count() or 2) // 2) // workers)  # share physical cores
    ctx = multiprocessing.get_context('spawn')  # safe with torch on every platform
    manager = ctx.Manager()
    progress_queue = manager.Queue()
    base_processed = stats.processed_chars  # chars from already-rendered (skipped) chapters
    per_chapter = {}  # chapter_index -> chars synthesized so far
    started = set()   # chapters a worker has actually begun
    t_start = None    # wall-clock when the first real chars were synthesized

    def push_progress():
        nonlocal t_start
        if not post_event:
            return
        done_now = sum(per_chapter.values())
        if per_chapter:
            stats.note = None  # real work started; drop the "loading" message
            if t_start is None:
                t_start = time.time()
        stats.processed_chars = base_processed + done_now
        stats.progress = min(100, stats.processed_chars * 100 // stats.total_chars)
        remaining = max(0, stats.total_chars - stats.processed_chars)
        elapsed = (time.time() - t_start) if t_start else 0
        if elapsed > 5 and done_now > 0:  # adaptive: measure the real rate on this machine
            stats.eta = strfdelta(remaining / (done_now / elapsed))
        else:
            stats.eta = strfdelta(remaining / stats.chars_per_sec)
        post_event('CORE_PROGRESS', stats=stats)

    completed = set()  # chapters fully finished (ignore late streamed stragglers)
    stats.note = f'Starting {workers} workers (loading model)…'
    push_progress()
    print(f'Synthesizing {len(todo)} chapters with {workers} workers ({threads} torch threads each)')
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            fut_meta = {}
            for chapter_index, i, text, wav_path in todo:
                args = (text, voice, speed, str(wav_path), threads, max_sentences, chapter_index, progress_queue)
                fut_meta[ex.submit(_synth_chapter_worker, args)] = (chapter_index, i, len(text))
            pending = set(fut_meta)
            while pending:
                while True:  # drain streamed progress
                    try:
                        ck, done = progress_queue.get_nowait()
                    except queue_mod.Empty:
                        break
                    except (EOFError, OSError):  # manager gone; stop draining
                        break
                    if ck in completed:
                        continue  # late straggler for a finished chapter -> never go backward
                    if ck not in started:  # first sign of life from this chapter
                        started.add(ck)
                        if post_event: post_event('CORE_CHAPTER_STARTED', chapter_index=ck)
                    per_chapter[ck] = done
                push_progress()
                for fut in [f for f in pending if f.done()]:
                    pending.discard(fut)
                    chapter_index, i, char_len = fut_meta[fut]
                    try:
                        wrote_audio = fut.result()
                    except Exception as e:
                        print(f'Chapter {i} failed: {e}')
                        wrote_audio = False
                    completed.add(chapter_index)
                    per_chapter[chapter_index] = char_len  # count the whole chapter as done
                    stats.chapters_done += 1
                    if not wrote_audio:
                        print(f'Warning: No audio generated for chapter {i}')
                    elif post_event:
                        if chapter_index not in started:
                            post_event('CORE_CHAPTER_STARTED', chapter_index=chapter_index)
                        post_event('CORE_CHAPTER_FINISHED', chapter_index=chapter_index)
                    print(f'Chapter {i} done — {stats.chapters_done}/{stats.chapters_total}')
                push_progress()
                time.sleep(0.3)
    finally:
        manager.shutdown()


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


_nlp = None


def get_sentencizer():
    """Load the spaCy sentence splitter once and reuse it. It was previously
    reloaded inside gen_audio_segments on every chapter (~1-2s each)."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load('xx_ent_wiki_sm')
        _nlp.add_pipe('sentencizer')
    return _nlp


def gen_audio_segments(pipeline, text, voice, speed, stats=None, max_sentences=None, post_event=None):
    nlp = get_sentencizer()
    audio_segments = []
    doc = nlp(text)
    lang_code = voice[0]

    if lang_code in 'ab':
        sentences = [s.text for s in doc.sents]
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

    if max_sentences:
        sentences = sentences[:max_sentences]

    for sent_text in sentences:
        for gs, ps, audio in pipeline(sent_text, voice=voice, speed=speed, split_pattern=r'\n\n\n'):
            audio_segments.append(audio)
        if stats:
            stats.processed_chars += len(sent_text)
            stats.progress = min(100, stats.processed_chars * 100 // stats.total_chars)
            stats.eta = strfdelta(max(0, stats.total_chars - stats.processed_chars) / stats.chars_per_sec)
            if post_event: post_event('CORE_PROGRESS', stats=stats)
            print(f'Estimated time remaining: {stats.eta}')
            print('Progress:', f'{stats.progress}%\n')
    return audio_segments


def gen_text(text, voice='af_heart', output_file='text.wav', speed=1, play=False):
    lang_code = voice[:1]
    pipeline = KPipeline(lang_code=lang_code, repo_id='hexgrad/Kokoro-82M')
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
    # Matroska holds raw PCM on every ffmpeg build; PCM-in-MP4 only works on
    # ffmpeg >= 7, so use .mka for the lossless intermediate.
    concat_file_path = Path(output_folder) / filename.replace('.epub', '.tmp.mka')
    subprocess.run([
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', wav_list_txt,
        '-c', 'copy',  # mux wavs as-is; create_m4b does the single AAC encode
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
