#!/usr/bin/env python3
# audiblez - A program to convert e-books into audiobooks using
# Kokoro-82M model for high-quality text-to-speech synthesis.
# by Claudio Santini 2025 - https://claudio.uk
import argparse
import sys
import time
import shutil
import subprocess
import soundfile
import ebooklib
import warnings
import re
import torch
from pathlib import Path
from string import Formatter
from bs4 import BeautifulSoup
from kokoro import KPipeline
from ebooklib import epub
from pydub import AudioSegment
from pick import pick
from gui import start_gui
from tempfile import NamedTemporaryFile

sample_rate = 24000
voices = [
    'af_alloy', 'af_aoede', 'af_bella', 'af_jessica', 'af_kore', 'af_nicole',
    'af_nova', 'af_river', 'af_sarah', 'af_sky', 'am_adam', 'am_echo', 'am_eric',
    'am_fenrir', 'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'bf_alice',
    'bf_emma', 'bf_isabella', 'bf_lily', 'bm_daniel', 'bm_fable',
    'bm_george', 'bm_lewis'
]


def main(pipeline, file_path, voice, pick_manually, speed, chapters_by_name):
    filename = Path(file_path).name
    warnings.simplefilter("ignore")
    book = epub.read_epub(file_path)
    meta_title = book.get_metadata('DC', 'title')
    title = meta_title[0][0] if meta_title else ''
    meta_creator = book.get_metadata('DC', 'creator')
    creator = meta_creator[0][0] if meta_creator else ''

    cover_maybe = [c for c in book.get_items() if c.get_type() == ebooklib.ITEM_COVER]
    cover_image = cover_maybe[0].get_content() if cover_maybe else b""
    if cover_maybe:
        print(f'Found cover image {cover_maybe[0].file_name} in {cover_maybe[0].media_type} format')

    intro = f'{title} by {creator}'
    print(intro)

    if chapters_by_name:
        chapters = []
        for chapter_by_name in chapters_by_name:
            for item in book.get_items():
                if item.get_name() == chapter_by_name:
                    chapters.append(item)
                    break
    else:
        print('Found Chapters:', [c.get_name() for c in book.get_items() if c.get_type() == ebooklib.ITEM_DOCUMENT])
        if pick_manually:
            chapters = pick_chapters(book)
        else:
            chapters = find_chapters(book)
            print('Automatically selected chapters:', [c.get_name() for c in chapters])

    texts = extract_texts(chapters)

    has_ffmpeg = shutil.which('ffmpeg') is not None
    if not has_ffmpeg:
        print('\033[91m' + 'ffmpeg not found. Please install ffmpeg to create mp3 and m4b audiobook files.' + '\033[0m')

    total_chars, processed_chars = sum(map(len, texts)), 0
    print('Started at:', time.strftime('%H:%M:%S'))
    print(f'Total characters: {total_chars:,}')
    print('Total words:', len(' '.join(texts).split()))
    chars_per_sec = 500 if torch.cuda.is_available() else 50  # assume 50 or 500 chars per second at the beginning
    print(f'Estimated time remaining (assuming {chars_per_sec} chars/sec): {strfdelta((total_chars - processed_chars) / chars_per_sec)}')

    chapter_mp3_files = []
    durations = {}

    for i, text in enumerate(texts, start=1):
        chapter_filename = filename.replace('.epub', f'_chapter_{i}.wav')
        chapter_mp3_files.append(chapter_filename)
        if Path(chapter_filename).exists():
            print(f'File for chapter {i} already exists. Skipping')
            continue
        if len(text.strip()) < 10:
            print(f'Skipping empty chapter {i}')
            chapter_mp3_files.remove(chapter_filename)
            continue
        print(f'Reading chapter {i} ({len(text):,} characters)...')
        if i == 1:
            text = intro + '.\n\n' + text
        start_time = time.time()
        generator = pipeline(text, voice=voice, speed=speed)
        for gs, ps, audio in generator:
            soundfile.write(chapter_filename, audio, sample_rate)
        end_time = time.time()
        delta_seconds = end_time - start_time
        chars_per_sec = len(text) / delta_seconds
        processed_chars += len(text)
        print(f'Estimated time remaining: {strfdelta((total_chars - processed_chars) / chars_per_sec)}')
        print('Chapter written to', chapter_filename)
        print(f'Chapter {i} read in {delta_seconds:.2f} seconds ({chars_per_sec:.0f} characters per second)')
        progress = processed_chars * 100 // total_chars
        print('Progress:', f'{progress}%\n')

    if has_ffmpeg:
        create_index_file(title, creator, chapter_mp3_files, durations)
        create_m4b(chapter_mp3_files, filename, title, creator, cover_image)


def extract_texts(chapters):
    texts = []
    for chapter in chapters:
        xml = chapter.get_body_content()
        soup = BeautifulSoup(xml, features='lxml')
        chapter_text = ''
        html_content_tags = ['title', 'p', 'h1', 'h2', 'h3', 'h4', 'li']
        for child in soup.find_all(html_content_tags):
            inner_text = child.text.strip() if child.text else ""
            if inner_text:
                chapter_text += inner_text + '\n'
        texts.append(chapter_text)
    return texts


def is_chapter(c):
    name = c.get_name().lower()
    return bool(
        'chapter' in name.lower()
        or re.search(r'part\d{1,3}', name)
        or re.search(r'ch\d{1,3}', name)
        or re.search(r'chap\d{1,3}', name)
    )


def find_chapters(book, verbose=False):
    chapters = [c for c in book.get_items() if c.get_type() == ebooklib.ITEM_DOCUMENT and is_chapter(c)]
    if verbose:
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                print(f"'{item.get_name()}'" + ', #' + str(len(item.get_body_content())))
                # print(f'{item.get_name()}'.ljust(60), str(len(item.get_body_content())).ljust(15), 'X' if item in chapters else '-')
    if len(chapters) == 0:
        print('Not easy to find the chapters, defaulting to all available documents.')
        chapters = [c for c in book.get_items() if c.get_type() == ebooklib.ITEM_DOCUMENT]
    return chapters


def pick_chapters(book):
    all_chapters_names = [c.get_name() for c in book.get_items() if c.get_type() == ebooklib.ITEM_DOCUMENT]
    title = 'Select which chapters to read in the audiobook'
    selected_chapters_names = pick(all_chapters_names, title, multiselect=True, min_selection_count=1)
    selected_chapters_names = [c[0] for c in selected_chapters_names]
    selected_chapters = [c for c in book.get_items() if c.get_name() in selected_chapters_names]
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


def create_m4b(chapter_files, filename, title, author, cover_image):
    tmp_filename = filename.replace('.epub', '.tmp.mp4')
    if not Path(tmp_filename).exists():
        combined_audio = AudioSegment.empty()
        for wav_file in chapter_files:
            audio = AudioSegment.from_wav(wav_file)
            combined_audio += audio
        print('Converting to Mp4...')
        combined_audio.export(tmp_filename, format="mp4", codec="aac", bitrate="64k")
    final_filename = filename.replace('.epub', '.m4b')
    print('Creating M4B file...')

    if cover_image:
        cover_image_file = NamedTemporaryFile("wb")
        cover_image_file.write(cover_image)
        cover_image_args = ["-i", cover_image_file.name, "-map", "0:a", "-map", "2:v"]
    else:
        cover_image_args = []

    proc = subprocess.run([
        'ffmpeg',
        '-i', f'{tmp_filename}',
        '-i', 'chapters.txt',
        *cover_image_args,
        '-map', '0',
        '-map_metadata', '1',
        '-c:a', 'copy',
        '-c:v', 'copy',
        '-disposition:v', 'attached_pic',
        '-c', 'copy',
        '-f', 'mp4',
        f'{final_filename}'
    ])
    Path(tmp_filename).unlink()
    if proc.returncode == 0:
        print(f'{final_filename} created. Enjoy your audiobook.')
        print('Feel free to delete the intermediary .wav chapter files, the .m4b is all you need.')


def probe_duration(file_name):
    args = ['ffprobe', '-i', file_name, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'default=noprint_wrappers=1:nokey=1']
    proc = subprocess.run(args, capture_output=True, text=True, check=True)
    return float(proc.stdout.strip())


def create_index_file(title, creator, chapter_mp3_files, durations):
    with open("chapters.txt", "w") as f:
        f.write(f";FFMETADATA1\ntitle={title}\nartist={creator}\n\n")
        start = 0
        i = 0
        for c in chapter_mp3_files:
            if c not in durations:
                durations[c] = probe_duration(c)
            end = start + (int)(durations[c] * 1000)
            f.write(f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={start}\nEND={end}\ntitle=Chapter {i}\n\n")
            i += 1
            start = end


def cli_main():
    voices_str = ', '.join(voices)
    epilog = 'example:\n' + \
             '  audiblez book.epub -l en-us -v af_sky'
    default_voice = 'af_sky' if 'af_sky' in voices else voices[0]

    parser = argparse.ArgumentParser(epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('epub_file_path', help='Path to the epub file')
    parser.add_argument('-v', '--voice', default=default_voice, help=f'Choose narrating voice: {voices_str}')
    parser.add_argument('-p', '--pick', default=False, help=f'Interactively select which chapters to read in the audiobook', action='store_true')
    parser.add_argument('-s', '--speed', default=1.0, help=f'Set speed from 0.5 to 2.0', type=float)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    if torch.cuda.is_available():
        print('CUDA GPU available')
        torch.set_default_device('cuda')
    else:
        print('CUDA GPU not available. Defaulting to CPU')

    pipeline = KPipeline(lang_code=args.voice[0])  # a for american or b for british
    main(pipeline, args.epub_file_path, args.voice, args.pick, args.speed, None)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--gui':
        start_gui()
    else:
        cli_main(sys.argv)
