"""Sweep batch_sentences max_length to find the sweet spot.

For each (device, max_length), time generation of chapter 2 of the Poe EPUB
and also count Kokoro's internal forward passes (its generator yields once
per phoneme chunk).
"""
import warnings; warnings.filterwarnings("ignore")
import time

import spacy
from ebooklib import epub

from audiblez.core import (
    batch_sentences, build_pipeline, find_document_chapters_and_extract_texts,
    load_spacy, set_device, set_espeak_library,
)


def get_chapter_text():
    book = epub.read_epub("/tmp/audiblez_smoke/poe.epub")
    chapters = find_document_chapters_and_extract_texts(book)
    return max((c.extracted_text for c in chapters), key=len)


def run(pipe, sentences, voice="af_sky"):
    """Iterate pipeline, counting internal phoneme-chunk yields."""
    n_calls = 0
    n_yields = 0
    audio = []
    t0 = time.time()
    for s in sentences:
        n_calls += 1
        for gs, ps, a in pipe(s, voice=voice, speed=1.0, split_pattern=r"\n\n\n"):
            n_yields += 1
            audio.append(a)
    return time.time() - t0, n_calls, n_yields, audio


def main():
    set_espeak_library(); load_spacy()
    text = get_chapter_text()
    print(f"Chapter: {len(text):,} chars\n")

    nlp = spacy.load("xx_ent_wiki_sm")
    nlp.add_pipe("sentencizer")
    raw = [s.text for s in nlp(text).sents]
    print(f"spaCy sentences: {len(raw)}\n")

    configs = [
        ("max_length=400 (current)", 400),
        ("max_length=600",           600),
        ("max_length=800",           800),
        ("max_length=1200",         1200),
        ("max_length=2000",         2000),  # almost certain to be Kokoro-chunked internally
    ]

    print(f"{'device':<5} {'config':<26} {'chunks':>7} {'fwd':>5} {'wall':>7} {'cps':>6}")
    for dev_pref in ("cpu", "mps"):
        dev = set_device(dev_pref)
        # build pipeline once per device to amortize warmup across the sweep
        pipe = build_pipeline("a", dev)
        # warmup so first config doesn't pay shader-compile tax
        list(pipe("This is a warmup sentence for the model.", voice="af_sky", speed=1.0))
        for label, mlen in configs:
            chunks = batch_sentences(raw, max_length=mlen)
            wall, n_calls, n_yields, _ = run(pipe, chunks)
            cps = len(text) / wall
            print(f"{dev:<5} {label:<26} {len(chunks):>7} {n_yields:>5} {wall:>6.2f}s {cps:>6.0f}")
        print()


if __name__ == "__main__":
    main()
