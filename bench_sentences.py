"""Diagnose CPU vs MPS perf vs sentence-length distribution on real EPUB prose."""
import warnings
warnings.filterwarnings("ignore")

import time
import statistics
from collections import Counter

import spacy
from ebooklib import epub
from bs4 import BeautifulSoup

from audiblez.core import (
    build_pipeline, find_document_chapters_and_extract_texts,
    load_spacy, set_device, set_espeak_library,
)


def get_chapter2_text():
    book = epub.read_epub("/tmp/audiblez_smoke/poe.epub")
    chapters = find_document_chapters_and_extract_texts(book)
    # The earlier run wrote "_chapter_2_..._1064-h-1.htm.html.wav" — index 1 in selected set.
    # Just pick the longest chapter.
    return max((c.extracted_text for c in chapters), key=len)


def sentence_stats(text):
    nlp = spacy.load("xx_ent_wiki_sm")
    nlp.add_pipe("sentencizer")
    doc = nlp(text)
    lens = [len(s.text) for s in doc.sents]
    buckets = Counter()
    for n in lens:
        if n < 20: buckets["<20"] += 1
        elif n < 50: buckets["20-49"] += 1
        elif n < 100: buckets["50-99"] += 1
        elif n < 200: buckets["100-199"] += 1
        elif n < 400: buckets["200-399"] += 1
        else: buckets["400+"] += 1
    return lens, buckets


def bench_per_sentence(device, sentences, voice="af_sky"):
    pipe = build_pipeline("a", device)
    # warmup
    for _ in range(3):
        list(pipe("Hello world.", voice=voice, speed=1.0))
    per_sent = []
    t_total0 = time.time()
    for s in sentences:
        t0 = time.time()
        for _ in pipe(s, voice=voice, speed=1.0, split_pattern=r"\n\n\n"):
            pass
        per_sent.append((len(s), time.time() - t0))
    wall = time.time() - t_total0
    return per_sent, wall


def main():
    set_espeak_library(); load_spacy()
    text = get_chapter2_text()
    print(f"Chapter text: {len(text):,} chars")

    lens, buckets = sentence_stats(text)
    print(f"Sentences: {len(lens)}")
    print(f"  mean {statistics.mean(lens):.1f}  median {statistics.median(lens)}  "
          f"min {min(lens)}  max {max(lens)}")
    print("  distribution:")
    for k in ["<20", "20-49", "50-99", "100-199", "200-399", "400+"]:
        if buckets[k]:
            print(f"    {k:>8s}: {buckets[k]}")

    # Time per-sentence on both devices. Limit to a sample so we don't sit here forever.
    nlp = spacy.load("xx_ent_wiki_sm")
    nlp.add_pipe("sentencizer")
    sentences = [s.text for s in nlp(text).sents][:40]
    print(f"\nBenchmarking on {len(sentences)} sentences...")

    for dev_pref in ("cpu", "mps"):
        dev = set_device(dev_pref)
        per_sent, wall = bench_per_sentence(dev, sentences)
        total_chars = sum(c for c, _ in per_sent)
        total_time = sum(t for _, t in per_sent)
        print(f"\n  {dev}: {total_chars} chars in {total_time:.2f}s -> {total_chars/total_time:.0f} cps (wall {wall:.2f}s)")
        # Per-sentence time vs length:
        short = [(c, t) for c, t in per_sent if c < 80]
        long_ = [(c, t) for c, t in per_sent if c >= 80]
        if short:
            print(f"    short (<80 chars):  n={len(short)} mean_chars={statistics.mean(c for c,_ in short):.0f}  mean_t={statistics.mean(t for _,t in short)*1000:.0f}ms  cps={sum(c for c,_ in short)/sum(t for _,t in short):.0f}")
        if long_:
            print(f"    long (>=80 chars):  n={len(long_)} mean_chars={statistics.mean(c for c,_ in long_):.0f}  mean_t={statistics.mean(t for _,t in long_)*1000:.0f}ms  cps={sum(c for c,_ in long_)/sum(t for _,t in long_):.0f}")


if __name__ == "__main__":
    main()
