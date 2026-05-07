"""Verify the two fixes against EPUBs known to trigger each bug."""
import os
from ebooklib import epub
from audiblez import core


print('=== Fix B: dedupe nested matches ===')
mind_play = 'c:/Users/magee/repos/k2go/converted/Mind Play__105.epub'
book = epub.read_epub(mind_play)
docs = core.find_document_chapters_and_extract_texts(book)
total = sum(len(c.extracted_text) for c in docs)
# Original (broken): 454,383 chars  | Fixed: 393,349 chars (61,034 saved, 13.4%)
print(f'Mind Play extracted text: {total:,} chars')
print(f'Expected ~393,349 (would be 454,383 with the bug); '
      f'{"FIX WORKING" if total < 400000 else "FIX NOT APPLIED"}')

print()
print('=== Fix A: chapter detection threshold ===')
communicate = 'c:/Users/magee/repos/k2go/converted/Communicate About Sex (without making it awkward)_ Healthy and Shame-Free Ways to Talk with Your Partner about Sexual De__102.epub'
book2 = epub.read_epub(communicate)
docs2 = core.find_document_chapters_and_extract_texts(book2)
chapters = core.find_good_chapters(docs2)
print(f'Communicate About Sex selected chapters: {len(chapters)}')
print(f'Expected many (~30+); '
      f'{"FIX WORKING" if len(chapters) >= 10 else "FIX NOT APPLIED — only 1 chapter selected"}')
