"""Hermetic unit tests for the pure helper functions in audiblez.core.

These exercise logic that needs no audio model, network, or ffmpeg. The import is
guarded so the module degrades to a skip when the heavy TTS stack (torch/spacy/
kokoro) is not installed, instead of erroring at collection time.
"""
import unittest

try:
    from audiblez.core import (
        strfdelta,
        split_long_sentence,
        is_chapter,
        chapter_beginning_one_liner,
        _escape_concat_path,
        _escape_ffmetadata,
    )
    _IMPORT_ERR = None
except Exception as e:  # heavy deps (torch/spacy/kokoro) may be absent
    _IMPORT_ERR = e


class FakeChapter:
    """Minimal stand-in for an ebooklib item with the attrs is_chapter() reads."""

    def __init__(self, name, text):
        self._name = name
        self.extracted_text = text

    def get_name(self):
        return self._name


@unittest.skipIf(_IMPORT_ERR is not None, f"audiblez.core unavailable: {_IMPORT_ERR}")
class TextUtilsTest(unittest.TestCase):
    def test_strfdelta_zero(self):
        self.assertEqual(strfdelta(0), '00d 00h 00m 00s')

    def test_strfdelta_values(self):
        self.assertEqual(strfdelta(90), '00d 00h 01m 30s')
        self.assertEqual(strfdelta(3661), '00d 01h 01m 01s')
        self.assertEqual(strfdelta(90061), '01d 01h 01m 01s')

    def test_split_short_sentence_unchanged(self):
        self.assertEqual(split_long_sentence('a short one', 400), ['a short one'])

    def test_split_long_sentence_chunks_within_limit(self):
        text = ' '.join(['word'] * 500)
        parts = split_long_sentence(text, 100)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(p) <= 100 for p in parts))
        # no words are lost across the split
        self.assertEqual(' '.join(parts).split(), text.split())

    def test_split_no_whitespace_falls_back_to_hard_cut(self):
        text = 'x' * 250
        parts = split_long_sentence(text, 100)
        self.assertEqual(''.join(parts), text)
        self.assertTrue(all(len(p) <= 100 for p in parts))

    def test_is_chapter_positive(self):
        self.assertTrue(is_chapter(FakeChapter('OEBPS/chapter_1.xhtml', 'x' * 200)))
        self.assertTrue(is_chapter(FakeChapter('Text/part_03.xhtml', 'y' * 200)))

    def test_is_chapter_rejects_short_or_nonchapter(self):
        self.assertFalse(is_chapter(FakeChapter('chapter_1.xhtml', 'too short')))
        self.assertFalse(is_chapter(FakeChapter('cover.xhtml', 'z' * 500)))

    def test_chapter_one_liner(self):
        c = FakeChapter('x', '  Hello\nthere world  ')
        self.assertEqual(chapter_beginning_one_liner(c, 20), 'Hello there world…')
        self.assertEqual(chapter_beginning_one_liner(FakeChapter('x', ''), 20), '')

    def test_escape_concat_path(self):
        self.assertEqual(_escape_concat_path('/a/b.wav'), '/a/b.wav')
        # a single quote becomes: close-quote, escaped-quote, reopen-quote
        self.assertEqual(_escape_concat_path("/a/it's.wav"), "/a/it'\\''s.wav")

    def test_escape_ffmetadata(self):
        self.assertEqual(_escape_ffmetadata('plain'), 'plain')
        self.assertEqual(_escape_ffmetadata('a=b;c#d\\e'), 'a\\=b\\;c\\#d\\\\e')
        self.assertEqual(_escape_ffmetadata('line1\nline2'), 'line1 line2')


if __name__ == '__main__':
    unittest.main()
