"""Tests for the book trailer assembly (core.make_trailer).

Guarded import so it skips without the heavy stack. The epub, spaCy, synthesizer and
soundfile write are all faked, so the test exercises the assembly wiring (label +
body per chapter, max_sentences cap, empty-chapter skip) with no model or IO.
"""
import unittest
from unittest import mock
from types import SimpleNamespace

try:
    import numpy as np
    import audiblez.core as core
    _ERR = None
except Exception as e:
    _ERR = e


def _chapter(text):
    return SimpleNamespace(extracted_text=text)


@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class MakeTrailerTest(unittest.TestCase):
    def _patches(self, gen_return=None):
        if gen_return is None:
            gen_return = [np.ones(2, dtype=np.float32)]
        return (
            mock.patch.object(core, 'epub'),
            mock.patch.object(core, 'find_document_chapters_and_extract_texts', return_value=[]),
            mock.patch.object(core, 'build_synthesizer', return_value=lambda t, s: [np.ones(2)]),
            mock.patch.object(core, 'load_spacy'),
            mock.patch.object(core, 'gen_audio_segments', return_value=gen_return),
            mock.patch.object(core, 'soundfile'),
        )

    def test_label_and_body_per_chapter_with_max_sentences(self):
        chapters = [_chapter('A real chapter with text.'), _chapter('Another chapter here.')]
        p = self._patches()
        with p[0], p[1], p[2], p[3], p[4] as gen, p[5] as msf:
            out = core.make_trailer('book.epub', 'af_sky', 'trailer.wav',
                                    selected_chapters=chapters, sentences_per_chapter=2)
        self.assertEqual(out, 'trailer.wav')
        self.assertTrue(msf.write.called)
        self.assertEqual(gen.call_count, 4)  # label + body, x2 chapters
        body_calls = [c for c in gen.call_args_list if c.kwargs.get('max_sentences') == 2]
        self.assertEqual(len(body_calls), 2)

    def test_empty_chapters_are_skipped(self):
        chapters = [_chapter('   '), _chapter('Real text content here.')]
        p = self._patches()
        with p[0], p[1], p[2], p[3], p[4] as gen, p[5]:
            core.make_trailer('book.epub', 'af_sky', 'trailer.wav', selected_chapters=chapters)
        self.assertEqual(gen.call_count, 2)  # only the non-empty chapter

    def test_max_chapters_caps_sampling(self):
        # texts must clear make_trailer's <10-char skip guard
        chapters = [_chapter('Chapter one body.'), _chapter('Chapter two body.'),
                    _chapter('Chapter three body.')]
        p = self._patches()
        with p[0], p[1], p[2], p[3], p[4] as gen, p[5]:
            core.make_trailer('book.epub', 'af_sky', 'trailer.wav',
                              selected_chapters=chapters, max_chapters=1)
        self.assertEqual(gen.call_count, 2)  # one chapter -> label + body

    def test_no_usable_chapters_returns_none(self):
        p = self._patches()
        with p[0], p[1], p[2], p[3], p[4], p[5] as msf:
            out = core.make_trailer('book.epub', 'af_sky', 'trailer.wav', selected_chapters=[])
        self.assertIsNone(out)
        self.assertFalse(msf.write.called)


if __name__ == '__main__':
    unittest.main()
