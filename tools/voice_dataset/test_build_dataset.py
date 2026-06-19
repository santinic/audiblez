"""Hermetic tests for the pure logic in build_dataset.py (segmentation, manifests, text).

No ML deps needed — build_dataset only imports numpy + soundfile at module scope (the
heavy whisperx/demucs/speechbrain imports are lazy). Run from the repo root:

    python -m unittest discover -s tools/voice_dataset -p 'test_*.py' -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset as B  # noqa: E402


def W(word, start, end, speaker='S0'):
    return {'word': word, 'start': start, 'end': end, 'speaker': speaker}


class EndsSentenceTest(unittest.TestCase):
    def test_plain_terminals(self):
        self.assertTrue(B.ends_sentence('done.'))
        self.assertTrue(B.ends_sentence('really?'))
        self.assertTrue(B.ends_sentence('wow!'))
        self.assertFalse(B.ends_sentence('word'))

    def test_trailing_quotes_and_brackets(self):
        self.assertTrue(B.ends_sentence('go!"'))
        self.assertTrue(B.ends_sentence('really?”'))
        self.assertTrue(B.ends_sentence("(yes.)"))
        self.assertFalse(B.ends_sentence('mid"'))


class NormalizeTextTest(unittest.TestCase):
    def test_adds_terminal(self):
        self.assertEqual(B.normalize_text('hello world'), 'hello world.')

    def test_collapses_whitespace(self):
        self.assertEqual(B.normalize_text('a   b\n c'), 'a b c.')

    def test_keeps_existing_terminal_with_quote(self):
        self.assertEqual(B.normalize_text('go!"'), 'go!"')


class SegmentWordsTest(unittest.TestCase):
    def seg(self, words, min_dur=1.0, max_dur=15.0, min_chars=6, max_gap=1.0):
        return B.segment_words(words, min_dur, max_dur, min_chars, max_gap)

    def test_every_clip_ends_a_sentence(self):
        words = [W('Hi.', 0, 0.6), W('A', 0.6, 0.9), W('B', 0.9, 1.2),
                 W('ok?', 1.2, 1.8), W('frag', 1.8, 2.2)]
        segs = self.seg(words, min_dur=0.5, min_chars=3)
        self.assertTrue(all(B.ends_sentence(t) for _, _, t in segs))

    def test_punctuationless_runon_dropped(self):
        runon = [W('word', i * 0.5, i * 0.5 + 0.5) for i in range(40)]
        self.assertEqual(self.seg(runon, max_dur=6.0), [])

    def test_gap_breaks_into_separate_clips(self):
        words = [W('First', 0, 0.5), W('part.', 0.5, 1.2),
                 W('Second', 6.5, 7.0), W('part.', 7.0, 7.8)]
        segs = self.seg(words)
        self.assertEqual(len(segs), 2)
        self.assertTrue(all(B.ends_sentence(t) for _, _, t in segs))

    def test_mid_sentence_fragment_before_gap_dropped(self):
        words = [W('A', 0, 0.5), W('mid', 0.5, 1.0), W('fragment', 1.0, 1.6), W('Done.', 7.0, 7.6)]
        segs = self.seg(words)
        self.assertTrue(all(B.ends_sentence(t) for _, _, t in segs))

    def test_overflow_cut_respects_max_dur_and_sentence(self):
        long = [W('word.' if i % 5 == 4 else 'word', i * 0.5, i * 0.5 + 0.5) for i in range(40)]
        segs = self.seg(long, max_dur=6.0)
        self.assertTrue(all((e - s) <= 6.01 for s, e, _ in segs))
        self.assertTrue(all(B.ends_sentence(t) for _, _, t in segs))

    def test_closing_quote_clip_kept(self):
        words = [W('He', 0, 0.4), W('said', 0.4, 0.8), W('go!"', 0.8, 1.4)]
        segs = self.seg(words)
        self.assertEqual(len(segs), 1)
        self.assertTrue(segs[0][2].endswith('go!"'))


class SelectNarratorTest(unittest.TestCase):
    def test_dominant_speaker_fallback(self):
        words = [W('a', 0, 3, 'S0'), W('b', 3, 3.5, 'S1'), W('c', 3.5, 4, 'S1')]
        spk, _info = B.select_narrator(words, None, 24000, None, None, 0.5)
        self.assertEqual(spk, 'S0')

    def test_no_words_returns_none(self):
        spk, _ = B.select_narrator([], None, 24000, None, None, 0.5)
        self.assertIsNone(spk)


class NarratorWordsTest(unittest.TestCase):
    def test_none_speaker_words_excluded(self):
        # WhisperX-shaped result: some words carry speaker=None (un-diarized regions).
        result = {'segments': [
            {'speaker': 'SPEAKER_00', 'words': [
                {'word': 'Hello', 'start': 0.0, 'end': 0.4, 'speaker': 'SPEAKER_00'},
                {'word': 'world', 'start': 0.4, 'end': 0.8, 'speaker': None},
            ]},
            {'speaker': None, 'words': [
                {'word': 'noise', 'start': 1.0, 'end': 1.3, 'speaker': None},
                {'word': 'kept', 'start': 1.3, 'end': 1.6, 'speaker': 'SPEAKER_01'},
            ]},
        ]}
        out = B.narrator_words(result)
        self.assertEqual([w['word'] for w in out], ['Hello', 'kept'])
        self.assertTrue(all(w['speaker'] is not None for w in out))

    def test_word_inherits_segment_speaker_when_key_missing(self):
        # A word with no 'speaker' key falls back to the segment's speaker.
        result = {'segments': [
            {'speaker': 'SPEAKER_00', 'words': [{'word': 'Hi', 'start': 0.0, 'end': 0.3}]},
            {'speaker': None, 'words': [{'word': 'gone', 'start': 0.5, 'end': 0.9}]},
        ]}
        out = B.narrator_words(result)
        self.assertEqual([(w['word'], w['speaker']) for w in out], [('Hi', 'SPEAKER_00')])


class LoadRecordsTest(unittest.TestCase):
    def test_truncated_final_line_recovered(self):
        # A kill mid-write leaves a partial JSON fragment on the last line.
        good = ['{"id": "host-000000", "dur": 1.5}', '{"id": "host-000001", "dur": 2.0}']
        with tempfile.TemporaryDirectory() as td:
            rp = Path(td) / 'records.jsonl'
            rp.write_text('\n'.join(good) + '\n{"id": "host-000002", "du')  # truncated tail
            recs = B.load_records(rp)
            self.assertEqual([r['id'] for r in recs], ['host-000000', 'host-000001'])

    def test_blank_lines_skipped_and_missing_file_empty(self):
        with tempfile.TemporaryDirectory() as td:
            rp = Path(td) / 'records.jsonl'
            self.assertEqual(B.load_records(rp), [])  # no file yet
            rp.write_text('{"id": "a"}\n\n   \n{"id": "b"}\n')
            self.assertEqual([r['id'] for r in B.load_records(rp)], ['a', 'b'])


class ResumeStateTest(unittest.TestCase):
    def test_failed_file_recorded_in_state_and_skipped_on_resume(self):
        # The per-file error path marks the source 'done' so a permanently-bad file is
        # not retried on resume. write_state_atomic is the helper it uses; round-trip it
        # the same way main() reads it back, then assert the resume skip predicate.
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / 'state.json'
            key = '/abs/path/bad-episode.mp3'
            done = set()
            done.add(key)
            B.write_state_atomic(state_path, {'done': sorted(done)})

            # Resume reads state.json exactly as main() does.
            reloaded = json.loads(state_path.read_text())
            resumed_done = set(reloaded['done'])
            self.assertIn(key, resumed_done)            # failed file persisted to state.json
            self.assertTrue(key in resumed_done)        # so main()'s 'if key in done' skips it

    def test_resume_does_not_reprocess_done_source(self):
        # Already-'done' sources are skipped, so their clips are never rebuilt/duplicated.
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / 'state.json'
            srcs = ['/a/ep1.mp3', '/a/ep2.mp3', '/a/ep3.mp3']
            B.write_state_atomic(state_path, {'done': sorted([srcs[0], srcs[2]])})
            done = set(json.loads(state_path.read_text())['done'])
            to_process = [s for s in srcs if s not in done]   # main()'s 'if key in done: continue'
            self.assertEqual(to_process, ['/a/ep2.mp3'])

    def test_write_state_atomic_leaves_no_tmp_file(self):
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / 'state.json'
            B.write_state_atomic(state_path, {'done': ['/x.mp3']})
            self.assertTrue(state_path.exists())
            self.assertFalse(state_path.with_suffix('.json.tmp').exists())


class ManifestsTest(unittest.TestCase):
    def test_three_formats(self):
        recs = [{'id': f'host-{i:06d}', 'raw': f'Line {i}', 'norm': f'Line {i}.'} for i in range(20)]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            B.write_manifests(out, recs, 'host', val_fraction=0.1)
            lj = (out / 'metadata.csv').read_text().splitlines()
            self.assertEqual(lj[0], 'host-000000|Line 0|Line 0.')      # no header, no .wav
            sty = (out / 'train_list.txt').read_text().splitlines()
            val = (out / 'val_list.txt').read_text().splitlines()
            self.assertTrue(sty[0].endswith('|0') and '.wav|' in sty[0])  # .wav + int speaker
            self.assertEqual((len(val), len(sty)), (2, 18))               # ~10% val split
            xt = (out / 'metadata_xtts.csv').read_text().splitlines()
            self.assertEqual(xt[0], 'audio_file|text|speaker_name')       # header
            self.assertTrue(xt[1].startswith('wavs/host-'))


if __name__ == '__main__':
    unittest.main()
