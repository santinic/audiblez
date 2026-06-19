"""Hermetic tests for the pronunciation lexicon (audiblez/lexicon.py).

Pure stdlib module, so these run anywhere (no skip guard needed).
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from audiblez import lexicon


class ApplyTest(unittest.TestCase):
    def test_whole_word_replacement(self):
        m = {'Kade': 'Kaid'}
        self.assertEqual(lexicon.apply_lexicon('Kade ran.', m), 'Kaid ran.')
        # substring inside another word is not replaced
        self.assertEqual(lexicon.apply_lexicon('Kadence sang.', m), 'Kadence sang.')

    def test_identity_and_empty_entries_are_noops(self):
        self.assertEqual(lexicon.apply_lexicon('NASA rocks', {'NASA': 'NASA'}), 'NASA rocks')
        self.assertEqual(lexicon.apply_lexicon('NASA rocks', {'NASA': ''}), 'NASA rocks')

    def test_longer_terms_applied_first(self):
        m = {'New York': 'noo york', 'York': 'jork'}
        # "New York" must win over the shorter "York" inside it
        self.assertEqual(lexicon.apply_lexicon('New York', m), 'noo york')

    def test_empty_mapping_returns_text_unchanged(self):
        self.assertEqual(lexicon.apply_lexicon('anything', {}), 'anything')

    def test_empty_or_blank_key_is_ignored(self):
        # An empty/whitespace key would otherwise become \b\b and inject at every boundary.
        self.assertEqual(lexicon.apply_lexicon('hello world', {'': 'X'}), 'hello world')
        self.assertEqual(lexicon.apply_lexicon('hello world', {'   ': 'X'}), 'hello world')

    def test_chained_replacement_does_not_corrupt_longer_output(self):
        # Single-pass: the shorter 'AI' rule must NOT re-match inside 'AIME' -> 'AI-me'.
        m = {'AIME': 'AI-me', 'AI': 'ay-eye'}
        self.assertEqual(lexicon.apply_lexicon('Meet AIME today.', m), 'Meet AI-me today.')
        # a standalone AI is still replaced
        self.assertEqual(lexicon.apply_lexicon('AI rules', m), 'ay-eye rules')

    def test_replacement_value_with_backslash_is_literal(self):
        # function-based sub: no \1/\g backslash interpretation in the replacement value
        self.assertEqual(lexicon.apply_lexicon('Kade', {'Kade': r'Ka\1de'}), r'Ka\1de')


class SeedTest(unittest.TestCase):
    def test_acronyms_always_seeded(self):
        terms = lexicon.seed_terms('The NASA probe used a USB cable.')
        self.assertIn('NASA', terms)
        self.assertIn('USB', terms)

    def test_accented_proper_nouns_are_seeded(self):
        # Unicode-aware title-case detection: accented recurring names must be seeded.
        text = 'Muñoz spoke. Muñoz left. André waved. André smiled. Björn nodded. Björn ran.'
        terms = lexicon.seed_terms(text, min_count=2)
        for name in ('Muñoz', 'André', 'Björn'):
            self.assertIn(name, terms)

    def test_lowercase_words_not_seeded_as_proper_nouns(self):
        terms = lexicon.seed_terms('the cat sat and the cat ran and the cat ate', min_count=2)
        self.assertNotIn('cat', terms)
        self.assertNotIn('the', terms)

    def test_recurring_proper_nouns_seeded_singletons_not(self):
        text = 'Kade went home. Kade slept. A Random name appeared once.'
        terms = lexicon.seed_terms(text, min_count=2)
        self.assertIn('Kade', terms)
        self.assertNotIn('Random', terms)

    def test_stopwords_excluded(self):
        text = 'The the The There There There When When When'
        self.assertEqual(lexicon.seed_terms(text), [])

    def test_build_seed_is_identity_mapping(self):
        seed = lexicon.build_seed_lexicon('NASA NASA bonjour')
        self.assertEqual(seed.get('NASA'), 'NASA')


class RoundTripTest(unittest.TestCase):
    def test_save_then_load(self):
        with TemporaryDirectory() as tmp:
            path = lexicon.lexicon_path('book.epub', tmp)
            lexicon.save_lexicon(path, {'Kade': 'Kaid'})
            self.assertEqual(lexicon.load_lexicon(path), {'Kade': 'Kaid'})
            self.assertEqual(path.name, 'book.lexicon.json')

    def test_load_missing_or_malformed_returns_empty(self):
        with TemporaryDirectory() as tmp:
            self.assertEqual(lexicon.load_lexicon(Path(tmp) / 'nope.json'), {})
            bad = Path(tmp) / 'bad.json'
            bad.write_text('{not json')
            self.assertEqual(lexicon.load_lexicon(bad), {})


class FingerprintTest(unittest.TestCase):
    def test_empty_and_identity_have_blank_fingerprint(self):
        self.assertEqual(lexicon.fingerprint({}), '')
        self.assertEqual(lexicon.fingerprint({'NASA': 'NASA'}), '')

    def test_active_overrides_have_stable_nonblank_fingerprint(self):
        fp1 = lexicon.fingerprint({'Kade': 'Kaid'})
        fp2 = lexicon.fingerprint({'Kade': 'Kaid'})
        self.assertTrue(fp1)
        self.assertEqual(fp1, fp2)
        self.assertNotEqual(fp1, lexicon.fingerprint({'Kade': 'Kayd'}))


if __name__ == '__main__':
    unittest.main()
