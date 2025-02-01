import os
import unittest
from pathlib import Path

from kokoro import KPipeline

from audiblez import main


class MainTest(unittest.TestCase):
    def base(self, name, url, **kwargs):
        if not Path(f'{name}.epub').exists():
            os.system(f'wget {url} -O {name}.epub')
        Path(f'{name}.m4b').unlink(missing_ok=True)
        os.system(f'rm {name}_chapter_*.wav')
        merged_args = dict(voice='af_sky', pick_manually=False, speed=1.0, max_chapters=2)
        merged_args.update(kwargs)
        main(f'{name}.epub', **merged_args)
        m4b_file = Path(f'{name}.m4b')
        self.assertTrue(m4b_file.exists())
        self.assertTrue(m4b_file.stat().st_size > 256 * 1024)

    def test_poe(self):
        url = 'https://www.gutenberg.org/ebooks/1064.epub.images'
        self.base('poe', url)

    @unittest.skip('too slow for CI')
    def test_orwell(self):
        url = 'https://archive.org/download/AnimalFarmByGeorgeOrwell/Animal%20Farm%20by%20George%20Orwell.epub'
        self.base('orwell', url)

    def test_italian_pirandello(self):
        url = 'https://www.liberliber.eu/mediateca/libri/p/pirandello/cosi_e_se_vi_pare_1925/epub/pirandello_cosi_e_se_vi_pare_1925.epub'
        self.base('pirandello', url, voice='im_nicola')
        self.assertTrue(Path('pirandello.m4b').exists())

    @unittest.skip('too slow for CI')
    def test_italian_manzoni(self):
        url = 'https://www.liberliber.eu/mediateca/libri/m/manzoni/i_promessi_sposi/epub/manzoni_i_promessi_sposi.epub'
        self.base('manzoni', url, voice='im_nicola', max_chapters=1)

    def test_french_baudelaire(self):
        url = 'http://gallica.bnf.fr/ark:/12148/bpt6k70861t.epub'
        self.base('baudelaire', url, voice='ff_siwis')
