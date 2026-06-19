import os
import subprocess
import unittest
from pathlib import Path

from ebooklib import epub

from audiblez.core import main, find_document_chapters_and_extract_texts

# from phonemizer.backend.espeak.wrapper import EspeakWrapper

# EspeakWrapper.set_library('/opt/homebrew/Cellar/espeak-ng/1.52.0/lib/libespeak-ng.1.dylib')


class MainTest(unittest.TestCase):
    def base(self, name, url='', **kwargs):
        epub_path = Path(f'{name}.epub')
        if not epub_path.exists() and url:
            os.system(f'wget -q "{url}" -O "{name}.epub"')
        # Integration test: needs the epub (downloaded or a local fixture). Skip cleanly
        # when it can't be obtained (offline / source down / fixture absent) instead of
        # erroring, so `unittest discover` stays green without network or fixtures.
        if not epub_path.exists() or epub_path.stat().st_size == 0:
            epub_path.unlink(missing_ok=True)
            self.skipTest(f'could not obtain {name}.epub (no network/source/fixture)')
        Path(f'{name}.m4b').unlink(missing_ok=True)
        os.system(f'rm {name}_chapter_*.wav')
        merged_args = dict(voice='af_sky', pick_manually=False, speed=1.0, max_chapters=1, max_sentences=2)
        merged_args.update(kwargs)
        main(f'{name}.epub', **merged_args)
        m4b_file = Path(f'{name}.m4b')
        self.assertTrue(m4b_file.exists())
        self.assertTrue(m4b_file.stat().st_size > 1024)

    def test_poe(self):
        url = 'https://www.gutenberg.org/ebooks/1064.epub.images'
        self.base('poe', url)

    def test_orwell(self):
        url = 'https://archive.org/download/AnimalFarmByGeorgeOrwell/Animal%20Farm%20by%20George%20Orwell.epub'
        self.base('orwell', url)

    def test_italian_pirandello_and_filename_with_spaces(self):
        url = 'https://www.liberliber.eu/mediateca/libri/p/pirandello/cosi_e_se_vi_pare_1925/epub/pirandello_cosi_e_se_vi_pare_1925.epub'
        self.base('pirandello e spazio', url, voice='im_nicola')
        self.assertTrue(Path('pirandello e spazio.m4b').exists())

    def test_italian_manzoni(self):
        url = 'https://www.liberliber.eu/mediateca/libri/m/manzoni/i_promessi_sposi/epub/manzoni_i_promessi_sposi.epub'
        self.base('manzoni', url, voice='im_nicola')

    def test_french_baudelaire(self):
        url = 'http://gallica.bnf.fr/ark:/12148/bpt6k70861t.epub'
        self.base('baudelaire', url, voice='ff_siwis')

    def test_chinese(self):
        url = 'https://www.gutenberg.org/ebooks/24225.epub3.images'
        self.base('chinese', url, voice='zf_xiaobei')

    def test_leigh_and_play_result(self):
        if not Path('leigh.epub').exists():
            self.skipTest('leigh.epub fixture not present')
        book = epub.read_epub('leigh.epub')
        document_chapters = find_document_chapters_and_extract_texts(book)
        chapters = [c for c in document_chapters if c.get_name() == 'Text/Chap07.xhtml']
        self.base('leigh', voice='af_heart', selected_chapters=chapters, max_sentences=5)
        # Playback is interactive/blocking; only attempt it when explicitly opted in.
        if os.environ.get('AUDIBLEZ_TEST_PLAY'):
            subprocess.run(['ffplay', '-nodisp', '-autoexit', 'leigh.m4b'], check=True)

    def test_long_italian_sentence(self):
        text = ('È vero che non ho mai saputo troppo bene quante camere, quanti atri, quante scale e quanti corridoi '
                'avesse e molte volte, dirigendomi verso la mia camera da letto, mi sono ritrovato in saloni o in '
                'bagni non solo totalmente sconosciuti, come se fossero spariti i muri tra la mia casa e un’altra a '
                'essa appiccicata, ma anche lontani nel tempo, nello spazio e nel ricordo, con mobili scintillanti e '
                'strani, con orologi a pendolo di noce e candelabri con piccole foglie di ottone, ma questa volta la '
                'mia casa buia, dalle cui finestre nevicava all’impazzata, mi è parsa sconfinata.')
        from audiblez.core import split_long_sentence
        parts = split_long_sentence(text, 400)
        self.assertTrue(all(len(p) <= 400 for p in parts))
        self.assertEqual(' '.join(parts).split(), text.split())
