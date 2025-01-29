import unittest
from pathlib import Path

from kokoro import KPipeline

from audiblez import main


class MainTest(unittest.TestCase):
    def base(self, file_path, **kwargs):
        pipeline = KPipeline(lang_code='a')  # a for american or b for british
        main(pipeline, file_path=file_path, voice='af_sky', pick_manually=False, speed=1, **kwargs)

    # def test_0_txt(self):
    #     Path('book.m4b').unlink(missing_ok=True)
    #     self.base(file_path='../txt/book.txt')
    #     self.assertTrue(Path('book.m4b').exists())

    def test_1_mini(self):
        Path('mini.m4b').unlink(missing_ok=True)
        self.base(file_path='../epub/mini.epub')
        self.assertTrue(Path('mini.m4b').exists())

    def test_2_allan_poe(self):
        Path('poe.m4b').unlink(missing_ok=True)
        self.base(file_path='../epub/poe.epub')
        self.assertTrue(Path('poe.m4b').exists())

    # def test_3_gene(self):
    #     Path('gene.m4b').unlink(missing_ok=True)
    #     self.base(file_path='../epub/gene.epub')
    #     self.assertTrue(Path('gene.m4b').exists())
