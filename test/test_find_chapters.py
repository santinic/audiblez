import unittest
from ebooklib import epub

from audiblez import find_good_chapters, find_document_chapters_and_extract_texts


@unittest.skip('Development only, not for CI')
class FindChaptersTest(unittest.TestCase):
    def base(self, file, expected_chapter_names):
        book = epub.read_epub(file)
        document_chapters = find_document_chapters_and_extract_texts(book)
        chapters = find_good_chapters(document_chapters)
        chapter_names = [c.get_name() for c in chapters]
        self.assertEqual(chapter_names, expected_chapter_names)

    def test_gene(self):
        self.base('../epub/gene.epub', [
            # '000_ACover.xhtml',  # 81
            # '002_FM_halftitle.xhtml',  # 2302
            # '003_FM_titlepage.xhtml',  # 550
            # '004_FM_titleverso.xhtml',  # 2438
            # '005_FM_contents.xhtml',  # 5243
            # '006_FM_other.xhtml',  # 23923
            # '007_FM_preface.xhtml',  # 9546
            # '008_FM_foreword.xhtml',  # 5421
            # '009_FM_preface.xhtml',  # 7079
            '010_chapter.xhtml',  # 30013
            '011_chapter.xhtml',  # 22985
            '012_chapter.xhtml',  # 70519
            '013_chapter.xhtml',  # 56550
            '014_chapter.xhtml',  # 62632
            '015_chapter.xhtml',  # 62847
            '016_chapter.xhtml',  # 39001
            '017_chapter.xhtml',  # 48388
            '018_chapter.xhtml',  # 72467
            '019_chapter.xhtml',  # 63803
            '020_chapter.xhtml',  # 35317
            '021_chapter.xhtml',  # 85500
            '022_chapter.xhtml',  # 91176
            # '023_BM_other.xhtml',  # 25800
            # '025_BM_endNotes.xhtml',  # 220098
            # '025_BM_bibliographyGroup.xhtml',  # 63300
            # '025_BM_regular.xhtml',  # 231385
            # '027_BM_other.xhtml',  # 21939
            # 'navigation.xhtml',  # 34263
        ])

    def test_spawn_of_dagon(self):
        self.base('../epub/kuttner-spawn-of-dagon.epub', [
            # 'bk01-toc.xhtml',  # 276
            # 'cover.xhtml',  # 76
            # 'index.xhtml',  # 1399
            'ch01.xhtml',  # 38426
        ])

    def test_solenoid(self):
        self.base('../epub/solenoid.epub', [
            'xhtml/part1.xhtml',
            'xhtml/part2.xhtml',
            'xhtml/part3.xhtml',
            'xhtml/part4.xhtml'
        ])

    def test_lewis_innocents(self):
        self.base('../epub/lewis-innocents.epub', [
            # 'bk01-toc.xhtml',  # 1603
            # 'cover.xhtml',  # 72
            # 'index.xhtml',  # 1554
            # 'pr01.xhtml',  # 1141
            'ch01.xhtml',  # 8884
            'ch02.xhtml',  # 7909
            'ch03.xhtml',  # 13934
            'ch04.xhtml',  # 14362
            'ch05.xhtml',  # 19451
            'ch06.xhtml',  # 19506
            'ch07.xhtml',  # 11392
            'ch08.xhtml',  # 11356
            'ch09.xhtml',  # 18890
            'ch10.xhtml',  # 19804
            'ch11.xhtml',  # 10024
            'ch12.xhtml',  # 4915
            'ch13.xhtml',  # 20006
            'ch14.xhtml',  # 14206
            'ch15.xhtml',  # 16461
            'ch16.xhtml',  # 13201
            'ch17.xhtml',  # 11470
            'ch18.xhtml',  # 19445
        ])

    def test_poe(self):
        self.base('../epub/poe.epub', [
            # 'bk01-toc.xhtml',  # 332
            # 'cover.xhtml',  # 72
            # 'index.xhtml',  # 1454
            'ch01.xhtml',  # 40993
        ])

    def test_chalmers(self):
        self.base('../epub/chalmers.epub', [
            # 'nav.xhtml',  # 38422
            # 'Text/00_Cover.xhtml',  # 175
            # 'Text/01_Also.xhtml',  # 359
            # 'Text/02_Title.xhtml',  # 603
            # 'Text/03_Dedi.xhtml',  # 211
            # 'Text/04_Contents.xhtml',  # 6302
            # 'Text/06_Intro.xhtml',  # 34726
            # 'Text/11_Part1.xhtml',  # 332
            'Text/12_Chapter01.xhtml',  # 33940
            'Text/12_Chapter02.xhtml',  # 47738
            # 'Text/12_Chapter02_Part2.xhtml',  # 328
            'Text/12_Chapter03.xhtml',  # 42010
            'Text/12_Chapter04.xhtml',  # 42182
            'Text/12_Chapter05.xhtml',  # 51283
            # 'Text/12_Chapter05_Part3.xhtml',  # 327
            'Text/12_Chapter06.xhtml',  # 46570
            'Text/12_Chapter07.xhtml',  # 48752
            'Text/12_Chapter08.xhtml',  # 52048
            'Text/12_Chapter09.xhtml',  # 35717
            # 'Text/12_Chapter09_Part4.xhtml',  # 343
            'Text/12_Chapter10.xhtml',  # 43114
            'Text/12_Chapter11.xhtml',  # 53469
            'Text/12_Chapter12.xhtml',  # 29441
            'Text/12_Chapter13.xhtml',  # 34265
            # 'Text/12_Chapter13_Part5.xhtml',  # 327
            'Text/12_Chapter14.xhtml',  # 42381
            'Text/12_Chapter15.xhtml',  # 43875
            'Text/12_Chapter16.xhtml',  # 28861
            # 'Text/12_Chapter16_Part6.xhtml',  # 328
            'Text/12_Chapter17.xhtml',  # 45652
            'Text/12_Chapter18.xhtml',  # 43328
            'Text/12_Chapter19.xhtml',  # 34453
            # 'Text/12_Chapter19_Part7.xhtml',  # 334
            'Text/12_Chapter20.xhtml',  # 40554
            'Text/12_Chapter21.xhtml',  # 30382
            'Text/12_Chapter22.xhtml',  # 57467
            'Text/12_Chapter23.xhtml',  # 37671
            'Text/12_Chapter24.xhtml',  # 53101
            # 'Text/24_Ack.xhtml',  # 7312
            # 'Text/25_Glossary.xhtml',  # 10956
            # 'Text/27_Note.xhtml',  # 166936
            # 'Text/31_Index.xhtml',  # 76649
            # 'Text/32_Copyright.xhtml',  # 2166
        ])

    def test_dracula_default_all_chapters(self):
        # If it cannot detect chapters, default to all available documents.
        self.base('../epub/dracula.epub', [
            '2903486949112998543_345-h-0.htm.xhtml',  # 2087
            '2903486949112998543_345-h-1.htm.xhtml',  # 3997
            '2903486949112998543_345-h-2.htm.xhtml',  # 561
            '2903486949112998543_345-h-3.htm.xhtml',  # 31676
            '2903486949112998543_345-h-4.htm.xhtml',  # 29401
            '2903486949112998543_345-h-5.htm.xhtml',  # 30682
            '2903486949112998543_345-h-6.htm.xhtml',  # 31268
            '2903486949112998543_345-h-7.htm.xhtml',  # 19487
            '2903486949112998543_345-h-8.htm.xhtml',  # 30530
            '2903486949112998543_345-h-9.htm.xhtml',  # 31189
            '2903486949112998543_345-h-10.htm.xhtml',  # 34156
            '2903486949112998543_345-h-11.htm.xhtml',  # 32106
            '2903486949112998543_345-h-12.htm.xhtml',  # 32400
            '2903486949112998543_345-h-13.htm.xhtml',  # 28597
            '2903486949112998543_345-h-14.htm.xhtml',  # 39876
            '2903486949112998543_345-h-15.htm.xhtml',  # 35877
            '2903486949112998543_345-h-16.htm.xhtml',  # 34678
            '2903486949112998543_345-h-17.htm.xhtml',  # 31502
            '2903486949112998543_345-h-18.htm.xhtml',  # 24786
            '2903486949112998543_345-h-19.htm.xhtml',  # 30615
            '2903486949112998543_345-h-20.htm.xhtml',  # 37264
            '2903486949112998543_345-h-21.htm.xhtml',  # 30288
            '2903486949112998543_345-h-22.htm.xhtml',  # 33094
            '2903486949112998543_345-h-23.htm.xhtml',  # 33339
            '2903486949112998543_345-h-24.htm.xhtml',  # 29104
            '2903486949112998543_345-h-25.htm.xhtml',  # 30735
            '2903486949112998543_345-h-26.htm.xhtml',  # 33429
            '2903486949112998543_345-h-27.htm.xhtml',  # 34229
            '2903486949112998543_345-h-28.htm.xhtml',  # 38840
            '2903486949112998543_345-h-29.htm.xhtml',  # 40006
            '2903486949112998543_345-h-30.htm.xhtml',  # 4081
            '2903486949112998543_345-h-31.htm.xhtml',  # 19692
            'toc.xhtml',  # 4900
            'wrap0000.xhtml',  # 374
        ])
