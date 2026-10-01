import unittest
from ebooklib import epub

from audiblez.core import find_document_chapters_and_extract_texts


class FootnoteMarkersTest(unittest.TestCase):
    def extract(self, body):
        book = epub.EpubBook()
        chapter = epub.EpubHtml(title='Chapter', file_name='ch1.xhtml')
        chapter.content = '<html><body>%s</body></html>' % body
        book.add_item(chapter)
        return find_document_chapters_and_extract_texts(book)[0].extracted_text

    def test_sup_marker_is_dropped(self):
        text = self.extract('<p>Most people would not worry about politics at all.'
                            '<sup><a href="notes.xhtml#en3" id="nr3">3</a></sup></p>')
        self.assertEqual(text, 'Most people would not worry about politics at all.\n')

    def test_epub3_noteref_is_dropped(self):
        text = self.extract('<p>Most people would not worry about politics at all.'
                            '<a epub:type="noteref" href="notes.xhtml#en3">3</a></p>')
        self.assertEqual(text, 'Most people would not worry about politics at all.\n')

    def test_ordinary_text_is_untouched(self):
        text = self.extract('<p>Most people would not worry about politics at all</p>')
        self.assertEqual(text, 'Most people would not worry about politics at all.\n')


if __name__ == '__main__':
    unittest.main()
