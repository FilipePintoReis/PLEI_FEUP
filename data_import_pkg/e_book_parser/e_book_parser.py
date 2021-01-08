'''
This module contains a parser for books taken from project gutenberg into word class.
'''

from glob import glob

from data_import_pkg.generic_parser.generic_parser import GenericParser
from data_import_pkg.generic_parser.word import Word

class EBookParser(GenericParser):
    '''
    This class contains a parser for books taken from project gutenberg into word class.
    '''
    def __init__(self, paths, languages):
        '''
        Constructor.
        Receives a path to all books of each language.
        Language of the books.
        '''
        super().__init__()
        self.paths = paths
        self.type = 'EBook'
        self.languages = languages
        self.counter = 0

    def get_annotations(self):
        '''
        Returns word annotations of all books
        '''

        for index, path in enumerate(self.paths):
            for book_path in glob(f'{path}/*.txt'):
                for word in EBookParser.get_book_annotations(book_path, self.counter, self.languages[index]):
                    yield word

    @staticmethod
    def get_book_annotations(path, counter, language):
        '''
        Returns word annotations
        '''

        file = open(path, "r", encoding="UTF-8")
        book_str = file.read()

        for paragraph in book_str.split('\n'):
            if paragraph != '':
                paragraph = paragraph.replace('\t', ' ')
                words = paragraph.split(' ')
                len_words = len(words)

                for index, word in enumerate(words):
                    if word != '':

                        precedent = counter - 1
                        postcedent = counter + 1

                        if index == 0:
                            precedent = None
                        if index == len_words - 1:
                            postcedent = None

                        lang = language

                        if not word.isalpha():
                            lang = 'other'

                        yield Word(word, lang, counter, precedent, postcedent)

                        counter += 1
