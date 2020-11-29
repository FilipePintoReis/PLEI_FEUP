'''
This module contains a parser for books taken from project gutenberg into word class.
'''

from src.generic_parser.generic_parser import GenericParser
from src.generic_parser.word import Word

class EBookParser(GenericParser):
    '''
    This class contains a parser for books taken from project gutenberg into word class.
    '''
    def __init__(self, path, language):
        '''
        Constructor.
        Receives path to the book
        '''
        super().__init__()
        self.path = path
        self.type = 'EBook'
        self.language = language
        self.counter = 0

    def get_annotations(self):
        '''
        Returns word annotations
        '''

        file = open(self.path, "r", encoding="UTF-8")
        book_str = file.read()

        for paragraph in book_str.split('\n'):
            if paragraph != '':
                paragraph = paragraph.replace('\t', ' ')
                words = paragraph.split(' ')
                len_words = len(words)

                for index, word in enumerate(words):
                    if word != '':

                        precedent = self.counter - 1
                        postcedent = self.counter + 1

                        if index == 0:
                            precedent = None
                        if index == len_words - 1:
                            postcedent = None
                
                        language = self.language

                        if not word.isalpha():
                            language = 'other'

                        yield Word(word, language, self.counter, precedent, postcedent)
                        
                        self.counter += 1
