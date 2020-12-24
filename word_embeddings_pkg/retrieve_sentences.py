'''
Module to retrieve sentences from parsers.
This sentences are going to be used to train the Word2Vec model.
'''

from  data_import_pkg.e_book_parser.e_book_parser import EBookParser
from data_import_pkg.tweet_parser.parse_lid_spaeng import TweetParser


class RetrieveSentences:
    '''
    RetrieveSentences class takes in the type of parser (Can either be ebook or tweet),
    a path and language (optional, only used when the path is to a book).
    Can also receive list of e-books.
    '''

    @staticmethod
    def retrieve_sentences(type_of_parser, path, language=None):
        '''
        retrieve_sentences is a generator function that yields the sentences.
        In the case of the e-book it's a paragraph, in the case of tweets it's a full tweet.
        @type_of_parser: e-book, e-books or tweet.
        @path: Path to tweet folder, path to e-book or list of paths to e-books.
        @language: a language or list of languages. Only applies to e-books.
        '''
        if type_of_parser == 'e-book':
            parsers = [EBookParser(path, language)]
        elif type_of_parser == 'e-books':
            parsers = [EBookParser(path[i], language[i]) for i in range(len(path))]
        elif type_of_parser == 'tweet':
            parsers = [TweetParser(path)]


        sentence = []

        for parser in parsers:
            for word in parser.get_annotations():
                if word.precedent_id is None:
                    sentence = [word.word]

                elif word.postcedent_id is None:
                    sentence.append(word.word)
                    yield sentence

                else:
                    sentence.append(word.word)
