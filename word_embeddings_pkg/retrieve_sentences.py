'''
Module to retrieve sentences from parsers.
This sentences are going to be used to train the Word2Vec model.
'''

from  data_import_pkg.e_book_parser.e_book_parser import EBookParser
from data_import_pkg.tweet_parser.parse_lid_spaeng import TweetParser


class RetrieveSentences:
    '''
    RetrieveSentences class takes in the type of parser (Can either be ebook or tweet),
    a path and language (optional, only used when the path is to a book)
    '''

    @staticmethod
    def retrieve_sentences(type_of_parser, path, language=None):
        '''
        retrieve_sentences is a generator function that yields the sentences.
        In the case of the e-book it's a paragraph, in the case of tweets it's a full tweet.
        '''
        if type_of_parser == 'e-book':
            parser = EBookParser(path, language)
        elif type_of_parser == 'tweet':
            parser = TweetParser(path)

        sentence = []

        for word in parser.get_annotations():
            if word.precedent_id is None:
                sentence = [word.word]

            elif word.postcedent_id is None:
                sentence.append(word.word)
                yield sentence

            else:
                sentence.append(word.word)
