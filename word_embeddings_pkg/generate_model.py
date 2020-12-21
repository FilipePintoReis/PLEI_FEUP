'''
This class is used to generate Word2Vec models from parsed sentences.
'''

from gensim.models.word2vec import Word2Vec
from word_embeddings_pkg.retrieve_sentences import RetrieveSentences

class GenerateModel:
    '''
    This class is responsible for generating Word2Vec models from scratch, 
    as well a training pre-existing models with new data.
    '''

    @staticmethod
    def generate_new_model(path_to_model, type_of_parser, path, language=None):
        '''
        path_to_model: Location where the model is going to be placed, as well as it's name.
        type_of_parser: type of parser to retrieve sentences from.
        path: can either the path to the book, or the prefix to the tweet's folder.
        language: Only required when type_of_parser is book.
        then it should be the language the book is in. Elsewise, argument ignored.
        '''
        sentences = [word for word
        in RetrieveSentences.retrieve_sentences(type_of_parser, path, language)]

        model = Word2Vec(sentences=sentences, size=100, window=5, min_count=1, workers=4)
        model.train(sentences=sentences, total_examples=1, epochs=5)

        model.save(path_to_model)

GenerateModel.generate_new_model(
    'trained_models/tweet/tweet.model',
    'tweet', 'data_import_pkg/lince_spaeng')
