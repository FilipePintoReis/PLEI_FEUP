'''
Module to create Util
'''
from math import ceil
from random import choice, randint
import string

import numpy as np

from Levenshtein import distance

class Hashable:
    '''
    Utilitary class to hash unhashable data structures.
    '''
    def __init__(self, unhashable):
        self.unhashable = unhashable

class Util:
    '''
    TODO
    '''
    @staticmethod
    def populate_char_set_word(char_set, received_word):
        '''
        Adds all characters of a word to given set.
        '''
        for char in received_word:
            char_set.add(char)

    @staticmethod
    def populate_char_set_list(char_set, list_words):
        '''
        Adds all characters of all words to given set.
        '''
        char_set = set()
        for word in list_words:
            Util.populate_char_set_word(char_set, word)
        Util.populate_char_set_princtable(char_set)

        return char_set

    @staticmethod
    def populate_char_set_princtable(char_set):
        '''
        Adds princtable chars from string class to the char_set
        '''
        for character in string.printable:
            char_set.add(character)

    @staticmethod
    def process_list_words(list_words):
        '''
        Receives list of words, creates character set and x and y train from it.
        '''
        x_train = []
        y_train = []
        char_set = set()
        for word in list_words:
            Util.populate_char_set_word(char_set, word)

            n_diff = Util.number_of_differences(len(word))
            x_train.append(Util.levenstein_diff_maker(word, n_diff))
            y_train.append(0)

            x_train.append(Util.different_word(word, list_words, len(list_words)))
            y_train.append(1)

        Util.populate_char_set_princtable(char_set)

        return (char_set, x_train, y_train)

    @staticmethod
    def number_of_differences(word_len):
        '''
        Calculates acceptable number of differences in a misspelled word
        '''
        if word_len < 4:
            return 1

        return ceil(word_len/4)

    @staticmethod
    def levenstein_diff_maker(word, n_changes):
        '''
        Following method takes a word and performs n changes.
        A change may be an addition, deletion or edition.
        Returns changed word.
        '''
        ade = ['a', 'd', 'e']
        new_word = word
        rand_char = lambda: choice(string.printable)
        rand_pos = lambda: randint(0, len(word) - 1)

        for _ in range(n_changes):
            var = choice(ade)
            if var == 'a':
                pos = rand_pos()
                new_word = new_word[:pos] + rand_char() + new_word[pos:]
            elif var == 'd':
                pos = rand_pos()
                if rand_pos == len(new_word) - 1:
                    new_word = new_word[:pos]
                    continue
                new_word = new_word[:pos] + new_word[pos + 1:]
            elif var == 'e':
                pos = rand_pos()
                new_word = new_word[:pos] + rand_char() + new_word[pos + 1:]

        return (word, new_word)

    @staticmethod
    def different_word(word, list_words, len_list):
        '''
        Function that joins random, different words in array.
        '''
        word_len = len(word)

        counter = 10
        while counter > 0:
            var = randint(0, len_list - 1)
            new_word = list_words[var]

            if distance(word, new_word) > Util.number_of_differences(word_len):
                return (word, new_word)

            counter -= 1

        print("\n\n No different word could be found after 10 retries.")
        return (word, new_word)

    @staticmethod
    def word_id_embed_dict(dict_word_embedding, words):
        dict_id_embedding = {}
        for word in words:
            dict_id_embedding[word.word_id] = dict_word_embedding[word.word]

        return dict_id_embedding

    @staticmethod
    def word_feature_mixer(dictionary, word, size):
        '''
        Receives a dictionary mapping word_ids and their embeddings,
        a word_id and the length of the embeddings.
        Returns the concatenation of the current word plus pre and pos words.
        '''
        precedent_emb = dictionary[word.precedent_id].unhashable if not word.precedent_id is None else np.array(np.zeros(size))
        current_emb = dictionary[word.word_id].unhashable
        postcedent_emb = dictionary[word.postcedent_id].unhashable if not word.postcedent_id is None else np.array(np.zeros(size))

        return np.concatenate((precedent_emb, current_emb, postcedent_emb))

    @staticmethod
    def dictionary_embeddings(c2v_model, words):
        '''
        For a list of words, retrieves the correspondent embeddings
        '''
        word_set = set()
        for word in words:
            if word not in word_set:
                word_set.add(word)

        new_list = list(word_set)
        vectorized = c2v_model.vectorize_words(new_list)

        word_dict = {}
        for i, word in enumerate(new_list):
            word_dict[word] = Hashable(vectorized[i])

        return word_dict
