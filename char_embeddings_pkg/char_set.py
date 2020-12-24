'''
Module to create CharSet
'''

class CharSet:
    '''
    TODO
    '''
    @staticmethod
    def feed_word(char_set, received_word):
        '''
        TODO
        '''
        for char in received_word:
            char_set.add(char)

    @staticmethod
    def feed_list_words(list_words, action):
        '''
        TODO
        '''
        s = set()
        for word in list_words:
            CharSet.feed_word(s, action(word))

        return s
