'''
Module to create CharSet
'''

class CharSet:
    @staticmethod
    def feed_word(s, word):
        for char in word:
            s.add(char)

    @staticmethod
    def feed_list_words(l, action):
        s = set()
        for word in l:
            self.feed_word(action(word))
            
        return s

