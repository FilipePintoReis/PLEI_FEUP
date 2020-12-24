'''
Utilitary class
'''

from random import choice, randint
import string

class Util:
    '''
    Utilitary class
    '''
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

        return new_word
