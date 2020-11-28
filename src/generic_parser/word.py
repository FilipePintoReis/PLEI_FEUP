'''
Contains word class
'''

class Word:
    '''
    This class is used to represent words and their precedents/postcedents
    '''

    def __init__(self, word, label, word_id, precedent_id, postcedent_id):
        '''
        Constructor
        '''
        self.word = word
        self.label = label
        self.word_id = word_id
        self.precedent_id = precedent_id
        self.postcedent_id = postcedent_id

    def __repr__(self):
        '''
        String representation of Word
        '''
        return f'Word: {self.word}\tLabel: {self.label}'

    def verbose_repr(self):
        '''
        Verbose representation of Word
        '''
        return f'''Word: {self.word}
                   Label: {self.label}
                   Word id: {self.word_id}
                   Pre id: {self.precedent_id}
                   Post id: {self.postcedent_id}
                  '''
