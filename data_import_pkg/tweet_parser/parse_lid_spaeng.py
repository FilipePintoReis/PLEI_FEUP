'''
Contains parsers
'''

from data_import_pkg.generic_parser.generic_parser import GenericParser
from data_import_pkg.generic_parser.word import Word
from data_import_pkg.tweet_parser.read_lid_spaeng import LinceFileReader

class TweetParser(GenericParser):
    '''
    Parser for tweets
    '''
    def __init__(self, prefix, data_type):
        '''
        Prefix to lince_spaeng folder
        '''
        super().__init__()
        self.type = 'Tweet'
        self.prefix = prefix
        self.data_type = data_type
        self.train_str = LinceFileReader.train_data(self.prefix)
        self.dev_str = LinceFileReader.dev_data(self.prefix)
        self.counter = 0

    def get_all_tweets_train(self):
        """Gets all train tweets separated"""
        for tweet in self.train_str.split('\n\n'):
            yield tweet

    def get_all_tweets_dev(self):
        """Gets all dev tweets separated"""
        for tweet in self.dev_str.split('\n\n'):
            yield tweet

    @staticmethod
    def parse_line(line):
        """Parses line"""
        line = line.strip()

        conditions = [len(line) == 0, len(line) > 0 and line[0] == "#"]
        for condition in conditions:
            if condition:
                return False

        return line.split('\t')

    def get_annotations(self):
        '''
        If self.data_type == train, then retrieves train data, else retrieves dev data.
        '''
        gen = (self.get_all_tweets_dev(), self.get_all_tweets_train())[self.data_type == 'train']
        for tweet in gen:
            lines = tweet.split('\n')
            lines = lines[1:]
            len_lines = len(lines)

            for index, line in enumerate(lines):
                parsed_line = TweetParser.parse_line(line)

                if not parsed_line:
                    pass
                else:
                    precedent = self.counter - 1
                    postcedent = self.counter + 1
                    if index == 0:
                        precedent = None
                    if index == len_lines - 1:
                        postcedent = None

                    yield Word(parsed_line[0], parsed_line[1], self.counter, precedent, postcedent)
                    self.counter += 1
