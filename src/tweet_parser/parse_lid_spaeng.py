'''
Contains parsers
'''

from src.generic_parser.generic_parser import GenericParser
from src.generic_parser.word import Word
from src.tweet_parser.read_lid_spaeng import LinceFileReader

class TweetParse(GenericParser):
    '''
    Parser for tweets
    '''
    def __init__(self, prefix):
        super().__init__()
        self.type = 'Tweet'
        self.prefix = prefix
        self.train_str = LinceFileReader.train_data(self.prefix)
        self.counter = 0

    def get_all_tweets(self):
        """Gets all tweets separated"""

        for tweet in self.train_str.split('\n\n'):
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
        for tweet in self.get_all_tweets():
            lines = tweet.split('\n')
            for index, line in enumerate(lines):
                parsed_line = TweetParse.parse_line(line)

                if not parsed_line:
                    yield None
                else:
                    precedent = self.counter - 1
                    postcedent = self.counter + 1
                    if index == 0:
                        precedent = None
                    if index == len(lines) - 1:
                        postcedent = None

                    yield Word(parsed_line[0], parsed_line[1], self.counter, precedent, postcedent)
