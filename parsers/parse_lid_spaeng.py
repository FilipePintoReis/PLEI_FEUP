'''
Contains parsers
'''

from .read_lid_spaeng import LinceFileReader


# TODO
# Refactor this into a class with statics functions

PREFIX = "lid_spaeng"

train_str = LinceFileReader.train_data(PREFIX)

def get_all_tweets(raw):
    """Gets all tweets separated"""

    for tweet in raw.split('\n\n'):
        yield tweet

def parse_line(line):
    """Parses line"""

    line = line.strip()

    conditions = [len(line) == 0, len(line) > 0 and line[0] == "#"]
    for condition in conditions:
        if condition:
            return False

    return line.split('\t')
