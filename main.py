'''
Main
'''

from src.tweet_parser.parse_lid_spaeng import TweetParse

t = TweetParse('src/lid_spaeng')

for index, word in enumerate(t.get_annotations()):
    if index > 10:
        break
    if word is not None:
        print(word.verbose_repr())
