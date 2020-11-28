'''
Main
'''

from src.tweet_parser.parse_lid_spaeng import TweetParse

t = TweetParse('src/lid_spaeng')

for index, word in enumerate(t.get_annotations()):
    if index > 50:
        break
    print(word)
