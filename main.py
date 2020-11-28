'''
Main
'''

# from src.tweet_parser.parse_lid_spaeng import TweetParser
from src.e_book_preprocess.preprocess import Preprocess

# t = TweetParser('src/lid_spaeng')

# for index, word in enumerate(t.get_annotations()):
#     if index > 10:
#         break
#     if word is not None:
#         print(word.verbose_repr())

Preprocess.generate_clean_book(
    'src/e_books/processed/english/Three_Contributions_to_the_Theory_of_Sex_by_Sigmund_Freud',
    'src/e_books/unprocessed/english/Three_Contributions_to_the_Theory_of_Sex_by_Sigmund_Freud'
    )
