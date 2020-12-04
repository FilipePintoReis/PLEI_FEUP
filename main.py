'''
Main
'''

# from data_import_pkg.tweet_parser.parse_lid_spaeng import TweetParser
# from data_import_pkg.e_book_preprocess.preprocess import Preprocess
from data_import_pkg.e_book_parser.e_book_parser import EBookParser

# t = TweetParser('data_import_pkg/LinCE_spaeng')

# for index, word in enumerate(t.get_annotations()):
#     if index > 10:
#         break
#     if word is not None:
#         print(word.verbose_repr())

# Preprocess.generate_clean_book(
#     'data_import_pkg/e_books/processed/english/Three_Contributions_to_the_Theory_of_Sex_by_Sigmund_Freud',
#     'data_import_pkg/e_books/unprocessed/english/Three_Contributions_to_the_Theory_of_Sex_by_Sigmund_Freud'
#     )

parser = EBookParser(
    'data_import_pkg/e_books/processed/english/Three_Contributions_to_the_Theory_of_Sex_by_Sigmund_Freud',
    'english'
    )

for index, word in enumerate(parser.get_annotations()):
    if index == 100:
        break
    print(word)
