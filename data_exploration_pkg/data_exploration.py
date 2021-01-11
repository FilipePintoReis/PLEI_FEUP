from data_import_pkg.tweet_parser.parse_lid_spaeng import TweetParser
from data_import_pkg.e_book_parser.e_book_parser import EBookParser

tweet_parser_train = TweetParser('data_import_pkg/lince_spaeng', 'train')
tweet_parser_dev = TweetParser('data_import_pkg/lince_spaeng', 'dev')

ebook_parser = EBookParser(['data_import_pkg/e_books/processed/english', 
                            'data_import_pkg/e_books/processed/spanish'],
                            ['english', 'spanish'])

tweet_dev_words = []
tweet_dev_labels = []
for word in tweet_parser_train.get_annotations():
    tweet_dev_words.append(word.word)
    tweet_dev_labels.append(word.label)

tweet_train_words = []
tweet_train_labels = []
for word in tweet_parser_dev.get_annotations():
    tweet_train_words.append(word.word)
    tweet_train_labels.append(word.label)

ebook_words = []
ebook_labels = []
for word in ebook_parser.get_annotations():
    ebook_words.append(word.word)
    ebook_labels.append(word.label)


list_of_lists_of_words = [tweet_dev_words, tweet_train_words, ebook_words]
list_of_lists_of_labels = [tweet_dev_labels, tweet_train_labels, ebook_labels]
list_of_lists = list_of_lists_of_words + list_of_lists_of_labels


for list_words in list_of_lists_of_words:
    counter = 0
    s = set()
    for word in list_words:
        s.add(word)
        counter += 1
    print('Number of words:', counter)    
    print('Distinct words:', len(s))
    print()



for list_labels in list_of_lists_of_labels:
    dic = {}
    for label in list_labels:
        if not label in dic:
            dic[label] = 1
        else:
            dic[label] += 1
    print('Occurences of each label is:', dic)
    