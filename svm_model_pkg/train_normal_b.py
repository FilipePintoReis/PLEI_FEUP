'''
Script to create svm
'''

from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn import svm
import chars2vec

from util import tensorflow_cuda
from data_import_pkg.e_book_parser.e_book_parser import EBookParser

path_1 = 'data_import_pkg/e_books/processed/english'
path_2 = 'data_import_pkg/e_books/processed/spanish'
paths = [path_1, path_2]

parser = EBookParser(paths, ['english', 'spanish'])

words = []
labels = []
for word in parser.get_annotations():
    words.append(word.word)
    labels.append(word.label)

c2v_model = chars2vec.load_model('trained_models/char_embeddings/ebooks')

word_embeddings = c2v_model.vectorize_words(words)

###############################      Polynomial        ###############################################
X_train, _, y_train, _ = train_test_split(word_embeddings,
                    labels,
                    train_size=0.9999,
                    random_state=109
                    )

clf = svm.SVC(cache_size=2000, kernel='poly', coef0=0.1, degree=3, gamma='scale', C=100)

clf.fit(X_train, y_train)

dump(clf, 'trained_models/svm/tweet/polynomial_normal_b')
###############################      Polynomial        ###############################################

###############################        Radial          ###############################################
X_train, X_test, y_train, y_test = train_test_split(word_embeddings,
                    labels,
                    train_size=0.9999,
                    random_state=109
                    )

clf = svm.SVC(cache_size=2000, kernel='rbf', C=100, gamma='auto')

clf.fit(X_train, y_train)

dump(clf, 'trained_models/svm/tweet/rbf_normal_b')
###############################        Radial          ###############################################
