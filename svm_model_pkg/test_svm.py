'''
This script evaluates the model performance with the test data.
'''

import time
from joblib import dump, load

from util import tensorflow_cuda

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import chars2vec

from char_embeddings_pkg.util import Util
from data_import_pkg.tweet_parser.parse_lid_spaeng import TweetParser

parser = TweetParser('data_import_pkg/lince_spaeng', 'dev')

words = []
labels = []
for word in parser.get_annotations():
    words.append(word.word)
    labels.append(word.label)

path_to_embedding_model = 'trained_models/char_embeddings/tweet'
c2v_model = chars2vec.load_model(path_to_embedding_model)

word_embeddings = c2v_model.vectorize_words(words)

clf = load('trained_models/svm/tweet/polynomial_kernel')

#Predict the response for test dataset
y_pred = clf.predict(word_embeddings)

print("Accuracy:",metrics.accuracy_score(labels, y_pred))
print("Precision:",metrics.precision_score(labels, y_pred, average='macro', zero_division=1))
print("Recall:",metrics.recall_score(labels, y_pred, average='macro', zero_division=1))
print("F1_score:",metrics.f1_score(labels, y_pred, average='macro', zero_division=1))
