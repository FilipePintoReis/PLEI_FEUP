'''
Script to create svm
'''

import time
from joblib import dump, load
from util import tensorflow_cuda


import chars2vec
from sklearn import svm, metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from char_embeddings_pkg.util import Util
from data_import_pkg.tweet_parser.parse_lid_spaeng import TweetParser

parser = TweetParser('data_import_pkg/lince_spaeng')

words = []
labels = []
for word in parser.get_annotations():
    words.append(word.word)
    labels.append(word.label)

path_to_embedding_model = 'trained_models/char_embeddings/tweet'
c2v_model = chars2vec.load_model(path_to_embedding_model)

word_embeddings = c2v_model.vectorize_words(words)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(word_embeddings,
                    labels,
                    test_size=0.1,
                    train_size=0.1,
                    random_state=109
                    )

scaler_train = StandardScaler().fit(X_train)
scaler_test = StandardScaler().fit(X_test)
X_train = scaler_train.transform(X_train)
X_test = scaler_test.transform(X_test)

clf = svm.SVC(cache_size=2000, kernel='poly')

start = time.time()
clf.fit(X_train, y_train)
end = time.time()

with open('fit_time.txt', 'w+') as f:
    f.write(str(end - start))


dump(clf, 'trained_models/svm/tweet/polynomial_kernel')
#clf = load('trained_models/svm/tweet/linear_kernel')

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average='macro', zero_division=1))
print("Recall:",metrics.recall_score(y_test, y_pred, average='macro',zero_division=1))
print("F1_score:",metrics.f1_score(y_test, y_pred, average='macro',zero_division=1))
