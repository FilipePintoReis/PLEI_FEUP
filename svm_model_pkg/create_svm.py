'''
Script to create svm
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
                    train_size=0.05,
                    test_size=0.05,
                    random_state=109
                    )

#Create a svm Classifier
clf = svm.SVC(cache_size=1000, kernel='poly')

# Polynomial: C, degree, coef0 and gamma
# Radial Base: C and gamma
# C, 0 - 1000.
# gamma, 'scale' or 'auto'
# degree, 1 - 10
# coef0, 0 - 1

start = time.time()
#Train the model using the training sets
clf.fit(X_train, y_train)
end = time.time()
with open('fit_time.txt', 'w+') as f:
    f.write(str(end - start))

#Save classifier
dump(clf, 'trained_models/svm/tweet/polynomial_kernel')
#clf = load('trained_models/svm/tweet/linear_kernel')

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average='macro', zero_division=1))
print("Recall:",metrics.recall_score(y_test, y_pred, average='macro',zero_division=1))
print("F1_score:",metrics.f1_score(y_test, y_pred, average='macro',zero_division=1))
