'''
Script to create svm
'''

import time
from joblib import dump, load
from util import tensorflow_cuda


import chars2vec
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

from char_embeddings_pkg.util import Util
from data_import_pkg.tweet_parser.parse_lid_spaeng import TweetParser

parser = TweetParser('data_import_pkg/lince_spaeng', 'train')

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
                    test_size=0.01,
                    train_size=0.01,
                    random_state=109
                    )


scaler_train = StandardScaler().fit(X_train)
scaler_test = StandardScaler().fit(X_test)
X_train = scaler_train.transform(X_train)
X_test = scaler_test.transform(X_test)

params_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
          'gamma': ['scale', 'auto'],
          'kernel': ['poly'],
          'degree': [2,3,4,5,6],
          'coef0': [0.001, 0.01, 0.1] }

# params_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
#             'kernel': ['rbf'],
#             'gamma': ['scale', 'auto'],}

start = time.time()


clf = svm.SVC(cache_size=2000, class_weight='balanced')

#Create the GridSearchCV object
grid_clf = GridSearchCV(clf, params_grid)

#Fit the data with the best possible parameters
grid_clf = grid_clf.fit(X_train, y_train)

end = time.time()

with open('howmuchtime2.txt', 'w+') as f:
    f.write(str(end - start))

with open('name.txt', 'w+') as f:
    f.write(str(grid_clf.cv_results_) + '_2')

with open('poly_best_params', 'w+') as file:
    file.write(str(grid_clf.best_params_))
