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
import numpy as np

from char_embeddings_pkg.util import Util
from data_import_pkg.tweet_parser.parse_lid_spaeng import TweetParser

EMBEDDING_ARRAY_SIZE = 32
parser = TweetParser('data_import_pkg/lince_spaeng', 'train')

words = []
labels = []
word_obj = []
for word in parser.get_annotations():
    words.append(word.word)
    labels.append(word.label)
    word_obj.append(word)

path_to_embedding_model = 'trained_models/char_embeddings/tweet'
c2v_model = chars2vec.load_model(path_to_embedding_model)

word_to_embedding = Util.dictionary_embeddings(c2v_model, words)
id_to_embedding = Util.word_id_embed_dict(word_to_embedding, word_obj)

word_embeddings = Util.word_feature_mixer(id_to_embedding, word_obj[0], EMBEDDING_ARRAY_SIZE)
word_embeddings = np.array([np.array(word_embeddings)])

print('Total:', len(word_obj))

for i in range(1, len(word_obj)):
    if (i % 10000) == 0:
        print('Progress:', i)
    new_word = np.array(Util.word_feature_mixer(id_to_embedding, word_obj[i], EMBEDDING_ARRAY_SIZE))
    new_word = np.array([new_word])
    word_embeddings = np.concatenate((word_embeddings, new_word))

print('\n\n\n\n', type(word_embeddings), len(word_embeddings))


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(word_embeddings,
                    labels,
                    train_size=0.1,
                    test_size=0.05,
                    random_state=109
                    )

#Create a svm Classifier
clf = svm.SVC(cache_size=2000, kernel='poly', coef0=0.1, degree=3, gamma='scale',C=100)


start = time.time()
#Train the model using the training sets
clf.fit(X_train, y_train)
end = time.time()
with open('fit_time_feature_mix.txt', 'w+') as f:
    f.write(str(end - start))

#Save classifier
dump(clf, 'trained_models/svm/tweet/polynomial_kernel_mixed')
#clf = load('trained_models/svm/tweet/linear_kernel')

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average='macro', zero_division=1))
print("Recall:",metrics.recall_score(y_test, y_pred, average='macro',zero_division=1))
print("F1_score:",metrics.f1_score(y_test, y_pred, average='macro',zero_division=1))
