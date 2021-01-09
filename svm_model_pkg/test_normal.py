'''
This script evaluates the model performance with the test data.
'''

import time
from joblib import dump, load
from traceback import print_exc

from util import tensorflow_cuda

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import chars2vec

from char_embeddings_pkg.util import Util
from data_import_pkg.tweet_parser.parse_lid_spaeng import TweetParser
from email_pkg.email_sender import EmailSender

server = EmailSender.start_server()

try:
    parser = TweetParser('data_import_pkg/lince_spaeng', 'dev')

    words = []
    labels = []
    for word in parser.get_annotations():
        words.append(word.word)
        labels.append(word.label)

    tweeter_embeddings_path = 'trained_models/char_embeddings/tweet'
    ebook_embeddings_path = 'trained_models/char_embeddings/ebooks'

    poly_tweet_model_path = 'trained_models/svm/tweet/polynomial_normal_t'
    poly_ebook_model_path = 'trained_models/svm/tweet/polynomial_normal_b'
    radial_tweet_model_path = 'trained_models/svm/tweet/rbf_normal_t'
    radial_ebook_model_path = 'trained_models/svm/tweet/rbf_normal_b'

    tweet_c2v_model = chars2vec.load_model(tweeter_embeddings_path)
    ebook_c2v_model = chars2vec.load_model(ebook_embeddings_path)

    embeddings_tweet = tweet_c2v_model.vectorize_words(words)
    embeddings_ebooks = ebook_c2v_model.vectorize_words(words)

    clf_poly_t = load(poly_tweet_model_path)
    clf_poly_b = load(poly_ebook_model_path)
    clf_radial_t = load(radial_tweet_model_path)
    clf_radial_b = load(radial_ebook_model_path)

    y_poly_t = clf_poly_t.predict(embeddings_tweet)
    EmailSender.send_email(server, 'filipepintodosreis@gmail.com', 'predicted poly t')
    y_poly_b = clf_poly_b.predict(embeddings_ebooks)
    EmailSender.send_email(server, 'filipepintodosreis@gmail.com', 'predicted poly b')
    y_radial_t = clf_radial_t.predict(embeddings_tweet)
    EmailSender.send_email(server, 'filipepintodosreis@gmail.com', 'predicted radial t')
    y_radial_b = clf_radial_b.predict(embeddings_ebooks)
    EmailSender.send_email(server, 'filipepintodosreis@gmail.com', 'predicted radial b')

    with open('results/poly_tweet_normal', 'w+') as file:
        file.write("Accuracy:",
                   metrics.accuracy_score(labels, y_poly_t))
        file.write("Precision:",
                metrics.precision_score(labels, y_poly_t, average='macro', zero_division=1))
        file.write("Recall:",
                   metrics.recall_score(labels, y_poly_t, average='macro', zero_division=1))
        file.write("F1_score:",
                   metrics.f1_score(labels, y_poly_t, average='macro', zero_division=1))

    with open('results/poly_ebook_normal', 'w+') as file:
        file.write("Accuracy:",
                   metrics.accuracy_score(labels, y_poly_b))
        file.write("Precision:",
                metrics.precision_score(labels, y_poly_b, average='macro', zero_division=1))
        file.write("Recall:",
                   metrics.recall_score(labels, y_poly_b, average='macro', zero_division=1))
        file.write("F1_score:",
                   metrics.f1_score(labels, y_poly_b, average='macro', zero_division=1))

    with open('results/radial_tweet_normal', 'w+') as file:
        file.write("Accuracy:",
                   metrics.accuracy_score(labels, y_radial_t))
        file.write("Precision:",
                metrics.precision_score(labels, y_radial_t, average='macro', zero_division=1))
        file.write("Recall:",
                   metrics.recall_score(labels, y_radial_t, average='macro', zero_division=1))
        file.write("F1_score:",
                   metrics.f1_score(labels, y_radial_t, average='macro', zero_division=1))

    with open('results/radial_ebook_normal', 'w+') as file:
        file.write("Accuracy:",
                   metrics.accuracy_score(labels, y_radial_b))
        file.write("Precision:",
                   metrics.precision_score(labels, y_radial_b, average='macro', zero_division=1))
        file.write("Recall:",
                   metrics.recall_score(labels, y_radial_b, average='macro', zero_division=1))
        file.write("F1_score:",
                   metrics.f1_score(labels, y_radial_b, average='macro', zero_division=1))
except Exception as ex:
    message = 'Failed during train_mixed_embeddings'
    template = "An exception of type {0} occurred.\nException: {1}\nStack trace: {2}"
    err = template.format(type(ex).__name__, ex, print_exc())
    print(err)
    EmailSender.send_email(server, 'filipepintodosreis@gmail.com', message + '\n\n' + err)

finally:
    server.quit()
