'''
Main
'''
import chars2vec

from char_embeddings_pkg.util import Util
from data_import_pkg.tweet_parser.parse_lid_spaeng import TweetParser
from data_import_pkg.e_book_parser.e_book_parser import EBookParser

parser = TweetParser('data_import_pkg/lince_spaeng')

list_words = [word.word for word in parser.get_annotations()]
results = Util.process_list_words(list_words)

dim = 50
path_to_model = 'trained_models/char_embeddings/tweet'

model_chars = list(results[0])
x_train = results[1]
y_train = results[2]
len_train = len(y_train)

old = 0
data_batch = 10000

for i in range(data_batch, len_train, data_batch):
    if i > len_train - 1:
        i = len_train
    # Create and train chars2vec model using given training data
    # def train_model(emb_dim, X_train, y_train, model_chars,
    #             max_epochs=200, patience=10, validation_split=0.05, batch_size=64):
    my_c2v_model = chars2vec.train_model(dim, x_train[old:i], y_train[old:i], model_chars)
    # Save your pretrained model
    chars2vec.save_model(my_c2v_model, path_to_model)
    old = i

    # Load your pretrained model 
    my_c2v_model = chars2vec.load_model(path_to_model)
