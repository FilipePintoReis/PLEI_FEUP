'''
Main
'''
import chars2vec

from char_embeddings_pkg.util import Util
from data_import_pkg.tweet_parser.parse_lid_spaeng import TweetParser
from data_import_pkg.e_book_parser.e_book_parser import EBookParser
from email_pkg.email_sender import EmailSender

server = EmailSender.start_server()

try:
    parser2 = EBookParser(['data_import_pkg/e_books/processed/english',
                        'data_import_pkg/e_books/processed/spanish'],
                        ['english', 'spanish'])

    list_words2 = [word.word for word in parser2.get_annotations()]

    parser1 = TweetParser('data_import_pkg/lince_spaeng', 'train')
    list_words1 = [word.word for word in parser1.get_annotations()]

    list_words = list_words1 + list_words2

    results = Util.process_list_words(list_words)

    EMBEDDING_ARRAY_SIZE = 32
    path_to_model = 'trained_models/char_embeddings/ebooks'

    model_chars = list(results[0])
    x_train = results[1]
    y_train = results[2]

    x_train = [e for i, e in enumerate(x_train) if i % 7 != 0 and i % 13 != 0 and i % 17]
    y_train = [e for i, e in enumerate(y_train) if i % 7 != 0 and i % 13 != 0 and i % 17]

    len_train = len(y_train)

    old = 32
    data_batch = 50000

    for i in range(data_batch, len_train, data_batch):
        if i > len_train - 1:
            i = len_train
        # Create and train chars2vec model using given training data
        # def train_model(emb_dim, X_train, y_train, model_chars,
        #             max_epochs=200, patience=10, validation_split=0.05, batch_size=64):
        my_c2v_model = chars2vec.train_model(EMBEDDING_ARRAY_SIZE,
                                            x_train[old:i],
                                            y_train[old:i],
                                            model_chars,
                                            max_epochs=100,
                                            patience=3,
                                            validation_split=0.15,
                                            batch_size=32)
        old = i

        # Save your pretrained model
        chars2vec.save_model(my_c2v_model, path_to_model)
        # # Load your pretrained model 
        # my_c2v_model = chars2vec.load_model(path_to_model)

    # Save your pretrained model
    #chars2vec.save_model(my_c2v_model, path_to_model)

    message = 'Finished train_mixed_embeddings'
    EmailSender.send_email(server, 'filipepintodosreis@gmail.com', message)

except:
    message = 'Failed during train_mixed_embeddings'
    EmailSender.send_email(server, 'filipepintodosreis@gmail.com', message)

finally:
    server.quit()
