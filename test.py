# from util import tensorflow_cuda

# import chars2vec
import numpy as np

# path_to_embedding_model = 'trained_models/char_embeddings/tweet'
# c2v_model = chars2vec.load_model(path_to_embedding_model)

# #word_embeddings = c2v_model.vectorize_words(['strog', 'strog'])

# def dictionary_embeddings(c2v_model, words):
#     word_set = set()
#     for word in words:
#         if word not in word_set:
#             word_set.add(word)

#     new_list = list(word_set)
#     vectorized = c2v_model.vectorize_words(new_list)

#     word_dict = {}
#     for i, word in enumerate(new_list):
#         word_dict[word] = vectorized[i]

#     return word_dict

# print(dictionary_embeddings(c2v_model, ['aa', 'aa', 'ds']))

# ar1 = np.array([np.array([1,2,3,4])])
# ar2 = np.array([np.array([4,5,6,7])])

# print(type(ar1))

# ar3 = np.concatenate((ar1, ar2))

# print(ar3)

import traceback

try:
    ar = []

    print(ar[0])
except Exception as e:
    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
    message = template.format(type(e).__name__, e.args)
    print(message)
    print(traceback.print_exc())