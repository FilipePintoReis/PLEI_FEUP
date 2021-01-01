'''
Use data in pre-trained model.
'''

from util import tensorflow_cuda

import chars2vec
import sklearn.decomposition
import matplotlib.pyplot as plt


path_to_model = 'trained_models/char_embeddings/tweet'

c2v_model = chars2vec.load_model(path_to_model)

# words = ['Natural', 'Language', 'Understanding',
#          'Naturael', 'Longuge', 'Updderctundjing',
#          'Motural', 'Lamnguoge', 'Understaating',
#          'Naturrow', 'Laguage', 'Unddertandink',
#          'Nattural', 'Languagge', 'Umderstoneding',
#          'Whatever', 'Wtv', 'quiero' 'quero',
#          'amor', 'amorzito', 'amori', 'papa',
#          'puta', 'puto', 'corazon', 'gracias', 'grazi']

words = ['Natural', 'Language', 'Understanding',
         'Naturael', 'Longuge', 'Updderctundjing']

# Create word embeddings
word_embeddings = c2v_model.vectorize_words(words)

print(len(word_embeddings))

# # Project embeddings on plane using the PCA
# projection_2d = sklearn.decomposition.PCA(n_components=2).fit_transform(word_embeddings)

# # Draw words on plane
# f = plt.figure(figsize=(8, 6))

# for j in range(len(projection_2d)):
#     plt.scatter(projection_2d[j, 0], projection_2d[j, 1],
#                 marker=('$' + words[j] + '$'),
#                 s=500 * len(words[j]), label=j,
#                 facecolors='green' if words[j]
#                             in ['Natural', 'Language', 'Understanding'] else 'black')

# plt.show()
