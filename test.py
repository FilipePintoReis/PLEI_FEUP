'''
fgv
'''

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

import chars2vec
import sklearn.decomposition
import matplotlib.pyplot as plt


# Load Inutition Engineering pretrained model
# Models names: 'eng_50', 'eng_100', 'eng_150', 'eng_200', 'eng_300'
c2v_model = chars2vec.load_model('eng_300')

words = ['Natural', 'Language', 'Understanding',
         'Naturael', 'Longuge', 'Updderctundjing',
         'Motural', 'Lamnguoge', 'Understaating',
         'Naturrow', 'Laguage', 'Unddertandink',
         'Nattural', 'Languagge', 'Umderstoneding',
         'Whatever', 'Wtv']

# Create word embeddings
word_embeddings = c2v_model.vectorize_words(words)

# Project embeddings on plane using the PCA
projection_2d = sklearn.decomposition.PCA(n_components=2).fit_transform(word_embeddings)

# Draw words on plane
f = plt.figure(figsize=(8, 6))

for j in range(len(projection_2d)):
    plt.scatter(projection_2d[j, 0], projection_2d[j, 1],
                marker=('$' + words[j] + '$'),
                s=500 * len(words[j]), label=j,
                facecolors='green' if words[j]
                            in ['Natural', 'Language', 'Understanding'] else 'black')

plt.show()
