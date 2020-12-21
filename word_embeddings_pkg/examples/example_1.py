from gensim.models.word2vec import Word2Vec
from gensim.test.utils import common_texts

MODEL_PREFIX = 'trained_models/model_1/'

model = Word2Vec(sentences=common_texts, size=100, window=5, min_count=1, workers=4)
model.save(f"{MODEL_PREFIX}word2vec.model")

model = Word2Vec.load(f"{MODEL_PREFIX}word2vec.model")
model.train([["hello", "world"]], total_examples=1, epochs=5)
