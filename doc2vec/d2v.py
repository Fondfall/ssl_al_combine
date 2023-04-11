from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile

documents = [TaggedDocument(words, [movie_id]) for movie_id, words in movie_profile["profile"].iteritems()]
# 训练模型并保存
model = Doc2Vec(documents, vector_size=100, window=3, min_count=1, workers=4, epochs=20)


fname = get_tmpfile("my_doc2vec_model")
model.save(fname)
