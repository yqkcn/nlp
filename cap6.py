from gensim.models.keyedvectors import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format("file", binary=True, limit=200000)


from gensim.models.word2vec import Word2Vec

num_features = 200
min_word_count = 3
num_workers = 2
window_size = 6
subsampling = 1e-3

model = Word2Vec(
    [["hello", "word"], ["python", "java", "php"]],
    workers=num_workers,
    vector_size=num_features,
    min_count=min_word_count,
    window=window_size,
    sample=subsampling
)