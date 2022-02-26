import re

import nltk
import numpy as np
import pandas as pd

sentence = """Tomas Jefferson began building Monticello at the age of 26.\n"""
token_sequence = str.split(sentence)
vocab = sorted(set(token_sequence))
num_tokens = len(token_sequence)
vocab_size = len(vocab)
# 独热向量
# 适合需要保存原始文本的需求，独热向量都很合适
onehot_vectors = np.zeros((num_tokens, vocab_size), int)
for i, word in enumerate(token_sequence):
    onehot_vectors[i, vocab.index(word)] = 1
# 构建 DataFrame
df = pd.DataFrame(onehot_vectors, columns=vocab)
# df = pd.DataFrame(pd.Series(dict((token, 1) for token in sentence.split())), columns=["sent"])

sentence += """Construction was done mostly by local masons and carpenters.\n"""
sentence += """He moved into the South Pavilion in 1770.\n"""
sentence += """Turning Monticello into a neoclassical masterpiece wa Jefferson's obsession."""

tokens = re.split(r'[-\s.,;!?]+', sentence)
corpus = {}
for i, sent in enumerate(sentence.split("\n")):
    corpus['sent{}'.format(i)] = dict((token, 1) for token in sent.split())
# 空间向量模型 VSM: vector space model
df = pd.DataFrame.from_records(corpus).fillna(0).astype(int).T

df = df.T
# print(df.sent0.dot(df.sent2))

# print(df[df.columns[:10]])

from nltk.util import ngrams

sentence = """Tomas Jefferson began building Monticello at the age of 26.\n"""
patter = re.compile(r"[-\s.,;!?]]+")
tokens = patter.split(sentence)
tokens = [x for x in tokens if x and x not in '- \t\n.,;!?']
gram_2 = list(ngrams(token_sequence, 2))
print(gram_2)
# print(list(ngrams(token_sequence, 3)))
gram_2 = [" ".join(x) for x in gram_2]

import nltk
# nltk.download("stopwords")
# stop_words = nltk.corpus.stopwords.words("english")
# print(stop_words)

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
a = " ".join(stemmer.stem(x).strip("'") for x in "dish washer's washed dishes".split())
print(a)