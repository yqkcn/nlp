import numpy as np

# topic = {}
# tfidf = dict(list(zip('cat dog apple lion NYC love'.split(), np.random.rand(6))))

import pandas as pd
from nlpia.data.loaders import get_data
pd.options.display.width = 120
sms = get_data("sms-spam")
index = ['sms{} {}'.format(i, "!"*j) for (i, j) in zip(range(len(sms)), sms.spam)]
sms = pd.DataFrame(sms.values, columns=sms.columns, index=index)
sms['spam'] = sms.spam.astype(int)
print(len(sms))
print(sms.spam.sum())
print(sms.head(6))

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize

tfidf_model = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf_model.fit_transform(raw_documents=sms.text).toarray()
print(tfidf_docs.shape)
print(sms.spam.sum())

print("*"*100)
mask = sms.spam.astype(bool).values
spam_centroid = tfidf_docs[mask].mean(axis=0)
ham_centroid = tfidf_docs[~mask].mean(axis=0)
print(spam_centroid.round(2))
print(ham_centroid.round(2))
spamminess_score = tfidf_docs.dot(spam_centroid - ham_centroid)
print(spamminess_score.round(2))

from sklearn.preprocessing import MinMaxScaler
sms['lda_score'] = MinMaxScaler().fit_transform(spamminess_score.reshape(-1, 1))
sms['lda_predict'] = (sms.lda_score > 0.5).astype(int)
print(sms['spam lda_predict lda_score'.split()].round(2).head(6))
