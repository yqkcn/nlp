import nltk

nltk.download("brown")
from nltk.corpus import brown

print(brown.words())

from collections import Counter

puncs = {",", ".", "--", "-", "!", "?", ":", ";", "``", "''", "(", ")", "[", "]"}
word_list = (x.lower() for x in brown.words() if x not in puncs)
token_counter = Counter(word_list)
# print(token_counter.most_common(20))


docs = ["The faster Harry got to the store, the faster and faster Garry would get home."]
docs.append("Harry is hairy and faster than Jill.")
docs.append("Jill is not as hairy as Hairy.")


from sklearn.feature_extraction.text import TfidfVectorizer
corpus = docs
vectorizer = TfidfVectorizer(min_df=1)
model = vectorizer.fit_transform(corpus)
print(model.todense().round(2))