from nltk.tokenize import TreebankWordTokenizer
from collections import Counter

sentence = """The faster Harry got to the store, the faster Harry, the faster, would get home."""
tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(sentence.lower())
# print(tokens)

bag_of_words = Counter(tokens)
# print(bag_of_words)
# print(bag_of_words.most_common(4))
# 某个词在文档中出现的频率称为 词项频率， TF

# 风筝
kite_text = """A kite is a tethered heavier-than-air or lighter-than-air craft with wing surfaces that react against the air to create lift and drag forces.[2] A kite consists of wings, tethers and anchors. Kites often have a bridle and tail to guide the face of the kite so the wind can lift it.[3] Some kite designs don’t need a bridle; box kites can have a single attachment point. A kite may have fixed or moving anchors that can balance the kite. One technical definition is that a kite is “a collection of tether-coupled wing sets“.[4] The name derives from its resemblance to a hovering bird.[5]
The lift that sustains the kite in flight is generated when air moves around the kite's surface, producing low pressure above and high pressure below the wings.[6] The interaction with the wind also generates horizontal drag along the direction of the wind. The resultant force vector from the lift and drag force components is opposed by the tension of one or more of the lines or tethers to which the kite is attached.[7] The anchor point of the kite line may be static or moving (e.g., the towing of a kite by a running person, boat, free-falling anchors as in paragliders and fugitive parakites[8][9] or vehicle).[10][11]
The same principles of fluid flow apply in liquids, so kites can be used in underwater currents.[12][13] Paravanes and otter boards operate underwater on an analogous principle.
Man-lifting kites were made for reconnaissance, entertainment and during development of the first practical aircraft, the biplane.
Kites have a long and varied history and many different types are flown individually and at festivals worldwide. Kites may be flown for recreation, art or other practical uses. Sport kites can be flown in aerial ballet, sometimes as part of a competition. Power kites are multi-line steerable kites designed to generate large forces which can be used to power activities such as kite surfing, kite landboarding, kite buggying and snow kiting."""

tokens = tokenizer.tokenize(kite_text.lower())
token_counts = Counter(tokens)
print(token_counts)

# 下载停词
import nltk
nltk.download("stopwords", quiet=True)
stopwords = nltk.corpus.stopwords.words("english")
tokens = [x for x in tokens if x not in stopwords]
kite_counts = Counter(tokens)
# print(kite_counts)

# 词频向量
document_vector = []
doc_length = len(tokens)
for k, v in kite_counts.items():
    document_vector.append(v/ doc_length)
# print(document_vector)

docs = ["The faster Harry got to the store, the faster and faster Garry would get home."]
docs.append("Harry is hairy and faster than Jill.")
docs.append("Jill is not as hairy as Hairy.")
doc_tokens = []
for doc in docs:
    doc_tokens += [sorted(tokenizer.tokenize(doc.lower()))]
print(doc_tokens)
all_doc_tokens = sum(doc_tokens, [])
print(all_doc_tokens)
lexicon = sorted(set(all_doc_tokens))
print(len(lexicon))

from collections import OrderedDict

zero_dict = OrderedDict((token, 0) for token in lexicon)
print(zero_dict)

import copy
doc_vectors = []
for doc in docs:
    vec = copy.deepcopy(zero_dict)
    tokens = tokenizer.tokenize(doc.lower())
    token_counts = Counter(tokens)
    for key, value in token_counts.items():
        vec[key] = value / len(lexicon)
    doc_vectors.append(vec)

