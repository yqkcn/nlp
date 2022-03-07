from nlpia.book.examples.ch04_catdog_lsa_3x6x16 import word_topic_vectors
a = word_topic_vectors.T.round(1)
print(a)

from nlpia.book.examples.ch04_catdog_lsa_sorted import lsa_models, prettify_tdm

bow_scd, tfidf_scd = lsa_models()
print(prettify_tdm(**bow_scd))

tdm = bow_scd['tdm']
print(tdm)

import numpy as np
U, s, vt = np.linalg.svd(tdm)
import pandas as pd
print(pd.DataFrame(U, index=tdm.index).round(2))

pd.set_option("display.max_columns", 6)
from sklearn.decomposition import PCA
import seaborn
from matplotlib import pyplot as plt
from nlpia.data.loaders import get_data
df = get_data('pointcloud').sample(1000)
pca = PCA(n_components=2)
df2d = pd.DataFrame(pca.fit_transform(df), columns=list("xy"))
df2d.plot(kind="scatter", x="x", y="y")
plt.show()