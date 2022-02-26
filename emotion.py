from nlpia.data.loaders import get_data
movies = get_data("hutto_movies")
# print(movies)
print(movies.head().round(2))
print(movies.text)

import pandas as pd
pd.set_option('display.width', 75)
from nltk.tokenize import casual_tokenize
bags_of_words = []
from collections import Counter
for text in move.text:
    bags_of_words.append(Counter(casual_tokenize(text)))
df_bows = pd.DataFrame.from_records(bags_of_words)
df_bows = df_bows.fillna(0).astype(int)
print(df.shape)
