from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sa = SentimentIntensityAnalyzer()
# print(sa.lexicon)
# print([(tok, score) for tok, score in sa.lexicon.items()])
print(sa.polarity_scores(text="Python is very readable and it's great for NLP."))
print(sa.polarity_scores(text="Fuck you!"))
print(sa.polarity_scores(text="Son of bitch!"))

