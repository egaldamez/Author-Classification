# Script to test the Bag of words representation in Parser.py using a Multnomial Bayes classifier






import Parser
from sklearn.naive_bayes import MultinomialNB

train_data = Parser.TRAIN_SPARSE
test_data = Parser.TRAIN_TARGET

clf = MultinomialNB()
clf.fit(train_data, test_data)
