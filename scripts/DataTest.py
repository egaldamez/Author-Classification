# Script to test the Bag of words representation in Parser.py using a Multnomial Bayes classifier




import Parser
from sklearn.naive_bayes import MultinomialNB
import numpy

train_data = Parser.TRAIN_SPARSE
train_targets = Parser.TRAIN_TARGET

test_data = Parser.TEST_SPARSE

Amap = Parser.author_target_map
vect = Parser.count_vect


clf = MultinomialNB()
clf.fit(train_data, train_targets)

predictions = clf.predict(test_data)

print("Beta Accuracy of MNB",numpy.mean(predictions == Parser.TEST_TARGET))


