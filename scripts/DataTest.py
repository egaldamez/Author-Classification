# Script to test the Bag of words representation in Parser.py using a Multnomial Bayes classifier




import Parser
from sklearn.naive_bayes import MultinomialNB
import numpy

train_data = Parser.TRAIN_SPARSE
train_targets = Parser.TRAIN_TARGET

test_data = Parser.TEST_SPARSE
test_targets = Parser.TEST_TARGET


Amap = Parser.author_target_map
vect = Parser.count_vect


clf = MultinomialNB()
clf.fit(train_data, train_targets)

train_predictions = clf.predict(train_data)
test_predictions = clf.predict(test_data)


print("Beat accuracy of MNB on Training Data",numpy.mean(train_predictions == train_targets))
print("Beta accuracy of MNB on Test Data",numpy.mean(test_predictions == test_targets))


