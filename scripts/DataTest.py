# Script to test the Bag of words representation in Parser.py using a Multnomial Bayes classifier

# UPDATED TO TEST BOTH KNN AND MNB
# Runs a loop a given number of times to get an average accuracy performance


import Parser, numpy
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

# Number of times to run predictions
iters = 25

# Lists of prediction accuracies from each run
mnb_tr_list = []
mnb_te_list = []
knn_tr_list = []
knn_te_list = []

for i in range(iters):
    # Re-run to create new train/test splits
    Parser.run_script()

    train_data = Parser.TRAIN_SPARSE
    train_targets = Parser.TRAIN_TARGET

    test_data = Parser.TEST_SPARSE

    # Not used currently
##    Amap = Parser.author_target_map
##    vect = Parser.count_vect

    k = 1

    # Create learners
    mnb_clf = MultinomialNB()
    mnb_clf.fit(train_data, train_targets)

    knn_clf = KNeighborsClassifier (n_neighbors=k, weights='uniform')
    knn_clf.fit(train_data, train_targets)

    # Run predictions
    knn_train_predictions = knn_clf.predict(train_data)
    knn_predictions = knn_clf.predict(test_data)

    mnb_train_predictions = mnb_clf.predict(train_data)
    mnb_predictions = mnb_clf.predict(test_data)

    # Record the prediction accuracies
    mnb_tr_list.append(numpy.mean(mnb_train_predictions == Parser.TRAIN_TARGET))
    mnb_te_list.append(numpy.mean(mnb_predictions == Parser.TEST_TARGET))
    knn_tr_list.append(numpy.mean(knn_train_predictions == Parser.TRAIN_TARGET))
    knn_te_list.append(numpy.mean(knn_predictions == Parser.TEST_TARGET))

    # Print accuracies for each run for debugging
##    print("ITER #" + str(i))
##    print("Train Accuracy of KNN",numpy.mean(knn_train_predictions == Parser.TRAIN_TARGET))
##    print("Test Accuracy of KNN",numpy.mean(knn_predictions == Parser.TEST_TARGET))
##
##    print("Train Accuracy of MNB",numpy.mean(mnb_train_predictions == Parser.TRAIN_TARGET))
##    print("Test Accuracy of MNB",numpy.mean(mnb_predictions == Parser.TEST_TARGET))

# Output average accuracies
print("\nAverage MNB train accuracy:", numpy.mean(mnb_tr_list))
print("Average MNB test accuracy:", numpy.mean(mnb_te_list))
print("Average KNN train accuracy:", numpy.mean(knn_tr_list))
print("Average KNN test accuracy:", numpy.mean(knn_te_list))
