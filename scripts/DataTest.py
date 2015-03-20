# Script to test the Bag of words representation in Parser.py using a Multnomial Bayes classifier

# UPDATED TO TEST BOTH KNN AND MNB
# Runs a loop a given number of times to get an average accuracy performance


import Parser, numpy, random
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB
from KNNClassifier import KNNClassifier

# Number of times to run predictions
iters = 1

# Lists of prediction accuracies from each run
total_mnb_tr_list = []
total_mnb_te_list = []
total_knn_tr_list = []
total_knn_te_list = []

for i in range(iters):
    filenames = Parser.get_filenames();
    author_target_map = Parser.map_author_value(filenames)

    # Create vectorizer of chosen type (uncomment the type you want)    
    tfid_vect = CountVectorizer() # Ignore the nonsensical name, this is for ease of use
##    tfid_vect = TfidfVectorizer()

    random.seed(0);
    random.shuffle(filenames);

    # Create feature matrix 
    FILENAMES = tfid_vect.fit_transform(Parser.cat_data(filenames))

    # Set up cross-validation
    n_samples,n_features = FILENAMES.shape
    kf = cross_validation.KFold(n_samples,5)

    # Set k for the knn classifier
    k = 1

    # Lists to average the folds together
    mnb_tr_list = []
    mnb_te_list = []
    knn_tr_list = []
    knn_te_list = []

    for train_index,test_index in kf:

        # Break training from validation
        ValidationSet = FILENAMES[test_index]
        TrainSet = FILENAMES[train_index]

        # Set up validations targets
        datadirectory = []
        for i in test_index:
            datadirectory.append(filenames[i])
        ValidationTarget = Parser.get_target_values(datadirectory,author_target_map)

        # Set up train targets
        datadirectory = []
        for i in train_index:
            datadirectory.append(filenames[i])
        TrainTarget = Parser.get_target_values(datadirectory,author_target_map)

        # Create learners
        mnb_clf = MultinomialNB()
        mnb_clf.fit(TrainSet, TrainTarget)

        knn_clf = KNNClassifier(k)
        knn_clf.Train(TrainSet, TrainTarget)

        # Run predictions
        knn_train_predictions = knn_clf.Predict(TrainSet)
        knn_predictions = knn_clf.Predict(ValidationSet)

        mnb_train_predictions = mnb_clf.predict(TrainSet)
        mnb_predictions = mnb_clf.predict(ValidationSet)

        # Record the prediction accuracies
        mnb_tr_list.append(numpy.mean(mnb_train_predictions == TrainTarget))
        mnb_te_list.append(numpy.mean(mnb_predictions == ValidationTarget))
        knn_tr_list.append(numpy.mean(knn_train_predictions == TrainTarget))
        knn_te_list.append(numpy.mean(knn_predictions == ValidationTarget))

    # Record the cross-validation accuracies
    total_mnb_tr_list.append(numpy.mean(mnb_tr_list))
    total_mnb_te_list.append(numpy.mean(mnb_te_list))
    total_knn_tr_list.append(numpy.mean(knn_tr_list))
    total_knn_te_list.append(numpy.mean(knn_te_list))

    # Print accuracies for each run for debugging
##    print("ITER #" + str(i))
##    print("Train Accuracy of KNN",numpy.mean(knn_train_predictions == Parser.TRAIN_TARGET))
##    print("Test Accuracy of KNN",numpy.mean(knn_predictions == Parser.TEST_TARGET))
##
##    print("Train Accuracy of MNB",numpy.mean(mnb_train_predictions == Parser.TRAIN_TARGET))
##    print("Test Accuracy of MNB",numpy.mean(mnb_predictions == Parser.TEST_TARGET))

# Output average accuracies
print("\nAverage MNB train accuracy:", numpy.mean(total_mnb_tr_list))
print("Average MNB test accuracy:", numpy.mean(total_mnb_te_list))
print("Average KNN train accuracy:", numpy.mean(total_knn_tr_list))
print("Average KNN test accuracy:", numpy.mean(total_knn_te_list))
