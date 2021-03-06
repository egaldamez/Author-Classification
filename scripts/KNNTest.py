import Parser

def cross_validate_knn(V,K):
    
    from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
    from sklearn import cross_validation
    from KNNClassifier import KNNClassifier
    import random
    
    filenames = Parser.get_filenames();
    author_target_map = Parser.map_author_value(filenames) # maps authors to integer values
    
    #bag of words representation
    count_vect = CountVectorizer(ngram_range=(2,3))
    
    # term frequency–inverse document frequency
    #tfid_vect = TfidfVectorizer(ngram_range=(2,3))
    
    random.seed(0);
    random.shuffle(filenames);
  
    print('Vectorizing Files\n');
    
    #SparseMatrix from Bag-of-Words
    FILENAMES = count_vect.fit_transform(Parser.cat_data(filenames))
    
    #SparseMatrix from Tf_IDF
    #FILENAMES = tfid_vect.fit_transform(cat_data(filenames))
    
    n_samples,n_features = FILENAMES.shape
    kf = cross_validation.KFold(n_samples,V)
    validationAccuracy = [];
    k = 1;
    for train_index,test_index in kf:
        ValidationSet = FILENAMES[test_index]
        TrainSet = FILENAMES[train_index]
        datadirectory = []
        for i in test_index:
            datadirectory.append(filenames[i])
        ValidationTarget = Parser.get_target_values(datadirectory,author_target_map)
        datadirectory = []
        for i in train_index:
            datadirectory.append(filenames[i])
        TrainTarget = Parser.get_target_values(datadirectory,author_target_map)
        clf = KNNClassifier(K)
        clf.Train(TrainSet, TrainTarget)
    
        predictions = clf.Predict(ValidationSet )
        va = numpy.mean(predictions == ValidationTarget)
        validationAccuracy.append(va);
        print("Beta Accuracy of KNN Iter# ",k," Result: ",va)
        k = k+1
    return validationAccuracy;

VA = cross_validate_knn(5,1)
print("Mean Accuracy: ", numpy.mean(VA))


