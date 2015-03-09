# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 14:45:43 2015

@Author: Robert M.
@Description: Main idea, group together words that distribute more or less the same over the classes
@Parameters: Number of word groups to define
@Algorithm: Rissanen's Minimum Description Length (MDL) principle to select number of word groups

"""
from __future__ import print_function
import Parser
import re
#     from nltk.book import *

def GetAuthorTargets():
    targets = []
    filenames = Parser.get_filenames();
    for filename in filenames:
        m = re.search('^[a-z]+-',filename)
        author = m.group(0)[:-1]
        targets.append(author);
    return targets

def ReadByAuthor():
    authors = set();
    filesByAuthor = {}
    filenames = Parser.get_filenames();
    for filename in filenames:
        m = re.search('^[a-z]+-',filename)
        author = m.group(0)[:-1]
        authors.add(author);
        if(author in filesByAuthor):
            files = filesByAuthor[author]
            files = files + Parser.get_data(filename)
        else:
            files = Parser.get_data(filename)
        filesByAuthor[author] = files;
    return authors,filesByAuthor
    
def LexicalDiversity(text):
    l = len(set(text))
    return len(set(text))/len(text)

def Percentage(count,total): 
    return 100 * count/total
    
def RemoveStopWords(word_list):
    from nltk.corpus import stopwords
    filtered_word_list = word_list[:]
    for word in word_list:
        if word in stopwords.words('english'):
            filtered_word_list.remove(word)
    return filtered_word_list
    
def GetParameterReport():
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.grid_search import RandomizedSearchCV
    
    #Define the range of neighbors to test. 1..20    
    k = np.arange(20)+1
    
    # load the datasets
    filenames = Parser.get_filenames()
    Ytr = GetAuthorTargets()
    
    tfid_vect = TfidfVectorizer(stop_words='english')
    Xtr = tfid_vect.fit_transform(Parser.cat_data(filenames))
    
    # prepare a parameter list to test on
    parameters = {'n_neighbors': k,
                  'weights':('uniform','distance'),
                  'algorithm':('auto','ball_tree','kd_tree')}
                  
    # create and fit a KNeighbors model
    model = KNeighborsClassifier()
    
    # Run a randomized search using cross validation with K=5
    rsearch = RandomizedSearchCV(model, parameters,cv=5)
    
    rsearch.fit(Xtr, Ytr)
    
    # summarize the results of the random parameter search
    print('Best Score:',rsearch.best_score_)
    print('Best Estimator:',rsearch.best_estimator_)
    
def AnalyzeFilteredTexts(authors,filesByAuthor):
    from nltk import word_tokenize
    from nltk.probability import FreqDist
    for author in authors:
        files = filesByAuthor[author]
        text = word_tokenize(files)
        text = RemoveStopWords(text)
        print("Length of documents by author: ",len(files)," Author:",author)
        fdist = FreqDist(text)
        # Write to file for analysis
        with open('MostCommon50-RemovedStopWords.txt','a')as file:
            line = "Length of documents by author: "+str(len(files))+" Author:"+author
            file.write(line)
            line = "\n50 most common\n"+str(fdist.most_common(50))+"\n-------------\n"
            file.write(line)
            #print("50 most common\n",fdist.most_common(50),"\n-------------\n")
  
#Still testing this  
def doPipelineStrategy():
    
    from pprint import pprint
    from time import time
    import logging
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.grid_search import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    
    #logging.basicConfig(level = logging.info,format='%(asctime)s %(levelname)s %(message)s')
    
    #################################################
    # Get Train Data, Targets
    filenames = Parser.get_filenames();
    
    #Ytr = GetAuthorTargets()
    #Xtr = Parser.cat_data(filenames)
    categories = [
    'alt.atheism',
    'talk.religion.misc',
    ]
    data = fetch_20newsgroups(subset='train', categories=categories)
    Xtr = data.data
    Ytr = data.target
    
    #######################################################################
    # define pipeline strategy using KNN
    pipeline = Pipeline([('vect',CountVectorizer()),
                         ('tfidf',TfidfTransformer()),
                         ('clf',KNeighborsClassifier())
                         ])
    
    #####################################################
    # Set up parameters we are going to test on for optimum feature extraction
    parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    'clf__n_neighbors': (1, 3, 5),
    #'clf__penalty': ('l2', 'elasticnet'),
    #'clf__n_iter': (10, 50, 80),
    }
    
    gridSearch = GridSearchCV(pipeline,parameters,n_jobs=-1,verbose=1)
    
    print("Performing grid search")
    
    print("pipeline", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    gridSearch.fit(Xtr,Ytr)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % gridSearch.best_score_)
    print("Best parameters set:")
    best_parameters = gridSearch.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    


GetParameterReport()


        
    