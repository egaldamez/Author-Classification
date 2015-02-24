# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 12:37:33 2015

@author: Eddie
"""

# This script is used to parse the data into bag of words
# and into n_grams representations


from DataInitializer import data_directory_read, shuffle_data, split_data
from sklearn.feature_extraction.text import CountVectorizer
import numpy


#DATA_DIRECTORY = 'C:\\Users\\Edwin\\Documents\\CS175\\Project\\filtereddata'
DATA_DIRECTORY  = 'C:\\Users\\Robert\\Documents\\UCI\\CS 175\\Project\\filtereddata' # change to directory of filtereddata on your machine

TRAIN_SPARSE = None
TEST_SPARSE = None

TRAIN_TARGET = []
TEST_TARGET = []

# debugging
count_vect = None
author_target_map = None

def _extract_target_value(filename):
    '''
        From a filename such as author-book_title.txt, we will extract the author as a target value.
    '''
    return filename.split('-')[0]

def map_author_value(corpus):
    '''
        This function will assign an integer number to an author.
    '''

    author_map = {}
    author_id = 0 
    for doc in corpus:
        author = _extract_target_value(doc)
        if author not in author_map:
            author_map[author] = author_id
            author_id += 1

    return author_map


def get_target_values(data_directory, author_id_map):
    '''
        Will build an integer list of target values.
        EX
            [0,1,2,1,0] will mean that the target of the sparse matrix at index 0 will be the author associates with id 0
                        and the target value of the sparse matrix at index 1 will be the author associated with id 2
    '''
    targets = []

    for doc in data_directory:
        author = _extract_target_value(doc)
        targets.append(author_id_map[author])

    return targets



def cat_data(data_directory):
    '''
        Create a large list where each element is a string representing a document.
    '''
    from nltk import word_tokenize
    
    corpus = []
    
    tokenCount = []
    for document in data_directory:
        # open every file and append its contents to corpus
        with open(DATA_DIRECTORY + '\\' + document) as doc:
            raw = doc.read();
            corpus.append(raw)
            #tokenCount.append(len(word_tokenize(raw)));
            
            
    print("Average Token Count: ",numpy.mean(tokenCount));

    return corpus



def run_script():
    '''
        Will run script.
    '''

    # some preprocessing
    filenames = data_directory_read(DATA_DIRECTORY)
    
    global author_target_map
    author_target_map = map_author_value(filenames) # maps authors to integer values

    shuffle_data(filenames)

#    train_data, test_data = split_data(filenames, .75)

    # Bag of words creation
    global count_vect
    count_vect = CountVectorizer()
    
    
    global TRAIN_TARGET, TRAIN_SPARSE, TEST_TARGET, TEST_SPARSE
    FILENAMES = count_vect.fit_transform(cat_data(filenames))
    TRAIN_TARGET = get_target_values(filenames[:40], author_target_map)
    TRAIN_SPARSE = FILENAMES[:40]
    TEST_TARGET = get_target_values(filenames[40:], author_target_map)
    TEST_SPARSE = FILENAMES[40:]

#
#    #TODO: N-grams representation

def get_filenames():
    return data_directory_read(DATA_DIRECTORY)



run_script()
