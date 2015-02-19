# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 10:29:04 2015

@author: Eddie
"""

# This script is designed to process and tokenize raw data (string) using the nltk library


import os
import nltk
from random import shuffle


DATA_DIRECTORY  = 'C:\\Users\\Edwin\\Documents\\CS175\\Project\\filtereddata'

# Need some data structure to represent the books 
# First attempt. Store as lists of dictionaries with author as key (target value) and 
# it's value a list of text objects (value) representing the books written by author

TRAIN_DATA = {} # Training data will be contained in a dictionary of author, books values
TEST_DATA = {} # Test data  ' ' '



def extract_data(file):
    '''
        will read data into nltk.text.
    '''
    with open(DATA_DIRECTORY + '\\' + file, 'r') as data:
        raw_text = data.read()
        tokens = nltk.word_tokenize(raw_text)
        text = nltk.Text(tokens)
    return text

def fill_data_structure(struct, data_set):
    '''
        struct: Either TRAIN_DATA or TEST_DATA
        data_set: list of data files to populat struct with
    '''
    for file in data_set:
        author = file.split('-')[0]
        text = extract_data(file)
        if author not in struct:
            struct[author] = [text]
        else:
            struct[author].append(text)

def data_directory_read(data_directory: str) -> [str]:
    '''
        Generate a list of data filenames from directory at every call.
    '''

    return os.listdir(data_directory)
    
def shuffle_data(directory_list: list):
    '''
        shuffle list of filenames.
    '''
    return shuffle(directory_list)  
    
    
def split_data(data_files, split_fraction ):
    '''
        Will split the data according to split_fraction
    '''
    
    split_var = round(split_fraction*len(data_files))
    return (data_files[:split_var], data_files[split_var:])
    
def populate(input_data):
    '''
        input_data: A two-tuple that cotains training data and test data
        
        Populate TRAIN_DATA and TEST_DATA
    '''
    train_data, test_data = input_data
    
    fill_data_structure(TRAIN_DATA, train_data)    
    fill_data_structure(TEST_DATA, test_data)
    
    
    
    
class DataInit:

    @staticmethod 
    def __init__():
        print("Initializing")
        data = data_directory_read(DATA_DIRECTORY)
        shuffle_data(data)
        data = split_data(data, .75) # split data 75 percent training 25 percent test
        populate(data)
    