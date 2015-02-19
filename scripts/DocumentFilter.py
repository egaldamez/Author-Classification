# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 19:45:14 2015

@author: Eddie
"""

# This script is intended to filter out as much noise from our data.
'''
    What is considered noise? 
        - Anything that wasn't written directly by the author
        - Prefaces, Gutenberg copyright information, Information about the book
        .
        .
        .
    This script will check if there is already a filtereddata directory located in the project folder
    If there is and it contains documents, then no filtered will be made since this is an expensive process.
    
    Filtered documents will be marked to ensure the don't get refiltered. This will be usefull when new data is 
    added to the rawdata file.
    
    TODO: create more robust Filtering system using Regular expressions (not part of the project, just thought it would be cool to do)
    
    
'''




FILTERWORDS = ['chapter', 'volume', 'part', 'epilogue']
FILTER_TOKEN = 'F'

    

def data_dir(directory: str): 
 
    files = os.listdir(directory)
    for file in files:
        with open(directory + '\\' + file,'r') as book:
            yield [file, book.read()]
            
def dest_dir(directory: str, filename, data):
    '''
        Will write filtered data to a new file.
        'data' is a list of all the filtered data. It will be joined.
        'filename' is the name of the file. It will be added to directory.
    '''
    try:
        open(directory + '\\' + filename, 'x')
    except:
        pass
        
    
    with open(directory + '\\' + filename, 'w') as data_file:
        data_file.write(''.join(data))
    


class Filter:
    
    def __init__(self, book):
        self.raw_data = book.splitlines(keepends = True) # split on newlines
        self.data_length = len(self.raw_data)
        self.filtered_data = []
        self.filter = False if book[-1] == FILTER_TOKEN else True
    
    
    def _find_string(self, string, rev=False):
        container_index = range(self.data_length) if not rev else range(-1, -self.data_length,-1)

        for index in container_index:
            if string in self.raw_data[index]:
                return index + 5 if not rev else index - 5
            
        
    def _clear_guten_info(self):
        cuttoff_begin = self._find_string('*** START ')
        cuttoff_end = self._find_string('*** END ',rev=True)
        
        self.raw_data = self.raw_data[cuttoff_begin:cuttoff_end]
        
    def _clear_filter_words(self):
        # Simple, if a string begins with any of the FILTERWORDS, then pop that value from the list
        max_filterword_len = len(max(FILTERWORDS, key=len))
        for sentence in self.raw_data:            
            if sentence != '\n' and len(sentence) > max_filterword_len or \
                sentence.split(' ')[0].lower() not in FILTERWORDS:
                self.filtered_data.append(sentence)
        self.filtered_data.append(FILTER_TOKEN)
    
    def get_filtered_data(self) -> str:
        return self.filtered_data
        
    def data_filter(self):
        self._clear_guten_info()
        self._clear_filter_words()


#if __name__ == '__main__':
#    
#    import os
#    
#    # TODO: write function to get correct directory (below is currently hard-coded)
#    data_directory = "C:\\Users\\Edwin\\Documents\\CS175\\Project\\rawdata"
#    filtered_directory = "C:\\Users\\Edwin\\Documents\\CS175\\Project\\filtereddata"
#    data_set = data_dir(data_directory)   
#    
#    
#    for title, book in data_set:        
#        data = Filter(book)
#        data.data_filter()
#        dest_dir(filtered_directory, title, data.get_filtered_data())

