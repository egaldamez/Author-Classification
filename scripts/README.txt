DataTest.py:		This script outputs the average of some number of iterations of cross-validation on our classifiers.

DocumentFilter.py:	This script filters out the header and footer text of the books added in by Project Gutenberg.

FilteredDataSplit.py:	This script takes the filtered files from DocumentFilters.py and splits them into smaller files.

KNNClassifier.py:	This script contains a wrapper class for the k-nearest-neighbor classifier in sklearn.

KNNTest.py:		This script performs cross-validation on the KNNClassifier class.

Parser.py:		This script provides the functionality to read in a list of files and vectorize them using either CountVectorizer
			or TfidfVectorizer from sklearn.