# Author-Classification
Classify documents by author. 


Project Goal
The goal of this project will be to develop classifiers using two supervised learning methods to attempt to identify the 
author of the document using Stylometry methods. Stylometry attempts to identify authors based on non contextual word 
combinations.

Project Overview (Description and background)
The problem that this project seeks to address is the classification of documents by author. Our program should be able to
read in a document and based on past training data, should be able to classify the document to an already witnessed author. 
To accomplish this, we will need to develop style based fingerprints for authors. This process is known as Stylometry and is 
an example of single-label classification of documents, for we are assigning one label, an author, to a document. This problem 
of document author classification has been explored as described in Document Author Classification using Generalized 
Discriminant Analysis by Todd K. Mood, Peg Howland, and Jacob H. Gunther. Our intent is to approach the same classification 
problem but with simpler classification models such as the Naive-Bayes and k-Nearest Neighbor classifiers. 

Data Sets 
For our data set we chose to use a subset of the books provided by The Gutenberg Project. The Gutenberg Project houses 
over 46,000 free books and are labeled by author and category such as: Animal, Fine Arts, History, etc… There are 
(approximately) 11 different classes for our documents, meaning that there are 11 different authors from which we can label 
a document. 

Proposed Technical Approach
As mentioned above, for this problem we will consider using the Naive-Bayes and k-Nearest Neighbor models. Because we have 
multiple authors from which to label a single document, we will generalize our models to work with multiple classes. For 
example, we will be using the multinomial version of the Naive-Bayes classifier and a  k-Nearest Neighbor classifier. We 
will also use different representations for our data. The first of which will be the common “bag of words” representation 
and the second will be N-grams/phrases.
