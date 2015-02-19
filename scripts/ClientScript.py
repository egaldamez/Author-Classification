# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 22:10:31 2015

@author: Eddie
"""

''' 
The purpose of this script is to download data from the Gutenberg Project Website
and create a structured representation of the data (ie save books into permanent storage).
This script is only intended to run once. If I decide to just keep the data in the project directories,
then this script will not need to be re-run.
'''

#Set up parameters for call to wget to download data from the Gutenberg Project website

# Additional information will go here

# Authors(labels). We've decided to inrease the number of labels for our classifier; therefore, I decided to 
# use the top 20 from the top 100 authors on the Gutenberg Website.
# Here is the link for the page where I got the Author list.

# http://www.gutenberg.org/browse/scores/top

# I also used the software wget to do the actual call to the webpage to download the data. Here's the link 
# to the download site of wget for windows (It's such a pain for windows :()

# http://gnuwin32.sourceforge.net/packages/wget.htm

# If you guys want to run this script, install wget, or how to add wget to your environment PATH let me know
# and I will help or maybe even write a script that does that automatically



# TODO: create a script that will download datat (ebooks) from the gutenberg website