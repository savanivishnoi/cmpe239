import nltk
import string
import os
import numpy as np
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

token_dict = {}
stemmer = PorterStemmer()
usr_input ="machine learning" #input("Enter search term")
print(usr_input)
lowers = usr_input.lower()
no_punctuation = lowers.translate(str.maketrans({key: None for key in string.punctuation}))#None, string.punctuation) #no punctuation & in lower case
text_feature_dictionary = {}

file_path = "feature_names.txt"
rpFile = open(file_path, 'r')
i = 0;
for line in rpFile:
	text_feature_dictionary[i] = line
query_features_arr = [] #declare array for storing index of words
for word in no_punctuation:
	if word in text_feature_dictionary: 
		query_features_arr.append(text_feature_dictionary.index(word)) #save indexes by putting a loop on punctuations and getting word index from dictionary

token_dict[0] = no_punctuation

tfidf = pickle.load(open("rp_vectorizer.pickle", "rb"))
print (tfidf.idf_)
vector = tfidf.fit_transform(token_dict.values()) #tf idf for query
print (vector) 
	
 


