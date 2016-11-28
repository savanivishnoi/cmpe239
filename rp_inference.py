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

topk_k = 10

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
usr_input ="natural language" #input("Enter search term")
print(usr_input)
lowers = usr_input.lower()
no_punctuation = lowers.translate(str.maketrans({key: None for key in string.punctuation}))#None, string.punctuation) #no punctuation & in lower case
#text_feature_dictionary = {}

"""file_path = "feature_names.txt"
rpFile = open(file_path, 'r')
i = 0;
for line in rpFile:
	text_feature_dictionary[i] = line
query_features_arr = [] #declare array for storing index of words
for word in no_punctuation:
	if word in text_feature_dictionary: 
		query_features_arr.append(text_feature_dictionary.index(word)) #save indexes by putting a loop on punctuations and getting word index from dictionary
"""
token_dict[0] = no_punctuation
#  ************* Code for TEXT in research paper ********  
#tfidf = pickle.load(open("rp_vectorizer.pickle", "rb"))
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', analyzer = 'word', vocabulary = pickle.load(open("rp_vocabulary.pickle", "rb")))
query = tfidf.fit_transform(token_dict.values()) #tf idf for query
print(query)
VT = pickle.load(open("rp_VT.pickle","rb"))
XT = pickle.load(open("rp_XT.pickle","rb"))
A = pickle.load(open("rp_A.pickle","rb"))
print(type(VT))
print(type(XT))
Xq = np.matmul(query.todense(), np.transpose(VT))
cosine_sim = np.array(np.matmul(Xq,XT))[0]
#sorted_sim = np.array(cosine_sim)[0].argsort()[::-1][:topk_k]
cosine_sim_nosvd = np.matmul(query.todense(),np.transpose(A.todense()))
print("nosvd cosine")
print(cosine_sim_nosvd.shape)
print(cosine_sim_nosvd)
print("Xq")
print(Xq)
print("cosine sim")
print(cosine_sim)

print("cosine sim shape")
print(cosine_sim.shape)
print("cosine sim sort")
#print(sorted_sim.shape)
#print("sorted sim")
#print(sorted_sim)
#documents = open("filenames", 'r').read().splitlines()
#top_documents = [documents[x] for x in sorted_sim]
#print(top_documents)

# ************** Code for ABSTRACT in research paper *********
tfidf_abstracts = TfidfVectorizer(tokenizer=tokenize, stop_words='english', analyzer = 'word', vocabulary = pickle.load(open("rp_vocabulary_abstracts.pickle", "rb")))
query_abstracts = tfidf_abstracts.fit_transform(token_dict.values()) #tf idf for query abstracts
print(query_abstracts)
VT_abstracts = pickle.load(open("rp_VT_abstracts.pickle","rb"))
XT_abstracts = pickle.load(open("rp_XT_abstracts.pickle","rb"))
A_abstracts = pickle.load(open("rp_A_abstracts.pickle","rb"))
print(type(VT_abstracts))
print(type(XT_abstracts))
Xq_abstracts = np.matmul(query_abstracts.todense(), np.transpose(VT_abstracts))
cosine_sim_abstracts = np.array(np.matmul(Xq_abstracts,XT_abstracts))[0]
#sorted_sim_abstracts = np.array(cosine_sim_abstracts)[0].argsort()[::-1][:topk_k]
#cosine_sim_nosvd = np.matmul(query.todense(),np.transpose(A.todense()))
#print("nosvd cosine")
#print(cosine_sim_nosvd.shape)
#print(cosine_sim_nosvd)
print("Xq_abs")
print(Xq_abstracts)
print("cosine sim abs")

print("cosine sim shape abs")
print(cosine_sim_abstracts.shape)
print("cosine sim sort abs")
#print(sorted_sim_abstracts.shape)
#print("sorted sim")
#print(sorted_sim_abstracts)
#documents_abstracts = open("filenames_abstracts", 'r').read().splitlines()
#top_documents_abstracts = [documents_abstracts[x] for x in sorted_sim_abstracts]
#print(top_documents_abstracts)

# *********** Code for Combining results ******
"""
dictionary for abstracts results and text results
""" 
print(cosine_sim)
print(cosine_sim_abstracts)
combined_sim = (0.50 * cosine_sim) + (1.00 * cosine_sim_abstracts) 
print(combined_sim)
sorted_sim = combined_sim.argsort()[::-1][:topk_k]
documents = open("filenames", 'r').read().splitlines()
top_documents = [documents[x] for x in sorted_sim]
print(top_documents)
