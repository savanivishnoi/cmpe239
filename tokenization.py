import nltk
import string

from nltk.stem.porter import *
from nltk.corpus import stopwords
from collections import Counter

def get_tokens():
   with open('A00-1005.txt', 'r') as doc1:
    text = doc1.read()
    lowers = text.lower()
    #remove the punctuation using the character deletion step of translate
    no_punctuation = lowers.translate(None, string.punctuation)
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


tokens = get_tokens()
filtered = [w for w in tokens if not w in stopwords.words('english')]

stemmer = PorterStemmer()
stemmed = stem_tokens(filtered, stemmer)
count = Counter(stemmed)
print "stemmed---- "
print count.most_common(100)

count = Counter(filtered)
print "filtered"
print count.most_common(100)
