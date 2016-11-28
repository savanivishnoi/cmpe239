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
"""
A= 21K X 1 mil tfidf
VT = SVD transform (1 mil X 100)
X = A * VT (21K X 100)

query: 1 X 1mil
Xq = quer * VT (1 X 100)

cosine similarity = Xq * X'

"""
path = 'test_papers'
token_dict = {}
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

for subdir, dirs, files in os.walk(path):
    files.sort()
    with open("filenames", 'w') as f:
    	f.write("\n".join(files)) 
    for file in files:
        file_path = subdir + os.path.sep + file
        rpFile = open(file_path, 'r')
        text = rpFile.read()
        lowers = text.lower()
        no_punctuation = lowers.translate(str.maketrans({key: None for key in string.punctuation}))
        token_dict[file] = no_punctuation
        
#this can take some time
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', analyzer = 'word')
print (tfidf)

A = tfidf.fit_transform(token_dict.values())  #vector after initial steps.. preprocessing.. (docs X words) position
feat_names = tfidf.get_feature_names() #vector of features i.e. words .. save it locally to be used for query 
filename = "./feature_names.txt"
np.savetxt(filename, feat_names, fmt = "%s")
print("feats shape ")
print(len(feat_names)) 
#print(feat_names) 

pickle.dump(A ,open("rp_A.pickle","wb"))
#save tfidf vectorizer locally to use it later..
#pickle.dump(tfidf ,open("rp_vectorizer.pickle","wb"))
pickle.dump(tfidf.vocabulary_ ,open("rp_vocabulary.pickle","wb"))
                       
svd = TruncatedSVD(n_components=100)  # SVD... components will be 100.. so result will be (docs X 100)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd)
print("before ")
print(A.shape)
#print(X)
#U, Sigma, VT = randomized_svd(X, n_components=100,
 #                                     
  #                                    random_state=None)
X = lsa.fit_transform(A,normalizer) # SVD transformation on A.. will give (docs X 100)
print("fit transformed ")
VT =svd.components_  #VT.. matrix: words X 100 
pickle.dump(VT ,open("rp_VT.pickle","wb"))
pickle.dump(np.transpose(X) ,open("rp_XT.pickle","wb"))
"""print(U.shape)
print(U)
print("Sigma")
print(Sigma.shape)
print(Sigma)
print("VT")
print(VT.shape)
print(VT)
"""
print(X.shape)
print("trans")
print(VT)
print("ratio : ")
print(svd.explained_variance_ratio_) 

print("ratio sum : ")
print(svd.explained_variance_ratio_.sum()) 

filename = "./outX.txt"
np.savetxt(filename, X)
#z = np.load(filename)
#print(z)

filename = "./outVT.txt"
np.savetxt(filename, VT)
"""km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1, verbose=True)
km.fit(X)

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))

print()

if not opts.use_hashing:
    print("Top terms per cluster:")
    if opts.n_components:
        original_space_centroids = svd.inverse_transform(km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i)
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind]) 
	print()
"""
filename = "./outfile11.dat"
#np.savetxt(filename, tfs.toarray(), delimiter=',', fmt='%.4f')
#z = np.load(filename)
#print(z)
