"""
=======================================
Clustering text documents using k-means
=======================================

This is an example showing how the scikit-learn can be used to cluster
documents by topics using a bag-of-words approach. This example uses
a scipy.sparse matrix to store the features instead of standard numpy arrays.

Two feature extraction methods can be used in this example:

  - TfidfVectorizer uses a in-memory vocabulary (a python dict) to map the most
    frequent words to features indices and hence compute a word occurrence
    frequency (sparse) matrix. The word frequencies are then reweighted using
    the Inverse Document Frequency (IDF) vector collected feature-wise over
    the corpus.

  - HashingVectorizer hashes word occurrences to a fixed dimensional space,
    possibly with collisions. The word count vectors are then normalized to
    each have l2-norm equal to one (projected to the euclidean unit-ball) which
    seems to be important for k-means to work in high dimensional space.

    HashingVectorizer does not provide IDF weighting as this is a stateless
    model (the fit method does nothing). When IDF weighting is needed it can
    be added by pipelining its output to a TfidfTransformer instance.

Two algorithms are demoed: ordinary k-means and its more scalable cousin
minibatch k-means.

Additionally, latent semantic analysis can also be used to reduce dimensionality
and discover latent patterns in the data.

It can be noted that k-means (and minibatch k-means) are very sensitive to
feature scaling and that in this case the IDF weighting helps improve the
quality of the clustering by quite a lot as measured against the "ground truth"
provided by the class label assignments of the 20 newsgroups dataset.

This improvement is not visible in the Silhouette Coefficient which is small
for both as this measure seem to suffer from the phenomenon called
"Concentration of Measure" or "Curse of Dimensionality" for high dimensional
datasets such as text data. Other measures such as V-measure and Adjusted Rand
Index are information theoretic based evaluation scores: as they are only based
on cluster assignments rather than distances, hence not affected by the curse
of dimensionality.

Note: as k-means is optimizing a non-convex objective function, it will likely
end up in a local optimum. Several runs with independent random init might be
necessary to get a good convergence.

"""

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck
# License: BSD 3 clause

from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import DistanceMetric
from sklearn.metrics.pairwise import cosine_similarity

import gensim.models as g
import codecs
import nltk

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")

#print(__doc__)
#op.print_help()


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)


# #############################################################################
# Load some categories from the training set
# Uncomment the following to do the analysis on all the categories
# categories = None

print("Loading 20 newsgroups dataset for categories:")

dataset = fetch_20newsgroups(subset='all', 
                             shuffle=True, random_state=42)

print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))

s = np.arange(len(dataset.target))
print("random", s)
np.random.shuffle(s)


print(s)

data_shuffle = []
test_shuffle = []

for i in range(0, 40):
  data_shuffle.append(dataset.data[s[i]])
  test_shuffle.append(dataset.target[s[i]])

#data_shuffle = dataset.data[s]
#test_shuffle = dataset.target_names[s]

#data_shuffle = data_shuffle[0:10]

labels = test_shuffle
print(len(labels))
true_k = np.unique(labels).shape[0]

print(true_k)

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
print("hoang cuong")
vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features, min_df=2, stop_words='english', use_idf=opts.use_idf)
X = vectorizer.fit_transform(data_shuffle)
X_to_arr = X.toarray()

def getVector(data_shuffle):
  model="enwiki_dbow/doc2vec.bin"
  #inference hyper-parameters
  start_alpha=0.01
  infer_epoch=1000
  #load model
  m = g.Doc2Vec.load(model)

  array_without_tokenizer = []
  array_with_tokenizer = []
  for i in range(0, len(data_shuffle)):
    array_without_tokenizer.append(embeding_with_doc2vec(data_shuffle[i], m))
    array_with_tokenizer.append(embeding_with_doc2vec_with_tokenizer(data_shuffle[i], m))
  return array_without_tokenizer, array_with_tokenizer


def embeding_with_doc2vec_with_tokenizer(text, m):
  #parameters
  
  #inference hyper-parameters
  start_alpha=0.01
  infer_epoch=1000
  #load model

  tokens = nltk.word_tokenize(text)

  #print(tokens)

  text = ''.join(tokens)
  #print(type(tokens))
  
  vector = m.infer_vector(text, alpha=start_alpha, steps=infer_epoch)
  return vector


def embeding_with_doc2vec(text, m):
  #parameters
  
  #inference hyper-parameters
  start_alpha=0.01
  infer_epoch=1000
  #load model

  #tokens = nltk.word_tokenize(text)
  
  vector = m.infer_vector(text, alpha=start_alpha, steps=infer_epoch)
  return vector


def cluster_and_evaluate_myown(X_to_arr, labels):
  distance_matrix = np.zeros((len(X_to_arr),len(X_to_arr)))
  for i in range(0, len(X_to_arr)):
    for j in range(0, len(X_to_arr)):
      distance_matrix[i][j] = np.linalg.norm(X_to_arr[i]-X_to_arr[j])
  Spectral_Cluster = SpectralClustering(true_k).fit_predict(distance_matrix)
  print("Spectral_Clustering")

  print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, Spectral_Cluster))
  print("Completeness: %0.3f" % metrics.completeness_score(labels, Spectral_Cluster))
  print("V-measure: %0.3f" % metrics.v_measure_score(labels, Spectral_Cluster))
  print("Adjusted Rand-Index: %.3f"
        % metrics.adjusted_rand_score(labels, Spectral_Cluster))
  print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(X, Spectral_Cluster, sample_size=1000))



def cluster_and_evaluate(X_to_arr, labels, metric):
  
  print(metric)
  if metric is not "cosine_similarity":
    dist = DistanceMetric.get_metric(metric)
    #print(X_to_arr)
    distance_matrix = dist.pairwise(X_to_arr)
  else:
    distance_matrix = np.zeros((len(X_to_arr), len(X_to_arr)))
    for i in range(0, len(X_to_arr)):
      for j in range(0, len(X_to_arr)):
        distance_matrix[i][j] = cosine_similarity([X_to_arr[i]], [X_to_arr[j]])

  Spectral_Cluster = SpectralClustering(true_k).fit_predict(distance_matrix)
  
  print("Spectral_Clustering with metric: ", metric)


  print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, Spectral_Cluster))
  print("Completeness: %0.3f" % metrics.completeness_score(labels, Spectral_Cluster))
  print("V-measure: %0.3f" % metrics.v_measure_score(labels, Spectral_Cluster))
  print("Adjusted Rand-Index: %.3f"
        % metrics.adjusted_rand_score(labels, Spectral_Cluster))
  print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(X, Spectral_Cluster, sample_size=1000))




data_shuffle_without_tokenizer, data_shuffle_with_tokenizer = getVector(data_shuffle)

#cluster_and_evaluate(data_shuffle_without_tokenizer, labels, "euclidean")
#cluster_and_evaluate(data_shuffle_without_tokenizer, labels, "manhattan")
#cluster_and_evaluate(data_shuffle_without_tokenizer, labels, "chebyshev")
#cluster_and_evaluate(data_shuffle_without_tokenizer, labels, "cosine_similarity")



cluster_and_evaluate(data_shuffle_with_tokenizer, labels, "euclidean")
cluster_and_evaluate(data_shuffle_with_tokenizer, labels, "manhattan")
cluster_and_evaluate(data_shuffle_with_tokenizer, labels, "chebyshev")
cluster_and_evaluate(data_shuffle_with_tokenizer, labels, "cosine_similarity")



#cluster_and_evaluate_myown(X_to_arr, labels)
cluster_and_evaluate(X_to_arr, labels, "euclidean")
cluster_and_evaluate(X_to_arr, labels, "manhattan")
cluster_and_evaluate(X_to_arr, labels, "chebyshev")
cluster_and_evaluate(X_to_arr, labels, "cosine_similarity")





