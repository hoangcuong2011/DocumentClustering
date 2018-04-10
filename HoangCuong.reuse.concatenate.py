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
op.add_option("--n-features", type=int, default=300,
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

data_shuffle = []
test_shuffle = []
Training_DATA = 160
for i in range(0, Training_DATA):
  fh = open("train"+str(i),"r")
  text = fh.read()
  data_shuffle.append(text)

for i in range(0, Training_DATA):
  fh = open("test"+str(i),"r")
  text = fh.read()
  test_shuffle.append(text)


fh = open("data_saving.txt","r")
data_shuffle_with_tokenizer = np.zeros((Training_DATA, 300))

for i in range(0, Training_DATA):
  for j in range(0, 300):
    data_shuffle_with_tokenizer[i][j] = fh.readline()
fh.close()

#print(data_shuffle_with_tokenizer)


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
        #print(distance_matrix[i][j])



  #print(distance_matrix[0][0])
  #print(distance_matrix[0][1])
  Spectral_Cluster = SpectralClustering(true_k, random_state=42).fit_predict(distance_matrix)

  #print(Spectral_Cluster)

  
  print("Spectral_Clustering with metric: ", metric)


  print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, Spectral_Cluster))
  print("Completeness: %0.3f" % metrics.completeness_score(labels, Spectral_Cluster))
  print("V-measure: %0.3f" % metrics.v_measure_score(labels, Spectral_Cluster))
  print("Adjusted Rand-Index: %.3f"
        % metrics.adjusted_rand_score(labels, Spectral_Cluster))
  print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(X, Spectral_Cluster, sample_size=1000))



def standardize_data(X_train):
    X_mean = np.mean(X_train, axis=0)
    #print(X_mean)
    X_std = np.std(X_train, axis=0)+0.0000001 #a small noise
    #print(X_std)
    X_train -= X_mean

    return X_train

def cluster_and_evaluate_concatenate(X1_to_arr, X2_to_arr, labels, metric):
  print(type(X1_to_arr))
  print(type(X2_to_arr))
  X_to_arr = np.concatenate((X1_to_arr, X2_to_arr), axis=1)

  #X_to_arr = np.concatenate((standardize_data(X1_to_arr), standardize_data(X2_to_arr)), axis=1)

  #X_to_arr = standardize_data(X_to_arr)

  #print(X1_to_arr)
  #print(X2_to_arr)
  #print(X_to_arr)
  
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
        #print(distance_matrix[i][j])



  #print(distance_matrix[0][0])
  #print(distance_matrix[0][1])
  Spectral_Cluster = SpectralClustering(true_k, random_state=42).fit_predict(distance_matrix)

  #print(Spectral_Cluster)

  
  print("Spectral_Clustering with metric: ", metric)


  print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, Spectral_Cluster))
  print("Completeness: %0.3f" % metrics.completeness_score(labels, Spectral_Cluster))
  print("V-measure: %0.3f" % metrics.v_measure_score(labels, Spectral_Cluster))
  print("Adjusted Rand-Index: %.3f"
        % metrics.adjusted_rand_score(labels, Spectral_Cluster))
  print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(X, Spectral_Cluster, sample_size=1000))





#cluster_and_evaluate(data_shuffle_without_tokenizer, labels, "euclidean")
#cluster_and_evaluate(data_shuffle_without_tokenizer, labels, "manhattan")
#cluster_and_evaluate(data_shuffle_without_tokenizer, labels, "chebyshev")
#cluster_and_evaluate(data_shuffle_without_tokenizer, labels, "cosine_similarity")



#cluster_and_evaluate(data_shuffle_with_tokenizer, labels, "euclidean")
#cluster_and_evaluate(data_shuffle_with_tokenizer, labels, "manhattan")
#cluster_and_evaluate(data_shuffle_with_tokenizer, labels, "chebyshev")
#cluster_and_evaluate(data_shuffle_with_tokenizer, labels, "cosine_similarity")



#cluster_and_evaluate_myown(X_to_arr, labels)
#cluster_and_evaluate(X_to_arr, labels, "euclidean")
#cluster_and_evaluate(X_to_arr, labels, "manhattan")
#cluster_and_evaluate(X_to_arr, labels, "chebyshev")
#cluster_and_evaluate(X_to_arr, labels, "cosine_similarity")


cluster_and_evaluate_concatenate(X_to_arr, data_shuffle_with_tokenizer, labels, "euclidean")

#cluster_and_evaluate_concatenate(X_to_arr, data_shuffle_with_tokenizer, labels, "cosine_similarity")




