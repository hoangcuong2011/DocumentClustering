import numpy as np
from sklearn.cluster import SpectralClustering
mat = np.matrix([[1.,.1,.6,.4],[.1,1.,.1,.2],[.6,.1,1.,.7],[.4,.2,.7,1.]])
array = SpectralClustering(3).fit_predict(mat)
print(array)
#>>> array([0, 1, 0, 0], dtype=int32)
