# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 20:25:47 2018

@author: Invisible-Tilkey
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# ======================================================
# 1st part
# Generating Datasets

# X1 : The generated samples. = numerical
# y1 : The integer labels (0 or 1) for class membership of each sample. = catagorical
X1, y1 = datasets.make_circles(n_samples=5000, factor=.5, noise=.05) 
X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2,1.2]], cluster_std=[[.1]], random_state=9)
X = np.concatenate((X1, X2))

plt.scatter(X[:, 0], X[:, 1], marker='*', color='chartreuse')
plt.show()
print ("=====Original DataSet======")

# ======================================================
# 2nd part
# K-Means
from sklearn.cluster import KMeans
# Define K Means with 3 Clusters and 9 Centroid
y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
print ("=====K-Means Cluster======")
# 可见对于非凸数据集非常垃圾

# ======================================================
# 3rd part
# Density-Based Spatial Cluster of Application with Noise
from sklearn.cluster import DBSCAN
y_pred = DBSCAN().fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
print ("=====DBSCAN raw======")

# ======================================================
# 4th part
# DBSCAN with eps = 0.1
y_pred = DBSCAN(eps = 0.1).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
print ("=====DBSCAN 'eps=0.1'======")

# ======================================================
# 5th part
# DBSCAN with eps = 0.1; min_samples = 10
y_pred = DBSCAN(eps = 0.1, min_samples = 10).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
print ("=====DBSCAN 'eps=0.1' 'min_samples=10'======")
