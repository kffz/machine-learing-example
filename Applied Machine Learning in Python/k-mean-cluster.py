# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 15:15:11 2018

@author: 123
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sb
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
#from fig_code import plot_kmeans_interactive
from sklearn.datasets import load_digits
from scipy.stats import mode
from sklearn.decomposition import PCA


def K_mean_cluster(digits):
    est = KMeans(n_clusters=10)
    clusters =est.fit_predict(digits.data)
    est.cluster_centers_.shape
    labels = np.zeros_like(clusters)
    for i in range(10):
        mask = (clusters == i)
        labels[mask] = mode(digits.target[mask])[0]
        print(labels)
    X = PCA(2).fit_transform(digits.data)
    kwags=dict(cmap=plt.cm.get_cmap('rainbow',10),edgecolor='none',alpha=0.6)
    fig,ax = plt.subplots(1,2,figsize=(8,4))
    ax[0].scatter(X[:,0],X[:,1],c=labels,**kwags)
    ax[0].set_title('learned cluster labels')
    ax[1].scatter(X[:,0],X[:,1],c=digits.target,**kwags)
    ax[1].set_title('true labels')

digits = load_digits()
#print(digits)
K_mean_cluster(digits)

