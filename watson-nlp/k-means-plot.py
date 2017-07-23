

import json
import os
import pandas as pd
import numpy as np
import json
import glob
from flask import jsonify


from scipy.spatial.distance import cdist
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import hierarchical, KMeans, MeanShift
from sklearn.model_selection import train_test_split

from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt


def plot_PCA(model, data):
    cluster_assign = model.predict(data)
    pca = PCA(2)
    plot_columns = pca.fit_transform(data)
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_assign,)
    plt.xlabel('Canonical variable 1')
    plt.ylabel('Canonical variable 2')
    plt.title('Scatterplot of Canonical Variables for 3 Clusters')
    plt.show()
    
def plot_Trunc(model, data):
    cluster_assign = model.predict(data)
    pca_2 = TruncatedSVD(2)
    plot_columns = pca_2.fit_transform(data)
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_assign,)
    plt.xlabel('Canonical variable 1')
    plt.ylabel('Canonical variable 2')
    plt.title('Scatterplot of Article Similarity')
    plt.show()

df = pd.read_csv(r'C:\Users\austi\Documents\data.csv')

# splits into 56, 24 
clus_train, clus_test = train_test_split(df, test_size=.3, random_state=46)

clusters=range(1,20)
meandist=[]

# This plots error as we increase clusters

#for k in clusters:
 #   model=KMeans(n_clusters=k)
#   model.fit(clus_train)
 #   clusassign=model.predict(clus_train)
#  meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1))
#    / clus_train.shape[0])


# plt.plot(clusters, meandist)
# plt.xlabel('Number of clusters')
# plt.ylabel('Average distance')
# plt.title('Selecting k with the Elbow Method')

model3=KMeans(n_clusters=5)
model3.fit(clus_train)
clusassign=model3.predict(clus_train)
clusassign

plot_PCA(model3, clus_train)


# New graph with new data based on trained k-means. I'm cheating for now
# since im using training data in my test data because I want to test
# article mapping to coordinates

plot_Trunc(model3, df)


number_to_urls = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}

# Add url to each coordinate from number to urls dictionary
num_urls = len(plot_columns)
urls = np.arange(num_urls)
print(urls)
url_coord = []
for i in range(num_urls):
    #plot_columns[i].('www.example.com')
    url_coord.append(np.insert(plot_columns[i], 0, i))



url_coord



