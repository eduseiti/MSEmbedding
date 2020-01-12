import pickle
import numpy as numpy
import torch


import os

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

import hdbscan

EMBEDDINGS_FILE = "spectra_embeddings_018222.pkl"
EMBEDDINGS_FOLDER = "data/linfeng"

CLUSTERS_NUMBER = 5000

KMEANS_RESULTS_FILE = "kmeans_clustering.pkl"
AGGLOMERATIVE_RESULTS_FILE = "agglomerative_clustering.pkl"
HDBSCAN_RESULTS_FILE = "hdbscan_clustering.pkl"



all_embeddings = []


with open(os.path.join(EMBEDDINGS_FOLDER, EMBEDDINGS_FILE), 'rb') as inputFile:
    all_embeddings = pickle.load(inputFile)

print("len all_embeddings: {}".format(len(all_embeddings)))

all_embeddings = torch.nn.functional.normalize(torch.cat(all_embeddings)).numpy()

print("all_embeddings.shape: {}".format(all_embeddings.shape))

# kmeans_result = KMeans(n_clusters=CLUSTERS_NUMBER, random_state=0).fit(all_embeddings)

# print("cluster centers: {}".format(kmeans_result.cluster_centers_))
# print("clustered labels: {}".format(kmeans_result.labels_))

# with open(os.path.join(EMBEDDINGS_FOLDER, KMEANS_RESULTS_FILE), "wb") as outputFile:
#     pickle.dump(kmeans_result, outputFile, pickle.HIGHEST_PROTOCOL)

# agglomerative_result = AgglomerativeClustering(n_clusters=None, affinity="cosine", linkage="complete", distance_threshold=1e-10).fit(all_embeddings)

# print("clustered labels: {}".format(agglomerative_result.labels_))

# with open(os.path.join(EMBEDDINGS_FOLDER, AGGLOMERATIVE_RESULTS_FILE), "wb") as outputFile:
#     pickle.dump(agglomerative_result, outputFile, pickle.HIGHEST_PROTOCOL)



hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
hdbscan_clusterer.fit(all_embeddings)

print("clustered labels: {}".format(hdbscan_clusterer.labels_))

with open(os.path.join(EMBEDDINGS_FOLDER, HDBSCAN_RESULTS_FILE), "wb") as outputFile:
    pickle.dump(hdbscan_clusterer, outputFile, pickle.HIGHEST_PROTOCOL)
