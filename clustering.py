import pickle
import numpy as np
import torch


import os

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

import hdbscan

import sys

#
# Parameters
#
# 1 = embeddings file name
# 2 = embeddings dimensions
# 3 = clustering technique: "kmeans", "aglomerative", "hdbscan"
#


EMBEDDINGS_FILE = "sample_embeddings_q0.001_transformer_64_pvalue_0.6_double_n_pair_091215.bin"
EMBEDDINGS_FOLDER = "data/linfeng"

CLUSTERS_NUMBER = 50000

KMEANS_RESULTS_FILE = "kmeans_clustering.pkl"
AGGLOMERATIVE_RESULTS_FILE = "agglomerative_clustering.pkl"
HDBSCAN_RESULTS_FILE = "hdbscan_clustering.pkl"


def read_embeddings(filename, embedding_dimensions):

    embeddings = []

    with open(filename, "rb") as inputFile:
        while True:
            embedding = inputFile.read(embedding_dimensions * 4)

            if embedding:
                embeddings.append(torch.from_numpy(np.frombuffer(embedding, dtype=np.float32)))
            else:
                break

    print("Number of embeddings read={}".format(len(embeddings)))

    return torch.stack(embeddings, 0)



all_embeddings = read_embeddings(sys.argv[1], int(sys.argv[2]))

print("len all_embeddings: {}".format(len(all_embeddings)))

all_embeddings = torch.nn.functional.normalize(all_embeddings).numpy()

print("all_embeddings.shape: {}".format(all_embeddings.shape))

if sys.argv[3].lower() == "kmeans":
    kmeans_result = KMeans(n_clusters=CLUSTERS_NUMBER, random_state=0).fit(all_embeddings)

    print("cluster centers: {}".format(kmeans_result.cluster_centers_))
    print("clustered labels: {}".format(kmeans_result.labels_))

    with open(KMEANS_RESULTS_FILE, "wb") as outputFile:
        pickle.dump(kmeans_result, outputFile, pickle.HIGHEST_PROTOCOL)

elif sys.argv[3].lower() == "aglomerative":
    agglomerative_result = AgglomerativeClustering(n_clusters=None, affinity="cosine", linkage="complete", distance_threshold=1e-10).fit(all_embeddings)

    print("clustered labels: {}".format(agglomerative_result.labels_))

    with open(AGGLOMERATIVE_RESULTS_FILE, "wb") as outputFile:
        pickle.dump(agglomerative_result, outputFile, pickle.HIGHEST_PROTOCOL)

elif sys.argv[3].lower() == "hdbscan":
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    hdbscan_clusterer.fit(all_embeddings)

    print("clustered labels: {}".format(hdbscan_clusterer.labels_))

    with open(HDBSCAN_RESULTS_FILE, "wb") as outputFile:
        pickle.dump(hdbscan_clusterer, outputFile, pickle.HIGHEST_PROTOCOL)


