import pickle
import numpy as numpy
import torch

import os

from sklearn.cluster import KMeans

EMBEDDINGS_FILE = "spectra_embeddings_018222.pkl"
EMBEDDINGS_FOLDER = "data/linfeng"

CLUSTERS_NUMBER = 5000

KMEANS_RESULTS_FILE = "kmeans_clustering.pkl"

all_embeddings = []


with open(os.path.join(EMBEDDINGS_FOLDER, EMBEDDINGS_FILE), 'rb') as inputFile:
    all_embeddings = pickle.load(inputFile)

print("len all_embeddings: {}".format(len(all_embeddings)))

all_embeddings = torch.cat(all_embeddings).numpy()

print("all_embeddings.shape: {}".format(all_embeddings.shape))

kmeans_result = KMeans(n_clusters=CLUSTERS_NUMBER, random_state=0).fit(all_embeddings)

print("cluster centers: {}".format(kmeans_result.cluster_centers_))
print("clustered labels: {}".format(kmeans_result.labels_))

with open(os.path.join(EMBEDDINGS_FOLDER, KMEANS_RESULTS_FILE), "wb") as outputFile:
    pickle.dump(kmeans_result, outputFile, pickle.HIGHEST_PROTOCOL)
