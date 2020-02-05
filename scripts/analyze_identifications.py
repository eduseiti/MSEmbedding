#!/usr/bin/env python3
# coding: utf-8

#
# analyze_identifications.py <identifications folder> <clusters folder> <cluster basename>
#

import pandas as pd
import os
import sys

IDENTIFICATION_FOLDER = sys.argv[1]
# "/media/eduseiti/bigdata01/unicamp/doutorado/clustering/linfeng/sample/identifications_sample_embeddings.clusters_p100.000000"

CLUSTERS_FOLDER = sys.argv[2]
# "/media/eduseiti/bigdata02/unicamp/doutorado/clustering/linfeng/sample/clusters_sample_0.1_embeddings"

BASENAME = sys.argv[3]
# "sample_embeddings.clusters_p100.000000"

CLUSTERS_FILE_EXTENSION = ".tsv"
IDENTIFICATIONS_FILE = "percolator.target.peptides.txt"
CLUSTERS_IDENTIFICATIONS_FILE_EXTENSION = "_identifications.tsv"


clusters = pd.read_csv(os.path.join(CLUSTERS_FOLDER, BASENAME + CLUSTERS_FILE_EXTENSION), 
                       names=["file", "scan", "cluster"], sep="\t")

total_spectra = clusters.shape[0]
total_clusters = clusters['cluster'].drop_duplicates().shape[0]

all_consensus_identifications = pd.read_csv(os.path.join(IDENTIFICATION_FOLDER, IDENTIFICATIONS_FILE), sep="\t")

high_quality_consensus_identifications = all_consensus_identifications[all_consensus_identifications['percolator q-value'] < 0.01].sort_values('scan')
num_identified_consensus = high_quality_consensus_identifications.shape[0]


clusters_size = clusters.groupby('cluster')['scan'].count().sort_values()


clusters_size_df = pd.DataFrame()
clusters_size_df['cluster'] = clusters_size.index
clusters_size_df['size'] = clusters_size.values

clusters_size_df = clusters_size_df.sort_values('cluster')


#
# Match the clusters with the high quality consensus spectra identifications
#

cluster_index = 0

cluster_identifications = []

for index, row in high_quality_consensus_identifications.sort_values('scan').iterrows():
    if clusters_size_df.iloc[cluster_index]['cluster'] <= row['scan']:
        while clusters_size_df.iloc[cluster_index]['cluster'] < row['scan']:
            cluster_identifications.append('unrecognized')
            cluster_index += 1

        if clusters_size_df.iloc[cluster_index]['cluster'] == row['scan']:
            cluster_identifications.append(row['sequence'])
        else:
            cluster_identifications.append('unrecognized')
            
        cluster_index += 1
            

# Complete the missing rows for the final unrecognized clusters' consensus spectra

for i in range(clusters_size_df.shape[0] - cluster_index):
    cluster_identifications.append('unrecognized')


clusters_size_df['sequence'] = cluster_identifications
clusters_size_df[clusters_size_df['sequence'] != 'unrecognized'].to_csv(os.path.join(CLUSTERS_FOLDER, BASENAME + CLUSTERS_IDENTIFICATIONS_FILE_EXTENSION), index=False, sep='\t')

print("{}, {}, {}, {}".format(BASENAME, total_clusters, total_spectra, num_identified_consensus))




