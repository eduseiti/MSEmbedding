#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
import plotly
import os
import sys
import numpy as np

CLUSTER = sys.argv[1]
# "/media/eduseiti/bigdata02/unicamp/doutorado/clustering/linfeng/sample/clusters_sample_0.1_embeddings_q0.001/sample_embeddings_q0.001.clusters_p95.897800.tsv"

CLUSTER_NAME = sys.argv[2]
# "sample_embeddings_q0.001.clusters_p95.897800.tsv"

CLUSTER_CONSENSUS_IDENTIFICATIONS = sys.argv[3]
# "/media/eduseiti/bigdata02/unicamp/doutorado/clustering/linfeng/sample/clusters_sample_0.1_embeddings_q0.001/sample_embeddings_q0.001.clusters_p95.897800_identifications.tsv"

ANALYSIS_OUTPUT_FILE = sys.argv[4]
# "/media/eduseiti/bigdata02/unicamp/doutorado/clustering/linfeng/sample/clusters_sample_0.1_embeddings_q0.001/sample_embeddings_q0.001.clusters_p95.897800_identifications_analysis"

ALL_IDENTIFICATIONS = sys.argv[5]
# "/media/eduseiti/bigdata02/unicamp/doutorado/clustering/linfeng/sample/identifications_sample_0.1_nterm/sample_experiment_identifications/percolator.target.psms.txt"

CLUSTERED_FILES_LIST = sys.argv[6]
# "/media/eduseiti/bigdata02/unicamp/doutorado/clustering/linfeng/sample/sample_0.1_files.txt"

RESULT_FILE_EXTENSION = ".tsv"

all_ids = pd.read_csv(ALL_IDENTIFICATIONS, sep="\t")

#
# Observations on identifications file:
#
# - File indexes start in 0
# - Scan indexes start in 1
#



clusters = pd.read_csv(CLUSTER, names=["file", "scan", "cluster"], sep="\t")

#
# Observations on clusters file:
#
# - Clusters index starts in 1
# - Scan index starts in 0
#



consensus_ids = pd.read_csv(CLUSTER_CONSENSUS_IDENTIFICATIONS, sep="\t")

spectra_files = {}
spectra_files_count = 0

with open(CLUSTERED_FILES_LIST, "r") as inputFile:
    while True:
        filename = inputFile.readline().strip()
        
        if filename != "":
            spectra_files[filename] = spectra_files_count
            spectra_files_count += 1
        else:
            break


#
# Create a file index column in clusters dataframe
#

file_index = np.zeros(clusters.shape[0])

def fileindex(row):
    file_index[row.name] = spectra_files[row['file']]  

clusters.apply(fileindex, axis=1)

clusters['file_idx'] = file_index.astype(int)


#
# Filter only the high quality identifications, and fix the scan indexing to match
# the cluster results
#

all_ids = all_ids[all_ids['percolator q-value'] < 0.01]
all_ids['scan'] = all_ids['scan'] - 1


#
# Sort both dataframes to speedup their processing
#

clusters = clusters.sort_values(['file_idx', 'scan'])
all_ids = all_ids.sort_values(['file_idx', 'scan'])



#
# For each cluster, analyzes its spectra identification
#

def identify_clusters_spectra_fast(which_clusters, identifications):

    clusters_spectra = {}

    how_many_not_identified = 0
    
    ids_index = 0
    
    max_ids_index = identifications.shape[0]

    #
    # Transform the dataframes into numpy arrays
    #

    clusters_array = which_clusters.to_numpy()
    identifications_array = identifications.to_numpy()
    
    #
    # Get required columns indexes
    #

    CLUSTERS_FILE_IDX = which_clusters.columns.get_loc("file_idx")
    CLUSTERS_SCAN = which_clusters.columns.get_loc("scan")
    CLUSTERS_CLUSTER = which_clusters.columns.get_loc("cluster")

    IDS_FILE_IDX = identifications.columns.get_loc("file_idx")
    IDX_SCAN = identifications.columns.get_loc("scan")
    IDX_QVALUE = identifications.columns.get_loc("percolator q-value")
    IDX_SEQUENCE = identifications.columns.get_loc("sequence")


    for index in range(len(clusters_array)):
        
        new = {}
        
        while (ids_index < max_ids_index) and (identifications_array[ids_index][IDS_FILE_IDX] < clusters_array[index][CLUSTERS_FILE_IDX]):
            ids_index += 1
            
        if (ids_index < max_ids_index) and identifications_array[ids_index][IDS_FILE_IDX] == clusters_array[index][CLUSTERS_FILE_IDX]:
            while (ids_index < max_ids_index) and (identifications_array[ids_index][IDX_SCAN] < clusters_array[index][CLUSTERS_SCAN]):
                ids_index += 1
                
            if (ids_index < max_ids_index) and identifications_array[ids_index][IDX_SCAN] == clusters_array[index][CLUSTERS_SCAN]:
                new = {"sequence":identifications_array[ids_index][IDX_SEQUENCE], 
                       "q-value":identifications_array[ids_index][IDX_QVALUE], 
                       "file":identifications_array[ids_index][IDS_FILE_IDX],
                       "scan":identifications_array[ids_index][IDX_SCAN]}

                ids_index += 1
            
        if len(new) == 0:
            new = {"sequence":'not identified', 
                   "q-value":1.0, 
                   "file":clusters_array[index][CLUSTERS_FILE_IDX],
                   "scan":clusters_array[index][CLUSTERS_SCAN]}
            
            how_many_not_identified += 1
            
        if clusters_array[index][CLUSTERS_CLUSTER] in clusters_spectra:
            clusters_spectra[clusters_array[index][CLUSTERS_CLUSTER]].append(new)
        else:
            clusters_spectra[clusters_array[index][CLUSTERS_CLUSTER]] = [new]
            
    return clusters_spectra


#
# Check which are the peptides included in each cluster, returning their occurrence number.
#
# clusters: dictionary of clusters, with each entry as follows:
#           <cluster_id>:[<peptides sequence>, <percolator q-value>, <scan number>]
#
# clusters_analysis:
#           [cluster_id:<cluster id>, size:<size>, sequences:{<sequence>:<count>}, purity:<len(sequences)/size]
#

def check_clusters(clusters, consensus_id):
    
    clusters_analysis = []
    
    for cluster_id, cluster_data in clusters.items():
        
        current_cluster_sequences = {}
        
        longest_sequence = ""
        longest_sequence_len = -1
        
        for spectrum in cluster_data:
            if spectrum['sequence'] in current_cluster_sequences.keys():
                current_cluster_sequences[spectrum['sequence']] += 1
            else:
                current_cluster_sequences[spectrum['sequence']] = 1
                
            if (longest_sequence_len < current_cluster_sequences[spectrum['sequence']]) and (spectrum['sequence'] != "not identified"):
                    
                longest_sequence_len = current_cluster_sequences[spectrum['sequence']]
                longest_sequence = spectrum['sequence']
                
        current_consensus = consensus_id[consensus_id['cluster'] == cluster_id]
        consensus_sequence = "not identified"
        
        purity_consensus = 0.0
        purity_majority = 0.0
        consensus_sequence_len = 0
        
        if current_consensus.shape[0] > 0:
            consensus_sequence = current_consensus['sequence'].values[0]
            
            # Calculate the cluster purity based on the consensus identification
            
            if consensus_sequence in current_cluster_sequences.keys():
                consensus_sequence_len = current_cluster_sequences[consensus_sequence]
                purity_consensus = consensus_sequence_len/len(cluster_data)
                
        if longest_sequence != "":
            purity_majority = current_cluster_sequences[longest_sequence]/len(cluster_data)

        clusters_analysis.append({"cluster_id":cluster_id, 
                                  "size":len(cluster_data),
                                  "sequences":current_cluster_sequences,
                                  "longest_sequence":longest_sequence,
                                  "longest_sequence_len":longest_sequence_len,
                                  "consensus_sequence":consensus_sequence,
                                  "consensus_sequence_len":consensus_sequence_len,
                                  "purity_consensus":purity_consensus, 
                                  "purity_majority":purity_majority})
        
    return clusters_analysis


cluster_ids = identify_clusters_spectra_fast(clusters, all_ids)

cluster_analysis_result = check_clusters(cluster_ids, consensus_ids)

cluster_analysis_result_pd = pd.DataFrame(cluster_analysis_result)


#
# Compile the results per cluster size bins, using original MaRaCluster work bins
#

grouped_clusters_by_size = cluster_analysis_result_pd.groupby(pd.cut(cluster_analysis_result_pd['size'],
                                                                   pd.IntervalIndex.from_tuples([(0, 1), 
                                                                                                 (2, 3), 
                                                                                                 (4, 7), 
                                                                                                 (8, 15),
                                                                                                 (16, 31),
                                                                                                 (32, 63),
                                                                                                 (64, 127),
                                                                                                 (128, 255),
                                                                                                 (256, 511),
                                                                                                 (512, sys.maxsize)],
                                                                                                closed="both")))


size_groups_result = []
total_clusters = 0

for size, size_group in grouped_clusters_by_size:
    all_clusters_mean = size_group['purity_consensus'].mean()
    
    identified_clusters = size_group[size_group['consensus_sequence'] != 'not identified']
    
    identified_clusters_mean = identified_clusters['purity_consensus'].mean()
    size_groups_result.append({"size":size, 
                               "clusters":size_group.shape[0], 
                               "all clusters mean":all_clusters_mean,
                               "identified clusters":identified_clusters.shape[0],
                               "identified clusters mean":identified_clusters_mean})
    
    total_clusters += size_group.shape[0]


size_groups_result_pd = pd.DataFrame(size_groups_result)



#
# Save both dataframes
#

size_groups_result_pd.to_csv(ANALYSIS_OUTPUT_FILE + "_clusters_size" + RESULT_FILE_EXTENSION, index=False, sep='\t')
cluster_analysis_result_pd.to_csv(ANALYSIS_OUTPUT_FILE + RESULT_FILE_EXTENSION, index=False, sep='\t')



#
# Output some cluster analysys data
#
# <cluster name> <# clusters> <# identified consensus> <purity_consensus mean> <purity_majority mean> <# clusters with purity_consensus > 0.0> <# clusters with purity_consensus == 1.0>
#

print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(CLUSTER_NAME, 
                                          cluster_analysis_result_pd.shape[0],
                                          consensus_ids.shape[0],
                                          cluster_analysis_result_pd['purity_consensus'].mean(),
                                          cluster_analysis_result_pd['purity_majority'].mean(),
                                          sum(cluster_analysis_result_pd['purity_consensus'] > 0.0),
                                          sum(cluster_analysis_result_pd['purity_consensus'] == 1.0)))




