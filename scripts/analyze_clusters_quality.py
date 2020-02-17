#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
import plotly
import os
import sys

CLUSTER = sys.argv[1]
# "/media/eduseiti/bigdata02/unicamp/doutorado/clustering/linfeng/sample/clusters_sample_0.1_embeddings_q0.001/sample_embeddings_q0.001.clusters_p95.897800.tsv"

CLUSTER_NAME = sys.argv[2]
# "sample_embeddings_q0.001.clusters_p95.897800.tsv"

CLUSTER_CONSENSUS_IDENTIFICATIONS = sys.argv[3]
# "/media/eduseiti/bigdata02/unicamp/doutorado/clustering/linfeng/sample/clusters_sample_0.1_embeddings_q0.001/sample_embeddings_q0.001.clusters_p95.897800_identifications.tsv"

ANALYSIS_OUTPUT_FILE = sys.argv[4]
# "/media/eduseiti/bigdata02/unicamp/doutorado/clustering/linfeng/sample/clusters_sample_0.1_embeddings_q0.001/sample_embeddings_q0.001.clusters_p95.897800_identifications_analysis.tsv"

ALL_IDENTIFICATIONS = sys.argv[5]
# "/media/eduseiti/bigdata02/unicamp/doutorado/clustering/linfeng/sample/identifications_sample_0.1_nterm/sample_experiment_identifications/percolator.target.psms.txt"

CLUSTERED_FILES_LIST = sys.argv[6]
# "/media/eduseiti/bigdata02/unicamp/doutorado/clustering/linfeng/sample/sample_0.1_files.txt"


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
# For each cluster, analyzes its spectra identification
#

def identify_clusters_spectra(which_clusters, identifications):
    
    clusters = {}
    
    how_many_not_identified = 0
    
    for index, row in which_clusters.iterrows():
        
        scanIdFound = identifications[(identifications['file_idx'] == spectra_files[row['file']]) & 
                                      (identifications['scan'] - 1 == row['scan'])]

        if len(scanIdFound['sequence'].values) > 0:
            new = {"sequence":scanIdFound['sequence'].values[0], 
                   "q-value":scanIdFound['percolator q-value'].values[0], 
                   "file":spectra_files[row['file']],
                   "scan":row['scan']}
        else:
            new = {"sequence":'not identified', 
                   "q-value":'no q-value', 
                   "file":spectra_files[row['file']],
                   "scan":row['scan']}
            
            how_many_not_identified += 1
    
        if row['cluster'] in clusters:
            clusters[row['cluster']].append(new)
        else:
            clusters[row['cluster']] = [new]
            
    # print("Not identified scans count = {}".format(how_many_not_identified))
            
    return clusters



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
                
            if (longest_sequence_len < current_cluster_sequences[spectrum['sequence']]) and                (spectrum['sequence'] != "not identified"):
                    
                longest_sequence_len = current_cluster_sequences[spectrum['sequence']]
                longest_sequence = spectrum['sequence']
                
        current_consensus = consensus_id[consensus_id['cluster'] == cluster_id]
        consensus_sequence = "not identified"
        
        purity_consensus = 0.0
        purity_majority = 0.0
        
        if current_consensus.shape[0] > 0:
            consensus_sequence = current_consensus['sequence'].values[0]
            
            # Calculate the cluster purity based on the consensus identification
            
            if consensus_sequence in current_cluster_sequences.keys():
                purity_consensus = current_cluster_sequences[consensus_sequence]/len(cluster_data)
                
        if longest_sequence != "":
            purity_majority = current_cluster_sequences[longest_sequence]/len(cluster_data)

        clusters_analysis.append({"cluster_id":cluster_id, 
                                  "size":len(cluster_data),
                                  "sequences":current_cluster_sequences,
                                  "longest_sequence":longest_sequence,
                                  "longest_sequence_len":longest_sequence_len,
                                  "consensus_sequence":consensus_sequence,
                                  "purity_consensus":purity_consensus, 
                                  "purity_majority":purity_majority})
        
    return clusters_analysis


cluster_ids = identify_clusters_spectra(clusters, all_ids)

cluster_analysis_result = check_clusters(cluster_ids, consensus_ids)

cluster_analysis_result_pd = pd.DataFrame(cluster_analysis_result)

cluster_analysis_result_pd.to_csv(ANALYSIS_OUTPUT_FILE, index=False, sep='\t')



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




