#!/bin/bash

#
# This script analyzes the clusters quality through the following procedure:
#
# - Checks the identification of each clustered spectra
# - Calculate the purity of each cluster based on the number of spectra identified as the cluster's consensus spectra
#
# It calls the "analyze_clusters_quality.py" script for analyzing each particular clusterization result.
#
# How to use: 
# 
# 	./analyze_clusters_quality.sh <clusters filename root>
#                                 <clusters consensus spectra high quality identification result folder>
#								  <crux all spectra identifications result folder> 
#                                 <clustered spectra files list> 
#                                 
#

GREEN='\033[0;32m\033[1m'
RED='\033[0;31m\033[1m'
BLUE='\033[0;34m\033[1m'
NC='\033[0m'

EXECUTION_PATH=$(echo $0 | sed -r "s/(.+\/)?(.+)$/\1/")

CLUSTERS_FOLDER_NAME=$(echo $1 | sed -r "s/(.+\/)?(.+)\/([^\/]+$)/\2/")

CLUSTERS_FILES=($(ls -d $1*))

CONSENSUS_PATH=$2
SPECTRA_IDENTIFICATION_FILE=$3/percolator.target.psms.txt
CLUSTERED_SPECTRA_FILES_LIST=$4

OUTPUT_FILE=$(echo ${CLUSTERS_FOLDER_NAME}_clusters_quality_$(date +%Y%m%d).tsv)

echo -e "cluster\t# clusters\t# identified consensus\tpurity_consensus mean\tpurity_majority mean\t# clusters purity_consensus > 0.0\t# clusters purity_consensus == 1.0" &>> $OUTPUT_FILE

startTime=$(date +%s.%N)

for cluster in "${CLUSTERS_FILES[@]}"; do
	clusterStartTime=$(date +%s.%N)
	
	BASENAME=$(echo $cluster | sed -r "s/(.+\/)?([^\/]+)\.tsv$/\2/")

	CONSENSUS_IDENTIFICATIONS_FILENAME=${CONSENSUS_PATH}/${BASENAME}_identifications.tsv
	ANALYSIS_OUTPUT_FILE=${CONSENSUS_PATH}/${BASENAME}_identifications_analysis.tsv

	echo
	echo -e ${BLUE}Start analyzing cluster \"$cluster\"...${NC}
	echo

	echo "${EXECUTION_PATH}analyze_clusters_quality.py ${cluster} ${BASENAME}.tsv ${CONSENSUS_IDENTIFICATIONS_FILENAME} ${ANALYSIS_OUTPUT_FILE} ${SPECTRA_IDENTIFICATION_FILE} ${CLUSTERED_SPECTRA_FILES_LIST} $>> $OUTPUT_FILE"
	${EXECUTION_PATH}analyze_clusters_quality.py ${cluster} ${BASENAME}.tsv ${CONSENSUS_IDENTIFICATIONS_FILENAME} ${ANALYSIS_OUTPUT_FILE} ${SPECTRA_IDENTIFICATION_FILE} ${CLUSTERED_SPECTRA_FILES_LIST} $>> $OUTPUT_FILE
	
	clusterTotalTime=$(echo "$(date +%s.%N) - $clusterStartTime" | bc)

	echo == Elapse time $clusterTotalTime seconds.
	echo
done

totalTime=$(echo "$(date +%s.%N) - $startTime" | bc)

echo Analyzing all identifications took $totalTime seconds.
echo

