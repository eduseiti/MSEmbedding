#!/bin/bash

#
# This script analyzes the crux proteome identification files, which should have been obtained after applying the "identify_consensus.sh" script.
#
# It calls the "analyze_identifications.py" script for each consensus spectra identifications result file.
#
# How to use; ./analyze_identifications.sh <identifications folders name root> <clusters folder>
#

GREEN='\033[0;32m\033[1m'
RED='\033[0;31m\033[1m'
BLUE='\033[0;34m\033[1m'
NC='\033[0m'

EXECUTION_PATH=$(echo $0 | sed -r "s/(.+\/)?(.+)$/\1/")

CLUSTERS_PATH=$2
CLUSTER_FOLDER_NAME=$(echo $CLUSTERS_PATH | sed -r "s/(.+\/)?([^\/]+$)/\2/")

IDENTIFICATION_FOLDERS=($(ls -d $1*))

OUTPUT_FILE=$(echo ${CLUSTER_FOLDER_NAME}_identification_analysis_$(date +%Y%m%d).csv)

echo cluster name, num clusters, total spectra, num identifications &>> $OUTPUT_FILE

startTime=$(date +%s.%N)

for folder in "${IDENTIFICATION_FOLDERS[@]}"; do
	folderStartTime=$(date +%s.%N)
	
	BASENAME=$(echo $folder | sed -r "s/(.+\/)?identifications_(.+)/\2/")

	echo
	echo -e ${BLUE}Start processing identification folder \"$folder\"...${NC}
	echo

	echo "./analyze_identifications.py $folder $CLUSTERS_PATH $BASENAME $>> $OUTPUT_FILE"
	
	${EXECUTION_PATH}analyze_identifications.py $folder $CLUSTERS_PATH $BASENAME $>> $OUTPUT_FILE

	folderTotalTime=$(echo "$(date +%s.%N) - $folderStartTime" | bc)

	echo == Elapse time $folderTotalTime seconds.
	echo
done

totalTime=$(echo "$(date +%s.%N) - $startTime" | bc)

echo Analyzing all identifications took $totalTime seconds.
echo

