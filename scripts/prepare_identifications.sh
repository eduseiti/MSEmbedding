#!/bin/bash

#
# This script analyzes the crux proteome identification files, preparing them to be used during the model training, 
#
# It calls the "prepare_identifications.py" script for each identifications result file.
#
# This script is supposed to be used over the identifications results produced by the "identify_experiments.sh" script.
#
# How to use; ./prepare_identifications.sh <identifications folders> <output folder> <q-score>
#
# <identifications folders> is the folder where multiple "identifications results folders" will be present. Each identification folder
# might potentially contain the identifications performed on several spectra files from a given experiment.
#
# The "identifications results folders" shall follow the naming convention "<experiment name>_identifications", since "<experiment name>" will
# will be used to name the resulting file. The resulting file will be a simplified version of the original "percolator.target.psms.txt" file.
#
# <q-score> is the maximum percolator q-score acceptable value: only identifications with score smaller that <q-score> will be accepted.
#

GREEN='\033[0;32m\033[1m'
RED='\033[0;31m\033[1m'
BLUE='\033[0;34m\033[1m'
NC='\033[0m'

EXECUTION_PATH=$(echo $0 | sed -r "s/(.+\/)?(.+)$/\1/")

IDENTIFICATION_FOLDERS=($(ls -d $1/*))
OUTPUT_FOLDER=$2
Q_SCORE=$3

IDENTIFICATION_LAST_PATH=$(echo $1 | sed -r "s/(.+\/)?(.+)$/\2/")

OUTPUT_FILE=$(echo ${IDENTIFICATION_LAST_PATH}_preparation_$(date +%Y%m%d).csv)

echo "experiment, q<${Q_SCORE} identifications, total identifications" &>> $OUTPUT_FILE

startTime=$(date +%s.%N)

for folder in "${IDENTIFICATION_FOLDERS[@]}"; do
	folderStartTime=$(date +%s.%N)
	
	BASENAME=$(echo $folder | sed -r "s/(.+\/)?(.+)_identifications/\2/")

	echo
	echo -e ${BLUE}Start processing identification folder \"$folder\"...${NC}
	echo

	echo "./prepare_identifications.py $folder $BASENAME $OUTPUT_FOLDER $Q_SCORE $>> $OUTPUT_FILE"
	
	${EXECUTION_PATH}prepare_identifications.py $folder $BASENAME $OUTPUT_FOLDER $Q_SCORE &>> $OUTPUT_FILE

	folderTotalTime=$(echo "$(date +%s.%N) - $folderStartTime" | bc)

	echo == Elapse time $folderTotalTime seconds.
	echo
done

totalTime=$(echo "$(date +%s.%N) - $startTime" | bc)

echo Preparing all identifications took $totalTime seconds.
echo

