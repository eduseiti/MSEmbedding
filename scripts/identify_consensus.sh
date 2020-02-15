#!/bin/bash

#
# 
# This script calls crux to identify the consensus spectra calculated for each cluster obtained using the "caclulate_consensus.sh" script.
#
# How to use; ./identify_consensus.sh <consensus folders name root> <crux executable path> <peptides identification database file> <crux parameters file>
#
# The consensus folders should be in the following naming format: <anything>consensus_<cluster identification>.
# 
# <cluster identification> will be used to identify the results folder, which will be "identifications_<cluster identification>", as well as the
# execution log file.
#

GREEN='\033[0;32m\033[1m'
RED='\033[0;31m\033[1m'
BLUE='\033[0;34m\033[1m'
NC='\033[0m'

CRUX_PATH=$2
DB_PATH=$3
CRUX_PARAMS_FILE=$4

CONSENSUS_FOLDERS=($(ls -d $1*))


for folder in "${CONSENSUS_FOLDERS[@]}"; do
	startTime=$(date +%s.%N)

	BASENAME=$(echo $folder | sed -r "s/(.+\/)?consensus_(.+)/\2/")
	LOGFILE=$(echo log_identify_consensus_${BASENAME}_$(date +%Y%m%d).txt)
	IDENTIFICATION_FOLDER=$(echo ./identifications_${BASENAME})

	echo
	echo -e ${BLUE}Start processing consensus folder \"$folder\"...${NC}
	echo

	echo "Start processing consensus folder \"$folder\"" &>> $LOGFILE
	echo "" &>> $LOGFILE

	echo "{ time $CRUX_PATH/crux tide-search ${folder}/* ${DB_PATH} --parameter-file $CRUX_PARAMS_FILE --output-dir ${IDENTIFICATION_FOLDER} --overwrite T ; } &>> $LOGFILE"
	echo

	echo "time $CRUX_PATH/crux tide-search ${folder}/* ${DB_PATH} --parameter-file $CRUX_PARAMS_FILE --output-dir ${IDENTIFICATION_FOLDER} --overwrite T " &>> $LOGFILE
	echo "" &>> $LOGFILE

	{ time $CRUX_PATH/crux tide-search ${folder}/* ${DB_PATH} --parameter-file $CRUX_PARAMS_FILE --output-dir ${IDENTIFICATION_FOLDER} --overwrite T ; } &>> $LOGFILE

	if [ $? -eq 0 ]; then

		echo "{ time $CRUX_PATH/crux percolator $IDENTIFICATION_FOLDER/tide-search.target.txt --parameter-file $CRUX_PARAMS_FILE --output-dir $IDENTIFICATION_FOLDER --overwrite T ; } &>> $LOGFILE"
		echo

		echo "time $CRUX_PATH/crux percolator $IDENTIFICATION_FOLDER/tide-search.target.txt --parameter-file $CRUX_PARAMS_FILE --output-dir $IDENTIFICATION_FOLDER --overwrite T" &>> $LOGFILE
		echo "" &>> $LOGFILE

		{ time $CRUX_PATH/crux percolator $IDENTIFICATION_FOLDER/tide-search.target.txt --parameter-file $CRUX_PARAMS_FILE --output-dir $IDENTIFICATION_FOLDER --overwrite T ; } &>> $LOGFILE

		if [ $? -eq 0 ]; then

			echo -e ${GREEN}= Successfully identified consensus folder $folder!!!${NC}

		else
			echo -e ${RED}= Error while executing percolator...${NC}
		fi
	else
		echo -e ${RED}= Error while executing tide-search...${NC}
	fi

	totalTime=$(echo "$(date +%s.%N) - $startTime" | bc)

	echo == Elapse time $totalTime seconds.
	echo
	echo "== Elapse time $totalTime seconds." &>> $LOGFILE
	echo "" &>> $LOGFILE
done
