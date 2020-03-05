#!/bin/bash

#
# This script calculates the consensus spectra using the "maracluster consensus" command, and receiving a folder with the .tsv cluster files.
# This script requires "maracluster" executable visible in the PATH.
#
# How to use; ./calculate_consensus.sh <clusters folder>
#

GREEN='\033[0;32m\033[1m'
RED='\033[0;31m\033[1m'
BLUE='\033[0;34m\033[1m'
NC='\033[0m'

CLUSTERS_FILES=($(ls $1/*.tsv))

startTimeAll=$(date +%s.%N)

for file in "${CLUSTERS_FILES[@]}"; do
	startTime=$(date +%s.%N)

	BASENAME=$(echo $file | sed -r "s/(.+\/)?(.+)\.tsv$/\2/")
	LOGFILE=$(echo log_consensus_${BASENAME}_$(date +%Y%m%d).txt)
	CONSENSUS_FOLDER=$(echo ./consensus_${BASENAME})

	echo
	echo -e ${BLUE}Start processing file \"$file\"...${NC}
	echo

	echo "Start processing file \"$file\"..." &>> $LOGFILE
	echo "" &>> $LOGFILE

	echo "{ time maracluster consensus -v 5 -l $file -f ${CONSENSUS_FOLDER} -o ${CONSENSUS_FOLDER}/${BASENAME}.ms2 ; } &>> $LOGFILE"
	echo

	echo "time maracluster consensus -v 5 -l $file -f ${CONSENSUS_FOLDER} -o ${CONSENSUS_FOLDER}/${BASENAME}.ms2" &>> $LOGFILE
	echo "" &>> $LOGFILE

	{ time maracluster consensus -v 5 -l $file -f $CONSENSUS_FOLDER -o $CONSENSUS_FOLDER/$BASENAME.ms2 ; } &>> $LOGFILE

	if [ $? -eq 0 ]; then
		echo -e ${GREEN}= Successfully processed file $file!!!${NC}
	else
		echo -e ${RED}= Error while executing maracluster consensus...${NC}
	fi

	totalTime=$(echo "$(date +%s.%N) - $startTime" | bc)

	echo == Elapse time $totalTime seconds.
	echo
	echo "== Elapse time $totalTime seconds." &>> $LOGFILE
	echo "" &>> $LOGFILE
done

totalTimeAll=$(echo "$(date +%s.%N) - $startTimeAll" | bc)

echo Elapse time for creating all consensus spectra: $totalTimeAll seconds.
echo
echo "Elapse time for creating all consensus spectra: $totalTimeAll seconds." &>> $LOGFILE
echo "" &>> $LOGFILE
