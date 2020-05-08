#!/bin/bash

#
# This script is used to identify using crux the .mgf files obtained from the selected experiments of:
#
# Kim, Min-Sik, Sneha M. Pinto, Derese Getnet, Raja Sekhar Nirujogi, Srikanth S. Manda, Raghothama Chaerkady, Anil K. Madugundu et al. "A draft map of the human proteome." Nature 509, no. 7502 (2014): 575-581.
#
# How to use; ./identify_experiments.sh <experiments folders folder> 
#										<identifications destination folder> 
#										<crux executable path> 
#										<peptides identification database file> 
#										<crux parameters file>
#

GREEN='\033[0;32m\033[1m'
RED='\033[0;31m\033[1m'
BLUE='\033[0;34m\033[1m'
NC='\033[0m'

IDENTIFICATIONS_PATH=$2
CRUX_PATH=$3
DB_PATH=$4
CRUX_PARAMS_FILE=$5

EXPERIMENTS_FOLDERS=($(ls -d $1/*))


for folder in "${EXPERIMENTS_FOLDERS[@]}"; do
	startTime=$(date +%s.%N)

	BASENAME=$(echo $folder | sed -r "s/(.+\/)?(.+)$/\2/")
	LOGFILE=$(echo log_identify_experiment_${BASENAME}_$(date +%Y%m%d).txt)
	IDENTIFICATION_FOLDER=$(echo ${IDENTIFICATIONS_PATH}/${BASENAME}_identifications)

	echo
	echo -e ${BLUE}Start processing experiment folder \"$folder\"...${NC}
	echo

	echo "Start processing experiment folder \"$folder\"" &>> $LOGFILE
	echo "" &>> $LOGFILE

	EXPERIMENT_FILES=($(ls ${folder}/))

	FILE_IDX=0

	for file in "${EXPERIMENT_FILES[@]}"; do	

		OUTPUT_FOLDER=$(echo ${IDENTIFICATION_FOLDER}/${file})

		echo "{ time $CRUX_PATH/crux tide-search ${folder}/${file} ${DB_PATH} --parameter-file $CRUX_PARAMS_FILE --output-dir ${OUTPUT_FOLDER} --overwrite T ; } &>> $LOGFILE"
		echo

		echo "time $CRUX_PATH/crux tide-search ${folder}/${file} ${DB_PATH} --parameter-file $CRUX_PARAMS_FILE --output-dir ${OUTPUT_FOLDER} --overwrite T " &>> $LOGFILE
		echo "" &>> $LOGFILE

		{ time $CRUX_PATH/crux tide-search ${folder}/${file} ${DB_PATH} --parameter-file $CRUX_PARAMS_FILE --output-dir ${OUTPUT_FOLDER} --overwrite T ; } &>> $LOGFILE

		if [ $? -eq 0 ]; then

			echo "{ time $CRUX_PATH/crux percolator $OUTPUT_FOLDER/tide-search.target.txt --parameter-file $CRUX_PARAMS_FILE --output-dir $OUTPUT_FOLDER --overwrite T ; } &>> $LOGFILE"
			echo

			echo "time $CRUX_PATH/crux percolator $OUTPUT_FOLDER/tide-search.target.txt --parameter-file $CRUX_PARAMS_FILE --output-dir $OUTPUT_FOLDER --overwrite T" &>> $LOGFILE
			echo "" &>> $LOGFILE

			{ time $CRUX_PATH/crux percolator $OUTPUT_FOLDER/tide-search.target.txt --parameter-file $CRUX_PARAMS_FILE --output-dir $OUTPUT_FOLDER --overwrite T ; } &>> $LOGFILE

			if [ $? -eq 0 ]; then

				echo -e ${GREEN}= Successfully identified experiment file $file!!!${NC}

				if [ $FILE_IDX -eq 0 ]; then
					cat $OUTPUT_FOLDER/percolator.target.psms.txt | sed -r "s/0(.+)$/${FILE_IDX}\1/" >> ${IDENTIFICATION_FOLDER}/percolator.target.psms.txt
				else
					tail --lines=+2 $OUTPUT_FOLDER/percolator.target.psms.txt | sed -r "s/0(.+)$/${FILE_IDX}\1/" >> ${IDENTIFICATION_FOLDER}/percolator.target.psms.txt
				fi
			else
				echo -e ${RED}= Error while executing percolator...${NC}
			fi
		else
			echo -e ${RED}= Error while executing tide-search...${NC}
		fi

		FILE_IDX=$((FILE_IDX+1))
	done



	totalTime=$(echo "$(date +%s.%N) - $startTime" | bc)

	echo == Elapse time $totalTime seconds.
	echo
	echo "== Elapse time $totalTime seconds." &>> $LOGFILE
	echo "" &>> $LOGFILE
done
