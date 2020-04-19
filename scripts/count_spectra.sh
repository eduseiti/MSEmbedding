#!/bin/bash

#
# Output a tab-separated list of .mgf filenames and the number of spectra within it.
#
# ./count_spectra.sh <file-with-the-list-of-all-mgf-files>
#

EXPERIMENTS_FILES=($(cat $1))

echo -e "file\tspectra count"

for file in "${EXPERIMENTS_FILES[@]}"; do

    spectra_count=$(grep "BEGIN IONS" $file | wc -l)
    only_filename=$(echo $file | sed -r "s/(.+\/)?(.+)$/\2/")

    echo -e "$only_filename\t$spectra_count"
done
