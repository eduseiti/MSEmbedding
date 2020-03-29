#!/bin/bash

#
# Output a tab-separated list of .mgf filenames and the number of spectra within it.
#
# ./count_spectra.sh <folder-where-mgf-files-are>
#

EXPERIMENTS_FILES=($(ls $1/*.mgf))

echo -e "file\tspectra count"

for file in "${EXPERIMENTS_FILES[@]}"; do
    spectra_count=$(grep "BEGIN IONS" $file | wc -l)

    only_filename=$(echo $file | sed -r "s/(.+\/)?(.+)$/\2/")

    echo -e "$only_filename\t$spectra_count"
done
