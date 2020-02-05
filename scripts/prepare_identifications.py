#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
import os
import sys

IDENTIFICATIONS_FOLDER = sys.argv[1]
# "/media/eduseiti/bigdata01/unicamp/doutorado/PXD000561/identifications/Adult_Adrenalgland_bRP_Velos_identifications"

BASENAME = sys.argv[2]
# "Adult_Adrenalgland_bRP_Velos"

OUTPUT_FOLDER = sys.argv[3]
# "/media/eduseiti/bigdata01/unicamp/doutorado/PXD000561/identifications"

FILE = "percolator.target.psms.txt"


all_ids = pd.read_csv(os.path.join(IDENTIFICATIONS_FOLDER, FILE), sep="\t")

high_quality_ids = all_ids[all_ids['percolator q-value'] < 0.01]
high_quality_ids = high_quality_ids.sort_values(["file_idx", "scan"])

high_quality_ids.to_csv(os.path.join(OUTPUT_FOLDER, BASENAME + "_q_lt_0.01_identifications.tsv"), index=False, sep="\t")

print("{}, {}, {}".format(BASENAME, high_quality_ids.shape[0], all_ids.shape[0]))
