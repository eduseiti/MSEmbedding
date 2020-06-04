from .HumanProteome import HumanProteome
from .PXD000561 import PXD000561
from .spectra import SpectraFound

import os
import torch
import torch.utils.data as data

from .BatchLoader import BatchLoader

from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger


class MixedSpectraCrux(data.Dataset):

    #
    # Existing datasets versions:
    #
    # v2.0: Initial set of experiments (8 train, 1 validation), with identifications with percolator q-score < 0.01 confidence
    # v2.1: Initial set of experiments (8 train, 1 validation), with identifications with percolator q-score < 0.001 confidence
    #
    # v3.0: Expanded set of experiments (19 train, 4 validation), with identifications with percolator q-score < 0.01 confidence
    # v3.1: Initial set of experiments (19 train, 4 validation), with identifications with percolator q-score < 0.001 confidence
    #
    #



    CURRENT_TRAIN_VERSION = "v3.0"
    CURRENT_TEST_VERSION = "v3.0"

    TRAIN_FILENAME = "train_mixedSpectraCrux_{}.pkl"
    TEST_FILENAME = "test_mixedSpectraCrux_{}.pkl"

    TRAIN_EXPERIMENTS_NAME_FORMAT = "TRAIN_EXPERIMENTS_DATA_{}"
    TEST_EXPERIMENTS_NAME_FORMAT = "TEST_EXPERIMENTS_DATA_{}"


    TRAIN_EXPERIMENTS_DATA_2_0 = {
        "Fetal_Brain_Gel_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "fetal_brain_gel_velos_crux_q_0.01.pkl", "filesList": None, "constructor" : HumanProteome},
        "Fetal_Ovary_bRP_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "fetal_ovary_brp_velos_crux_q_0.01.pkl", "filesList": None, "constructor" : HumanProteome},
        "Fetal_Ovary_bRP_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "fetal_ovary_brp_elite_crux_q_0.01.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Adrenalgland_Gel_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_adrenalgland_gel_elite_crux_q_0.01.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Adrenalgland_Gel_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_adrenalgland_gel_velos_crux_q_0.01.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Adrenalgland_bRP_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_adrenalgland_bRP_velos_crux_q_0.01.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Urinarybladder_Gel_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_urinarybladder_gel_elite_crux_q_0.01.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Platelets_Gel_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_platelets_gel_elite_crux_q_0.01.pkl", "filesList": None, "constructor" : HumanProteome}
    }

    TRAIN_EXPERIMENTS_DATA_2_1 = {
        "Fetal_Brain_Gel_Velos_q_lt_0.001_identifications.tsv" : {"peaksFile" : "fetal_brain_gel_velos_crux_q_0.001.pkl", "filesList": None, "constructor" : HumanProteome},
        "Fetal_Ovary_bRP_Velos_q_lt_0.001_identifications.tsv" : {"peaksFile" : "fetal_ovary_brp_velos_crux_q_0.001.pkl", "filesList": None, "constructor" : HumanProteome},
        "Fetal_Ovary_bRP_Elite_q_lt_0.001_identifications.tsv" : {"peaksFile" : "fetal_ovary_brp_elite_crux_q_0.001.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Adrenalgland_Gel_Elite_q_lt_0.001_identifications.tsv" : {"peaksFile" : "adult_adrenalgland_gel_elite_crux_q_0.001.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Adrenalgland_Gel_Velos_q_lt_0.001_identifications.tsv" : {"peaksFile" : "adult_adrenalgland_gel_velos_crux_q_0.001.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Adrenalgland_bRP_Velos_q_lt_0.001_identifications.tsv" : {"peaksFile" : "adult_adrenalgland_bRP_velos_crux_q_0.001.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Urinarybladder_Gel_Elite_q_lt_0.001_identifications.tsv" : {"peaksFile" : "adult_urinarybladder_gel_elite_crux_q_0.001.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Platelets_Gel_Elite_q_lt_0.001_identifications.tsv" : {"peaksFile" : "adult_platelets_gel_elite_crux_q_0.001.pkl", "filesList": None, "constructor" : HumanProteome}
    }

    TRAIN_EXPERIMENTS_DATA_3_0 = {
        "Fetal_Brain_Gel_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "fetal_brain_gel_velos_crux_q_0.01.pkl", "filesList": None, "constructor" : HumanProteome},
        "Fetal_Ovary_bRP_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "fetal_ovary_brp_velos_crux_q_0.01.pkl", "filesList": None, "constructor" : HumanProteome},
        "Fetal_Ovary_bRP_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "fetal_ovary_brp_elite_crux_q_0.01.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Adrenalgland_Gel_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_adrenalgland_gel_elite_crux_q_0.01.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Adrenalgland_Gel_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_adrenalgland_gel_velos_crux_q_0.01.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Adrenalgland_bRP_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_adrenalgland_bRP_velos_crux_q_0.01.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Urinarybladder_Gel_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_urinarybladder_gel_elite_crux_q_0.01.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Platelets_Gel_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_platelets_gel_elite_crux_q_0.01.pkl", "filesList": None, "constructor" : HumanProteome},

        "Adult_Bcells_bRP_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_bcells_brp_elite_crux_q_0.01.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Bcells_bRP_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_bcells_brp_velos_crux_q_0.01.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_CD4Tcells_Gel_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_cd4tcells_gel_velos_crux_q_0.01.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_CD8Tcells_Gel_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_cd8tcells_gel_elite_crux_q_0.01.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Colon_Gel_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_colon_gel_elite_crux_q_0.01.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Esophagus_Gel_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_esophagus_gel_velos_crux_q_0.01.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_NKcells_bRP_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_nkcells_brp_velos_crux_q_0.01.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Pancreas_bRP_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_pancreas_brp_elite_crux_q_0.01.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Gut_Gel_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "fetal_gut_gel_velos_crux_q_0.01.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Liver_bRP_Elite_23_q_lt_0.01_identifications.tsv" : {"peaksFile" : "fetal_liver_brp_elite_23_crux_q_0.01.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Testis_bRP_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "fetal_testis_brp_elite_crux_q_0.01.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Bcells_Gel_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_bcells_gel_elite_crux_q_0.01.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Liver_Gel_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "fetal_liver_gel_velos_crux_q_0.01.pkl", "filesList" : None, "constructor" : HumanProteome}
    }

    TRAIN_EXPERIMENTS_DATA_3_1 = {
        "Fetal_Brain_Gel_Velos_q_lt_0.001_identifications.tsv" : {"peaksFile" : "fetal_brain_gel_velos_crux_q_0.001.pkl", "filesList": None, "constructor" : HumanProteome},
        "Fetal_Ovary_bRP_Velos_q_lt_0.001_identifications.tsv" : {"peaksFile" : "fetal_ovary_brp_velos_crux_q_0.001.pkl", "filesList": None, "constructor" : HumanProteome},
        "Fetal_Ovary_bRP_Elite_q_lt_0.001_identifications.tsv" : {"peaksFile" : "fetal_ovary_brp_elite_crux_q_0.001.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Adrenalgland_Gel_Elite_q_lt_0.001_identifications.tsv" : {"peaksFile" : "adult_adrenalgland_gel_elite_crux_q_0.001.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Adrenalgland_Gel_Velos_q_lt_0.001_identifications.tsv" : {"peaksFile" : "adult_adrenalgland_gel_velos_crux_q_0.001.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Adrenalgland_bRP_Velos_q_lt_0.001_identifications.tsv" : {"peaksFile" : "adult_adrenalgland_bRP_velos_crux_q_0.001.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Urinarybladder_Gel_Elite_q_lt_0.001_identifications.tsv" : {"peaksFile" : "adult_urinarybladder_gel_elite_crux_q_0.001.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Platelets_Gel_Elite_q_lt_0.001_identifications.tsv" : {"peaksFile" : "adult_platelets_gel_elite_crux_q_0.001.pkl", "filesList": None, "constructor" : HumanProteome},

        "Adult_Bcells_bRP_Elite_q_lt_0.001_identifications.tsv" : {"peaksFile" : "adult_bcells_brp_elite_crux_q_0.001.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Bcells_bRP_Velos_q_lt_0.001_identifications.tsv" : {"peaksFile" : "adult_bcells_brp_velos_crux_q_0.001.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_CD4Tcells_Gel_Velos_q_lt_0.001_identifications.tsv" : {"peaksFile" : "adult_cd4tcells_gel_velos_crux_q_0.001.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_CD8Tcells_Gel_Elite_q_lt_0.001_identifications.tsv" : {"peaksFile" : "adult_cd8tcells_gel_elite_crux_q_0.001.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Colon_Gel_Elite_q_lt_0.001_identifications.tsv" : {"peaksFile" : "adult_colon_gel_elite_crux_q_0.001.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Esophagus_Gel_Velos_q_lt_0.001_identifications.tsv" : {"peaksFile" : "adult_esophagus_gel_velos_crux_q_0.001.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_NKcells_bRP_Velos_q_lt_0.001_identifications.tsv" : {"peaksFile" : "adult_nkcells_brp_velos_crux_q_0.001.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Pancreas_bRP_Elite_q_lt_0.001_identifications.tsv" : {"peaksFile" : "adult_pancreas_brp_elite_crux_q_0.001.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Gut_Gel_Velos_q_lt_0.001_identifications.tsv" : {"peaksFile" : "fetal_gut_gel_velos_crux_q_0.001.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Liver_bRP_Elite_23_q_lt_0.001_identifications.tsv" : {"peaksFile" : "fetal_liver_brp_elite_23_crux_q_0.001.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Testis_bRP_Elite_q_lt_0.001_identifications.tsv" : {"peaksFile" : "fetal_testis_brp_elite_crux_q_0.001.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Bcells_Gel_Elite_q_lt_0.001_identifications.tsv" : {"peaksFile" : "adult_bcells_gel_elite_crux_q_0.001.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Liver_Gel_Velos_q_lt_0.001_identifications.tsv" : {"peaksFile" : "fetal_liver_gel_velos_crux_q_0.001.pkl", "filesList" : None, "constructor" : HumanProteome}
    }

    TRAIN_EXPERIMENTS_DATA_4_0 = {
        "Fetal_Brain_Gel_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "fetal_brain_gel_velos_crux_q_0.01_pvalue{}.pkl", "filesList": None, "constructor" : HumanProteome},
        "Fetal_Ovary_bRP_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "fetal_ovary_brp_velos_crux_q_0.01_pvalue{}.pkl", "filesList": None, "constructor" : HumanProteome},
        "Fetal_Ovary_bRP_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "fetal_ovary_brp_elite_crux_q_0.01_pvalue{}.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Adrenalgland_Gel_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_adrenalgland_gel_elite_crux_q_0.01_pvalue{}.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Adrenalgland_Gel_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_adrenalgland_gel_velos_crux_q_0.01_pvalue{}.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Adrenalgland_bRP_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_adrenalgland_bRP_velos_crux_q_0.01_pvalue{}.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Urinarybladder_Gel_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_urinarybladder_gel_elite_crux_q_0.01_pvalue{}.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Platelets_Gel_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_platelets_gel_elite_crux_q_0.01_pvalue{}.pkl", "filesList": None, "constructor" : HumanProteome},

        "Adult_Bcells_bRP_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_bcells_brp_elite_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Bcells_bRP_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_bcells_brp_velos_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_CD4Tcells_Gel_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_cd4tcells_gel_velos_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_CD8Tcells_Gel_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_cd8tcells_gel_elite_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Colon_Gel_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_colon_gel_elite_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Esophagus_Gel_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_esophagus_gel_velos_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_NKcells_bRP_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_nkcells_brp_velos_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Pancreas_bRP_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_pancreas_brp_elite_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Gut_Gel_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "fetal_gut_gel_velos_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Liver_bRP_Elite_23_q_lt_0.01_identifications.tsv" : {"peaksFile" : "fetal_liver_brp_elite_23_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Testis_bRP_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "fetal_testis_brp_elite_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Bcells_Gel_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_bcells_gel_elite_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Liver_Gel_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "fetal_liver_gel_velos_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome}
    }

    TRAIN_EXPERIMENTS_DATA_4_6 = TRAIN_EXPERIMENTS_DATA_4_5 = TRAIN_EXPERIMENTS_DATA_4_4 = TRAIN_EXPERIMENTS_DATA_4_3 = TRAIN_EXPERIMENTS_DATA_4_2 = TRAIN_EXPERIMENTS_DATA_4_1 = TRAIN_EXPERIMENTS_DATA_4_0



    TRAIN_EXPERIMENTS_DATA_5_0 = {
        "Fetal_Brain_Gel_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "fetal_brain_gel_velos_crux_q_0.01_pvalue{}.pkl", "filesList": None, "constructor" : HumanProteome},
        "Fetal_Ovary_bRP_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "fetal_ovary_brp_velos_crux_q_0.01_pvalue{}.pkl", "filesList": None, "constructor" : HumanProteome},
        "Fetal_Ovary_bRP_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "fetal_ovary_brp_elite_crux_q_0.01_pvalue{}.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Adrenalgland_Gel_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_adrenalgland_gel_elite_crux_q_0.01_pvalue{}.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Adrenalgland_Gel_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_adrenalgland_gel_velos_crux_q_0.01_pvalue{}.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Adrenalgland_bRP_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_adrenalgland_bRP_velos_crux_q_0.01_pvalue{}.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Urinarybladder_Gel_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_urinarybladder_gel_elite_crux_q_0.01_pvalue{}.pkl", "filesList": None, "constructor" : HumanProteome},
        "Adult_Platelets_Gel_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_platelets_gel_elite_crux_q_0.01_pvalue{}.pkl", "filesList": None, "constructor" : HumanProteome},

        "Adult_Bcells_bRP_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_bcells_brp_elite_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Bcells_bRP_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_bcells_brp_velos_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_CD4Tcells_Gel_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_cd4tcells_gel_velos_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_CD8Tcells_Gel_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_cd8tcells_gel_elite_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Colon_Gel_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_colon_gel_elite_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Esophagus_Gel_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_esophagus_gel_velos_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_NKcells_bRP_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_nkcells_brp_velos_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Pancreas_bRP_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_pancreas_brp_elite_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Gut_Gel_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "fetal_gut_gel_velos_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Liver_bRP_Elite_23_q_lt_0.01_identifications.tsv" : {"peaksFile" : "fetal_liver_brp_elite_23_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Testis_bRP_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "fetal_testis_brp_elite_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Bcells_Gel_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_bcells_gel_elite_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Liver_Gel_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "fetal_liver_gel_velos_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},

        "Adult_CD4Tcells_bRP_Elite_28_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_CD4Tcells_bRP_Elite_28_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_CD4Tcells_bRP_Velos_29_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_CD4Tcells_bRP_Velos_29_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_CD8Tcells_Gel_Velos_45_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_CD8Tcells_Gel_Velos_45_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_CD8Tcells_bRP_Elite_77_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_CD8Tcells_bRP_Elite_77_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_CD8Tcells_bRP_Velos_43_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_CD8Tcells_bRP_Velos_43_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Colon_bRP_Elite_50_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Colon_bRP_Elite_50_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Esophagus_bRP_Velos_3_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Esophagus_bRP_Velos_3_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Frontalcortex_Gel_Elite_80_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Frontalcortex_Gel_Elite_80_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Frontalcortex_bRP_Elite_38_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Frontalcortex_bRP_Elite_38_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Frontalcortex_bRP_Elite_85_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Frontalcortex_bRP_Elite_85_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Gallbladder_Gel_Elite_52_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Gallbladder_Gel_Elite_52_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Gallbladder_bRP_Elite_53_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Gallbladder_bRP_Elite_53_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Heart_Gel_Elite_54_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Heart_Gel_Elite_54_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Heart_Gel_Velos_7_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Heart_Gel_Velos_7_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Heart_bRP_Elite_81_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Heart_bRP_Elite_81_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Kidney_Gel_Elite_55_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Kidney_Gel_Elite_55_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Kidney_Gel_Velos_9_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Kidney_Gel_Velos_9_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Kidney_bRP_Velos_8_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Kidney_bRP_Velos_8_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Liver_Gel_Elilte_83_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Liver_Gel_Elilte_83_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Liver_Gel_Velos_11_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Liver_Gel_Velos_11_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Liver_bRP_Elite_82_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Liver_bRP_Elite_82_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Liver_bRP_Velos_10_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Liver_bRP_Velos_10_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Lung_Gel_Elite_56_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Lung_Gel_Elite_56_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Lung_Gel_Velos_13_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Lung_Gel_Velos_13_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Lung_bRP_Velos_12_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Lung_bRP_Velos_12_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Monocytes_Gel_Velos_32_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Monocytes_Gel_Velos_32_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Monocytes_bRP_Elite_33_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Monocytes_bRP_Elite_33_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Monocytes_bRP_Velos_31_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Monocytes_bRP_Velos_31_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_NKcells_Gel_Elite_78_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_NKcells_Gel_Elite_78_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_NKcells_Gel_Velos_47_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_NKcells_Gel_Velos_47_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Ovary_Gel_Elite_58_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Ovary_Gel_Elite_58_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Ovary_bRP_Elite_57_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Ovary_bRP_Elite_57_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Pancreas_Gel_Elite_60_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Pancreas_Gel_Elite_60_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Platelets_Gel_Velos_36_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Platelets_Gel_Velos_36_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Platelets_bRP_Velos_35_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Platelets_bRP_Velos_35_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Prostate_Gel_Elite_62_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Prostate_Gel_Elite_62_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Prostate_bRP_Elite_61_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Prostate_bRP_Elite_61_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Rectum_Gel_Elite_63_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Rectum_Gel_Elite_63_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Rectum_bRP_Elite_84_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Rectum_bRP_Elite_84_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Retina_Gel_Elite_65_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Retina_Gel_Elite_65_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Retina_Gel_Velos_5_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Retina_Gel_Velos_5_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Retina_bRP_Elite_64_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Retina_bRP_Elite_64_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Spinalcord_Gel_Elite_67_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Spinalcord_Gel_Elite_67_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Spinalcord_bRP_Elite_66_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Spinalcord_bRP_Elite_66_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Testis_Gel_Elite_69_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Testis_Gel_Elite_69_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Testis_bRP_Elite_68_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Testis_bRP_Elite_68_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Urinarybladder_Gel_Elite_40_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Urinarybladder_Gel_Elite_40_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Urinarybladder_bRP_Elite_71_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Adult_Urinarybladder_bRP_Elite_71_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Brain_bRP_Elite_15_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Fetal_Brain_bRP_Elite_15_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Gut_bRP_Elite_17_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Fetal_Gut_bRP_Elite_17_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Gut_bRP_Elite_18_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Fetal_Gut_bRP_Elite_18_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome}
    }

    TRAIN_EXPERIMENTS_DATA_5_4 = TRAIN_EXPERIMENTS_DATA_5_3 = TRAIN_EXPERIMENTS_DATA_5_2 = TRAIN_EXPERIMENTS_DATA_5_1 = TRAIN_EXPERIMENTS_DATA_5_0

    TRAIN_EXPERIMENTS_DATA_6_3 = TRAIN_EXPERIMENTS_DATA_6_2 = TRAIN_EXPERIMENTS_DATA_6_1 = TRAIN_EXPERIMENTS_DATA_6_0 = TRAIN_EXPERIMENTS_DATA_5_4





    TEST_EXPERIMENTS_DATA_2_0 = {
        "Adult_Heart_bRP_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_heart_brp_velos_crux_q_0.01.pkl", "filesList": None, "constructor" : HumanProteome}
    }

    TEST_EXPERIMENTS_DATA_2_1 = {
        "Adult_Heart_bRP_Velos_q_lt_0.001_identifications.tsv" : {"peaksFile" : "adult_heart_brp_velos_crux_q_0.001.pkl", "filesList": None, "constructor" : HumanProteome}
    }

    TEST_EXPERIMENTS_DATA_3_0 = {
        "Adult_Heart_bRP_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_heart_brp_velos_crux_q_0.01.pkl", "filesList": None, "constructor" : HumanProteome},

        "Adult_NKcells_bRP_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_nkcells_brp_elite_crux_q_0.01.pkl", "filesList" : None, "constructor" : HumanProteome}
    }

    TEST_EXPERIMENTS_DATA_3_1 = {
        "Adult_Heart_bRP_Velos_q_lt_0.001_identifications.tsv" : {"peaksFile" : "adult_heart_brp_velos_crux_q_0.001.pkl", "filesList": None, "constructor" : HumanProteome},

        "Adult_NKcells_bRP_Elite_q_lt_0.001_identifications.tsv" : {"peaksFile" : "adult_nkcells_brp_elite_crux_q_0.001.pkl", "filesList" : None, "constructor" : HumanProteome}
    }

    TEST_EXPERIMENTS_DATA_4_0 = {
        "Adult_Heart_bRP_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_heart_brp_velos_crux_q_0.01_pvalue{}.pkl", "filesList": None, "constructor" : HumanProteome},

        "Adult_NKcells_bRP_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_nkcells_brp_elite_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome}
    }

    TEST_EXPERIMENTS_DATA_4_6 = TEST_EXPERIMENTS_DATA_4_5 = TEST_EXPERIMENTS_DATA_4_4 = TEST_EXPERIMENTS_DATA_4_3 = TEST_EXPERIMENTS_DATA_4_2 = TEST_EXPERIMENTS_DATA_4_1 = TEST_EXPERIMENTS_DATA_4_0


    TEST_EXPERIMENTS_DATA_5_0 = {
        "Adult_Heart_bRP_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_heart_brp_velos_crux_q_0.01_pvalue{}.pkl", "filesList": None, "constructor" : HumanProteome},

        "Adult_NKcells_bRP_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_nkcells_brp_elite_crux_q_0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},

        "Fetal_Heart_Gel_Velos_21_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Fetal_Heart_Gel_Velos_21_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Heart_Gel_Velos_73_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Fetal_Heart_Gel_Velos_73_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Heart_bRP_Elite_19_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Fetal_Heart_bRP_Elite_19_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Heart_bRP_Elite_20_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Fetal_Heart_bRP_Elite_20_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Ovary_Gel_Velos_74_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Fetal_Ovary_Gel_Velos_74_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Placenta_Gel_Velos_14_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Fetal_Placenta_Gel_Velos_14_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Placenta_bRP_Elite_79_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Fetal_Placenta_bRP_Elite_79_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Testis_Gel_Velos_27_q_lt_0.01_identifications.tsv" : {"peaksFile" : "Fetal_Testis_Gel_Velos_27_crux_q0.01_pvalue{}.pkl", "filesList" : None, "constructor" : HumanProteome}
    }


    TEST_EXPERIMENTS_DATA_5_4 = TEST_EXPERIMENTS_DATA_5_3 = TEST_EXPERIMENTS_DATA_5_2 = TEST_EXPERIMENTS_DATA_5_1 = TEST_EXPERIMENTS_DATA_5_0

    TEST_EXPERIMENTS_DATA_6_3 = TEST_EXPERIMENTS_DATA_6_2 = TEST_EXPERIMENTS_DATA_6_1 = TEST_EXPERIMENTS_DATA_6_0 = TEST_EXPERIMENTS_DATA_5_4




    def __init__(self, dataDirectory = 'data/mixedSpectraCrux', split = 'train', 
                 batch_size = 100, nb_threads = 1, trainingDataset = None):

        self.split = split
        self.nb_threads = nb_threads
        self.batch_size = batch_size
        self.dataDirectory = dataDirectory

        if trainingDataset:
            self.trainingDataset = trainingDataset.totalSpectra
        else:
            self.trainingDataset = None

        currentDirectory = os.getcwd()

        print('Working directory: ' + os.getcwd())

        # try:
        #     currentDirectory.index(dataDirectory)
        # except Exception:
        #     os.chdir(dataDirectory)


        #
        # Check if the provided dataset version exists
        #

        trainVersion = Options().get("dataset.train_set_version", MixedSpectraCrux.CURRENT_TRAIN_VERSION)
        testVersion = Options().get("dataset.eval_set_version", MixedSpectraCrux.CURRENT_TEST_VERSION)

        trainExperimentsName = MixedSpectraCrux.TRAIN_EXPERIMENTS_NAME_FORMAT.format("_".join(trainVersion.replace("v", "").split(".")))
        testExperimentsName = MixedSpectraCrux.TEST_EXPERIMENTS_NAME_FORMAT.format("_".join(testVersion.replace("v", "").split(".")))

        if not hasattr(MixedSpectraCrux, trainExperimentsName) or not hasattr(MixedSpectraCrux, testExperimentsName):
            raise ValueError("There is no test dataset {} or train dataset {}.".format(trainVersion, testVersion))


        #
        # Now, process the defined experiments
        #

        if split == 'train':
            peaksFile = MixedSpectraCrux.TRAIN_FILENAME.format(trainVersion)
            experimentsData = getattr(MixedSpectraCrux, trainExperimentsName)
        else:
            peaksFile = MixedSpectraCrux.TEST_FILENAME.format(testVersion)
            experimentsData = getattr(MixedSpectraCrux, testExperimentsName)

        peaksFilesFolder = os.path.join(self.dataDirectory, 'sequences')

        self.totalSpectra = SpectraFound(False, peaksFilesFolder)
        self.totalSpectra.load_spectra(peaksFile)

        if not self.totalSpectra.spectra:

            print("*** Need to create the {} dataset".format(split))

            if split != 'train' and not self.trainingDataset:
                print("***** Need to load training dataset to get normalization parameters")

                trainingPeaksFile = MixedSpectraCrux.TRAIN_FILENAME.format(Options().get("dataset.train_set_version", MixedSpectraCrux.CURRENT_TRAIN_VERSION))

                self.trainingDataset = SpectraFound(False, peaksFilesFolder)
                self.trainingDataset.load_spectra(trainingPeaksFile)

                if not self.trainingDataset.spectra:
                    raise ValueError("Missing training dataset to get normalization parameters !!!")


            # Make sure each experiment peaks file exists

            for experiment in experimentsData.keys():

                print("== Loading experiment {}...".format(experiment))

                spectraPeaksFilename = experimentsData[experiment]["peaksFile"]

                # Check if need to add the maxPvalue threshold to the peaks filename

                maxPvalue = Options().get("dataset.max_pvalue", None)

                if maxPvalue:
                    spectraPeaksFilename = spectraPeaksFilename.format(maxPvalue)

                    print("Changing peaksfilename from {} to {}".format(experimentsData[experiment]["peaksFile"], spectraPeaksFilename))

                newExperiment = experimentsData[experiment]["constructor"](dataDirectory = dataDirectory,
                                                                           split = split,
                                                                           identificationsFilename = experiment, 
                                                                           spectraFilename = spectraPeaksFilename,
                                                                           filesList = experimentsData[experiment]["filesList"],
                                                                           normalizeData = False,
                                                                           storeUnrecognized = False,
                                                                           cruxIdentifications = True)

                del newExperiment

                self.totalSpectra.merge_spectra(self.totalSpectra, peaksFilesFolder, spectraPeaksFilename)

                self.totalSpectra.save_spectra(peaksFile, True)


            # Now, analyze the sequences
            self.totalSpectra.list_single_and_multiple_scans_sequences()


            # And finally normalize the data

            if self.trainingDataset:
                self.totalSpectra.normalize_data(self.trainingDataset.normalizationParameters)
            else:
                self.totalSpectra.normalize_data()


            # Check if needs to discretize the data

            if Options().get("dataset.discretize", False):
                if self.trainingDataset:
                    self.totalSpectra.discretize_data(self.trainingDataset.discretizationParameters)
                else:
                    self.totalSpectra.discretize_data()


            # Save the entire data
            self.totalSpectra.save_spectra(peaksFile, True)


        Logger()("Dataset statistics ({}):".format(split))
        Logger()('- # of singleScanSequences: {}, # of multipleScansSequences: {}'.format(len(self.totalSpectra.singleScanSequences), 
                                                                                          len(self.totalSpectra.multipleScansSequences)))
        Logger()('- Total number of spectra: {}'.format(self.totalSpectra.spectraCount))

        numberOfSequences = len(self.totalSpectra.multipleScansSequences)

        examplesPerSequence = 2

        if Options().get("dataset.include_negative", False):
            examplesPerSequence = 3

        self.numberOfBatches = (numberOfSequences * examplesPerSequence) // self.batch_size

        if (numberOfSequences * examplesPerSequence) % self.batch_size != 0:
            self.numberOfBatches += 1

        print('=============> Initial number of batches: {}'.format(self.numberOfBatches))

        #
        # Make sure the initial working directory remains the same, to avoid breaking the
        # framework.
        #

        os.chdir(currentDirectory)



    def __getitem__(self, index):

        # print('********************* __getitem__: {}, index: {}'.format(self.batchSampler, index))
        # print('********************* __getitem__: epoch: {}'.format(id(self.batchSampler.epoch)))

        item = {}
        item['peaks'] = self.batchSampler.epoch[index]
        item['peaksLen'] = self.batchSampler.peaksLen[index]
        item['pepmass'] = self.batchSampler.pepmass[index]
        item['epoch_data'] = self.batchSampler.epoch_data[index]

        return item


    def __len__(self):

        print('--------------->>>> number of batches: {}'.format(self.numberOfBatches))

        return self.numberOfBatches


    def make_batch_loader(self):

        self.batchSampler = BatchLoader(self.totalSpectra, self.batch_size, dataDumpFolder = self.dataDirectory)

        self.numberOfBatches = len(self.batchSampler.epoch) // self.batch_size

        if len(self.batchSampler.epoch) % self.batch_size != 0:
            self.numberOfBatches += 1

        print('=============> Updated number of batches: {}'.format(self.numberOfBatches))


        print('********************* make_batch_loader: {}'.format(self.batchSampler))

        data_loader = data.DataLoader(self,
            num_workers = self.nb_threads,
            batch_sampler = self.batchSampler,
            drop_last = False)
        return data_loader


