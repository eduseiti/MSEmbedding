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
        "Fetal_Testis_bRP_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "fetal_testis_brp_elite_crux_q_0.01.pkl", "filesList" : None, "constructor" : HumanProteome}
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
        "Fetal_Testis_bRP_Elite_q_lt_0.001_identifications.tsv" : {"peaksFile" : "fetal_testis_brp_elite_crux_q_0.001.pkl", "filesList" : None, "constructor" : HumanProteome}
    }





    TEST_EXPERIMENTS_DATA_2_0 = {
        "Adult_Heart_bRP_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_heart_brp_velos_crux_q_0.01.pkl", "filesList": None, "constructor" : HumanProteome}
    }

    TEST_EXPERIMENTS_DATA_2_1 = {
        "Adult_Heart_bRP_Velos_q_lt_0.001_identifications.tsv" : {"peaksFile" : "adult_heart_brp_velos_crux_q_0.001.pkl", "filesList": None, "constructor" : HumanProteome}
    }

    TEST_EXPERIMENTS_DATA_3_0 = {
        "Adult_Heart_bRP_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_heart_brp_velos_crux_q_0.01.pkl", "filesList": None, "constructor" : HumanProteome},

        "Adult_NKcells_bRP_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_nkcells_brp_elite_crux_q_0.01.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Liver_Gel_Velos_q_lt_0.01_identifications.tsv" : {"peaksFile" : "fetal_liver_gel_velos_crux_q_0.01.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Bcells_Gel_Elite_q_lt_0.01_identifications.tsv" : {"peaksFile" : "adult_bcells_gel_elite_crux_q_0.01.pkl", "filesList" : None, "constructor" : HumanProteome}
    }

    TEST_EXPERIMENTS_DATA_3_1 = {
        "Adult_Heart_bRP_Velos_q_lt_0.001_identifications.tsv" : {"peaksFile" : "adult_heart_brp_velos_crux_q_0.001.pkl", "filesList": None, "constructor" : HumanProteome},

        "Adult_NKcells_bRP_Elite_q_lt_0.001_identifications.tsv" : {"peaksFile" : "adult_nkcells_brp_elite_crux_q_0.001.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Fetal_Liver_Gel_Velos_q_lt_0.001_identifications.tsv" : {"peaksFile" : "fetal_liver_gel_velos_crux_q_0.001.pkl", "filesList" : None, "constructor" : HumanProteome},
        "Adult_Bcells_Gel_Elite_q_lt_0.001_identifications.tsv" : {"peaksFile" : "adult_bcells_gel_elite_crux_q_0.001.pkl", "filesList" : None, "constructor" : HumanProteome}
    }




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


