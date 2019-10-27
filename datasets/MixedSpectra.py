from .HumanProteome import HumanProteome
from .PXD000561 import PXD000561
from .spectra import SpectraFound

import os
import torch
import torch.utils.data as data

from .BatchLoader import BatchLoader

from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger


class MixedSpectra(data.Dataset):

    CURRENT_TRAIN_VERSION = "v0.1"
    CURRENT_TEST_VERSION = "v0.1"

    TRAIN_FILENAME = "train_mixedSpectra_{}.pkl"
    TEST_FILENAME = "test_mixedSpectra_{}.pkl"

    TRAIN_EXPERIMENTS_DATA = {
        "fetal_brain_gel_velos.csv" : {"peaksFile" : "fetal_brain_gel_velos.pkl", "filesList": None, "constructor" : HumanProteome},
        "fetal_ovary_brp_velos.csv" : {"peaksFile" : "fetal_ovary_brp_velos.pkl", "filesList": None, "constructor" : HumanProteome},
        "fetal_ovary_brp_elite.csv" : {"peaksFile" : "fetal_ovary_brp_elite.pkl", "filesList": None, "constructor" : HumanProteome},
        "adult_adrenalgland_gel_elite.csv" : {"peaksFile" : "adult_adrenalgland_gel_elite.pkl", "filesList": None, "constructor" : HumanProteome},
        "adult_adrenalgland_gel_velos.csv" : {"peaksFile" : "adult_adrenalgland_gel_velos.pkl", "filesList": None, "constructor" : HumanProteome},
        "adult_adrenalgland_bRP_velos.csv" : {"peaksFile" : "adult_adrenalgland_bRP_velos.pkl", "filesList": None, "constructor" : HumanProteome},
        "adult_urinarybladder_gel_elite.csv" : {"peaksFile" : "adult_urinarybladder_gel_elite.pkl", "filesList": None, "constructor" : HumanProteome},
        "adult_platelets_gel_elite.csv" : {"peaksFile" : "adult_platelets_gel_elite.pkl", "filesList": None, "constructor" : HumanProteome}
    }

    TEST_EXPERIMENTS_DATA = {
        "adult_heart_brp_elite.csv" : {"peaksFile" : "adult_heart_brp_elite.pkl", "filesList": ["f02", "f13", "f23"], "constructor" : HumanProteome}
    }

    TEST_EXPERIMENTS_DATA_SMALL = {
        "adult_heart_brp_elite.csv" : {"peaksFile" : "adult_heart_brp_elite.pkl", "filesList": ["f13", "f23"], "constructor" : HumanProteome}
    }



    def __init__(self, dataDirectory = 'data/mixedSpectra', split = 'train', 
                 batch_size = 100, nb_threads = 1, trainingDataset = None):

        self.split = split
        self.nb_threads = nb_threads
        self.batch_size = batch_size
        self.dataDirectory = dataDirectory

        if split != 'train':
            self.trainingDataset = trainingDataset

        currentDirectory = os.getcwd()

        print('Working directory: ' + os.getcwd())

        # try:
        #     currentDirectory.index(dataDirectory)
        # except Exception:
        #     os.chdir(dataDirectory)


        if split == 'train':
            peaksFile = MixedSpectra.TRAIN_FILENAME.format(Options().get("dataset.train_set_version", MixedSpectra.CURRENT_TRAIN_VERSION))
            #peaksFile = MixedSpectra.TEST_FILENAME.format(Options().get("dataset.eval_set_version", MixedSpectra.CURRENT_TRAIN_VERSION))
            experimentsData = MixedSpectra.TRAIN_EXPERIMENTS_DATA
        else:
            testVersion = Options().get("dataset.eval_set_version", MixedSpectra.CURRENT_TEST_VERSION)

            if testVersion == MixedSpectra.CURRENT_TEST_VERSION:
                experimentsData = MixedSpectra.TEST_EXPERIMENTS_DATA
            else:
                experimentsData = MixedSpectra.TEST_EXPERIMENTS_DATA_SMALL

            peaksFile = MixedSpectra.TEST_FILENAME.format(testVersion)

        peaksFilesFolder = os.path.join(self.dataDirectory, 'sequences')

        self.totalSpectra = SpectraFound(False, peaksFilesFolder)
        self.totalSpectra.load_spectra(peaksFile)

        if not self.totalSpectra.spectra:

            print("*** Need to create the {} dataset".format(split))

            if not trainingDataset:
                print("***** Need to load training dataset to get normalization parameters")

                trainingPeaksFile = MixedSpectra.TRAIN_FILENAME.format(Options().get("dataset.train_set_version", MixedSpectra.CURRENT_TRAIN_VERSION))

                trainingDataset = SpectraFound(False, peaksFilesFolder)
                trainingDataset.load_spectra(trainingPeaksFile)

                if not trainingDataset:
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
                                                                                           storeUnrecognized = False)

                del newExperiment

                self.totalSpectra.merge_spectra(self.totalSpectra, peaksFilesFolder, spectraPeaksFilename)

                self.totalSpectra.save_spectra(peaksFile, True)


            # Now, analyze the sequences
            self.totalSpectra.list_single_and_multiple_scans_sequences()

            # And finally normalize the data
            self.totalSpectra.normalize_data(trainingDataset.normalizationParameters)

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

        if self.split != 'train':
            self.batchSampler = BatchLoader(self.totalSpectra, self.batch_size, dataDumpFolder = self.dataDirectory)
        else:
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


