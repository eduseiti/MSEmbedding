from .PXD000561 import PXD000561
from .spectra import MGF
import os
import torch
import torch.utils.data as data
from .BatchLoader import BatchLoader

from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger


class HumanProteome(data.Dataset):

    MATCHES_FILE = 'adult_adrenalgland_gel_elite.csv'
    TRAIN_DATASET = 'initialTest2.pkl'
    TEST_DATASET = 'initialTest2_b02.pkl'


    def __init__(self, dataDirectory = 'data/humanProteome', split = 'train', 
                 batch_size = 100, nb_threads = 1, trainingDataset = None, 
                 identificationsFilename = None, spectraFilename = None, normalizeData = True):

        self.split = split
        self.nb_threads = nb_threads
        self.batch_size = batch_size
        self.dataDirectory = dataDirectory

        if split != 'train':
            self.trainingDataset = trainingDataset

        currentDirectory = os.getcwd()

        print('Working directory: ' + os.getcwd())

        try:
            currentDirectory.index(dataDirectory)
        except Exception:
            os.chdir(dataDirectory)

        matchesFile = identificationsFilename
        peaksFile = spectraFilename
        filteredFilesList = None

        if split == 'train':
            if not matchesFile:
                matchesFile = Options().get("dataset.train_matches_file", HumanProteome.MATCHES_FILE)

            if not peaksFile:
                peaksFile = Options().get("dataset.train_peaks_file", HumanProteome.TRAIN_DATASET)

            filteredFilesList = Options().get("dataset.train_filtered_files_list", None)
        else:
            if not matchesFile:
                matchesFile = Options().get("dataset.eval_matches_file", HumanProteome.MATCHES_FILE)

            if not peaksFile:
                peaksFile = Options().get("dataset.eval_peaks_file", HumanProteome.TEST_DATASET)

            filteredFilesList = Options().get("dataset.eval_filtered_files_list", None)

        self.dataset = PXD000561(identificationsFilename = matchesFile, spectraFilename = peaksFile)
        self.dataset.load_identifications(filteredFilesList = filteredFilesList)


        # Check if has processed the spectra yet

        if not self.dataset.totalSpectra.spectra:

            print("+-+-+ Processing the original MGF files")

            # Read the MGF files

            self.dataset.read_spectra(MGF())

            # raise NotImplementedError('Missing implementation to generate the dataset spectra file.')

            # Now, analyze the sequences
            self.dataset.totalSpectra.list_single_and_multiple_scans_sequences()

            if normalizeData:
                # And finally normalize the data
                self.dataset.totalSpectra.normalize_data(trainingDataset)

            # Now, save the entire data
            self.dataset.totalSpectra.save_spectra(self.dataset.spectraFilename)

        elif not self.dataset.totalSpectra.multipleScansSequences:

            print("+-+-+ Processing the spectra to include sequence and normalization information")

            # Now, analyze the sequences
            self.dataset.totalSpectra.list_single_and_multiple_scans_sequences()

            if normalizeData:
                # And finally normalize the data
                self.dataset.totalSpectra.normalize_data(trainingDataset)

            # Now, save the entire data
            self.dataset.totalSpectra.save_spectra(self.dataset.spectraFilename)

        else:

            print("+-+-+ Spectra information complete")

            self.dataset.totalSpectra.multipleScansSequences

            Logger()('# of singleScanSequences: {}, # of multipleScansSequences: {}'.format(len(self.dataset.totalSpectra.singleScanSequences), 
                                                                                            len(self.dataset.totalSpectra.multipleScansSequences)))

            Logger()('mz mean: {}, mz std: {}'.format(self.dataset.totalSpectra.normalizationParameters['mz_mean'], 
                                                      self.dataset.totalSpectra.normalizationParameters['mz_std']))

            Logger()('intensity mean: {}, intensity std: {}'.format(self.dataset.totalSpectra.normalizationParameters['intensity_mean'], 
                                                                    self.dataset.totalSpectra.normalizationParameters['intensity_std']))




        numberOfSequences = len(self.dataset.totalSpectra.multipleScansSequences)

        self.numberOfBatches = (numberOfSequences * 3) // self.batch_size

        if (numberOfSequences * 3) % self.batch_size != 0:
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
            self.batchSampler = BatchLoader(self.dataset.totalSpectra, self.batch_size, dataDumpFolder = self.dataDirectory)
        else:
            self.batchSampler = BatchLoader(self.dataset.totalSpectra, self.batch_size, dataDumpFolder = self.dataDirectory)


        self.numberOfBatches = len(self.batchSampler.epoch) // self.batch_size

        if len(self.batchSampler.epoch) % self.batch_size != 0:
            self.numberOfBatches += 1

        print('=============> Updated number of batches: {}'.format(self.numberOfBatches))


        print('********************* make_batch_loader: {}'.format(self.batchSampler))

        data_loader = data.DataLoader(self,
            num_workers=self.nb_threads,
            batch_sampler=self.batchSampler,
            drop_last=False)
        return data_loader


