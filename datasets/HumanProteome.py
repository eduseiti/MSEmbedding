from .spectra import PXD000561
from .spectra import MGF
import os
import torch
import torch.utils.data as data
from .BatchLoader import BatchLoader

from bootstrap.lib.options import Options


class HumanProteome(data.Dataset):

    TRAIN_DATASET = 'initialTest3_b01_b03.pkl'

    TEST_DATASET = 'initialTest2_b02.pkl'


    def __init__(self, dataDirectory='data/humanProteome', split='train', batch_size=100, nb_threads=1):

        self.split = split
        self.nb_threads = nb_threads
        self.batch_size = batch_size

        currentDirectory = os.getcwd()

        print('Working directory: ' + os.getcwd())

        try:
            currentDirectory.index(dataDirectory)
        except Exception:
            os.chdir(dataDirectory)

        if split == 'train':
            self.dataset = PXD000561(spectraFilename = HumanProteome.TRAIN_DATASET)
            self.dataset.load_identifications(filteredFilesList = Options()['dataset']['train_filtered_files_list'])
        else:
            self.dataset = PXD000561(spectraFilename = HumanProteome.TEST_DATASET)
            self.dataset.load_identifications()


        if not self.dataset.totalSpectra.spectra:
            self.dataset.read_spectra(MGF())
            # raise NotImplementedError('Missing implementation to generate the dataset spectra file.')

        self.dataset.totalSpectra.listMultipleScansSequences()

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

        return self.batchSampler.epoch[index]


    def __len__(self):

        print('--------------->>>> number of batches: {}'.format(self.numberOfBatches))

        return self.numberOfBatches


    def make_batch_loader(self):

        self.batchSampler = BatchLoader(self.dataset.totalSpectra, self.batch_size)

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


