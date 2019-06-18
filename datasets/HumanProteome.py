from .spectra import PXD000561
import os
import torch
import torch.utils.data as data
from .BatchLoader import BatchLoader

class HumanProteome(data.Dataset):

    TRAIN_DATASET = 'initialTest2.pkl'

    TEST_DATASET = 'initialTest2_b02.pkl'


    def __init__(self, dataDirectory='data/humanProteome', split='train', batch_size=100, nb_threads=1):

        self.split = split
        self.nb_threads = nb_threads

        currentDirectory = os.getcwd()

        print('Working directory: ' + os.getcwd())

        try:
            currentDirectory.index(dataDirectory)
        except Exception:
            os.chdir(dataDirectory)

        if split == 'train':
            self.dataset = PXD000561(spectraFilename = HumanProteome.TRAIN_DATASET)
        else:
            self.dataset = PXD000561(spectraFilename = HumanProteome.TEST_DATASET)

        self.dataset.load_identifications()

        if not self.dataset.totalSpectra.spectra:
            raise NotImplementedError('Missing implementation to generate the dataset spectra file.')

        self.dataset.totalSpectra.listMultipleScansSequences()



    def __getitem__(self, index):

        # print('********************* __getitem__: {}, index: {}'.format(self.batchSampler, index))
        # print('********************* __getitem__: epoch: {}'.format(id(self.batchSampler.epoch)))

        return self.batchSampler.epoch[index]


    def __len__(self):
        return len(self.dataset.totalSpectra.multipleScansSequences) * 3


    def make_batch_loader(self):

        self.batchSampler = BatchLoader(self.dataset.totalSpectra, 128 * 3)

        print('********************* make_batch_loader: {}'.format(self.batchSampler))

        data_loader = data.DataLoader(self,
            num_workers=self.nb_threads,
            batch_sampler=self.batchSampler,
            drop_last=False)
        return data_loader

