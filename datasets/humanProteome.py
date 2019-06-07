from .spectra import PXD000561
import os
import torch
import torch.utils.data as data

class HumanProteome(data.Dataset):

    TRAIN_DATASET = 'initialTest2.pkl'

    TEST_DATASET = 'initialTest2_b02.pkl'


    def __init__(self, dir_data='data/humanProteome', split='train', batch_size=100, nb_threads=1):

        os.chdir(dir_data)

        if split == 'train':
            self.dataset = s.PXD000561(spectraFilename = TRAIN_DATASET)
        else:
            self.dataset = s.PXD000561(spectraFilename = TEST_DATASET)

        self.dataset.load_identifications()

        if not self.dataset.totalSpectra.spectra:
            raise NotImplementedError('Missing implementation to generate the dataset spectra file.')

        self.data.totalSpectra.listMultipleScansSequences()




    def __getitem__(self, index):

        return item



    def __len__(self):


    def make_batch_loader(self):
        data_loader = data.DataLoader(self,
            batch_size=self.batch_size,
            num_workers=self.nb_threads,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            drop_last=False)
        return data_loader


