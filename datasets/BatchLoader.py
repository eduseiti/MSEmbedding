from .spectra import Scan
import random
import torch
import numpy as np

from bootstrap.lib.logger import Logger

class BatchLoader(object):

    def __init__(self, originalData, batchSize, trainingDataset = None):

        self.totalSpectra = originalData
        self.batchSize = batchSize

        if trainingDataset:
            self.normalizationParameters = trainingDataset.batchSampler.normalizationParameters
        else:
            self.normalizationParameters = None

        self.createTripletBatch()


    def createTripletBatch(self):

        random.shuffle(self.totalSpectra.multipleScansSequences)

        negativeExamplesIndexes = list(range(len(self.totalSpectra.spectra[Scan.UNRECOGNIZED_SEQUENCE])))
        random.shuffle(negativeExamplesIndexes)


        print('*_*_**_*_*_*_*_*_*_*_*_*_*>>> New epoch initial sequences: {}'.format(self.totalSpectra.multipleScansSequences[0:10]))


        peaksList = []

        #
        # Every 3 peaks corresponds to the following sequence: anchor, positive and negative examples.
        #

        negativeIndexes = 0

        for sequence in self.totalSpectra.multipleScansSequences:

            positiveExamplesIndexes = list(range(len(self.totalSpectra.spectra[sequence])))
            random.shuffle(positiveExamplesIndexes)

            for positiveIndex in positiveExamplesIndexes[1:3]:
                peaksList.append(self.totalSpectra.spectra[sequence][positiveExamplesIndexes[0]]['nzero_peaks'])
                peaksList.append(self.totalSpectra.spectra[sequence][positiveIndex]['nzero_peaks'])
                peaksList.append(self.totalSpectra.spectra[Scan.UNRECOGNIZED_SEQUENCE][negativeExamplesIndexes[negativeIndexes]]['nzero_peaks'])

                # negativeIndexes = (negativeIndexes + 1) % len(self.totalSpectra.spectra[Scan.UNRECOGNIZED_SEQUENCE]) 
                negativeIndexes += 1


        self.epoch = torch.nn.utils.rnn.pad_sequence(peaksList, batch_first = True, padding_value = 0.0)

        print('********************* BatchLoader.createTripletBatch. self.epoch len: {}, shape: {}'.format(len(self.epoch), 
                                                                                                               self.epoch.shape))


        #
        # Normalize the epoch data, both m/z and the intensity values
        #

        if not self.normalizationParameters:

            self.normalizationParameters = {}

            self.normalizationParameters['mz_mean'] = self.epoch[:, :, 0].mean()
            self.normalizationParameters['mz_std'] = self.epoch[:, :, 0].std()

            self.normalizationParameters['intensity_mean'] = self.epoch[:, :, 1].mean()
            self.normalizationParameters['intensity_std']  = self.epoch[:, :, 1].std()

            Logger()('mz mean: {}, mz std: {}'.format(self.normalizationParameters['mz_mean'], self.normalizationParameters['mz_std']))
            Logger()('intensity mean: {}, intensity std: {}'.format(self.normalizationParameters['intensity_mean'], self.normalizationParameters['intensity_std']))

        self.epoch[:, :, 0] = (self.epoch[:, :, 0] - self.normalizationParameters['mz_mean']) / self.normalizationParameters['mz_std']
        self.epoch[:, :, 1] = (self.epoch[:, :, 1] - self.normalizationParameters['intensity_mean']) / self.normalizationParameters['intensity_std']

        return self.epoch


    def __iter__(self):

        print('********************* BatchLoader.__iter__. self: {}'.format(self))

        #
        # Generate a single list of peaks
        #

        howManyCompleteBatches = len(self.epoch) // self.batchSize

        if len(self.epoch) % self.batchSize != 0:
            self.numberOfBatches = howManyCompleteBatches + 1
        else:
            self.numberOfBatches = howManyCompleteBatches

        for batch in range(howManyCompleteBatches):

            print('Batch {}: from {} to {}'.format(batch, batch * self.batchSize, batch * self.batchSize + self.batchSize - 1))

            # print('Batch Indexes: {}'.format(list(range(batch * self.batchSize, batch * self.batchSize + self.batchSize))))

            yield(list(range(batch * self.batchSize, batch * self.batchSize + self.batchSize)))

            print('\nWill get next batch')
      
        print('Last batch {}: from {} to {}; size {}'.format(batch + 1, 
                                                    howManyCompleteBatches * self.batchSize, 
                                                    howManyCompleteBatches * self.batchSize + len(self.epoch) % self.batchSize - 1,
                                                    len(self.epoch) % self.batchSize))

        # print('Batch Indexes: {}'.format(list(range(howManyCompleteBatches * self.batchSize, howManyCompleteBatches * self.batchSize + len(self.epoch) % self.batchSize - 1))))

        yield(list(range(howManyCompleteBatches * self.batchSize, howManyCompleteBatches * self.batchSize+ len(self.epoch) % self.batchSize)))

        

    
    def __len__(self):

        # The total length will be the number of sequences * 3 = anchor, positive and negative examples.

        return self.numberOfBatches