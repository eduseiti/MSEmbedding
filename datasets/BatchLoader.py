from .spectra import Scan
import random
import torch
import numpy as np


class BatchLoader(object):

    def __init__(self, originalData, batchSize):

        self.totalSpectra = originalData
        self.batchSize = batchSize

        self.createTripletBatch()


    def createTripletBatch(self):

        random.shuffle(self.totalSpectra.multipleScansSequences)

        negativeExamplesIndexes = random.sample(range(len(self.totalSpectra.spectra[Scan.UNRECOGNIZED_SEQUENCE])), 
                                                k = len(self.totalSpectra.multipleScansSequences))

        peaksList = []

        #
        # Every 3 peaks corresponds to the following sequence: anchor, positive and negative examples.
        #

        for i, sequence in enumerate(self.totalSpectra.multipleScansSequences):

            positiveExamplesIndexes = random.sample(range(len(self.totalSpectra.spectra[sequence])), k = 2)

            peaksList.append(self.totalSpectra.spectra[sequence][positiveExamplesIndexes[0]]['nzero_peaks'])
            peaksList.append(self.totalSpectra.spectra[sequence][positiveExamplesIndexes[1]]['nzero_peaks'])
            peaksList.append(self.totalSpectra.spectra[Scan.UNRECOGNIZED_SEQUENCE][negativeExamplesIndexes[i]]['nzero_peaks'])


        self.epoch = torch.nn.utils.rnn.pad_sequence(peaksList, batch_first = True, padding_value = 0.0)

        print('********************* BatchLoader.createTripletBatch. self.epoch: {}'.format(id(self.epoch)))

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

            yield(list(range(batch * self.batchSize, self.batchSize)))

            print('\nWill get next batch')
      
        print('Last batch {}: from {} to {}; size {}'.format(batch + 1, 
                                                    howManyCompleteBatches * self.batchSize, 
                                                    howManyCompleteBatches * self.batchSize + len(self.epoch) % self.batchSize - 1,
                                                    len(self.epoch) % self.batchSize - 1))

        yield(list(range(howManyCompleteBatches * self.batchSize, len(self.epoch) % self.batchSize - 1)))

        

    
    def __len__(self):

        # The total length will be the number of sequences * 3 = anchor, positive and negative examples.

        return self.numberOfBatches