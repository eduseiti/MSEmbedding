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

        self.epoch = {}

        peaksIndex = 0

        #
        # Every 3 peaks corresponds to the following sequence: anchor, positive and negative examples.
        #

        for i, sequence in enumerate(self.totalSpectra.multipleScansSequences):

            positiveExamplesIndexes = random.sample(range(len(self.totalSpectra.spectra[sequence])), k = 2)

            self.epoch[peaksIndex] = positiveExamplesIndexes[0]
            self.epoch[peaksIndex + 1] = positiveExamplesIndexes[1]
            self.epoch[peaksIndex + 2] = negativeExamplesIndexes[i]

            peaksIndex += 3


        print('********************* BatchLoader.createTripletBatch. self.epoch: {}'.format(id(self.epoch)))

        return self.epoch


    def __iter__(self):

        print('********************* BatchLoader.__iter__. self: {}'.format(self))

        #
        # Generate a single list of peaks
        #

        howManyCompleteBatches = len(self.epoch) // self.batchSize

        for batch in range(howManyCompleteBatches):

            print('Batch {}: from {} to {}'.format(batch, batch * self.batchSize, batch * self.batchSize + self.batchSize - 1))

            yield(list(range(batch * self.batchSize, self.batchSize)))

            print('\nWill get next batch')
      
        print('Last batch {}: from {} to {}'.format(batch, howManyCompleteBatches * self.batchSize, howManyCompleteBatches * self.batchSize + self.batchSize - 1))

        yield(list(range(howManyCompleteBatches * self.batchSize, len(self.epoch))))

        

    
    def __len__(self):

        # The total length will be the number of sequences * 3 = anchor, positive and negative examples.

        return len(self.epoch) * 3