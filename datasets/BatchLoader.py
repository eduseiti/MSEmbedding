from .spectra import Scan
import random
import torch
import numpy as np


class BatchLoader:

    def __init__(self, originalData, batchSize, randomSeed = 1234):

        self.totalSpectra = originalData
        self.batchSize = batchSize

        random.seed(randomSeed)


    def createTripletBatch(self):

        random.shuffle(self.totalSpectra.multipleScansSequences)

        negativeExamplesIndexes = random.sample(range(len(self.totalSpectra.spectra[Scan.UNRECOGNIZED_SEQUENCE])), k = len(self.multipleScansSequences))

        self.epoch = {}

        for i, sequence in enumerate(self.multipleScansSequences):

            examples = {}

            positiveExamplesIndexes = random.sample(range(len(self.totalSpectra.spectra[sequence])), k = 2)

            examples['anchor']   = positiveExamplesIndexes[0]
            examples['positive'] = positiveExamplesIndexes[1]
            examples['negative'] = negativeExamplesIndexes[i]

            self.epoch[sequence] = examples

        return self.epoch


    def __iter__(self):

        i = 0

        for tripletKey in currentBatch.keys():

            triplet = self.epoch[tripletKey]

            self.currentBatch[i] = self.totalSpectra.spectra[tripletKey][triplet['anchor']]['nzero_peaks']
            self.currentBatch[i + 1] = self.totalSpectra.spectra[tripletKey][triplet['positive']]['nzero_peaks']
            self.currentBatch[i + 2] = self.totalSpectra.spectra[sp.Scan.UNRECOGNIZED_SEQUENCE][triplet['negative']]['nzero_peaks']

            i += 3

            # print('Grouping sequence {}: {}'.format(i, tripletKey))

            if i % self.batchSize == 0:
                yield list(self.currentBatch.keys())

                del self.currentBatch
                torch.cuda.empty_cache()

                self.currentBatch = {}

                i = 0

                print('\nGetting next batch')

    
    def __len__(self):

        # The total length will be the number of sequences * 3 = anchor, positive and negative examples.

        return len(self.epoch) * 3