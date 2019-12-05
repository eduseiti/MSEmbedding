from .spectra import Scan
import random
import torch
import numpy as np

from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options


import os
import pickle

class BatchLoaderEncoder(object):

    numberOfEpochs = 0


    def __init__(self, spectraList, batchSize):

        self.spectraList = spectraList
        self.batchSize = batchSize

        self.currentFilename = ""
        self.currentExperiment = None
        self.currentIndexInExperiment = 0
        self.currentBatchStartingIndex = 0



    def load_batch(self, firstIndex, lastIndex):

        print('load_batch: from {} to {}'.format(firstIndex, lastIndex)
        print("Current experiment file: {}, current index in experiment: {}".format(self.currentFilename, self.currentIndexInExperiment))

        peaksList = []
        self.peaksLen = []
        currentIndexInBatch = 0
        self.currentBatchStartingIndex = firstIndex

        while self.currentIndexInExperiment <= lastIndex:
            if self.currentFilename != self.spectraList[self.currentIndexInExperiment]['filename']:

                self.currentFilename = self.spectraList[self.currentIndexInExperiment]['filename']
                print("Opening new experiment file: {}".format(self.currentFilename))

                with open(self.currentFilename, 'rb') as inputFile:
                    self.currentExperiment = pickle.load(inputFile)

                self.currentIndexInExperiment = 0

            peaksList.append(self.currentExperiment['spectra']['unrecognized'][self.currentIndexInExperiment])
            self.peaksLen.append(len(self.currentExperiment['spectra']['unrecognized'][self.currentIndexInExperiment]))

            currentIndexInBatch += 1
            self.currentIndexInExperiment += 1
        
        self.currentBatch = torch.nn.utils.rnn.pad_sequence(peaksList, batch_first = True, padding_value = 0.0)
        self.currentBatchSize = currentIndexInBatch

        print('********************* BatchLoaderEncoder.load_batch. self.currentBatch len: {}, shape: {}'.format(self.currentBatchSize, self.currentBatch.shape))



    #
    # index: item index considering the entire epoch, i.e., all spectra in spectraList.
    # 
    # return spectrum, spectrum len
    #

    def getItem(self, index):

        return self.currentBatch[index - self.currentBatchStartingIndex], self.peaksLen[index - self.currentBatchStartingIndex]



    def __iter__(self):

        print('********************* BatchLoader.__iter__. self: {}'.format(self))

        #
        # Generate a single list of peaks
        #

        howManyCompleteBatches = len(self.epoch) // self.batchSize
        additionalBatch = False

        if len(self.epoch) % self.batchSize != 0:
            self.numberOfBatches = howManyCompleteBatches + 1

            additionalBatch = True
        else:
            self.numberOfBatches = howManyCompleteBatches

        for batch in range(howManyCompleteBatches):

            print('Batch {}: from {} to {}'.format(batch, batch * self.batchSize, batch * self.batchSize + self.batchSize - 1))

            self.load_batch(batch * self.batchSize, batch * self.batchSize + self.batchSize - 1)

            # print('Batch Indexes: {}'.format(list(range(batch * self.batchSize, batch * self.batchSize + self.batchSize))))

            batchExamples = list(range(batch * self.batchSize, batch * self.batchSize + self.batchSize))

            # batch = {'peaks' : batchExamples, "peaksLen" : self.peaksLen[batchExamples[0]:len(batchExamples)]}

            yield batchExamples


        # Check if there is a last batch

        if additionalBatch:
            print('Last batch {}: from {} to {}; size {}'.format(batch + 1, 
                                                        howManyCompleteBatches * self.batchSize, 
                                                        howManyCompleteBatches * self.batchSize + len(self.epoch) % self.batchSize - 1,
                                                        len(self.epoch) % self.batchSize))

            # print('Batch Indexes: {}'.format(list(range(howManyCompleteBatches * self.batchSize, howManyCompleteBatches * self.batchSize + len(self.epoch) % self.batchSize - 1))))


            self.load_batch(howManyCompleteBatches * self.batchSize, howManyCompleteBatches * self.batchSize + len(self.epoch) % self.batchSize - 1)

            batchExamples = list(range(howManyCompleteBatches * self.batchSize, howManyCompleteBatches * self.batchSize + len(self.epoch) % self.batchSize))

            # batch = {'peaks' : batchExamples, "peaksLen" : self.peaksLen[batchExamples[0]:len(batchExamples)]}

            yield batchExamples

        

    def __len__(self):

        # The total length will be the number of sequences * 3 = anchor, positive and negative examples.

        return self.numberOfBatches



