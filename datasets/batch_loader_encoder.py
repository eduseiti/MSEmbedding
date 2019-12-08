from .spectra import Scan
import random
import torch
import numpy as np

from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options


import os
import pickle

import time


class BatchLoaderEncoder(object):

    MAX_LOADED_BATCHES = 10


    def __init__(self, spectraList, batchSize, dataFolder):

        self.spectraList = spectraList
        self.batchSize = batchSize

        self.dataFolder = dataFolder

        self.currentBatches = {}

        self.currentFilename = ""
        self.currentExperiment = None

        self.currentSpectrumIndex = 0
        self.currentIndexInExperiment = 0
        self.currentBatchesStartingIndex = 0

        self.batchLimits = self.define_batches()

        # Already read the first batch, to avoid racing conditions...

        self.load_batch(0, self.batchSize - 1)



    def define_batches(self):

        batchLimits = []

        howManyCompleteBatches = len(self.spectraList) // self.batchSize
        additionalBatch = False

        if len(self.spectraList) % self.batchSize != 0:
            self.numberOfBatches = howManyCompleteBatches + 1

            additionalBatch = True
        else:
            self.numberOfBatches = howManyCompleteBatches

        for batch in range(howManyCompleteBatches):

            newBatch = {}
            newBatch['firstIndex'] = batch * self.batchSize
            newBatch['lastIndex'] = batch * self.batchSize + self.batchSize - 1

            batchLimits.append(newBatch)


        # Check if there is a last batch

        if additionalBatch:
            print('Last batch {}: from {} to {}; size {}'.format(batch + 1, 
                                                        howManyCompleteBatches * self.batchSize, 
                                                        howManyCompleteBatches * self.batchSize + len(self.spectraList) % self.batchSize - 1,
                                                        len(self.spectraList) % self.batchSize))

            # print('Batch Indexes: {}'.format(list(range(howManyCompleteBatches * self.batchSize, howManyCompleteBatches * self.batchSize + len(self.epoch) % self.batchSize - 1))))


            newBatch = {}
            newBatch['firstIndex'] = howManyCompleteBatches * self.batchSize
            newBatch['lastIndex'] = howManyCompleteBatches * self.batchSize + len(self.spectraList) % self.batchSize - 1

            batchLimits.append(newBatch)

        return batchLimits



    def load_batch(self, firstIndex, lastIndex):
        self.load_batch_internal(firstIndex, lastIndex)

        nextBatchIndex = (lastIndex + 1) // self.batchSize

        for i in range(BatchLoaderEncoder.MAX_LOADED_BATCHES // 2):
            if nextBatchIndex + i < len (self.batchLimits):
                self.load_batch_internal(self.batchLimits[nextBatchIndex + i]['firstIndex'], 
                                         self.batchLimits[nextBatchIndex + i]['lastIndex'])
            else:
                break



    def load_batch_internal(self, firstIndex, lastIndex):

        print('load_batch: from {} to {}. currentSpectrumIndex={}'.format(firstIndex, lastIndex, self.currentSpectrumIndex))

        if firstIndex not in self.currentBatches.keys():
            print("Current experiment file: {}, current index in experiment: {}".format(self.currentFilename, self.currentIndexInExperiment))

            peaksList = []
            peaksLen = []

            if len(self.currentBatches.keys()) == 0:
                self.currentBatchStartingIndex = firstIndex

            while self.currentSpectrumIndex <= lastIndex:
                splittedFilename = self.spectraList[self.currentSpectrumIndex]['filename'].split('_')
                pklFilename = splittedFilename[2] + "_" + splittedFilename[1] + ".pkl"

                if self.currentFilename != pklFilename:
                    self.currentFilename = pklFilename
                    print("Opening new experiment file: {}".format(self.currentFilename))

                    with open(os.path.join(self.dataFolder, self.currentFilename), 'rb') as inputFile:
                        self.currentExperiment = pickle.load(inputFile)

                    self.currentIndexInExperiment = 0

                peaksList.append(self.currentExperiment['spectra']['unrecognized'][self.currentIndexInExperiment]['nzero_peaks'])
                peaksLen.append(len(self.currentExperiment['spectra']['unrecognized'][self.currentIndexInExperiment]['nzero_peaks']))

                self.currentIndexInExperiment += 1
                self.currentSpectrumIndex += 1
            
            newBatch = {}
            newBatch['peaksList'] = torch.nn.utils.rnn.pad_sequence(peaksList, batch_first = True, padding_value = 0.0)
            newBatch['peaksLen'] = peaksLen

            self.currentBatches[firstIndex] = newBatch

            print('********************* BatchLoaderEncoder.load_batch. firstIndex={}, lastIndex={}, shape: {}'.format(firstIndex,
                                                                                                                       lastIndex,
                                                                                                                       self.currentBatches[firstIndex]['peaksList'].shape))

            currentLoadedBatchesStartingIndex = list(self.currentBatches.keys())

            if len(currentLoadedBatchesStartingIndex) > BatchLoaderEncoder.MAX_LOADED_BATCHES:
                currentLoadedBatchesStartingIndex.sort()

                print("===========> Removing batch starting on spectrum={} from memory".format(currentLoadedBatchesStartingIndex[0]))

                self.currentBatchesStartingIndex = currentLoadedBatchesStartingIndex[1]

                del self.currentBatches[currentLoadedBatchesStartingIndex[0]]

        # else:
        #     print("********************* Batch already loaded")


    #
    # index: item index considering the entire epoch, i.e., all spectra in spectraList.
    # 
    # return spectrum, spectrum len
    #

    def getItem(self, index):

        whichLoadedBatch = (index - self.currentBatchesStartingIndex) // self.batchSize
        spectrumIndexWithinBatch = (index - self.currentBatchesStartingIndex) % self.batchSize

        batchKeys = list(self.currentBatches.keys())

        # print("BatchLoadedEncoder.getItem: index={}, batch={}, index in batch={}, loaded={}".format(index, 
        #                                                                                             whichLoadedBatch,
        #                                                                                             spectrumIndexWithinBatch,
        #                                                                                             batchKeys))

        peaksList = self.currentBatches[batchKeys[whichLoadedBatch]]['peaksList'][spectrumIndexWithinBatch]
        peaksLen = self.currentBatches[batchKeys[whichLoadedBatch]]['peaksLen'][spectrumIndexWithinBatch]

        return peaksList, peaksLen
               



    def __iter__(self):

        print('********************* BatchLoader.__iter__. self: {}'.format(self))

        #
        # Generate a single list of peaks
        #

        howManyCompleteBatches = len(self.spectraList) // self.batchSize
        additionalBatch = False

        if len(self.spectraList) % self.batchSize != 0:
            self.numberOfBatches = howManyCompleteBatches + 1

            additionalBatch = True
        else:
            self.numberOfBatches = howManyCompleteBatches

        for batch in range(howManyCompleteBatches):

            print('Batch {}: from {} to {}'.format(batch, batch * self.batchSize, batch * self.batchSize + self.batchSize - 1))

            self.load_batch(batch * self.batchSize, batch * self.batchSize + self.batchSize - 1)

            # print('Batch Indexes: {}'.format(list(range(batch * self.batchSize, batch * self.batchSize + self.batchSize))))

            batchExamples = list(range(batch * self.batchSize, batch * self.batchSize + self.batchSize))

            yield batchExamples


        # Check if there is a last batch

        if additionalBatch:
            print('Last batch {}: from {} to {}; size {}'.format(batch + 1, 
                                                        howManyCompleteBatches * self.batchSize, 
                                                        howManyCompleteBatches * self.batchSize + len(self.spectraList) % self.batchSize - 1,
                                                        len(self.spectraList) % self.batchSize))

            # print('Batch Indexes: {}'.format(list(range(howManyCompleteBatches * self.batchSize, howManyCompleteBatches * self.batchSize + len(self.epoch) % self.batchSize - 1))))


            self.load_batch(howManyCompleteBatches * self.batchSize, howManyCompleteBatches * self.batchSize + len(self.spectraList) % self.batchSize - 1)

            batchExamples = list(range(howManyCompleteBatches * self.batchSize, howManyCompleteBatches * self.batchSize + len(self.spectraList) % self.batchSize))

            yield batchExamples

        

    def __len__(self):

        # The total length will be the number of sequences * 3 = anchor, positive and negative examples.

        return self.numberOfBatches



