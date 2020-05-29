from .spectra import Scan
import random
import torch
import numpy as np

from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options


import os
import pickle

class BatchLoader(object):

    numberOfEpochs = 0

    SAVE_EPOCH_DATA_FIRST = 28
    SAVE_EPOCH_DATA_LAST = 30

    DATA_DUMP_FILENAME = "data_epoch_{}.pkl"

    def __init__(self, originalData, batchSize, dataDumpFolder = None):

        self.totalSpectra = originalData
        self.batchSize = batchSize
        self.dataDumpFolder = dataDumpFolder
        self.includeNegative = Options().get("dataset.include_negative", False)

        if Options().get("dataset.discretize", False):
            self.createTripletBatch_MLP()
        else:
            self.createTripletBatch()



    def createTripletBatch(self):

        random.shuffle(self.totalSpectra.multipleScansSequences)

        random.shuffle(self.totalSpectra.singleScanSequences)

        if self.includeNegative:
            singleScanSequencesCount = len(self.totalSpectra.singleScanSequences)

            print("++++++++ singleScanSequencesCount={}".format(singleScanSequencesCount))

        Logger()('>>> New epoch initial sequences: {}'.format(self.totalSpectra.multipleScansSequences[0:10]))


        peaksList = []
        self.peaksLen = []
        self.pepmass = []


        #
        # When the epoch data needs to be externalized
        # 

        self.epoch_data = []


        #
        # If "includeNegative" enabled, Every 3 peaks corresponds to the following sequence: anchor, positive and negative examples.
        #
        # Otherwise, every 2 peaks corresponds to: anchr, positive example.
        #

        for i, sequence in enumerate(self.totalSpectra.multipleScansSequences):

            positiveExamplesIndexes = random.sample(range(len(self.totalSpectra.spectra[sequence])), k = 2)

            anchor = self.totalSpectra.spectra[sequence][positiveExamplesIndexes[0]]['nzero_peaks']
            peaksList.append(anchor)
            self.peaksLen.append(len(anchor))
            self.pepmass.append(self.totalSpectra.spectra[sequence][positiveExamplesIndexes[0]]['pepmass'][0])
            self.epoch_data.append({"sequence": sequence, "index": positiveExamplesIndexes[0]})

            positive = self.totalSpectra.spectra[sequence][positiveExamplesIndexes[1]]['nzero_peaks']
            peaksList.append(positive)
            self.peaksLen.append(len(positive))
            self.pepmass.append(self.totalSpectra.spectra[sequence][positiveExamplesIndexes[1]]['pepmass'][0])
            self.epoch_data.append({"sequence": sequence, "index": positiveExamplesIndexes[1]})

            if self.includeNegative:
                negative = self.totalSpectra.spectra[self.totalSpectra.singleScanSequences[i % singleScanSequencesCount]][0]['nzero_peaks']
                peaksList.append(negative)
                self.peaksLen.append(len(negative))
                self.pepmass.append(self.totalSpectra.spectra[self.totalSpectra.singleScanSequences[i % singleScanSequencesCount]][0]['pepmass'][0])
                self.epoch_data.append({"sequence": self.totalSpectra.singleScanSequences[i % singleScanSequencesCount], "index": 0})


        self.pepmass = np.array(self.pepmass, dtype = np.float32)

        #
        # Now, pad the sequence still on the original order: the batches will be sorted before going through the LSTM...
        #

        self.epoch = torch.nn.utils.rnn.pad_sequence(peaksList, batch_first = True, padding_value = 0.0)

        print('********************* BatchLoader.createTripletBatch. self.epoch len: {}, shape: {}'.format(len(self.epoch), self.epoch.shape))


        BatchLoader.numberOfEpochs += 1

        # if not self.normalizationParameters:
        #     if (BatchLoader.numberOfEpochs >= BatchLoader.SAVE_EPOCH_DATA_FIRST and BatchLoader.numberOfEpochs <= BatchLoader.SAVE_EPOCH_DATA_LAST):
        #         self.dumpData(BatchLoader.numberOfEpochs, self.totalSpectra.multipleScansSequences, self.totalSpectra.singleScanSequences, self.epoch)

        return (self.epoch, self.peaksLen, self.pepmass)


    def createTripletBatch_MLP(self):

        random.shuffle(self.totalSpectra.multipleScansSequences)

        random.shuffle(self.totalSpectra.singleScanSequences)


        Logger()('>>> createTripletBatch_MLP: New epoch initial sequences: {}'.format(self.totalSpectra.multipleScansSequences[0:10]))


        peaksList = []
        self.peaksLen = []
        self.pepmass = []


        #
        # When the epoch data needs to be externalized
        # 

        self.epoch_data = []


        #
        # If "includeNegative" enabled, Every 3 peaks corresponds to the following sequence: anchor, positive and negative examples.
        #
        # Otherwise, every 2 peaks corresponds to: anchr, positive example.
        #

        for i, sequence in enumerate(self.totalSpectra.multipleScansSequences):

            positiveExamplesIndexes = random.sample(range(len(self.totalSpectra.spectra[sequence])), k = 2)

            anchor = self.totalSpectra.spectra[sequence][positiveExamplesIndexes[0]]['bins']
            peaksList.append(anchor)
            self.peaksLen.append(len(anchor))
            self.pepmass.append(self.totalSpectra.spectra[sequence][positiveExamplesIndexes[0]]['pepmass'][0])
            self.epoch_data.append({"sequence": sequence, "index": positiveExamplesIndexes[0]})

            positive = self.totalSpectra.spectra[sequence][positiveExamplesIndexes[1]]['bins']
            peaksList.append(positive)
            self.peaksLen.append(len(positive))
            self.pepmass.append(self.totalSpectra.spectra[sequence][positiveExamplesIndexes[1]]['pepmass'][0])
            self.epoch_data.append({"sequence": sequence, "index": positiveExamplesIndexes[1]})


        self.pepmass = np.array(self.pepmass, dtype = np.float32)

        self.epoch = peaksList

        print('********************* BatchLoader.createTripletBatch_MLP. self.epoch len: {}, element 0 shape: {}'.format(len(self.epoch), self.epoch[0].shape))


        BatchLoader.numberOfEpochs += 1

        return (self.epoch, self.peaksLen, self.pepmass)



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

            # print('Batch Indexes: {}'.format(list(range(batch * self.batchSize, batch * self.batchSize + self.batchSize))))

            batchExamples = list(range(batch * self.batchSize, batch * self.batchSize + self.batchSize))

            # batch = {'peaks' : batchExamples, "peaksLen" : self.peaksLen[batchExamples[0]:len(batchExamples)]}

            yield batchExamples


        # Check if there is a last batch

        if additionalBatch:
            print('Last batch {}: from {} to {}; size {}'.format(howManyCompleteBatches, 
                                                        howManyCompleteBatches * self.batchSize, 
                                                        howManyCompleteBatches * self.batchSize + len(self.epoch) % self.batchSize - 1,
                                                        len(self.epoch) % self.batchSize))

            # print('Batch Indexes: {}'.format(list(range(howManyCompleteBatches * self.batchSize, howManyCompleteBatches * self.batchSize + len(self.epoch) % self.batchSize - 1))))


            batchExamples = list(range(howManyCompleteBatches * self.batchSize, howManyCompleteBatches * self.batchSize + len(self.epoch) % self.batchSize))

            # batch = {'peaks' : batchExamples, "peaksLen" : self.peaksLen[batchExamples[0]:len(batchExamples)]}

            yield batchExamples

        
    def __len__(self):

        # The total length will be the number of sequences * 3 = anchor, positive and negative examples.

        return self.numberOfBatches


    def dumpData(self, epochNumber, multipleScansSequences, singleScanSequences, epoch):
        
        if os.path.isdir(self.dataDumpFolder) == False:
            os.makedirs(self.dataDumpFolder)

        with open(os.path.join(self.dataDumpFolder, DATA_DUMP_FILENAME.format(epochNumber)), 'wb') as outputFile:

            dumpedData = {}
            dumpedData['multipleScansSequences'] = multipleScansSequences
            dumpedData['singleScanSequences'] = singleScanSequences
            dumpedData['epoch'] = epoch

            pickle.dump(dumpedData, outputFile, pickle.HIGHEST_PROTOCOL)