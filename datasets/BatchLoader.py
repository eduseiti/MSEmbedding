from .spectra import Scan
import random
import torch
import numpy as np

from bootstrap.lib.logger import Logger

class BatchLoader(object):

    numberOfEpochs = 0

    SAVE_EPOCH_DATA_FIRST = 28
    SAVE_EPOCH_DATA_LAST = 30

    DATA_DUMP_FILENAME = "data_epoch_{}.pkl"

    def __init__(self, originalData, batchSize, trainingDataset = None, dataDumpFolder = None):

        self.totalSpectra = originalData
        self.batchSize = batchSize

        self.dataDumpFolder = dataDumpFolder

        if trainingDataset:
            self.normalizationParameters = trainingDataset.batchSampler.normalizationParameters
        else:
            self.normalizationParameters = None

        self.createTripletBatch()


    def createTripletBatch(self):

        random.shuffle(self.totalSpectra.multipleScansSequences)

        negativeExamplesIndexes = random.sample(range(len(self.totalSpectra.spectra[Scan.UNRECOGNIZED_SEQUENCE])), 
                                                k = len(self.totalSpectra.multipleScansSequences))


        Logger()('>>> New epoch initial sequences: {}'.format(self.totalSpectra.multipleScansSequences[0:10]))


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

        BatchLoader.numberOfEpochs += 1

        if (BatchLoader.numberOfEpochs >= BatchLoader.SAVE_EPOCH_DATA_FIRST and BatchLoader.numberOfEpochs <= BatchLoader.SAVE_EPOCH_DATA_LAST):
            self.dumpData(BatchLoader.numberOfEpochs, self.totalSpectra.multipleScansSequences, negativeExamplesIndexes, self.epoch)

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


    def dumpData(self, epochNumber, multipleScansSequences, negativeExamplesIndexes, epoch):
        
        if os.path.isdir(self.dataDumpFolder) == False:
            os.makedirs(self.dataDumpFolder)

        with open(os.path.join(self.dataDumpFolder, DATA_DUMP_FILENAME.format(epochNumber)), 'wb') as outputFile:

            dumpedData = {}
            dumpedData['multipleScansSequences'] = multipleScansSequences
            dumpedData['negativeExamplesIndexes'] = negativeExamplesIndexes
            dumpedData['epoch'] = epoch

            pickle.dump(dumpedData, outputFile, pickle.HIGHEST_PROTOCOL)