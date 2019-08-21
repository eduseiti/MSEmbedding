from .spectra import Scan
import random
import torch
import numpy as np

from bootstrap.lib.logger import Logger

import os
import pickle

class BatchLoader(object):

    PADDING_VALUE_FOR_MASK = -999.999

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

        random.shuffle(self.totalSpectra.singleScanSequences)

        singleScanSequencesCount = len(self.totalSpectra.singleScanSequences)

        print("++++++++ singleScanSequencesCount={}".format(singleScanSequencesCount))

        Logger()('>>> New epoch initial sequences: {}'.format(self.totalSpectra.multipleScansSequences[0:10]))


        peaksList = []
        self.peaksLen = []


        # negativeExamplesIndexes = random.sample(range(len(self.totalSpectra.spectra[Scan.UNRECOGNIZED_SEQUENCE])), 
        #                                         k = len(self.totalSpectra.multipleScansSequences))

        #
        # Every 3 peaks corresponds to the following sequence: anchor, positive and negative examples.
        #

        for i, sequence in enumerate(self.totalSpectra.multipleScansSequences):

            positiveExamplesIndexes = random.sample(range(len(self.totalSpectra.spectra[sequence])), k = 2)

            anchor = self.totalSpectra.spectra[sequence][positiveExamplesIndexes[0]]['nzero_peaks']
            positive = self.totalSpectra.spectra[sequence][positiveExamplesIndexes[1]]['nzero_peaks']

            # negativeExample = random.choices(range(singleScanSequencesCount))[0]


            # print("++++ which negative={}.".format(negativeExample))
            # print("++++ negative seq={}.".format(self.totalSpectra.singleScanSequences[negativeExample]))
            # print("++++ negative peaks={}.".format(self.totalSpectra.spectra[self.totalSpectra.singleScanSequences[negativeExample]][0]['nzero_peaks']))


            negative = self.totalSpectra.spectra[self.totalSpectra.singleScanSequences[i % singleScanSequencesCount]][0]['nzero_peaks']
            # negative = self.totalSpectra.spectra[Scan.UNRECOGNIZED_SEQUENCE][negativeExamplesIndexes[i]]['nzero_peaks']

            peaksList.append(anchor)
            self.peaksLen.append(len(anchor))

            peaksList.append(positive)
            self.peaksLen.append(len(positive))

            peaksList.append(negative)
            self.peaksLen.append(len(negative))


        #
        # Now, pad the sequence still on the original order: the batches will be sorted before going through the LSTM...
        #

        # use a specific padding value to allow creating a mask

        self.epoch = torch.nn.utils.rnn.pad_sequence(peaksList, batch_first = True, padding_value = BatchLoader.PADDING_VALUE_FOR_MASK)

        print('********************* BatchLoader.createTripletBatch. self.epoch len: {}, shape: {}'.format(len(self.epoch), self.epoch.shape))


        #
        # Normalize the epoch data, both m/z and the intensity values
        #

        # obtain a mask to ignore the padding

        paddingMask = (self.epoch != BatchLoader.PADDING_VALUE_FOR_MASK).float()

        if not self.normalizationParameters:

            self.normalizationParameters = {}

            totalNonZeroPeaks = sum(self.peaksLen)

            paddedEpoch = self.epoch * paddingMask

            mzMean = paddedEpoch[:, :, 0].sum() / totalNonZeroPeaks
            intensityMean = paddedEpoch[:, :, 1].sum() / totalNonZeroPeaks

            squaredMeanReduced = torch.pow((paddedEpoch - torch.tensor([mzMean, intensityMean])) * paddingMask, 2)

            mzStd = torch.sqrt(squaredMeanReduced[:, :, 0].sum() / totalNonZeroPeaks)
            intensityStd = torch.sqrt(squaredMeanReduced[:, :, 1].sum() / totalNonZeroPeaks)

            self.normalizationParameters['mz_mean'] = mzMean
            self.normalizationParameters['mz_std'] = mzStd
            self.normalizationParameters['intensity_mean'] = intensityMean
            self.normalizationParameters['intensity_std']  = intensityStd

            Logger()('mz mean: {}, mz std: {}'.format(self.normalizationParameters['mz_mean'], self.normalizationParameters['mz_std']))
            Logger()('intensity mean: {}, intensity std: {}'.format(self.normalizationParameters['intensity_mean'], self.normalizationParameters['intensity_std']))


        self.epoch[:, :, 0] = (self.epoch[:, :, 0] - self.normalizationParameters['mz_mean']) / self.normalizationParameters['mz_std']
        self.epoch[:, :, 1] = (self.epoch[:, :, 1] - self.normalizationParameters['intensity_mean']) / self.normalizationParameters['intensity_std']

        self.epoch = self.epoch * paddingMask


        # if not self.normalizationParameters:

        #     self.normalizationParameters = {}

        #     totalNonZeroPeaks = sum(self.peaksLen)

        #     self.normalizationParameters['mz_mean'] = self.epoch[:, :, 0].mean()
        #     self.normalizationParameters['mz_std'] = self.epoch[:, :, 0].std()

        #     self.normalizationParameters['intensity_mean'] = self.epoch[:, :, 1].mean()
        #     self.normalizationParameters['intensity_std']  = self.epoch[:, :, 1].std()

        #     Logger()('mz mean: {}, mz std: {}'.format(self.normalizationParameters['mz_mean'], self.normalizationParameters['mz_std']))
        #     Logger()('intensity mean: {}, intensity std: {}'.format(self.normalizationParameters['intensity_mean'], self.normalizationParameters['intensity_std']))

        # self.epoch[:, :, 0] = (self.epoch[:, :, 0] - self.normalizationParameters['mz_mean']) / self.normalizationParameters['mz_std']
        # self.epoch[:, :, 1] = (self.epoch[:, :, 1] - self.normalizationParameters['intensity_mean']) / self.normalizationParameters['intensity_std']

        # self.epoch[:, :, 0] = torch.nn.functional.normalize(self.epoch[:, :, 0])
        # self.epoch[:, :, 1] = torch.nn.functional.normalize(self.epoch[:, :, 1])


        BatchLoader.numberOfEpochs += 1

        # if not self.normalizationParameters:
        #     if (BatchLoader.numberOfEpochs >= BatchLoader.SAVE_EPOCH_DATA_FIRST and BatchLoader.numberOfEpochs <= BatchLoader.SAVE_EPOCH_DATA_LAST):
        #         self.dumpData(BatchLoader.numberOfEpochs, self.totalSpectra.multipleScansSequences, self.totalSpectra.singleScanSequences, self.epoch)

        return (self.epoch, self.peaksLen)


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

            batchExamples = list(range(batch * self.batchSize, batch * self.batchSize + self.batchSize))

            # batch = {'peaks' : batchExamples, "peaksLen" : self.peaksLen[batchExamples[0]:len(batchExamples)]}

            yield batchExamples

            print('\nWill get next batch')
      
        print('Last batch {}: from {} to {}; size {}'.format(batch + 1, 
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