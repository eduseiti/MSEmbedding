import spectra as sp
import random
import torch
import numpy as np


class batchLoader:

    def __init__(self, originalData, randomSeed = 1234):

        self.totalSpectra = originalData

        random.seed(randomSeed)


    def listMultipleScansSequences(self):

        maxLen   = 0
        totalLen = 0

        sequenceMaxLen = ''
        numSpectrum    = 0

        maxScansInSequence        = 0
        sequenceWithMultipleScans = 0

        self.multipleScansSequences = []

        for key in self.totalSpectra.spectra.keys():
            
            if key != sp.scan.UNRECOGNIZED_SEQUENCE:
                scansLen = len(self.totalSpectra.spectra[key])

                if scansLen > 1:
                    sequenceWithMultipleScans += 1
                    self.multipleScansSequences.append(key)

                if key != scansLen > maxScansInSequence:
                    maxScansInSequence = scansLen
            
            for spectrum in self.totalSpectra.spectra[key]:
                
                # spectrumLen = len(spectrum['peaks'][spectrum['peaks'][:,1]>0])
                spectrumLen = len(spectrum['nzero_peaks'])
                
                totalLen += spectrumLen
                numSpectrum += 1
                
                if spectrumLen > maxLen:
                    maxLen = spectrumLen
                    sequenceMaxLen = key

        print('Maximum non-zero peaks list len = {}. key = {}'.format(maxLen, sequenceMaxLen))
        print('Average peaks list len = {}'.format(totalLen / numSpectrum))
        print('Number of sequences with more than 1 scan = {}'.format(sequenceWithMultipleScans))
        print('Max number of scans in a single sequence = {}'.format(maxScansInSequence))


    def createTripletBatch(self):

        random.shuffle(self.multipleScansSequences)

        negativeExamplesIndexes = random.choices(range(len(self.totalSpectra.spectra[sp.scan.UNRECOGNIZED_SEQUENCE])), k = len(self.multipleScansSequences))

        newBatch = {}

        for i, sequence in enumerate(self.multipleScansSequences):

            examples = {}

            positiveExamplesIndexes = random.choices(range(len(self.totalSpectra.spectra[sequence])), k = 2)

            examples['anchor']   = positiveExamplesIndexes[0]
            examples['positive'] = positiveExamplesIndexes[1]
            examples['negative'] = negativeExamplesIndexes[i]

            newBatch[sequence] = examples

        return newBatch


    def groupTriplets(self, currentBatch):

        groupedBatch = []

        for i, tripletKey in enumerate(currentBatch.keys()):

            triplet = currentBatch[tripletKey]

            sampleTriplet = []

            sampleTriplet.append(self.totalSpectra.spectra[tripletKey][triplet['anchor']]['nzero_peaks'].cuda())
            sampleTriplet.append(self.totalSpectra.spectra[tripletKey][triplet['positive']]['nzero_peaks'].cuda())
            sampleTriplet.append(self.totalSpectra.spectra[sp.scan.UNRECOGNIZED_SEQUENCE][triplet['negative']]['nzero_peaks'].cuda())

            groupedBatch.append(sampleTriplet)

            print('\Grouping sequence {}: {}'.format(i, tripletKey))

        return groupedBatch