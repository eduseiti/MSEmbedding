import os
import pandas as pd
import json
import time
import pickle

import datetime
import math

import numpy as np
import re

import torch

from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger




class Scan:
    
    UNRECOGNIZED_SEQUENCE = "unrecognized"
    
    def __init__(self, filenamePrefix, maxDigitsSequence):
        
        self.scan          = 0
        self.retentionTime = 0
        self.pepmass       = []
        self.charge        = ''
        
        self.peaks = []
        
        self.maxDigitsSequence = maxDigitsSequence

        self.filenamePrefix = filenamePrefix
        
        
    def to_dict(self, saveFilename):
        
        if saveFilename:
            result = {'filename' : self.filenamePrefix + str(self.scan).zfill(self.maxDigitsSequence) + '.pkl'}
        else:
            result = {}
        
        # result['peaks'] = np.array(self.peaks)
        tmpResult = np.array(self.peaks, dtype = np.float32)

        result['nzero_peaks'] = torch.from_numpy(tmpResult[tmpResult[:, 1] > 0])
        
        result['pepmass'] = self.pepmass
        result['charge']  = self.charge
        result['scan'] = self.scan
        
        return result
        
        
        
class SpectraFound:

    PERCENTILES_TO_CALCULATE = (1.0, 5.0, 10.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.99)
    LIMIT_PERCENTILE_INDEX = -2

    BIN_WIDTH = 1.000508
    NUM_OF_BINS = 2000


    def __init__(self, saveFiles, filesFolder):
                
        #
        # Dictionary to hold the spectrum found
        #
        # {<sequence>:[{'filename' : <filename>,
        #               'peaks' : [[<mz>, <intensity>], ...],
        #               'nzero_peaks' : [[<mz>, <intensity>], ...],
        #               'pepmass' : [<value>],
        #               'charge' : <value>}, ...]
        #
        #
        # Entry "<sequence> == UNRECOGNIZED_SEQUENCE" holds all the unrecognized spectra.
        #
        
        self.spectra = {}
        self.saveFiles = saveFiles
        
        if filesFolder != None:
            self.filesFolder = filesFolder
        else:
            self.filesFolder = os.getcwd()
        
        
    def add_scan(self, whichScan, whichSequence):
                
        sequenceDict = whichScan.to_dict(self.saveFiles)

        # print('- add_scan. sequence={}, scan={}, # non-zero peaks={}'.format(whichSequence, 
        #     whichScan.scan,
        #     len(sequenceDict['nzero_peaks'])))
        
        if whichSequence in self.spectra:
            self.spectra[whichSequence].append(sequenceDict)
        else:
            self.spectra[whichSequence] = [sequenceDict]

        #
        # Does not save each individual file, even if saveFiles == True;
        #
        # if self.saveFiles:
        #     sequenceFolder = os.path.join(self.filesFolder, whichSequence)
            
        #     if os.path.isdir(sequenceFolder) == False:
        #         os.makedirs(sequenceFolder)
               
        #     with open(os.path.join(sequenceFolder, sequenceDict['filename']), 'w') as outputFile:
        #         pickle.dump(sequenceDict, outputFile, pickle.HIGHEST_PROTOCOL)



    def list_single_and_multiple_scans_sequences(self):

        self.maxPeaksListLen = 0
        totalLen = 0

        sequenceMaxLen = ''

        maxScansInSequence = 0
        sequencesWithMultipleScans = 0
        sequencesWithSingleScan = 0

        self.spectraCount = 0
        self.multipleScansSequences = []
        self.singleScanSequences = []

        for key in self.spectra.keys():
            
            if key != Scan.UNRECOGNIZED_SEQUENCE:
                scansLen = len(self.spectra[key])

                if scansLen > 1:
                    sequencesWithMultipleScans += 1
                    self.multipleScansSequences.append(key)
                else:
                    sequencesWithSingleScan += 1    
                    self.singleScanSequences.append(key)

                if key != scansLen > maxScansInSequence:
                    maxScansInSequence = scansLen
            
            for spectrum in self.spectra[key]:
                
                # spectrumLen = len(spectrum['peaks'][spectrum['peaks'][:,1]>0])
                spectrumLen = len(spectrum['nzero_peaks'])
                
                totalLen += spectrumLen
                self.spectraCount += 1
                
                if spectrumLen > self.maxPeaksListLen:
                    self.maxPeaksListLen = spectrumLen
                    sequenceMaxLen = key

        Logger()('Maximum non-zero peaks list len = {}. key = {}'.format(self.maxPeaksListLen, sequenceMaxLen))
        Logger()('Average peaks list len = {}'.format(float('NaN') if self.spectraCount == 0 else totalLen / self.spectraCount))
        Logger()('Number of sequences with a single scan = {}'.format(sequencesWithSingleScan))
        Logger()('Number of sequences with more than 1 scan = {}'.format(sequencesWithMultipleScans))
        Logger()('Max number of scans in a single sequence = {}'.format(maxScansInSequence))
        Logger()('Total number of spectra in this dataset = {}'.format(self.spectraCount))



    def load_spectra(self, spectraName):
    
        print("Loading spectra file {}".format(spectraName))

        self.multipleScansSequences = None
        self.singleScanSequences = None
        self.normalizationParameters = None
        self.spectraCount = 0

        try:
            with open(os.path.join(self.filesFolder, spectraName), 'rb') as inputFile:
                entireData = pickle.load(inputFile)

            # Check if the data is already saved in the complete format

            if 'spectra' in entireData.keys():
                self.spectra = entireData['spectra']
                self.multipleScansSequences = entireData['multipleScansSequences']
                self.singleScanSequences = entireData['singleScanSequences']
                self.normalizationParameters = entireData['normalizationParameters']
                self.spectraCount = entireData['spectraCount']

                if 'discretizationParameters' in entireData:                
                    self.discretizationParameters = entireData['discretizationParameters']
            else:
                self.spectra = entireData



        except Exception:
            print('Could not open spectra file {}'.format(os.path.join(self.filesFolder, spectraName)))
                    


    def merge_spectra(self, destinationSpectra, spectraToMergeFolder, spectraToMergeFilename):

        print("\nmerge_spectra. spectraToMergeFolder={}, spectraToMergeFilename={}".format(spectraToMergeFolder, spectraToMergeFilename))

        spectraToMerge = SpectraFound(False, spectraToMergeFolder)
        spectraToMerge.load_spectra(spectraToMergeFilename)

        numberOfExistingSequences = 0
        numberOfSpectraFromExistingSequences = 0
        totalNumberOfSpectraMerged = 0

        for whichSequence in spectraToMerge.spectra.keys():      
            if whichSequence in destinationSpectra.spectra:

                destinationSpectra.spectra[whichSequence] += spectraToMerge.spectra[whichSequence]

                numberOfExistingSequences += 1
                numberOfSpectraFromExistingSequences += len(spectraToMerge.spectra[whichSequence])
            else:
                destinationSpectra.spectra[whichSequence] = spectraToMerge.spectra[whichSequence]

            totalNumberOfSpectraMerged += len(spectraToMerge.spectra[whichSequence])

        del spectraToMerge

        Logger()(">> Total # spectra merged: {}. # of existing sequences: {}. # of spectra from existing sequences: {}".format(totalNumberOfSpectraMerged,
                                                                                                                               numberOfExistingSequences,
                                                                                                                               numberOfSpectraFromExistingSequences))



    def normalize_data(self, normalizationParameters = None):

        if not normalizationParameters:

            #
            # Need to calculate the normalization parameters
            #

            # Sum everything to calculate the mean

            mzSum = 0.0
            intensitySum = 0.0
            pepmassSum = 0.0

            totalNonZeroPeaks = 0

            winsorizing_index = Options().get("dataset.winsorizing_index", SpectraFound.LIMIT_PERCENTILE_INDEX)

            all_intensities = []

            print("\nStart to calculate the intensities distribution for Winsorizing...")

            for key in self.multipleScansSequences + self.singleScanSequences:
                for peaksList in self.spectra[key]:
                    all_intensities.append(peaksList['nzero_peaks'][:, 1])


            all_intensities = torch.cat(all_intensities)

            print("all_intensities.shape={}".format(all_intensities.shape))


            percentiles = np.percentile(all_intensities, SpectraFound.PERCENTILES_TO_CALCULATE)

            for i in range(len(SpectraFound.PERCENTILES_TO_CALCULATE)):
                print("Percentile={}, intensity={}".format(SpectraFound.PERCENTILES_TO_CALCULATE[i], percentiles[i]))


            print("\nStart to calculate the mzMean, intensityMean, and pepmassMean applying the intensity limit of={}...".format(percentiles[winsorizing_index]))

            for key in self.multipleScansSequences + self.singleScanSequences:
                for peaksList in self.spectra[key]:
                    if winsorizing_index != 0:
                        #
                        # First of all, limit the intensity value
                        #

                        peaksList['nzero_peaks'][:, 1][peaksList['nzero_peaks'][:, 1] > percentiles[winsorizing_index]] = percentiles[winsorizing_index]

                    #
                    # Now, calculate the sums
                    #

                    mzSum += peaksList['nzero_peaks'][:, 0].sum()
                    intensitySum += peaksList['nzero_peaks'][:, 1].sum()
                    totalNonZeroPeaks += len(peaksList['nzero_peaks'])

                    pepmassSum += peaksList['pepmass'][0]


            self.normalizationParameters = {}

            mzMean = mzSum / totalNonZeroPeaks
            intensityMean = intensitySum / totalNonZeroPeaks
            pepmassMean = pepmassSum / totalNonZeroPeaks


            print("Start to calculate the mzStd, intensityStd, and pepmassStd...")

            mzSquaredMeanReducedSum = 0.0
            intensitySquaredMeanReducedSum = 0.0
            pepmassSquaredMeanReducedSum = 0.0

            for key in self.multipleScansSequences + self.singleScanSequences:
                for peaksList in self.spectra[key]:
                    mzSquaredMeanReducedSum += torch.pow(peaksList['nzero_peaks'][:, 0] - mzMean, 2).sum()
                    intensitySquaredMeanReducedSum += torch.pow(peaksList['nzero_peaks'][:, 1] - intensityMean, 2).sum()

                    pepmassSquaredMeanReducedSum += (peaksList['pepmass'][0] - pepmassMean) ** 2

            mzStd = math.sqrt(mzSquaredMeanReducedSum / totalNonZeroPeaks)
            intensityStd = math.sqrt(intensitySquaredMeanReducedSum / totalNonZeroPeaks)
            pepmassStd = math.sqrt(pepmassSquaredMeanReducedSum / totalNonZeroPeaks)

            self.normalizationParameters['mz_mean'] = mzMean
            self.normalizationParameters['mz_std'] = mzStd
            self.normalizationParameters['intensity_mean'] = intensityMean
            self.normalizationParameters['intensity_std']  = intensityStd
            self.normalizationParameters['intensity_percentiles'] = list(zip(SpectraFound.PERCENTILES_TO_CALCULATE, percentiles))
            self.normalizationParameters['winsorizing_index'] = winsorizing_index
            self.normalizationParameters['pepmass_mean'] = pepmassMean
            self.normalizationParameters['pepmass_std']  = pepmassStd

            Logger()('mz mean: {}, mz std: {}'.format(self.normalizationParameters['mz_mean'], self.normalizationParameters['mz_std']))
            Logger()('intensity mean: {}, intensity std: {}'.format(self.normalizationParameters['intensity_mean'], self.normalizationParameters['intensity_std']))
            Logger()('intensities percentiles: {}'.format(self.normalizationParameters['intensity_percentiles']))
            Logger()('pepmass mean: {}, pepmass std: {}'.format(self.normalizationParameters['pepmass_mean'], self.normalizationParameters['pepmass_std']))

        else:
            self.normalizationParameters = normalizationParameters

            Logger()("\nWill apply the following normalization parameters, from training dataset")
            Logger()('mz mean: {}, mz std: {}'.format(self.normalizationParameters['mz_mean'], self.normalizationParameters['mz_std']))
            Logger()('intensity mean: {}, intensity std: {}'.format(self.normalizationParameters['intensity_mean'], self.normalizationParameters['intensity_std']))
            Logger()('intensities percentiles: {}'.format(self.normalizationParameters['intensity_percentiles']))
            Logger()('pepmass mean: {}, pepmass std: {}'.format(self.normalizationParameters['pepmass_mean'], self.normalizationParameters['pepmass_std']))
            

        #
        # Now, normalize the data
        #

        print("Now, normalize the entire spectra (excluding unrecognized scans).")

        apply_winsorizing = ('winsorizing_index' in self.normalizationParameters) and (self.normalizationParameters['winsorizing_index'] != 0)

        for key in self.multipleScansSequences + self.singleScanSequences:
            for peaksList in self.spectra[key]:
                if apply_winsorizing:
                    #
                    # First of all, limit the intensity value
                    #

                    peaksList['nzero_peaks'][:, 1][peaksList['nzero_peaks'][:, 1] > self.normalizationParameters['intensity_percentiles'][self.normalizationParameters['winsorizing_index']][1]] = self.normalizationParameters['intensity_percentiles'][self.normalizationParameters['winsorizing_index']][1]

                peaksList['nzero_peaks'][:, 0] = (peaksList['nzero_peaks'][:, 0] - self.normalizationParameters['mz_mean']) / self.normalizationParameters['mz_std']
                peaksList['nzero_peaks'][:, 1] = (peaksList['nzero_peaks'][:, 1] - self.normalizationParameters['intensity_mean']) / self.normalizationParameters['intensity_std']
                peaksList['pepmass'][0] = (peaksList['pepmass'][0] - self.normalizationParameters['pepmass_mean']) / self.normalizationParameters['pepmass_std']



    def discretize_data(self, discretizationParameters = None):

        if discretizationParameters:
            bin_width = discretizationParameters['bin_width']
            num_of_bins = discretizationParameters['num_of_bins']

            self.discretizationParameters = discretizationParameters

            Logger()("Discretize data using the received parameters from training dataset:")
            Logger()("bin_width={}, num_of_bins={}".format(bin_width, num_of_bins))
        else:
            bin_width = SpectraFound.BIN_WIDTH
            num_of_bins = SpectraFound.NUM_OF_BINS

            self.discretizationParameters = {}

            self.discretizationParameters['bin_width'] = bin_width
            self.discretizationParameters['num_of_bins'] = num_of_bins

            Logger()("Discretize data using default parameters:")
            Logger()("bin_width={}, num_of_bins={}".format(bin_width, num_of_bins))


        for key in self.multipleScansSequences + self.singleScanSequences:
            for peaksList in self.spectra[key]:
                peaksList['bins'] = torch.from_numpy(np.zeros(num_of_bins, dtype = np.float32))

                original_mz = peaksList['nzero_peaks'][:, 0] * self.normalizationParameters['mz_std'] + self.normalizationParameters['mz_mean']
                bin_indexes_mz = (original_mz / bin_width).long()

                peaksList['bins'][bin_indexes_mz] = peaksList['nzero_peaks'][:, 1][range(peaksList['nzero_peaks'].shape[0])]

                # for i in range(peaksList['nzero_peaks'].shape[0]):
                #     print("original_mz={}".format(original_mz[i]))
                #     print("bin_indexes_mz={}".format(bin_indexes_mz[i]))
                #     print("intensity={}".format(peaksList['nzero_peaks'][:, 1][i]))

                # for i in range(peaksList['bins'].shape[0]):
                #     print("peaksList['bins'][{}]={}".format(i, peaksList['bins'][i]))



    def save_spectra(self, spectraName, overwrite = False):
        
        entireData = {}
        entireData['spectra'] = self.spectra
        entireData['multipleScansSequences'] = self.multipleScansSequences
        entireData['singleScanSequences'] = self.singleScanSequences
        entireData['normalizationParameters'] = self.normalizationParameters
        entireData['spectraCount'] = self.spectraCount

        if hasattr(self, "discretizationParameters"):
            entireData['discretizationParameters'] = self.discretizationParameters

        completeFilename = os.path.join(self.filesFolder, spectraName) 

        if os.path.isdir(self.filesFolder) == False:
            os.makedirs(self.filesFolder)
        else:
            if not overwrite and os.path.exists(completeFilename):
                os.rename(completeFilename, completeFilename + ".bkp_" + str(datetime.datetime.now()))

        with open(completeFilename, 'wb') as outputFile:
            pickle.dump(entireData, outputFile, pickle.HIGHEST_PROTOCOL)






class MGF:

    SCAN_SEQUENCE_MAX_DIGITS = 8
    SCAN_SEQUENCE_OVER_LIMIT = 999999999

    BEGIN_IONS_FIELD  = 0
    TITLE_FIELD       = 1
    RTINSECONDS_FIELD = 2
    PEPMASS_FIELD     = 3
    CHARGE_FIELD      = 4
    PEAKS_FIELD       = 5
    END_IONS_FIELD    = 6
    FIELDS_COUNTER    = 7


    MGF_PARSE = [
        '^BEGIN IONS',
        '^TITLE=.+\.(\d+)\..+',
        '^RTINSECONDS=(\d+\.*\d*)',
        '^PEPMASS=(\d+\.*\d*)',
        '^CHARGE=(\d+\+*)',
        '^(\d+\.\d+)\s(\d+\.\d+)',
        '^END IONS'
    ]    


#
# Reads the scans inside a MGF file and stores in a spectra dictionary. Assumes the scans are in ascending order
# inside the MGF file.
#
# whichFile: MGF file being analyzed.
# scanFilenamePrefix: scan filename prefix, to be optionally added in each scan dictionary.
# searchedScan: scan number being searched inside the file. If "None", all scans found in the file will be stored.
#               If the "searchedScan" is found, it is added into the "speactraFound" dictionary and the function
#               returns.
#
# decodedSequence: sequence corresponding to the scan number being searched. Ignored if "searchedScan" == "None".
# spectraFound: spectra dictionary holding the scans found.
# currentScan: last scan read from the previous file. Is matched against the "searchedScan"
# storeUnrecognized: if True, while searching for a non-null "searchedScan" value, stores in the "spectraFound"
#                    dictionary all the scans smaller than "searchedScan". Must be "True" if "searchedScan" == "None".
#
       
    def read_spectrum(self, whichFile, scanFilenamePrefix, searchedScan, decodedSequence, spectraFound, 
                      currentScan = None,
                      storeUnrecognized = True, useScanIndex = False):

        totalScansAdded = 0

        # scan index starts with 1 in crux identifications
        currentScanIndex = 0

        # Logger()('read_spectrum: file={}, scan={}, sequence={}'.format(whichFile.name, searchedScan, decodedSequence))

        hasFoundScan = False

        #
        # Check if previously found scan isn't also the same for the new sequence
        #

        if currentScan != None:

            if useScanIndex:
                currentScanIndex = currentScan.scan

            if currentScan.scan == searchedScan:                            
                spectraFound.add_scan(currentScan, decodedSequence)

                hasFoundScan = True

            elif currentScan.scan < searchedScan:
                currentScan = None

            else:    
                raise ValueError('Scan {} not found. File={}'.format(searchedScan, whichFile.name))

        newScan = currentScan

        while not hasFoundScan:
            line = whichFile.readline()

            if line != '':
                for i, fieldRegex in enumerate(MGF.MGF_PARSE):

                    parsing = re.match(fieldRegex, line)

                    if (parsing != None):
                        if (i == MGF.PEAKS_FIELD):
                            newScan.peaks.append([float(parsing.group(1)), float(parsing.group(2))])

                        elif (i == MGF.BEGIN_IONS_FIELD):
                            if (newScan != None):
                                raise ValueError('\"END IONS\" statement missing. File={}. Scan={}'.format(whichFile.name,
                                                                                                           json.dump(newScan.to_dict)))

                            newScan = Scan(scanFilenamePrefix, MGF.SCAN_SEQUENCE_MAX_DIGITS)
                            currentScanIndex += 1

                        elif (i == MGF.TITLE_FIELD):
                            if useScanIndex:
                                newScan.scan = currentScanIndex
                            else:
                                newScan.scan = int(parsing.group(1))

                        elif (i == MGF.RTINSECONDS_FIELD):
                            newScan.retentionTime = float(parsing.group(1))

                        elif (i == MGF.PEPMASS_FIELD):
                            newScan.pepmass.append(float(parsing.group(1)))

                        elif (i == MGF.CHARGE_FIELD):
                            newScan.charge = parsing.group(1)

                        elif (i == MGF.END_IONS_FIELD):
                            if (newScan != None):
                                if newScan.scan == searchedScan:                            
                                    spectraFound.add_scan(newScan, decodedSequence)

                                    hasFoundScan = True

                                    totalScansAdded += 1 

                                elif not searchedScan or newScan.scan < searchedScan:
                                    if storeUnrecognized:
                                        spectraFound.add_scan(newScan, Scan.UNRECOGNIZED_SEQUENCE)

                                        totalScansAdded += 1 

                                    newScan = None
                                else:    
                                    raise ValueError('Scan {} not found. File={}'.format(searchedScan, whichFile.name))
                            else:
                                raise ValueError('\"BEGIN IONS\" statement missing. File={}. Scan={}'.format(whichFile.name,
                                                                                                             searchedScan))

                        break

    #                 else:
    #                     raise ValueError('Unexpected line. File={}. Scan={}. Line:\n{}'.format(whichFile.name,
    #                                                                                            searchedScan,
    #                                                                                            line))
            else:
                print('End of file. File={}'.format(whichFile.name))

                newScan = False

                break


        return hasFoundScan, newScan, totalScansAdded