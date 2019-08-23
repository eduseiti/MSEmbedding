import os
import pandas as pd
import json
import time
import pickle

import datetime

import numpy as np
import re

import torch

from bootstrap.lib.logger import Logger


class PXD000561:

    SCAN_SEQUENCE_MAX_DIGITS = 8
    SCAN_SEQUENCE_OVER_LIMIT = 999999999
    
    SPECTRA_FILES = {
        'b01' : 'Adult_Adrenalgland_Gel_Elite_49_f01.mgf',
        'b02' : 'Adult_Adrenalgland_Gel_Elite_49_f02.mgf',
        'b03' : 'Adult_Adrenalgland_Gel_Elite_49_f03.mgf',
        'b04' : 'Adult_Adrenalgland_Gel_Elite_49_f04.mgf',
        'b05' : 'Adult_Adrenalgland_Gel_Elite_49_f05.mgf',
        'b06' : 'Adult_Adrenalgland_Gel_Elite_49_f06.mgf',
        'b07' : 'Adult_Adrenalgland_Gel_Elite_49_f07.mgf',
        'b08' : 'Adult_Adrenalgland_Gel_Elite_49_f08.mgf',
        'b09' : 'Adult_Adrenalgland_Gel_Elite_49_f09.mgf',
        'b10' : 'Adult_Adrenalgland_Gel_Elite_49_f10.mgf',
        'b11' : 'Adult_Adrenalgland_Gel_Elite_49_f11.mgf',
        'b12' : 'Adult_Adrenalgland_Gel_Elite_49_f12.mgf',
        'b13' : 'Adult_Adrenalgland_Gel_Elite_49_f13.mgf',
        'b14' : 'Adult_Adrenalgland_Gel_Elite_49_f14.mgf',
        'b15' : 'Adult_Adrenalgland_Gel_Elite_49_f15.mgf',
        'b16' : 'Adult_Adrenalgland_Gel_Elite_49_f16.mgf',
        'b17' : 'Adult_Adrenalgland_Gel_Elite_49_f17.mgf',
        'b18' : 'Adult_Adrenalgland_Gel_Elite_49_f18.mgf',
        'b19' : 'Adult_Adrenalgland_Gel_Elite_49_f19.mgf',
        'b20' : 'Adult_Adrenalgland_Gel_Elite_49_f20.mgf',
        'b21' : 'Adult_Adrenalgland_Gel_Elite_49_f21.mgf',
        'b22' : 'Adult_Adrenalgland_Gel_Elite_49_f22.mgf',
        'b23' : 'Adult_Adrenalgland_Gel_Elite_49_f23.mgf',
        'b24' : 'Adult_Adrenalgland_Gel_Elite_49_f24.mgf'
    }

    def __init__(self, identificationsFilename = 'Gel_Elite_49.csv', spectraFilename = None):
    
        self.identificationsFilename = identificationsFilename
        self.spectraFilename = spectraFilename
    
        self.totalSpectra = SpectraFound(False, 'sequences')
    
    
    def load_identifications(self, verbose = False, filteredFilesList = None):

        
        #
        # First check if the spectra has already been loaded
        #
        
        self.totalSpectra.load_spectra(self.spectraFilename)
        
        if self.totalSpectra.spectra: 
            return
        
        print('Loading file: {}. dir:{}'.format(self.identificationsFilename, os.getcwd()))

        gel_elite_49 = pd.read_csv(self.identificationsFilename)

        if verbose:
            print(gel_elite_49)

        print('Number of unique sequences found: {}'.format(len(gel_elite_49['Sequence'].unique())))

        #
        # Inject new columns to hold the scan sequence within the file and the file name, recovered from the 
        # "Spectrum Title" information.
        #

        gel_elite_49['File Sequence'] = \
            gel_elite_49['Spectrum Title'].str.split("_", expand = True).iloc[:, 3].str.zfill(PXD000561.SCAN_SEQUENCE_MAX_DIGITS)
        
        gel_elite_49['File'] = gel_elite_49['Spectrum Title'].str.split("_", expand = True).iloc[:, 2]

        ordered = gel_elite_49.sort_values(['File', 'File Sequence'])


        #
        # Select only unique "Sequence" + "First Scan" combination:
        #
        # - Same spectrum can contain different sequences
        # - Different sequences can be read in different scans

        duplicatesIndication = ordered.duplicated(['Spectrum Title', 'Sequence', 'First Scan'])

        self.uniqueCombination = ordered[~duplicatesIndication]
        print('Unique combinations found: {}'.format(self.uniqueCombination.shape))

        print('Unique combinations file {}: {}'.format('b01', 
                                                       self.uniqueCombination[self.uniqueCombination['File'] == 'b01'].shape))

        if filteredFilesList:
            self.uniqueCombination = self.uniqueCombination[self.uniqueCombination['File'].isin(filteredFilesList)]


    def read_spectra(self, spectraParser):
        currentFileName           = ''
        currentScanFileNamePrefix = ''

        currentFile  = None
        lastScan     = None

        startTime = time.time()

        for index, row in self.uniqueCombination.iterrows():
            if (PXD000561.SPECTRA_FILES[row['File']] != currentFileName):
                if (currentFile != None):

                    #
                    # Dumps the rest of current file as unrecognized spectra
                    #

                    spectraParser.read_spectrum(currentFile, currentFileNamePrefix + '_', 
                                                PXD000561.SCAN_SEQUENCE_OVER_LIMIT, 
                                                '', 
                                                self.totalSpectra)

                    print('File {}. Processing time of {} seconds'.format(currentFile.name, 
                                                                          time.time() - startTime))

                    currentFile.close()

                    # break

                currentFileNamePrefix = row['File']
                currentFileName = PXD000561.SPECTRA_FILES[currentFileNamePrefix]

                print('Will open file \"{}\"'.format(currentFileName))

                currentFile = open(currentFileName, 'r')
                lastScan    = None

            _, lastScan = spectraParser.read_spectrum(currentFile, 
                                                      currentFileNamePrefix + '_', 
                                                      row['First Scan'], 
                                                      row['Sequence'], 
                                                      self.totalSpectra, 
                                                      lastScan)


        print('Total processing time {} seconds'.format(time.time() - startTime))
        
        self.totalSpectra.save_spectra(self.spectraFilename)


class Scan:
    
    UNRECOGNIZED_SEQUENCE = "unrecognized"    
    
    def __init__(self, filenamePrefix):
        
        self.scan          = 0
        self.retentionTime = 0
        self.pepmass       = []
        self.charge        = ''
        
        self.peaks = []
        
        self.filenamePrefix = filenamePrefix
        
        
    def to_dict(self, saveFilename):
        
        if saveFilename:
            result = {'filename' : self.filenamePrefix + str(self.scan).zfill(PXD000561.SCAN_SEQUENCE_MAX_DIGITS) + '.json'}
        else:
            result = {}
        
        # result['peaks'] = np.array(self.peaks)
        tmpResult = np.array(self.peaks, dtype = np.float32)

        result['nzero_peaks'] = torch.from_numpy(tmpResult[tmpResult[:, 1] > 0])
        
        result['pepmass'] = self.pepmass
        result['charge']  = self.charge
        
        return result
        
        
        
class SpectraFound:
    
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

        print('- add_scan. sequence={}, scan={}, # non-zero peaks={}'.format(whichSequence, 
            whichScan.scan,
            len(sequenceDict['nzero_peaks'])))
        
        if whichSequence in self.spectra:
            self.spectra[whichSequence].append(sequenceDict)
        else:
            self.spectra[whichSequence] = [sequenceDict]

        if self.saveFiles:
            sequenceFolder = os.path.join(self.filesFolder, whichSequence)
            
            if os.path.isdir(sequenceFolder) == False:
                os.makedirs(sequenceFolder)
               
            with open(os.path.join(sequenceFolder, sequenceDict['filename']), 'w') as outputFile:
                json.dump(sequenceDict, outputFile)
                
    
    def load_spectra(self, spectraName):
    
        try:
            with open(os.path.join(self.filesFolder, spectraName), 'rb') as inputFile:
                entireData = pickle.load(inputFile)

        except Exception:
            print('Could not open spectra file {}'.format(os.path.join(self.filesFolder, spectraName)))
                    
        # Check if the data is already saved in the complete format

        if 'spectra' in entireData.keys():
            self.spectra = entireData['spectra']
            self.multipleScansSequences = entireData['multipleScansSequences']
            self.singleScanSequences = entireData['singleScanSequences']
            self.normalizationParameters = entireData['normalizationParameters']
        else:
            self.spectra = entireData

            self.multipleScansSequences = None
            self.singleScanSequences = None
            self.normalizationParameters = None



    def save_spectra(self, spectraName):
        
        entireData = {}
        entireData['spectra'] = self.spectra
        entireData['multipleScansSequences'] = self.multipleScansSequences
        entireData['singleScanSequences'] = self.singleScanSequences
        entireData['normalizationParameters'] = self.normalizationParameters

        completeFilename = os.path.join(self.filesFolder, spectraName) 

        if os.path.isdir(self.filesFolder) == False:
            os.makedirs(self.filesFolder)
        else:
            if os.path.exists(completeFilename):
                os.rename(completeFilename, completeFilename + ".bkp_" + datetime.datetime.now())

        with open(completeFilename, 'wb') as outputFile:
            pickle.dump(self.spectra, outputFile, pickle.HIGHEST_PROTOCOL)



    def list_single_and_multiple_scans_sequences(self):

        self.maxPeaksListLen = 0
        totalLen = 0

        sequenceMaxLen = ''
        numSpectrum = 0

        maxScansInSequence = 0
        sequencesWithMultipleScans = 0
        sequencesWithSingleScan = 0

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
                numSpectrum += 1
                
                if spectrumLen > self.maxPeaksListLen:
                    self.maxPeaksListLen = spectrumLen
                    sequenceMaxLen = key

        Logger()('Maximum non-zero peaks list len = {}. key = {}'.format(self.maxPeaksListLen, sequenceMaxLen))
        Logger()('Average peaks list len = {}'.format(totalLen / numSpectrum))
        Logger()('Number of sequences with a single scan = {}'.format(sequencesWithSingleScan))
        Logger()('Number of sequences with more than 1 scan = {}'.format(sequencesWithMultipleScans))
        Logger()('Max number of scans in a single sequence = {}'.format(maxScansInSequence))


    def normalize_data(self, trainingDataset = None):

        if not trainingDataset:

            #
            # Need to calculate the normalization parameters
            #

            # First, create a list with all non-zero peaks lists

            allPeaksLists = []
            allPeaksListsLen = []

            for key in self.multipleScansSequences + self.singleScanSequences:
                for peaksList in self.spectra[key]:
                    allPeaksLists.append(peaksList['nzero_peaks'])
                    allPeaksListsLen.append(len(peaksList['nzero_peaks']))

            # Now, create a huge tensor will all lists padded
            allLists = torch.nn.utils.rnn.pad_sequence(allPeaksLists, batch_first = True, padding_value = 0.0)

            # And calculate the parameters

            self.normalizationParameters = {}

            totalNonZeroPeaks = sum(allPeaksListsLen)

            mzMean = allLists[:, :, 0].sum() / totalNonZeroPeaks
            intensityMean = allLists[:, :, 1].sum() / totalNonZeroPeaks

            squaredMeanReduced = torch.pow(allLists - torch.tensor([mzMean, intensityMean]), 2)

            for i in range(allLists.shape[0]):
                allLists[i, allLists[i]:, :] = torch.tensor([0.0, 0.0])

            mzStd = torch.sqrt(squaredMeanReduced[:, :, 0].sum() / totalNonZeroPeaks)
            intensityStd = torch.sqrt(squaredMeanReduced[:, :, 1].sum() / totalNonZeroPeaks)

            self.normalizationParameters['mz_mean'] = mzMean
            self.normalizationParameters['mz_std'] = mzStd
            self.normalizationParameters['intensity_mean'] = intensityMean
            self.normalizationParameters['intensity_std']  = intensityStd

            Logger()('mz mean: {}, mz std: {}'.format(self.normalizationParameters['mz_mean'], self.normalizationParameters['mz_std']))
            Logger()('intensity mean: {}, intensity std: {}'.format(self.normalizationParameters['intensity_mean'], self.normalizationParameters['intensity_std']))

        else:

            Logger()("Will apply the following normalization parameters, from training dataset")
            Logger()('mz mean: {}, mz std: {}'.format(self.normalizationParameters['mz_mean'], self.normalizationParameters['mz_std']))
            Logger()('intensity mean: {}, intensity std: {}'.format(self.normalizationParameters['intensity_mean'], self.normalizationParameters['intensity_std']))
            
            self.normalizationParameters = trainingDataset.dataset.totalSpectra.normalizationParameters

        #
        # Now, normalize the data
        #

        print("Now, normalize the entire spectra (excluding unrecognized scans).")

        for key in self.multipleScansSequences + self.singleScanSequences:
            for peaksList in self.spectra[key]:
                peaksList['nzero_peaks'][:, 0] = (peaksList['nzero_peaks'][:, 0] - self.normalizationParameters['mz_mean']) / self.normalizationParameters['mz_std']
                peaksList['nzero_peaks'][:, 1] = (peaksList['nzero_peaks'][:, 1] - self.normalizationParameters['intensity_mean']) / self.normalizationParameters['intensity_std']



class MGF:
    
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
        '^TITLE=.+\".+scan=(\d+)\"',
        '^RTINSECONDS=(\d+\.*\d*)',
        '^PEPMASS=(\d+\.*\d*)',
        '^CHARGE=(\d+\+*)',
        '^(\d+\.\d+)\s(\d+\.\d+)',
        '^END IONS'
    ]    

       
    def read_spectrum(self, whichFile, scanFilenamePrefix, searchedScan, decodedSequence, spectraFound, currentScan = None):

        print('read_spectrum: file={}, scan={}, sequence={}'.format(whichFile.name, searchedScan, decodedSequence))

        hasFoundScan = False

        #
        # Check if previously found scan isn't also the same for the new sequence
        #

        if currentScan != None:
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

                            newScan = Scan(scanFilenamePrefix)

                        elif (i == MGF.TITLE_FIELD):
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

                                elif newScan.scan < searchedScan:
                                    spectraFound.add_scan(newScan, Scan.UNRECOGNIZED_SEQUENCE)

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


        return hasFoundScan, newScan