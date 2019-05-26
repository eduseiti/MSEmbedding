import os
import pandas as pd
import json
import time
import pickle

import numpy as np
import re

import torch


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

    def __init__(self, identificationsFilename, spectraFilename):
    
        self.identificationsFilename = identificationsFilename
        self.spectraFilename = spectraFilename
    
        self.totalSpectra = spectraFound(False, 'sequences')
    
    
    def load_identifications(self, verbose = False, singleFile = None):

        
        #
        # First check if the spectra has already been loaded
        #
        
        self.totalSpectra.load_spectra(self.spectraFilename)
        
        if self.totalSpectra.spectra: 
            return
        
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

        if singleFile:
            self.uniqueCombination = self.uniqueCombination[self.uniqueCombination['File'] == singleFile]


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


class scan:
    
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
        
        # result['peaks']       = np.array(self.peaks)
        tmpResult = np.array(self.peaks, dtype = np.float32)

        result['nzero_peaks'] = torch.from_numpy(tmpResult[tmpResult[:, 1] > 0])
        
        result['pepmass'] = self.pepmass
        result['charge']  = self.charge
        
        return result
        
        
        
class spectraFound:
    
    def __init__(self, saveFiles, filesFolder):
                
        #
        # Dictionary to hold the spectrum found
        #
        # {<sequence>:[{'filename' : <filename>,
        #               'peaks' : [[<mz>, <intensity>], ...],
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
                self.spectra = pickle.load(inputFile)
                
        except Exception as e:
            print('Could not open spectra file {}'.format(os.path.join(self.filesFolder, spectraName)))
                      
            
    def save_spectra(self, spectraName):
        
        if os.path.isdir(self.filesFolder) == False:
            os.makedirs(self.filesFolder)

        with open(os.path.join(self.filesFolder, spectraName), 'wb') as outputFile:
            pickle.dump(self.spectra, outputFile, pickle.HIGHEST_PROTOCOL)



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

                            newScan = scan(scanFilenamePrefix)

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
                                    spectraFound.add_scan(newScan, scan.UNRECOGNIZED_SEQUENCE)

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