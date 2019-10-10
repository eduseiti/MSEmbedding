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

from bootstrap.lib.logger import Logger


class PXD000561:

    SCAN_SEQUENCE_MAX_DIGITS = 8
    SCAN_SEQUENCE_OVER_LIMIT = 999999999

    FETAL_BRAIN_GEL_VELOS_FILES = {
		"E1" : "Fetal_Brain_Gel_Velos_16_f01.mgf",
		"E1rep" : "Fetal_Brain_Gel_Velos_16_f02.mgf",
		"E2" : "Fetal_Brain_Gel_Velos_16_f03.mgf",
		"E3" : "Fetal_Brain_Gel_Velos_16_f04.mgf",
		"E3rep" : "Fetal_Brain_Gel_Velos_16_f05.mgf",
		"E4" : "Fetal_Brain_Gel_Velos_16_f06.mgf",
		"E5" : "Fetal_Brain_Gel_Velos_16_f07.mgf",
		"E6" : "Fetal_Brain_Gel_Velos_16_f08.mgf",
		"E7" : "Fetal_Brain_Gel_Velos_16_f09.mgf",
		"E8" : "Fetal_Brain_Gel_Velos_16_f10.mgf",
		"E9" : "Fetal_Brain_Gel_Velos_16_f11.mgf",
		"E10" : "Fetal_Brain_Gel_Velos_16_f12.mgf",
		"E11" : "Fetal_Brain_Gel_Velos_16_f13.mgf",
		"E12" : "Fetal_Brain_Gel_Velos_16_f14.mgf",
		"F1" : "Fetal_Brain_Gel_Velos_16_f15.mgf",
		"F2" : "Fetal_Brain_Gel_Velos_16_f16.mgf",
		"F3" : "Fetal_Brain_Gel_Velos_16_f17.mgf",
		"F4" : "Fetal_Brain_Gel_Velos_16_f18.mgf",
		"F5" : "Fetal_Brain_Gel_Velos_16_f19.mgf",
		"F6" : "Fetal_Brain_Gel_Velos_16_f20.mgf",
		"F7" : "Fetal_Brain_Gel_Velos_16_f21.mgf",
		"F8" : "Fetal_Brain_Gel_Velos_16_f22.mgf",
		"F9" : "Fetal_Brain_Gel_Velos_16_f23.mgf",
		"F10" : "Fetal_Brain_Gel_Velos_16_f24.mgf",
		"F11" : "Fetal_Brain_Gel_Velos_16_f25.mgf",
		"F12" : "Fetal_Brain_Gel_Velos_16_f26.mgf",
		"G1" : "Fetal_Brain_Gel_Velos_16_f27.mgf",
		"G2" : "Fetal_Brain_Gel_Velos_16_f28.mgf",
		"G3" : "Fetal_Brain_Gel_Velos_16_f29.mgf"
    }


    ADULT_ADRENALGLAND_GEL_ELITE_FILES = {
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



    ADULT_ADRENALGLAND_GEL_VELOS_FILES = {
        'D1' : 'Adult_Adrenalgland_Gel_Velos_2_f01.mgf',
        'D2' : 'Adult_Adrenalgland_Gel_Velos_2_f02.mgf',
        'D3' : 'Adult_Adrenalgland_Gel_Velos_2_f03.mgf',
        'D4' : 'Adult_Adrenalgland_Gel_Velos_2_f04.mgf',
        'D5' : 'Adult_Adrenalgland_Gel_Velos_2_f05.mgf',
        'D6' : 'Adult_Adrenalgland_Gel_Velos_2_f06.mgf',
        'D7' : 'Adult_Adrenalgland_Gel_Velos_2_f07.mgf',
        'D8' : 'Adult_Adrenalgland_Gel_Velos_2_f08.mgf',
        'D9' : 'Adult_Adrenalgland_Gel_Velos_2_f09.mgf',
        'D10' : 'Adult_Adrenalgland_Gel_Velos_2_f10.mgf',
        'D11' : 'Adult_Adrenalgland_Gel_Velos_2_f11.mgf',
        'D12' : 'Adult_Adrenalgland_Gel_Velos_2_f12.mgf',
        'E1' : 'Adult_Adrenalgland_Gel_Velos_2_f13.mgf',
        'E1Re' : 'Adult_Adrenalgland_Gel_Velos_2_f14.mgf',
        'E2' : 'Adult_Adrenalgland_Gel_Velos_2_f15.mgf',
        'E3' : 'Adult_Adrenalgland_Gel_Velos_2_f16.mgf',
        'E4' : 'Adult_Adrenalgland_Gel_Velos_2_f17.mgf',
        'E5' : 'Adult_Adrenalgland_Gel_Velos_2_f18.mgf',
        'E6' : 'Adult_Adrenalgland_Gel_Velos_2_f19.mgf',
        'E6-110108181623' : 'Adult_Adrenalgland_Gel_Velos_2_f20.mgf',
        'E7' : 'Adult_Adrenalgland_Gel_Velos_2_f21.mgf',
        'E8' : 'Adult_Adrenalgland_Gel_Velos_2_f22.mgf',
        'E9' : 'Adult_Adrenalgland_Gel_Velos_2_f23.mgf',
        'E10' : 'Adult_Adrenalgland_Gel_Velos_2_f24.mgf',
        'E11' : 'Adult_Adrenalgland_Gel_Velos_2_f25.mgf',
        'E12' : 'Adult_Adrenalgland_Gel_Velos_2_f26.mgf'
    }



    ADULT_ADRENALGLAND_BRP_VELOS_FILES = {
        'A1' : 'Adult_Adrenalgland_bRP_Velos_1_f01.mgf',
        'A2' : 'Adult_Adrenalgland_bRP_Velos_1_f02.mgf',
        'A3' : 'Adult_Adrenalgland_bRP_Velos_1_f03.mgf',
        'A4' : 'Adult_Adrenalgland_bRP_Velos_1_f04.mgf',
        'A5' : 'Adult_Adrenalgland_bRP_Velos_1_f05.mgf',
        'A6' : 'Adult_Adrenalgland_bRP_Velos_1_f06.mgf',
        'A7' : 'Adult_Adrenalgland_bRP_Velos_1_f07.mgf',
        'A8' : 'Adult_Adrenalgland_bRP_Velos_1_f08.mgf',
        'A9' : 'Adult_Adrenalgland_bRP_Velos_1_f09.mgf',
        'A10' : 'Adult_Adrenalgland_bRP_Velos_1_f10.mgf',
        'A11' : 'Adult_Adrenalgland_bRP_Velos_1_f11.mgf',
        'A12' : 'Adult_Adrenalgland_bRP_Velos_1_f12.mgf',
        'B1' : 'Adult_Adrenalgland_bRP_Velos_1_f13.mgf',
        'B2' : 'Adult_Adrenalgland_bRP_Velos_1_f14.mgf',
        'B3' : 'Adult_Adrenalgland_bRP_Velos_1_f15.mgf',
        'B4' : 'Adult_Adrenalgland_bRP_Velos_1_f16.mgf',
        'B5' : 'Adult_Adrenalgland_bRP_Velos_1_f17.mgf',
        'B6' : 'Adult_Adrenalgland_bRP_Velos_1_f18.mgf',
        'B7' : 'Adult_Adrenalgland_bRP_Velos_1_f19.mgf',
        'B8' : 'Adult_Adrenalgland_bRP_Velos_1_f20.mgf',
        'B9' : 'Adult_Adrenalgland_bRP_Velos_1_f21.mgf',
        'B10' : 'Adult_Adrenalgland_bRP_Velos_1_f22.mgf',
        'B11' : 'Adult_Adrenalgland_bRP_Velos_1_f23.mgf',
        'B12' : 'Adult_Adrenalgland_bRP_Velos_1_f24.mgf',
        'C1' : 'Adult_Adrenalgland_bRP_Velos_1_f25.mgf',
        'C2' : 'Adult_Adrenalgland_bRP_Velos_1_f26.mgf',
        'C3' : 'Adult_Adrenalgland_bRP_Velos_1_f27.mgf',
        'C4' : 'Adult_Adrenalgland_bRP_Velos_1_f28.mgf',
        'C5' : 'Adult_Adrenalgland_bRP_Velos_1_f29.mgf',
        'C6' : 'Adult_Adrenalgland_bRP_Velos_1_f30.mgf',
        'C7' : 'Adult_Adrenalgland_bRP_Velos_1_f31.mgf',
        'C8' : 'Adult_Adrenalgland_bRP_Velos_1_f32.mgf',
        'C9' : 'Adult_Adrenalgland_bRP_Velos_1_f33.mgf',
        'C10' : 'Adult_Adrenalgland_bRP_Velos_1_f34.mgf',
        'C11' : 'Adult_Adrenalgland_bRP_Velos_1_f35.mgf',
        'C12' : 'Adult_Adrenalgland_bRP_Velos_1_f36.mgf'
    }


    ADULT_PLATELETS_GEL_ELITE_FILES = {
        'b1' : 'Adult_Platelets_Gel_Elite_48_f01.mgf',
        'b2' : 'Adult_Platelets_Gel_Elite_48_f02.mgf',
        'b3' : 'Adult_Platelets_Gel_Elite_48_f03.mgf',
        'b4' : 'Adult_Platelets_Gel_Elite_48_f04.mgf',
        'b5' : 'Adult_Platelets_Gel_Elite_48_f05.mgf',
        'b6' : 'Adult_Platelets_Gel_Elite_48_f06.mgf',
        'b7' : 'Adult_Platelets_Gel_Elite_48_f07.mgf',
        'b8' : 'Adult_Platelets_Gel_Elite_48_f08.mgf',
        'b9' : 'Adult_Platelets_Gel_Elite_48_f09.mgf',
        'b10' : 'Adult_Platelets_Gel_Elite_48_f10.mgf',
        'b11' : 'Adult_Platelets_Gel_Elite_48_f11.mgf',
        'b12' : 'Adult_Platelets_Gel_Elite_48_f12.mgf',
        'b13' : 'Adult_Platelets_Gel_Elite_48_f13.mgf',
        'b14' : 'Adult_Platelets_Gel_Elite_48_f14.mgf',
        'b15' : 'Adult_Platelets_Gel_Elite_48_f15.mgf',
        'b16' : 'Adult_Platelets_Gel_Elite_48_f16.mgf',
        'b17' : 'Adult_Platelets_Gel_Elite_48_f17.mgf',
        'b18' : 'Adult_Platelets_Gel_Elite_48_f18.mgf',
        'b19' : 'Adult_Platelets_Gel_Elite_48_f19.mgf',
        'b20' : 'Adult_Platelets_Gel_Elite_48_f20.mgf',
        'b21' : 'Adult_Platelets_Gel_Elite_48_f21.mgf',
        'b22' : 'Adult_Platelets_Gel_Elite_48_f22.mgf',
        'b23' : 'Adult_Platelets_Gel_Elite_48_f23.mgf',
        'b24' : 'Adult_Platelets_Gel_Elite_48_f24.mgf',
    }


    ADULT_URINARYBLADDER_GEL_ELITE_FILES = {
        'b1' : 'Adult_Urinarybladder_Gel_Elite_70_f01.mgf',
        'b2' : 'Adult_Urinarybladder_Gel_Elite_70_f02.mgf',
        'b3' : 'Adult_Urinarybladder_Gel_Elite_70_f03.mgf',
        'b4' : 'Adult_Urinarybladder_Gel_Elite_70_f04.mgf',
        'b5' : 'Adult_Urinarybladder_Gel_Elite_70_f05.mgf',
        'b6' : 'Adult_Urinarybladder_Gel_Elite_70_f06.mgf',
        'b7' : 'Adult_Urinarybladder_Gel_Elite_70_f07.mgf',
        'b8' : 'Adult_Urinarybladder_Gel_Elite_70_f08.mgf',
        'b9' : 'Adult_Urinarybladder_Gel_Elite_70_f09.mgf',
        'b9r' : 'Adult_Urinarybladder_Gel_Elite_70_f1.mgf',
        'b10' : 'Adult_Urinarybladder_Gel_Elite_70_f11.mgf',
        'b11' : 'Adult_Urinarybladder_Gel_Elite_70_f12.mgf',
        'b12' : 'Adult_Urinarybladder_Gel_Elite_70_f13.mgf',
        'b13' : 'Adult_Urinarybladder_Gel_Elite_70_f14.mgf',
        'b14' : 'Adult_Urinarybladder_Gel_Elite_70_f15.mgf',
        'b15' : 'Adult_Urinarybladder_Gel_Elite_70_f16.mgf',
        'b16' : 'Adult_Urinarybladder_Gel_Elite_70_f17.mgf',
        'b17' : 'Adult_Urinarybladder_Gel_Elite_70_f18.mgf',
        'b18' : 'Adult_Urinarybladder_Gel_Elite_70_f19.mgf',
        'b19' : 'Adult_Urinarybladder_Gel_Elite_70_f20.mgf',
        'b20' : 'Adult_Urinarybladder_Gel_Elite_70_f21.mgf',
        'b21' : 'Adult_Urinarybladder_Gel_Elite_70_f22.mgf',
        'b22' : 'Adult_Urinarybladder_Gel_Elite_70_f23.mgf',
        'b23' : 'Adult_Urinarybladder_Gel_Elite_70_f24.mgf',
        'b24' : 'Adult_Urinarybladder_Gel_Elite_70_f25.mgf',
    }


    FETAL_LIVER_BRP_ELITE_FILES = {
        '01' : 'Fetal_Liver_bRP_Elite_22_f01.mgf',
        '02' : 'Fetal_Liver_bRP_Elite_22_f02.mgf',
        '03' : 'Fetal_Liver_bRP_Elite_22_f03.mgf',
        '04' : 'Fetal_Liver_bRP_Elite_22_f04.mgf',
        '05' : 'Fetal_Liver_bRP_Elite_22_f05.mgf',
        '06' : 'Fetal_Liver_bRP_Elite_22_f06.mgf',
        '07' : 'Fetal_Liver_bRP_Elite_22_f07.mgf',
        '08' : 'Fetal_Liver_bRP_Elite_22_f08.mgf',
        '08-NCE27' : 'Fetal_Liver_bRP_Elite_22_f09.mgf',
        '08-NCE27-1' : 'Fetal_Liver_bRP_Elite_22_f10.mgf',
        '08-NCE27-32' : 'Fetal_Liver_bRP_Elite_22_f11.mgf',
        '08-NCE27-32-1' : 'Fetal_Liver_bRP_Elite_22_f12.mgf',
        '08-1' : 'Fetal_Liver_bRP_Elite_22_f13.mgf',
        '09' : 'Fetal_Liver_bRP_Elite_22_f14.mgf',
        '10' : 'Fetal_Liver_bRP_Elite_22_f15.mgf',
        '11' : 'Fetal_Liver_bRP_Elite_22_f16.mgf',
        '12' : 'Fetal_Liver_bRP_Elite_22_f17.mgf',
        '13-120502235957' : 'Fetal_Liver_bRP_Elite_22_f18.mgf',
        '13' : 'Fetal_Liver_bRP_Elite_22_f19.mgf',
        '14-120503014940' : 'Fetal_Liver_bRP_Elite_22_f20.mgf',
        '14' : 'Fetal_Liver_bRP_Elite_22_f21.mgf',
        '15-120503033941' : 'Fetal_Liver_bRP_Elite_22_f22.mgf',
        '15' : 'Fetal_Liver_bRP_Elite_22_f23.mgf',
        '16-120503052933' : 'Fetal_Liver_bRP_Elite_22_f24.mgf',
        '16' : 'Fetal_Liver_bRP_Elite_22_f25.mgf',
        '17-120503071920' : 'Fetal_Liver_bRP_Elite_22_f26.mgf',
        '17' : 'Fetal_Liver_bRP_Elite_22_f27.mgf',
        '18' : 'Fetal_Liver_bRP_Elite_22_f28.mgf',
        '19' : 'Fetal_Liver_bRP_Elite_22_f29.mgf',
        '20' : 'Fetal_Liver_bRP_Elite_22_f30.mgf',
        '21' : 'Fetal_Liver_bRP_Elite_22_f31.mgf',
        '22' : 'Fetal_Liver_bRP_Elite_22_f32.mgf',
        '23401' : 'Fetal_Liver_bRP_Elite_22_f33.mgf',
        '23' : 'Fetal_Liver_bRP_Elite_22_f34.mgf',
        '24' : 'Fetal_Liver_bRP_Elite_22_f35.mgf',
    }


    MATCHES_TO_FILES_LIST = {
        "Gel_Elite_49.csv" : ADULT_ADRENALGLAND_GEL_ELITE_FILES,
        "fetal_brain_gel_velos_16.csv" : FETAL_BRAIN_GEL_VELOS_FILES,
        "adult_adrenalgland_gel_velos.csv" : ADULT_ADRENALGLAND_GEL_VELOS_FILES,
        "adult_adrenalgland_bRP_velos.csv" : ADULT_ADRENALGLAND_BRP_VELOS_FILES,
        "adult_urinarybladder_gel_elite.csv" : ADULT_URINARYBLADDER_GEL_ELITE_FILES,
        "adult_platelets_gel_elite.csv" : ADULT_PLATELETS_GEL_ELITE_FILES
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

        matches_file = pd.read_csv(self.identificationsFilename)

        if verbose:
            print(matches_file)

        print('Number of unique sequences found: {}'.format(len(matches_file['Sequence'].unique())))

        #
        # Inject new columns to hold the scan sequence within the file and the file name, recovered from the 
        # "Spectrum Title" information.
        #

        matches_file['File Sequence'] = \
            matches_file['Spectrum Title'].str.split("_", expand = True).iloc[:, 3].str.zfill(PXD000561.SCAN_SEQUENCE_MAX_DIGITS)
        
        matches_file['File'] = matches_file['Spectrum Title'].str.split("_", expand = True).iloc[:, 2]

        ordered = matches_file.sort_values(['File', 'File Sequence'])


        #
        # Select only unique "Sequence" + "First Scan" combination:
        #
        # - Same spectrum can contain different sequences
        # - Different sequences can be read in different scans

        duplicatesIndication = ordered.duplicated(['Spectrum Title', 'Sequence', 'First Scan'])

        self.uniqueCombination = ordered[~duplicatesIndication]
        print('Unique combinations found: {}'.format(self.uniqueCombination.shape))

        # print('Unique combinations file {}: {}'.format('b01', 
        #                                                self.uniqueCombination[self.uniqueCombination['File'] == 'b01'].shape))

        if filteredFilesList:
            self.uniqueCombination = self.uniqueCombination[self.uniqueCombination['File'].isin(filteredFilesList)]


    def read_spectra(self, spectraParser):
        currentFileName           = ''
        currentScanFileNamePrefix = ''

        currentFile  = None
        lastScan     = None

        startTime = time.time()

        spectraFiles = PXD000561.MATCHES_TO_FILES_LIST[self.identificationsFilename]

        for index, row in self.uniqueCombination.iterrows():
            if (spectraFiles[row['File']] != currentFileName):
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
                currentFileName = spectraFiles[currentFileNamePrefix]

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
    
        self.multipleScansSequences = None
        self.singleScanSequences = None
        self.normalizationParameters = None

        try:
            with open(os.path.join(self.filesFolder, spectraName), 'rb') as inputFile:
                entireData = pickle.load(inputFile)

            # Check if the data is already saved in the complete format

            if 'spectra' in entireData.keys():
                self.spectra = entireData['spectra']
                self.multipleScansSequences = entireData['multipleScansSequences']
                self.singleScanSequences = entireData['singleScanSequences']
                self.normalizationParameters = entireData['normalizationParameters']
            else:
                self.spectra = entireData



        except Exception:
            print('Could not open spectra file {}'.format(os.path.join(self.filesFolder, spectraName)))
                    



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
                os.rename(completeFilename, completeFilename + ".bkp_" + str(datetime.datetime.now()))

        with open(completeFilename, 'wb') as outputFile:
            pickle.dump(entireData, outputFile, pickle.HIGHEST_PROTOCOL)



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

            # Sum everything to calculate the mean

            mzSum = 0.0
            intensitySum = 0.0
            totalNonZeroPeaks = 0

            print("Start to calculate the mzMean and intensityMean...")

            for key in self.multipleScansSequences + self.singleScanSequences:
                for peaksList in self.spectra[key]:
                    mzSum += peaksList['nzero_peaks'][:, 0].sum()
                    intensitySum += peaksList['nzero_peaks'][:, 1].sum()
                    totalNonZeroPeaks += len(peaksList['nzero_peaks'])


            self.normalizationParameters = {}

            mzMean = mzSum / totalNonZeroPeaks
            intensityMean = intensitySum / totalNonZeroPeaks

            print("Start to calculate the mzStd and intensityStd...")

            mzSquaredMeanReducedSum = 0.0
            intensitySquaredMeanReducedSum = 0.0

            for key in self.multipleScansSequences + self.singleScanSequences:
                for peaksList in self.spectra[key]:
                    mzSquaredMeanReducedSum += torch.pow(peaksList['nzero_peaks'][:, 0] - mzMean, 2).sum()
                    intensitySquaredMeanReducedSum += torch.pow(peaksList['nzero_peaks'][:, 1] - intensityMean, 2).sum()

            mzStd = math.sqrt(mzSquaredMeanReducedSum / totalNonZeroPeaks)
            intensityStd = math.sqrt(intensitySquaredMeanReducedSum / totalNonZeroPeaks)

            self.normalizationParameters['mz_mean'] = mzMean
            self.normalizationParameters['mz_std'] = mzStd
            self.normalizationParameters['intensity_mean'] = intensityMean
            self.normalizationParameters['intensity_std']  = intensityStd

            Logger()('mz mean: {}, mz std: {}'.format(self.normalizationParameters['mz_mean'], self.normalizationParameters['mz_std']))
            Logger()('intensity mean: {}, intensity std: {}'.format(self.normalizationParameters['intensity_mean'], self.normalizationParameters['intensity_std']))

        else:
            self.normalizationParameters = trainingDataset.dataset.totalSpectra.normalizationParameters

            Logger()("Will apply the following normalization parameters, from training dataset")
            Logger()('mz mean: {}, mz std: {}'.format(self.normalizationParameters['mz_mean'], self.normalizationParameters['mz_std']))
            Logger()('intensity mean: {}, intensity std: {}'.format(self.normalizationParameters['intensity_mean'], self.normalizationParameters['intensity_std']))
            

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