from .spectra import SpectraFound
from .spectra import MGF
from .spectra import Scan

import os
import torch
import torch.utils.data as data

import pickle

import gc

import math

from .batch_loader_encoder import BatchLoaderEncoder
from MSEmbedding.models.metrics.SaveEmbeddings import SaveEmbeddings

from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger


class Linfeng(data.Dataset):

    SPECTRA_LIST_FILE_DEFAULT = "linfeng_spectra_index"
    SPECTRA_FOLDER = "sequence"
    SPECTRA_FILES_EXTENSION = ".pkl"

    TMP_EMBEDDINGS_FILENAME = "tmp_files_list.txt"


    EXPERIMENTS_FOLDERS_ALL = [
        "HapMap9_080510",
        "HapMap10_080810",
        "HapMap11_080910",
        "HapMap12_091210",
        "HapMap18_092610",
        "HapMap19_092810",
        "HapMap20_093010",
        "HapMap21_111710",
        "HapMap22_111510",
        "HapMap23_111910",
        "HapMap24_112110",
        "HapMap25_112310",
        "HapMap26_112510",
        "HapMap27_112710",
        "HapMap28_112910",
        "HapMap29_120110",
        "HapMap30_120610",
        "HapMap31_010411",
        "HapMap32_010211",
        "HapMap33_011511",
        "HapMap34_010811",
        "HapMap35_011011",
        "HapMap36_012711",
        "HapMap37_111911",
        "HapMap38_012111",
        "HapMap39_012511",
        "HapMap40_020511",
        "HapMap41_020911",
        "HapMap42_020211",
        "HapMap43_030711",
        "HapMap44_031911",
        "HapMap45_022011",
        "HapMap46_030911",
        "HapMap47_022811",
        "HapMap48_030211",
        "HapMap56_081211",
        "HapMap57_080311",
        "HapMap58_080511",
        "HapMap59_080711",
        "HapMap60_081511",
        "HapMap63_081911",
        "HapMap64_082111",
        "HapMap65_082411",
        "HapMap66_082611",
        "HapMap67_111311",
        "HapMap68_090211",
        "HapMap69_090611",
        "HapMap70_090811",
        "HapMap71_091112",
        "HapMap72_110911",
        "HapMap73_110711"
    ]


    SAMPLE_EXPERIMENTS_FOLDERS = [
        "sample_experiment"
    ]



    def read_spectra(self, folders, mgfFolder, outputFolder, normalizationParameters):

        spectraParser = MGF()

        totalSpectraCount = 0

        spectraList = []

        print("HERE: {}".format(os.getcwd()))

        with open(os.path.join(self.currentDirectory, self.embeddingsFolder, Linfeng.TMP_EMBEDDINGS_FILENAME), "w") as outputFile:
            for folder in folders:

                print("Processing folder {}".format(folder))

                spectraFound = SpectraFound(True, outputFolder)

                spectraFound.multipleScansSequences = None
                spectraFound.singleScanSequences = None
                spectraFound.normalizationParameters = normalizationParameters
                spectraFound.spectraCount = 0

                files = os.listdir(os.path.join(mgfFolder, folder))
                files.sort()

                for fileName in files:

                    spectraCountInFile = 0

                    if fileName.lower().endswith(".mgf"):
                        Logger()("- Processing file {}".format(fileName))

                        currentFile = open(os.path.join(mgfFolder, folder, fileName), 'r')

                        _, _, spectraCountInFile = spectraParser.read_spectrum(currentFile, 
                                                                            fileName + '_', 
                                                                            None, None, spectraFound)
    
                    spectraFound.spectraCount += spectraCountInFile

                Logger()("Folder {} files had {} spectra.".format(folder, spectraFound.spectraCount))

                totalSpectraCount += spectraFound.spectraCount

                # Normalize the spectra with the training set normalization parameters
                # Also, populate the list of all spectra

                currentFile = ""
                spectrumIndexInFile = 0

                for peaksList in spectraFound.spectra[Scan.UNRECOGNIZED_SEQUENCE]:

                    if currentFile != '_'.join(peaksList['filename'].split('_')[:-1]):
                        spectrumIndexInFile = 0
                        currentFile = '_'.join(peaksList['filename'].split('_')[:-1])
                    else:
                        spectrumIndexInFile += 1

                    spectraList.append({'filename' : currentFile, 
                                        'index' : spectrumIndexInFile,
                                        'pepmass' : peaksList['pepmass'],
                                        'charge' : peaksList['charge'],
                                        'scan' : peaksList['scan']})


                    peaksList['nzero_peaks'][:, 0] = (peaksList['nzero_peaks'][:, 0] - normalizationParameters['mz_mean']) / normalizationParameters['mz_std']
                    peaksList['nzero_peaks'][:, 1] = (peaksList['nzero_peaks'][:, 1] - normalizationParameters['intensity_mean']) / normalizationParameters['intensity_std']

                    # Save the original filename in the spectra filelist file

                    outputFile.write(currentFile + "\n")


                # Save spectra for this folder in a separated file to be processed by the batch loader

                spectraFound.save_spectra(folder + Linfeng.SPECTRA_FILES_EXTENSION)

        os.rename(os.path.join(self.currentDirectory, self.embeddingsFolder, Linfeng.TMP_EMBEDDINGS_FILENAME), 
                  os.path.join(self.currentDirectory, self.embeddingsFolder, 
                               self.fileListFilename.format(str(math.ceil(totalSpectraCount / self.batch_size)).zfill(6)) + SaveEmbeddings.EMBEDDINGS_FILES_LIST_FILE_EXTENSION))

        return totalSpectraCount, spectraList



    def __init__(self, dataDirectory = 'data/linfeng', split = 'test', 
                 batch_size = 100, nb_threads = 1, trainingDataset = None):

        self.split = split
        self.nb_threads = nb_threads
        self.batch_size = batch_size
        self.dataDirectory = dataDirectory

        self.spectraListFilename = Options().get("dataset.spectra_list_file", Linfeng.SPECTRA_LIST_FILE_DEFAULT) + Linfeng.SPECTRA_FILES_EXTENSION
        self.spectraExperimentsFolder = getattr(Linfeng, Options().get("dataset.mgf_experiments", "EXPERIMENTS_FOLDERS_ALL"))
        self.fileListFilename = SaveEmbeddings.build_embeddings_filename()
        self.embeddingsFolder = Options().get("dataset.embeddings_dir", SaveEmbeddings.EMBEDDINGS_FOLDER)

        self.currentDirectory = os.getcwd()

        print('Working directory: ' + os.getcwd())

        if not os.path.exists(os.path.join(dataDirectory, Linfeng.SPECTRA_FOLDER, self.spectraListFilename)):

            print("*** Need to create the spectra list !!!")

            print("***** Need to load training dataset to get normalization parameters")

            trainingPeaksCompleteFile = Options().get("dataset.train_normalization_file", None)

            mgfFilesFolder = Options().get("dataset.mgf_dir", None)

            trainingDataset = SpectraFound(False, os.path.dirname(trainingPeaksCompleteFile))
            trainingDataset.load_spectra(os.path.basename(trainingPeaksCompleteFile))

            if not trainingDataset:
                raise ValueError("Missing training dataset to get normalization parameters !!!")

            normalizationParameters = trainingDataset.normalizationParameters

            del trainingDataset
            gc.collect()

            os.chdir(dataDirectory)

            self.totalSpectraCount, self.spectraList = self.read_spectra(self.spectraExperimentsFolder,
                                                                         mgfFilesFolder, 
                                                                         Linfeng.SPECTRA_FOLDER, 
                                                                         normalizationParameters)

            with open(os.path.join(Linfeng.SPECTRA_FOLDER, self.spectraListFilename), 'wb') as outputFile:
                pickle.dump(self.spectraList, outputFile, pickle.HIGHEST_PROTOCOL)            
        else:
            with open(os.path.join(dataDirectory, Linfeng.SPECTRA_FOLDER, self.spectraListFilename), 'rb') as inputFile:
                self.spectraList = pickle.load(inputFile)

            self.totalSpectraCount = len(self.spectraList)            

        self.numberOfBatches = self.totalSpectraCount // self.batch_size

        print('=============> Initial number of batches: {}'.format(self.numberOfBatches))

        #
        # Make sure the initial working directory remains the same, to avoid breaking the
        # framework.
        #

        os.chdir(self.currentDirectory)



    def __getitem__(self, index):

        item = {}
        item['peaks'], item['peaksLen'] = self.batchSampler.getItem(index)

        return item



    def __len__(self):

        print('--------------->>>> number of batches: {}'.format(self.numberOfBatches))

        return self.numberOfBatches



    def make_batch_loader(self):

        self.batchSampler = BatchLoaderEncoder(self.spectraList, self.batch_size, os.path.join(self.dataDirectory, Linfeng.SPECTRA_FOLDER))

        self.numberOfBatches = len(self.batchSampler.spectraList) // self.batch_size

        if len(self.batchSampler.spectraList) % self.batch_size != 0:
            self.numberOfBatches += 1

        print('=============> Updated number of batches: {}'.format(self.numberOfBatches))

        print('********************* make_batch_loader: {}'.format(self.batchSampler))

        data_loader = data.DataLoader(self,
            num_workers = self.nb_threads,
            batch_sampler = self.batchSampler,
            drop_last = False)
        return data_loader


