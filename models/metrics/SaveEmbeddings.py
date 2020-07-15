import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
from scipy.spatial.distance import cdist

import pandas as pd

import os
import pickle
import sys

class SaveEmbeddings(torch.nn.Module):

    EMBEDDINGS_FOLDER = "data/linfeng"

    EMBEDDINGS_FILE_EXTENSION = ".pkl"
    # EMBEDDINGS_FILE_EXTENSION = ".tsv"
    EMBEDDINGS_BIN_FILE_EXTENSION = ".bin"
    EMBEDDINGS_FILES_LIST_FILE_EXTENSION = ".txt"

    EMBEDDINGS_FILENAME_DEFAULT = "spectra_embeddings"

    def build_embeddings_filename(which_format):
        if which_format == "complete":
            return Options().get("dataset.embeddings_file", SaveEmbeddings.EMBEDDINGS_FILENAME_DEFAULT)
        else:
            return Options().get("dataset.embeddings_file", SaveEmbeddings.EMBEDDINGS_FILENAME_DEFAULT) + "_{}"


    SAVE_COUNT = 40000

    def __init__(self, engine = None, mode = 'test'):
        super(SaveEmbeddings, self).__init__()

        self.currentBatch = 0

        self.mode = mode
        self.allEmbeddings = []
        self.allEmbeddingsBin = []

        self.embeddingsFolder = Options().get("dataset.embeddings_dir", SaveEmbeddings.EMBEDDINGS_FOLDER)

        self.save_embeddings_format = Options().get("model.metric.embeddings_format", "binary")

        self.embeddingsFilename = SaveEmbeddings.build_embeddings_filename(self.save_embeddings_format)

        if engine and mode == 'test':
            engine.register_hook('eval_on_end_epoch', self.save_embeddings)


    def forward(self, criterionOutput, networkOutput, batch):

        embeddings = networkOutput

        print("**** size={}".format(embeddings.element_size() * embeddings.nelement()))
        print("**** element_size={}, nelement={}".format(embeddings.element_size(), embeddings.nelement()))
        print("**** shape={}".format(embeddings.shape))
        
        lastCellState = embeddings.cpu()

        print("**** shape lastCellState={}".format(lastCellState.shape))

        if self.save_embeddings_format == "complete":
            batch_embeddings = list(lastCellState.unbind())

            for i in range(len(batch_embeddings)):

                print("sequence {}:{}".format(i, batch['sequence'][i]))
                print("index {}:{}".format(i, batch['index'][i]))

                self.allEmbeddings.append((batch['sequence'][i], batch['index'][i].item(), batch_embeddings[i]))

            print("Len allEmbeddings={}".format(len(self.allEmbeddings)))
        else:
            self.allEmbeddingsBin += [whichEmbedding.numpy().tobytes() for whichEmbedding in lastCellState.unbind()]

            print("Len allEmbeddingsBin={}".format(len(self.allEmbeddingsBin)))


        self.currentBatch += 1

        if self.currentBatch % SaveEmbeddings.SAVE_COUNT == 0:
            self.save_embeddings()




    def save_embeddings(self):

        print("-- save_embeddings. currentBatch={}".format(self.currentBatch))

        if self.save_embeddings_format == "complete":
            with open(os.path.join(self.embeddingsFolder, self.embeddingsFilename + SaveEmbeddings.EMBEDDINGS_FILE_EXTENSION), "ab") as outputFile:
                pickle.dump(self.allEmbeddings, outputFile, pickle.HIGHEST_PROTOCOL)
                # for i in range(len(self.allEmbeddings)):
                #     outputFile.write("{}\t{}\t{}\n".format(self.allEmbeddings[i][0], self.allEmbeddings[i][1], self.allEmbeddings[i][2]))

        else:
            with open(os.path.join(self.embeddingsFolder, (self.embeddingsFilename + SaveEmbeddings.EMBEDDINGS_BIN_FILE_EXTENSION).format(str(self.currentBatch).zfill(6))), "wb") as outputFile:
                for whichEmbedding in self.allEmbeddingsBin:
                    outputFile.write(whichEmbedding)

        self.allEmbeddings = []
        self.allEmbeddingsBin = []