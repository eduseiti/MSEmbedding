import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
from scipy.spatial.distance import cdist

import os
import pickle
import sys

class SaveEmbeddings(torch.nn.Module):

    EMBEDDINGS_FOLDER = "data/linfeng"
    EMBEDDINGS_FILENAME = "spectra_embeddings_{}.pkl"

    SAVE_COUNT = 20000

    def __init__(self, engine = None, mode = 'test'):
        super(SaveEmbeddings, self).__init__()

        self.currentBatch = 0

        self.mode = mode
        self.allEmbeddings = []

        if engine and mode == 'test':
            engine.register_hook('eval_on_end_epoch', self.save_embeddings)


    def forward(self, criterionOutput, networkOutput, batch):

        originalIndexes = torch.zeros(len(networkOutput[1]), dtype = torch.int32)

        for i in range(len(networkOutput[1])):
            originalIndexes[networkOutput[1][i]] = i

        embeddings = networkOutput[0]
        originalIndexes = originalIndexes.tolist()

        print("**** size={}".format(embeddings.element_size() * embeddings.nelement()))
        print("**** element_size={}, nelement={}".format(embeddings.element_size(), embeddings.nelement()))
        print("**** shape={}".format(embeddings.shape))
        
        lastHiddenState = embeddings[originalIndexes].cpu()[range(embeddings.shape[0]), batch['peaksLen'] - 1, :]

        print("**** shape lastHiddenState={}".format(lastHiddenState.shape))

        self.allEmbeddings.append(lastHiddenState)

        self.currentBatch += 1

        if self.currentBatch % SaveEmbeddings.SAVE_COUNT == 0:
            self.save_embeddings()


    def save_embeddings(self):

        print("-- save_embeddings. currentBatch={}".format(self.currentBatch))

        with open(os.path.join(SaveEmbeddings.EMBEDDINGS_FOLDER, SaveEmbeddings.EMBEDDINGS_FILENAME.format(str(self.currentBatch).zfill(6))), "wb") as outputFile:
            pickle.dump(self.allEmbeddings, outputFile, pickle.HIGHEST_PROTOCOL)

        self.allEmbeddings = []