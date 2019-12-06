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

class SaveEmbeddings(torch.nn.Module):

    EMBEDDINGS_FILENAME = "spectra_embeddings.pkl"

    wrongCount = 0

    def __init__(self, engine = None, mode = 'test'):
        super(SaveEmbeddings, self).__init__()

        self.mode = mode
        self.allEmbeddings = []

        if engine and mode == 'test':
            engine.register_hook('eval_on_end_epoch', self.save_all_embeddings)


    def forward(self, criterionOutput, networkOutput, batch):

        embeddings = networkOutput[0]
        originalIndexes = criterionOutput['originalIndexes']

        self.allEmbeddings.append(embeddings[originalIndexes])



    def save_all_embeddings(self):

        print("save_all_embeddings. Current directory: {}".format(os.getcwd()))

        with open(EMBEDDINGS_FILENAME, "wb") as outputFile:
            pickle.dump(self.allEmbeddings, outputFile, pickle.HIGHEST_PROTOCOL)


        self.allEmbeddings = []