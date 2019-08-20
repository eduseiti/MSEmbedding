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

class EmbeddingsDistance(torch.nn.Module):

    wrongCount = 0

    def __init__(self, engine = None, mode = 'train'):
        super(EmbeddingsDistance, self).__init__()

        self.mode = mode
        self.allEmbeddings = []

        if engine and mode == 'eval':
            self.split = engine.dataset[mode].split
            engine.register_hook('eval_on_end_epoch', self.calculate_MedR)


    def forward(self, criterionOutput, networkOutput, batch):

        embeddings = networkOutput[0]
        originalIndexes = criterionOutput['originalIndexes']

        self.allEmbeddings.append(embeddings[originalIndexes])



    def calculate_MedR(self):

        #
        # concatenate all embeddings
        #

        epochEmbeddings = torch.cat(self.allEmbeddings)

        print("epochEmbeddings.shape={}".format(epochEmbeddings.shape))

        epochEmbeddingsNorm = epochEmbeddings.reshape(epochEmbeddings.shape[0], -1)

        print("epochEmbeddingsNorm.shape={}".format(epochEmbeddingsNorm.shape))

        epochEmbeddingsNorm = nn.functional.normalize(epochEmbeddingsNorm)
        allCosineDistances = 1 - torch.mm(epochEmbeddingsNorm, epochEmbeddingsNorm.t())

        # ltZeroCount = (allCosineDistances < 0).sum()
        # gtOneCount = (allCosineDistances > 1).sum()

        # if (ltZeroCount > 0) or (gtOneCount > 0):

        #     EmbeddingsDistance.wrongCount += 1

        #     print("*** Saving data due to unexpected allCosineDistances value. Count={}, Folder={}".format(EmbeddingsDistance.wrongCount, os.getcwd()))
        #     print("*** ltZeroCount={}, gtOneCount={}".format(ltZeroCount, gtOneCount))

        #     with open("allCosineDistances", 'wb') as outputFile:
        #         dataDump = {}
        #         dataDump['allCosineDistances'] = allCosineDistances
        #         dataDump["anchors"] = anchors
        #         dataDump["positive"] = positive
        #         dataDump["negative"] = negative
        #         pickle.dump(dataDump, outputFile)


        ranks = []

        for i in range(len(epochEmbeddings) // 3):

            allCosineDistances[i * 3, i * 3] = -1 # Make sure the same embedding distance is always the first after sorting

            orderedDistancesFast = torch.argsort(allCosineDistances[i * 3])
            orderedListFast = orderedDistancesFast.tolist()

            #
            # Keep the rank of the positive example of the given anchor - it should be the nearest point.
            # Since the anchor itself is still in the list, and the distance to it will always be zero, decrement the
            # positive example ranking.
            #


            sameRankFast = orderedListFast.index(i * 3)
            positiveExampleRankFast = orderedListFast.index(i * 3 + 1) - 1
            negativeExampleRankFast = orderedListFast.index(i * 3 + 2) - 1

            # if (sameRankFast > 0 or allCosineDistances[orderedListFast[sameRankFast], orderedListFast[sameRankFast]] < 0):

            #     print('{} - Same rank Fast={}, Same distance Fast={}, Positive rank Fast={}, Negative rank Fast={}'.format(i, 
            #         sameRankFast, allCosineDistances[orderedListFast[sameRankFast], orderedListFast[sameRankFast]], 
            #         positiveExampleRankFast, negativeExampleRankFast))

            #     print("allCosineDistances={}".format(allCosineDistances[i * 3]))

            #     print("cwd={}".format(os.getcwd()))

            #     with open("allCosineDistances.pkl", 'wb') as outputFile:
            #         pickle.dump(allCosineDistances, outputFile, pickle.HIGHEST_PROTOCOL)                

            #     quit()



            ranks.append(positiveExampleRankFast)

            Logger()('{} - Same rank Fast={}, Same distance Fast={}, Positive rank Fast={}, Negative rank Fast={}'.format(i, 
                sameRankFast, allCosineDistances[orderedListFast[sameRankFast], orderedListFast[sameRankFast]], 
                positiveExampleRankFast, negativeExampleRankFast))


        out = {}

        MedR = np.mean(ranks)

        print('Validation MedR={}'.format(MedR))

        out['MedR'] = MedR

        for key, value in out.items():
            Logger().log_value('{}_epoch.metric.{}'.format(self.mode, key), float(value), should_print = True)

        self.allEmbeddings = []