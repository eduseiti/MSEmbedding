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

    def __init__(self, engine = None, mode = 'train'):
        super(EmbeddingsDistance, self).__init__()

        self.mode = mode
        self.allEmbeddings = []

        if engine and mode == 'eval':
            self.split = engine.dataset[mode].split
            engine.register_hook('eval_on_end_epoch', self.calculate_MedR)


    def forward(self, criterionOutput, networkOutput, batch):

        self.allEmbeddings.append(networkOutput)



    def calculate_MedR(self):

        #
        # concatenate all embeddings
        #

        epochEmbeddings = torch.cat(self.allEmbeddings)

        print("epochEmbeddings.shape={}".format(epochEmbeddings.shape))

        epochEmbeddingsNorm = epochEmbeddings.reshape(epochEmbeddings.shape[0], -1)

        print("epochEmbeddingsNorm.shape={}".format(epochEmbeddingsNorm.shape))

        epochEmbeddingsNorm = nn.functional.normalize(epochEmbeddingsNorm)
        allCosineDistances = torch.max(1 - torch.mm(epochEmbeddingsNorm, epochEmbeddingsNorm.t()), torch.zeros(1).cuda())


        # # scipy.cdist - begin

        # epochEmbeddings = epochEmbeddings.cpu()
        # epochEmbeddings = epochEmbeddings.numpy()

        # # scipy.cdist - end

        # print('Dimensions: {}'.format(epochEmbeddings.shape))

        ranks = []

        for i in range(len(epochEmbeddings) // 3):

            # # scipy.cdist - begin

            # distances = cdist(epochEmbeddings[i * 3].reshape(1, -1), epochEmbeddings.reshape(epochEmbeddings.shape[0], -1), metric = 'cosine')

            # print('==> distances.shape={}'.format(distances.shape))

            # orderedDistances = np.argsort(distances[0])
            # orderedList      = orderedDistances.tolist()

            # # scipy.cdist - end

            orderedDistancesFast = torch.argsort(allCosineDistances[i * 3])
            orderedListFast = orderedDistancesFast.tolist()


            # torch.cdist - begin

            # epochEmbeddingsFixed = epochEmbeddings.contiguous()

            # distances = torch.cdist(epochEmbeddingsFixed[i * 3].view(1, -1), 
            #                              epochEmbeddingsFixed.view(epochEmbeddingsFixed.shape[0], -1))

            # print('==> distances.shape={}'.format(distances.shape))

            # orderedDistancesTorch = torch.argsort(distances[0])
            # orderedList           = orderedDistancesTorch.tolist()

            # torch.cdist - end


            #
            # Keep the rank of the positive example of the given anchor - it should be the nearest point.
            # Since the anchor itself is still in the list, and the distance to it will always be zero, decrement the
            # positive example ranking.
            #


            sameRankFast = orderedListFast.index(i * 3)
            positiveExampleRankFast = orderedListFast.index(i * 3 + 1) - 1
            negativeExampleRankFast = orderedListFast.index(i * 3 + 2) - 1

            if (sameRankFast > 0 or allCosineDistances[orderedListFast[sameRankFast], orderedListFast[sameRankFast]] < 0):

                print('{} - Same rank Fast={}, Same distance Fast={}, Positive rank Fast={}, Negative rank Fast={}'.format(i, 
                    sameRankFast, allCosineDistances[orderedListFast[sameRankFast], orderedListFast[sameRankFast]], 
                    positiveExampleRankFast, negativeExampleRankFast))

                print("allCosineDistances={}".format(allCosineDistances[i * 3]))

                print("cwd={}".format(os.getcwd()))

                with open("allCosineDistances.pkl", 'wb') as outputFile:
                    pickle.dump(allCosineDistances, outputFile, pickle.HIGHEST_PROTOCOL)                

                quit()



            ranks.append(positiveExampleRankFast)

            # sameRank = orderedList.index(i * 3)
            # positiveExampleRank = orderedList.index(i * 3 + 1) - 1
            # negativeExampleRank = orderedList.index(i * 3 + 2) - 1

            # ranks.append(positiveExampleRank)

            # Logger()('{} - Same rank={}, Same distance={}, Positive rank={}, Negative rank={}'.format(i, 
            #     sameRank, distances[0, orderedList[sameRank]], positiveExampleRank, negativeExampleRank))

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