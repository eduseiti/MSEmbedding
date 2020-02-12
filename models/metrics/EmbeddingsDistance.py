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

    PERCENTILES = list(range(0, 105, 5))

    wrongCount = 0

    def __init__(self, engine = None, mode = 'train'):
        super(EmbeddingsDistance, self).__init__()

        self.mode = mode
        self.allEmbeddings = []

        if Options().get("dataset.include_negative", False):
            self.examplesPerSequence = 3
        else:
            self.examplesPerSequence = 2

        if engine and mode == 'eval':
            self.split = engine.dataset[mode].split
            engine.register_hook('eval_on_end_epoch', self.calculate_MedR)


    def forward(self, criterionOutput, networkOutput, batch):

        embeddings = networkOutput[0]
        originalIndexes = criterionOutput['originalIndexes']

        self.allEmbeddings.append(embeddings[originalIndexes])
        # self.allEmbeddings.append(networkOutput)



    def calculate_MedR(self):

        #
        # concatenate all embeddings
        #

        epochEmbeddings = torch.cat(self.allEmbeddings)

        print("epochEmbeddings.shape={}".format(epochEmbeddings.shape))

        epochEmbeddingsNorm = nn.functional.normalize(epochEmbeddings)
        allCosineDistances = 1 - torch.mm(epochEmbeddingsNorm, epochEmbeddingsNorm.t())


        #
        # Calculate distances statistics
        #

        allCosineDistances_np = allCosineDistances.cpu().numpy()

        cosDist_percentiles = np.percentile(allCosineDistances_np, EmbeddingsDistance.PERCENTILES)
        cosDist_mean = np.mean(allCosineDistances_np)
        cosDist_std = np.std(allCosineDistances_np)
        cosDist_max = np.amax(allCosineDistances_np)
        cosDist_min = np.amin(allCosineDistances_np)

        cosDist_histogram, cosDist_bin_edges = np.histogram(allCosineDistances_np, 1000)

        Logger()('cosine distance stats\nmean={}, std={}, max={}, min={}'.format(cosDist_mean, cosDist_std, cosDist_max, cosDist_min))


        output = ""

        for i in range(len(EmbeddingsDistance.PERCENTILES)):
            output = output + "{}\t{:.6f}\n".format(EmbeddingsDistance.PERCENTILES[i], cosDist_percentiles[i])

        Logger()('Percentiles:\n{}\n'.format(output))


        output = ""

        for i in range(len(cosDist_histogram)):
            output = output + "a{}\t{:.6f}\t{:.6f}\n".format(cosDist_histogram[i], cosDist_bin_edges[i], cosDist_bin_edges[i + 1])

        Logger()('Histogram:\n{}\n'.format(output))



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

        recall_at_1 = 0
        recall_at_5 = 0
        recall_at_10 = 0

        for i in range(len(epochEmbeddings) // self.examplesPerSequence):

            allCosineDistances[i * self.examplesPerSequence, i * self.examplesPerSequence] = -1 # Make sure the same embedding distance is always the first after sorting

            orderedDistancesFast = torch.argsort(allCosineDistances[i * self.examplesPerSequence])
            orderedListFast = orderedDistancesFast.tolist()

            #
            # Keep the rank of the positive example of the given anchor - it should be the nearest point.
            # Since the anchor itself is still in the list, and the distance to it will always be zero, decrement the
            # positive example ranking.
            #


            sameRankFast = orderedListFast.index(i * self.examplesPerSequence)
            positiveExampleRankFast = orderedListFast.index(i * self.examplesPerSequence + 1) - 1

            # if (sameRankFast > 0 or allCosineDistances[orderedListFast[sameRankFast], orderedListFast[sameRankFast]] < 0):

            #     print('{} - Same rank Fast={}, Same distance Fast={}, Positive rank Fast={}'.format(i, 
            #         sameRankFast, allCosineDistances[orderedListFast[sameRankFast], orderedListFast[sameRankFast]], 
            #         positiveExampleRankFast))

            #     print("allCosineDistances={}".format(allCosineDistances[i * self.examplesPerSequence]))

            #     print("cwd={}".format(os.getcwd()))

            #     with open("allCosineDistances.pkl", 'wb') as outputFile:
            #         pickle.dump(allCosineDistances, outputFile, pickle.HIGHEST_PROTOCOL)                

            #     quit()


            ranks.append(positiveExampleRankFast)

            if positiveExampleRankFast == 0:
                recall_at_1 += 1

            if positiveExampleRankFast <= 4:
                recall_at_5 += 1

            if positiveExampleRankFast <= 9:
                recall_at_10 += 1

            Logger()('{} - Same rank Fast={}, Same distance Fast={}, Positive rank Fast={}'.format(i, 
                sameRankFast, allCosineDistances[orderedListFast[sameRankFast], orderedListFast[sameRankFast]], 
                positiveExampleRankFast))


        out = {}

        MedR = np.median(ranks)

        print('Validation MedR={}'.format(MedR))

        out['MedR'] = MedR
        out['recall_at_1'] = recall_at_1 / len(ranks)
        out['recall_at_5'] = recall_at_5 / len(ranks)
        out['recall_at_10'] = recall_at_10 / len(ranks)

        for key, value in out.items():
            Logger().log_value('{}_epoch.metric.{}'.format(self.mode, key), float(value), should_print = True)

        self.allEmbeddings = []