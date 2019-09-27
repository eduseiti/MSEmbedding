import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .datasets import HumanProteome

import numpy as np

import random

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import time

import pickle


BATCH_SIZE = 1002
EPOCHS = 700

MedRs = []
recalls_at_1 = []
recalls_at_5 = []
recalls_at_10 = []
recalls_at_100 = []

def main():

    dataset = HumanProteome.HumanProteome(
        "data/humanProteome",
        "eval",
        BATCH_SIZE,
        nb_threads = 1, 
        trainingDataset = None)


    for epoch in range(EPOCHS):

        batch_loader = dataset.make_batch_loader()

        spectra = []

        for i, batch in enumerate(batch_loader):
            # print(batch)
            # print("Appending spectrum {}".format(i))

            spectra.append(batch['peaks'])

        all_spectra = torch.cat(spectra)

        if torch.cuda.is_available():
            all_spectra = all_spectra.cuda()

        print("all_spectra.shape={}".format(all_spectra.shape))

        # print("Current directory: {}".format(os.getcwd()))

        # with open("all_spectra.pkl", 'wb') as outputFile:
        #     pickle.dump(all_spectra, outputFile, pickle.HIGHEST_PROTOCOL)      

        # quit()

        all_spectra = all_spectra[:, :, 0] * all_spectra[:, :, 1]

        all_spectra_norm = nn.functional.normalize(all_spectra)

        print("new all_spectra.shape={}".format(all_spectra_norm.shape))

        allCosineDistances = 1 - torch.mm(all_spectra_norm, all_spectra_norm.t())


        ranks = []

        recall_at_1 = 0
        recall_at_5 = 0
        recall_at_10 = 0
        recall_at_100 = 0

        for i in range(len(all_spectra) // 3):

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

            if positiveExampleRankFast == 0:
                recall_at_1 += 1

            if positiveExampleRankFast <= 4:
                recall_at_5 += 1

            if positiveExampleRankFast <= 9:
                recall_at_10 += 1

            if positiveExampleRankFast <= 99:
                recall_at_100 += 1


            # print('{} - Same rank Fast={}, Same distance Fast={}, Positive rank Fast={}, Negative rank Fast={}'.format(i, 
            #     sameRankFast, allCosineDistances[orderedListFast[sameRankFast], orderedListFast[sameRankFast]], 
            #     positiveExampleRankFast, negativeExampleRankFast))



        MedR = np.median(ranks)


        MedRs.append(MedR)
        recalls_at_1.append(recall_at_1 / len(ranks))
        recalls_at_5.append(recall_at_5 / len(ranks))
        recalls_at_10.append(recall_at_10 / len(ranks))
        recalls_at_100.append(recall_at_100 / len(ranks))

        print("Epoch {}".format(epoch))
        print('- Validation MedR={}'.format(MedR))
        print("- Recall@1={}".format(recall_at_1 / len(ranks)))
        print("- Recall@5={}".format(recall_at_5 / len(ranks)))
        print("- Recall@10={}".format(recall_at_10 / len(ranks)))
        print("- Recall@100={}".format(recall_at_100 / len(ranks)))


    print("\n\nOverall results after {} epochs".format(EPOCHS))
    print('- MedR={}'.format(np.mean(MedRs)))
    print("- Recall@1={}".format(np.mean(recalls_at_1)))
    print("- Recall@5={}".format(np.mean(recalls_at_5)))
    print("- Recall@10={}".format(np.mean(recalls_at_10)))
    print("- Recall@100={}".format(np.mean(recalls_at_100)))

    print("\n\nBest results after {} epochs".format(EPOCHS))
    print('- MedR={}'.format(np.min(MedRs)))
    print("- Recall@1={}".format(np.max(recalls_at_1)))
    print("- Recall@5={}".format(np.max(recalls_at_5)))
    print("- Recall@10={}".format(np.max(recalls_at_10)))
    print("- Recall@100={}".format(np.max(recalls_at_100)))


if __name__ == '__main__':
    main()

    