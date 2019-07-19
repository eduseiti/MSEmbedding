import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
from scipy.spatial.distance import cdist

class EmbeddingsDistance(torch.nn.Module):

    def __init__(self):
        super(EmbeddingsDistance, self).__init__()


    def forward(self, criterionOutput, networkOutput, batch):

        embeddings = networkOutput.cpu()
        embeddings = embeddings.numpy()

        print('Dimensions: {}'.format(embeddings.shape))

        ranks = []
        ranksTorch = []

        # validationResult = np.zeros((len(batch) // 3 * 2, 2), dtype = bool)

        for i in range(len(networkOutput) // 3):

            distances = cdist(embeddings[i * 3].reshape(1, -1), embeddings.reshape(embeddings.shape[0], -1))

            netOutputFixed = networkOutput.contiguous()

            distancesTorch = torch.cdist(netOutputFixed[i * 3].view(1, -1), netOutputFixed.view(netOutputFixed.shape[0], -1))

            orderedDistances = np.argsort(distances)

            orderedDistancesTorch = np.argsort(distancesTorch.cpu())

            #
            # Keep the rank of the positive example of the given anchor - it should be the nearest point.
            # Since the anchor itself is still in the list, and the distance to it will always be zero, decrement the
            # positive example ranking.
            #

            positiveExampleRank = orderedDistances[0].tolist().index(i * 3 + 1) - 1
            negativeExampleRank = orderedDistances[0].tolist().index(i * 3 + 2) - 1

            postiveExampleRankTorch = orderedDistancesTorch[0].tolist().index(i * 3 + 1) - 1

            ranks.append(positiveExampleRank)
            ranksTorch.append(postiveExampleRankTorch)

            Logger()('{} - Positive rank={}, Positive rank torch={} Negative rank={}'.format(i, positiveExampleRank, postiveExampleRankTorch, negativeExampleRank))


        out = {}

        MedR = np.mean(ranks)
        MedR_torch = np.mean(ranksTorch)

        print('Validation MedR={}, MedR_torch={}'.format(MedR, MedR_torch))

        out['MedR'] = MedR
        out['MedR_torch'] = MedR_torch

        return out