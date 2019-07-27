import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
from scipy.spatial.distance import cdist

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

        # embeddings = networkOutput.cpu()
        # embeddings = embeddings.numpy()

        print('Dimensions: {}'.format(epochEmbeddings.shape))

        ranks = []
        ranksTorch = []

        # validationResult = np.zeros((len(batch) // 3 * 2, 2), dtype = bool)

        for i in range(len(epochEmbeddings) // 3):

            # distances = cdist(embeddings[i * 3].reshape(1, -1), embeddings.reshape(embeddings.shape[0], -1))

            epochEmbeddingsFixed = epochEmbeddings.contiguous()

            distancesTorch = torch.cdist(epochEmbeddingsFixed[i * 3].view(1, -1), 
                                         epochEmbeddingsFixed.view(epochEmbeddingsFixed.shape[0], -1))

            # print('==> distances.shape={}'.format(distances.shape))

            # orderedDistances = np.argsort(distances[0])

            print('==> distancesTorch.shape={}'.format(distancesTorch.shape))

            orderedDistancesTorch = torch.argsort(distancesTorch[0])

            #
            # Keep the rank of the positive example of the given anchor - it should be the nearest point.
            # Since the anchor itself is still in the list, and the distance to it will always be zero, decrement the
            # positive example ranking.
            #

            # orderedList = orderedDistances.tolist()
            orderedList = orderedDistancesTorch.tolist()

            sameRank = orderedList.index(i * 3)
            positiveExampleRank = orderedList.index(i * 3 + 1) - 1
            negativeExampleRank = orderedList.index(i * 3 + 2) - 1

            ranks.append(positiveExampleRank)
            # ranksTorch.append(postiveExampleRankTorch)

            # Logger()('{} - Same rank={}, Same distance={}, Positive rank={}, Negative rank={}'.format(i, 
            #     sameRank, distances[0, orderedList[sameRank]], positiveExampleRank, negativeExampleRank))

            Logger()('{} - Same rank={}, Same distance={}, Positive rank={}, Negative rank={}'.format(i, 
                sameRank, distancesTorch[0, orderedList[sameRank]], positiveExampleRank, negativeExampleRank))


        out = {}

        MedR = np.mean(ranks)
        # MedR_torch = np.mean(ranksTorch)

        print('Validation MedR={}'.format(MedR))
        # print('Validation MedR={}, MedR_torch={}'.format(MedR, MedR_torch))

        out['MedR'] = MedR
        # out['MedR_torch'] = MedR_torch

        for key, value in out.items():
            Logger().log_value('{}_epoch.metric.{}'.format(self.mode, key), float(value), should_print = True)

        self.allEmbeddings = []