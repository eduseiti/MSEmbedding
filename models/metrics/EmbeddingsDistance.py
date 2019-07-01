import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from bootstrap.lib.options import Options

class EmbeddingsDistance(torch.nn.Module):

    def __init__(self):
        super(EmbeddingsDistance, self).__init__()


    def forward(self, criterionOutput, networkOutput, batch):

        validationResult = np.zeros((len(batch) // 3 * 2, 2), dtype = bool)

        for i in range(len(networkOutput) // 3):

            positiveDistance = torch.dist(networkOutput[i * 3], networkOutput[i * 3 + 1])
            negativeDistance = torch.dist(networkOutput[i * 3], networkOutput[i * 3 + 2])

            # print('{} - Positive distance={}, Negative distance={}'.format(i, positiveDistance, negativeDistance))

            lossMargin = Options()['model']['criterion']['loss_margin']

            if torch.lt(positiveDistance, lossMargin):
                validationResult[i * 2] = (True, True)
            else:
                validationResult[i * 2] = (True, False)
            
            if torch.lt(negativeDistance, lossMargin):
                validationResult[i * 2 + 1] = (True, False)
            else:
                validationResult[i * 2 + 1] = (True, True)


        out = {}

        totalAccuracyScore = accuracy_score(validationResult[:, 0], validationResult[:, 1])

        print('Validation accuracy: {}'.format(totalAccuracyScore))

        out['accuracy'] = totalAccuracyScore

        return out