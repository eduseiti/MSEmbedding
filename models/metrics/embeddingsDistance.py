import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class EmbeddingDistance(nn.Module):

    def __init__(self):
        super(EmbeddingDistance, self).__init__()


    def forward(self, criterionOutput, networkOutput, batch):

        validationResult = np.zeros((batch.len / 3 * 2, 2), dtype = bool)

        for i in range(len(networkOutput) // 3):

            positiveDistance = torch.dist(networkOutput[i * 3], networkOutput[i * 3 + 1])
            negativeDistance = torch.dist(networkOutput[i * 3], networkOutput[i * 3 + 2])

            # print('{} - Positive distance={}, Negative distance={}'.format(i, positiveDistance, negativeDistance))

            if torch.lt(positiveDistance, LOSS_MARGIN):
                validationResult[i * 2] = (True, True)
            else:
                validationResult[i * 2] = (True, False)
            
            if torch.lt(negativeDistance, LOSS_MARGIN):
                validationResult[i * 2 + 1] = (True, False)
            else:
                validationResult[i * 2 + 1] = (True, True)


        totalAccuracyScore = accuracy_score(validationResult[:, 0], validationResult[:, 1])

        print('Validation accuracy: {}'.format(totalAccuracyScore))

        return totalAccuracyScore