import torch
from torch import nn
from bootstrap.lib.options import Options

import pickle
import os


class MatrixTripletMargin(nn.Module):

    wrongCount = 0

    def __init__(self):

        super(MatrixTripletMargin, self).__init__()

        self.margin = Options()['model']['criterion']['loss_margin']
        self.aggregation = Options()['model']['criterion'].get('aggregation', 'mean')
        self.epsilon = Options()['model']['criterion'].get('epsilon', 0.00000001)


    def forward(self, networkOutput, batch):

        out = {}

        originalIndexes = torch.zeros(len(networkOutput[1]), dtype = torch.int32)

        for i in range(len(networkOutput[1])):
            originalIndexes[networkOutput[1][i]] = i

        embeddings = networkOutput[0]
        originalIndexes = originalIndexes.tolist()

        anchors = embeddings[originalIndexes[::3]]
        positive = embeddings[originalIndexes[1::3]]
        negative = embeddings[originalIndexes[2::3]]

        anchors = anchors.reshape(anchors.shape[0], -1)
        positive = positive.reshape(positive.shape[0], -1)
        negative = negative.reshape(negative.shape[0], -1)

        normalizedAnchors = nn.functional.normalize(anchors)

        allPositiveCosineDistances = 1 - torch.mm(normalizedAnchors, nn.functional.normalize(positive).t())
        allNegativeCosineDistances = 1 - torch.mm(normalizedAnchors, nn.functional.normalize(negative).t())

        # ltZeroCount = (allCosineDistances < 0).sum()
        # gtOneCount = (allCosineDistances > 1).sum()

        # if (ltZeroCount > 0) or (gtOneCount > 0):

        #     MatrixTripletMargin.wrongCount += 1

        #     print("*** Saving data due to unexpected allCosineDistances value. Count={}, Folder={}".format(MatrixTripletMargin.wrongCount, os.getcwd()))
        #     print("*** ltZeroCount={}, gtOneCount={}".format(ltZeroCount, gtOneCount))

        #     with open("allCosineDistances", 'wb') as outputFile:
        #         dataDump = {}
        #         dataDump['allCosineDistances'] = allCosineDistances
        #         dataDump["anchors"] = anchors
        #         dataDump["positive"] = positive
        #         dataDump["negative"] = negative
        #         pickle.dump(dataDump, outputFile)

        anchorPositiveDistances = allPositiveCosineDistances.diag().unsqueeze(1)

        loss = torch.max(anchorPositiveDistances - torch.cat((allPositiveCosineDistances, allNegativeCosineDistances), dim = 1) + self.margin, 
                         anchorPositiveDistances.new_zeros(1))

        loss[range(loss.shape[0]), range(loss.shape[0])] = 0.0

        print("-------------> aggregation={}".format(self.aggregation))

        if self.aggregation == "valid":
            non_zeroed_losses = (loss > self.epsilon).float().sum()

            out['loss'] = torch.sum(loss) / non_zeroed_losses
        else:
            out['loss'] = torch.mean(loss)

        out['originalIndexes'] = originalIndexes

        return out