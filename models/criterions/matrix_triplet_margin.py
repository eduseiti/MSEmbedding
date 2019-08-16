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


    def forward(self, networkOutput, batch):

        out = {}

        anchors = networkOutput[::3]
        positive = networkOutput[1::3]
        negative = networkOutput[2::3]

        anchors = anchors.reshape(anchors.shape[0], -1)
        positive = positive.reshape(positive.shape[0], -1)
        negative = negative.reshape(negative.shape[0], -1)

        allPositiveCosineDistances = 1 - torch.mm(nn.functional.normalize(anchors), nn.functional.normalize(positive).t())
        allNegativeCosineDistances = 1 - torch.mm(nn.functional.normalize(anchors), nn.functional.normalize(negative).t())

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

        loss = torch.max(anchorPositiveDistances - torch.cat((allPositiveCosineDistances, allNegativeCosineDistances), dim = 1) + self.margin, torch.zeros(1).cuda())

        loss[range(loss.shape[0]), range(loss.shape[0])] = 0.

        out['loss'] = torch.mean(loss)

        return out