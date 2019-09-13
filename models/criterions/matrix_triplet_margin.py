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
        self.epsilon = Options()['model']['criterion'].get('epsilon', 0.000001)
        self.variableMargin = Options()['model']['criterion'].get('variable_margin', False)
        
        if self.variableMargin:
            self.variableMarginStep = Options()['model']['criterion'].get('variable_margin_step', 0.1)


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

        anchorDistances = anchorPositiveDistances - torch.cat((allPositiveCosineDistances, allNegativeCosineDistances), dim = 1) 

        comparissonBase = anchorPositiveDistances.new_zeros(1)

        loss = torch.max(anchorDistances + self.margin, comparissonBase)

        loss[range(loss.shape[0]), range(loss.shape[0])] = 0.0

        if self.aggregation == "valid":
            if self.variableMargin:
                zeroedLosses = loss <= self.epsilon
                countOfNonZeroedLosses = loss.numel() - zeroedLosses.float().sum()

                loss2 = torch.max(anchorDistances + self.margin + self.variableMarginStep, comparissonBase)
                loss2[range(loss.shape[0]), range(loss.shape[0])] = 0.0
                
                countOfNonZeroedLosses_2 = (loss2[zeroedLosses] > self.epsilon).float().sum()

                out['loss'] = torch.sum(loss) / countOfNonZeroedLosses + torch.sum(loss2) / countOfNonZeroedLosses_2
            else:
                countOfNonZeroedLosses = (loss > self.epsilon).float().sum()

                out['loss'] = torch.sum(loss) / countOfNonZeroedLosses
        else:
            out['loss'] = torch.mean(loss)

        out['originalIndexes'] = originalIndexes

        return out