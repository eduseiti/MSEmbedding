import torch
from torch import nn
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

import pickle
import os


#
# This criterion only works with batches containing only (anchor, positive example), 
# not including negative examples.
#
#

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

        # print("criterion: embeddings.shape={}".format(embeddings.shape))

        anchors = embeddings[originalIndexes[::2]]
        positive = embeddings[originalIndexes[1::2]]

        # print("criterion: anchors.shape={}, positive.shape={}".format(anchors.shape, positive.shape))
        # print("anchors[0]={}".format(anchors[0]))
        # print("positive[0]={}".format(positive[0]))


        # anchors = anchors.reshape(anchors.shape[0], -1)
        # positive = positive.reshape(positive.shape[0], -1)

        # print("criterion 2: anchors.shape={}, positive.shape={}".format(anchors.shape, positive.shape))
        # print("anchors[0]={}".format(anchors[0]))
        # print("positive[0]={}".format(positive[0]))

        normalizedAnchors = nn.functional.normalize(anchors)

        allPositiveCosineDistances = 1 - torch.mm(normalizedAnchors, nn.functional.normalize(positive).t())

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
        #         pickle.dump(dataDump, outputFile)

        anchorPositiveDistances = allPositiveCosineDistances.diag().unsqueeze(1)

        anchorDistances = anchorPositiveDistances - allPositiveCosineDistances
        comparissonBase = anchorPositiveDistances.new_zeros(1)

        loss = torch.max(anchorDistances + self.margin, comparissonBase)

        loss[range(loss.shape[0]), range(loss.shape[0])] = 0.0

        if self.aggregation == "valid":
            if self.variableMargin:
                nonZeroedLosses = loss > self.epsilon
                zeroedLosses = loss <= self.epsilon
                countOfNonZeroedLosses = loss.numel() - zeroedLosses.float().sum()

                Logger()("zeroedLosses={}, countOfNonZeroedLosses={}".format(zeroedLosses.float().sum(), countOfNonZeroedLosses))

                aggregatedLoss = torch.sum(loss[nonZeroedLosses]) / countOfNonZeroedLosses
                aggregatedLoss2 = 0.0

                if countOfNonZeroedLosses > 0:

                    loss2 = torch.max(anchorDistances + self.margin + self.variableMarginStep, comparissonBase)
                    loss2[range(loss2.shape[0]), range(loss2.shape[0])] = 0.0

                    countOfNonZeroedLosses_2 = (loss2[zeroedLosses] > self.epsilon).float().sum()

                    Logger()("countOfNonZeroedLosses_2={}".format(countOfNonZeroedLosses_2))
                    
                    if countOfNonZeroedLosses_2 > 0:
                        aggregatedLoss2 = torch.sum(loss2[zeroedLosses]) / countOfNonZeroedLosses_2

                    Logger()("Loss={}, Loss2={}".format(aggregatedLoss, aggregatedLoss2))

                out['loss'] = aggregatedLoss + aggregatedLoss2
            else:
                countOfNonZeroedLosses = (loss > self.epsilon).float().sum()

                Logger()("countOfNonZeroedLosses={}".format(countOfNonZeroedLosses))

                out['loss'] = torch.sum(loss) / countOfNonZeroedLosses
        else:
            out['loss'] = torch.mean(loss)

        out['originalIndexes'] = originalIndexes

        return out
