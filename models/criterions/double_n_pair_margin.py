import torch
from torch import nn
from bootstrap.lib.options import Options

import pickle
import os

#
# Implemented according to:
# 
# Liu, Xiaofeng, B. V. K. Vijaya Kumar, Jane You, and Ping Jia. "Adaptive deep metric learning 
# for identity-aware facial expression recognition." In Proceedings of the IEEE Conference on 
# Computer Vision and Pattern Recognition Workshops, pp. 20-29. 2017.
#



#
# This criterion only works with batches containing only (anchor, positive example), 
# not including negative examples.
#
#

class DoubleNPairMargin(nn.Module):

    def __init__(self):

        super(DoubleNPairMargin, self).__init__()

        self.margin = Options()['model']['criterion']['loss_margin']
        self.aggregation = Options()['model']['criterion'].get('aggregation', 'mean')
        self.epsilon = Options()['model']['criterion'].get('epsilon', 0.00000001)

        self.batch_size = Options().get("dataset.batch_size", 100)

        # torch.autograd.set_detect_anomaly(True)

        self.access_mask = torch.ones([self.batch_size, self.batch_size], dtype=torch.bool)

        for i in range(self.batch_size):
            self.access_mask[i, i] = False

        self.access_mask[1::2] = False

        if torch.cuda.is_available():
            self.access_mask = self.access_mask.cuda()  


    def forward(self, networkOutput, batch):

        out = {}

        embeddings = networkOutput

        # normalize the anchors and positive examples separately 

        anchors = embeddings[::2]
        positive = embeddings[1::2]

        anchors = nn.functional.normalize(anchors)
        positive = nn.functional.normalize(positive)

        print("criterion: embeddings.shape={}".format(embeddings.shape))

        allEmbeddingsCosineDistances = 1 - torch.mm(embeddings, embeddings.t())

        anchorPositiveCosineDistances = allEmbeddingsCosineDistances.diag(diagonal=1).unsqueeze(1)[0::2]
        print("anchorPositiveCosineDistances.shape={}".format(anchorPositiveCosineDistances.shape))

        if embeddings.shape[0] < self.batch_size:
            anchorAllOtherCosineDistance = torch.masked_select(allEmbeddingsCosineDistances, self.access_mask[:embeddings.shape[0], :embeddings.shape[0]]).view(int(embeddings.shape[0] / 2), embeddings.shape[0] - 1)
        else:
            anchorAllOtherCosineDistance = torch.masked_select(allEmbeddingsCosineDistances, self.access_mask).view(int(self.batch_size / 2), self.batch_size - 1)

        print("anchorAllOtherCosineDistance.shape={}".format(anchorAllOtherCosineDistance.shape))


        # print("anchorAllOtherCosineDistance[0]={}".format(anchorAllOtherCosineDistance[0]))

        dotProductDiferences = anchorPositiveCosineDistances - anchorAllOtherCosineDistance + self.margin

        # print("dotProductDiferences[0]={}".format(dotProductDiferences[0]))

        partialLoss = torch.exp(dotProductDiferences)

        # print("partialLoss[0]={}".format(partialLoss[0]))
        # print("partialLoss.shape={}".format(partialLoss.shape))

        expSum = torch.sum(partialLoss, dim=1)

        # print("expSum[0]={}".format(expSum[0]))

        loss = torch.log(expSum)

        # print("loss.shape={}".format(loss.shape))

        if self.aggregation == "valid":
            non_zeroed_losses = (loss > self.epsilon).float().sum()

            print("non_zeroed_losses={}".format(non_zeroed_losses))

            if non_zeroed_losses > 0.0:
                out['loss'] = torch.sum(loss) / non_zeroed_losses
            else:
                out['loss'] = torch.mean(loss)
        else:
            out['loss'] = torch.mean(loss)

        return out