import torch
from torch import nn
from bootstrap.lib.options import Options

import pickle
import os


#
# This criterion only works with batches containing only (anchor, positive example), 
# not including negative examples.
#
#

class NPair(nn.Module):

    def __init__(self):

        super(NPair, self).__init__()

        self.aggregation = Options()['model']['criterion'].get('aggregation', 'mean')
        self.epsilon = Options()['model']['criterion'].get('epsilon', 0.00000001)

        # torch.autograd.set_detect_anomaly(True)



    def forward(self, networkOutput, batch):

        out = {}

        embeddings = networkOutput

        # print("criterion: embeddings.shape={}".format(embeddings.shape))

        anchors = embeddings[::2]
        positive = embeddings[1::2]

        print("anchors[0]={}".format(anchors[0]))
        print("positive[0]={}".format(positive[0]))

        allAnchorsAndPositivesDotProduct = torch.mm(anchors, positive.t())

        anchorPositiveDotProduct = allAnchorsAndPositivesDotProduct.diag().unsqueeze(1)

        # print("anchorPositiveDotProduct[0]={}".format(anchorPositiveDotProduct[0]))

        dotProductDiferences = allAnchorsAndPositivesDotProduct - anchorPositiveDotProduct

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