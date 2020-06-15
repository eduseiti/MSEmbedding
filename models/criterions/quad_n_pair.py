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

class QuadNPair(nn.Module):

    def __init__(self):

        super(QuadNPair, self).__init__()

        self.aggregation = Options()['model']['criterion'].get('aggregation', 'mean')
        self.epsilon = Options()['model']['criterion'].get('epsilon', 0.00000001)

        self.batch_size = Options().get("dataset.batch_size", 100)

        # torch.autograd.set_detect_anomaly(True)

        self.anchor_access_mask = torch.ones([self.batch_size, self.batch_size], dtype=torch.bool)
        self.positive_access_mask = torch.ones([self.batch_size, self.batch_size], dtype=torch.bool)

        for i in range(self.batch_size):
            self.anchor_access_mask[i, i] = False
            self.positive_access_mask[i, i] = False

        self.anchor_access_mask[1::2] = False
        self.positive_access_mask[0::2] = False

        if torch.cuda.is_available():
            self.anchor_access_mask = self.anchor_access_mask.cuda()  
            self.positive_access_mask = self.positive_access_mask.cuda()  


    def forward(self, networkOutput, batch):

        out = {}

        embeddings = networkOutput

        print("criterion: embeddings.shape={}".format(embeddings.shape))

        allEmbeddingsDotProduct = torch.mm(embeddings, embeddings.t())

        anchorPositiveDotProduct = allEmbeddingsDotProduct.diag(diagonal=1).unsqueeze(1)[0::2]
        positiveAnchorDotProduct = allEmbeddingsDotProduct.diag(diagonal=-1).unsqueeze(1)[0::2]
        print("anchorPositiveDotProduct.shape={}".format(anchorPositiveDotProduct.shape))
        print("positiveAnchorDotProduct.shape={}".format(positiveAnchorDotProduct.shape))

        if embeddings.shape[0] < self.batch_size:
            anchorAllOtherDotProduct = torch.masked_select(allEmbeddingsDotProduct, 
                                                           self.anchor_access_mask[:embeddings.shape[0], :embeddings.shape[0]]).view(int(embeddings.shape[0] / 2), 
                                                                                                                                         embeddings.shape[0] - 1)

            positiveAllOtherDotProduct = torch.masked_select(allEmbeddingsDotProduct, 
                                                             self.positive_access_mask[:embeddings.shape[0], :embeddings.shape[0]]).view(int(embeddings.shape[0] / 2), 
                                                                                                                                             embeddings.shape[0] - 1)
        else:
            anchorAllOtherDotProduct = torch.masked_select(allEmbeddingsDotProduct, self.anchor_access_mask).view(int(self.batch_size / 2), self.batch_size - 1)
            positiveAllOtherDotProduct = torch.masked_select(allEmbeddingsDotProduct, self.positive_access_mask).view(int(self.batch_size / 2), self.batch_size - 1)

        print("anchorAllOtherDotProduct.shape={}".format(anchorAllOtherDotProduct.shape))
        print("positiveAllOtherDotProduct.shape={}".format(positiveAllOtherDotProduct.shape))


        # print("anchorAllOtherDotProduct[0]={}".format(anchorAllOtherDotProduct[0]))

        anchorDotProductDiferences = anchorAllOtherDotProduct - anchorPositiveDotProduct
        positiveDotProductDiferences = positiveAllOtherDotProduct - positiveAnchorDotProduct

        # print("dotProductDiferences[0]={}".format(dotProductDiferences[0]))

        anchorPartialLoss = torch.exp(anchorDotProductDiferences)
        positivePartialLoss = torch.exp(positiveDotProductDiferences)

        # print("partialLoss[0]={}".format(partialLoss[0]))
        # print("partialLoss.shape={}".format(partialLoss.shape))

        anchorExpSum = torch.sum(anchorPartialLoss, dim=1)
        positiveExpSum = torch.sum(positivePartialLoss, dim=1)

        print("anchorExpSum.shape={}".format(anchorExpSum.shape))
        print("positiveExpSum.shape={}".format(positiveExpSum.shape))

        print("cat.shape={}".format(torch.cat([anchorExpSum, positiveExpSum]).shape))

        loss = torch.log(anchorExpSum) + torch.log(positiveExpSum)

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