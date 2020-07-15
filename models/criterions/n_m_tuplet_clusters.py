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

class NMTupletClusters(nn.Module):

    def __init__(self):

        super(NMTupletClusters, self).__init__()

        self.reference_distance = Options()['model']['criterion']['reference_distance']
        self.margin = Options()['model']['criterion']['loss_margin']
        self.aggregation = Options()['model']['criterion'].get('aggregation', 'mean')
        self.epsilon = Options()['model']['criterion'].get('epsilon', 0.00000001)

        self.batch_size = Options().get("dataset.batch_size", 100)


    def forward(self, networkOutput, batch):

        out = {}

        embeddings = networkOutput

        # split the samples by their clusters

        if 'numOfClusters' in batch:
            numOfClusters = batch['numOfClusters']
            samplesClusters = embeddings[:-batch['paddedSamples']].chunk(numOfClusters)
        else:
            numOfClusters = embeddings.shape[0] // 2
            samplesClusters = embeddings.chunk(numOfClusters)

        # compute the normalized positive clusters centroids

        positiveCentroids = nn.functional.normalize(torch.mean(torch.stack(samplesClusters), 1))

        print("positiveCentroids.shape={}".format(positiveCentroids.shape))


        # normalize the clusters positive examples separately 

        # embeddings = torch.cat([nn.functional.normalize(cluster) for cluster in samplesClusters])
        embeddings = nn.functional.normalize(embeddings)

        print("embeddings.shape={}".format(embeddings.shape))


        allEmbeddingsCosineDistances = 1 - torch.mm(positiveCentroids, embeddings.t())

        print("allEmbeddingsCosineDistances.shape={}".format(allEmbeddingsCosineDistances.shape))


        clustersChunkedAllEmbeddingsCosineDistances = torch.stack(allEmbeddingsCosineDistances.chunk(numOfClusters, dim=1), dim=0)

        print("clustersChunkedAllEmbeddingsCosineDistances.shape={}".format(clustersChunkedAllEmbeddingsCosineDistances.shape))


        positiveClusterDistances = clustersChunkedAllEmbeddingsCosineDistances.diagonal().t()

        print("positiveClusterDistances.shape={}".format(positiveClusterDistances.shape))


        # create a access mask to get the diagonal clusters, which corresponds to the positive samples distance

        non_diagonal_mask = torch.ones([numOfClusters, numOfClusters, embeddings.shape[0] // numOfClusters], dtype=torch.bool)
        non_diagonal_mask[range(numOfClusters), range(numOfClusters)] = False

        if torch.cuda.is_available():
            non_diagonal_mask = non_diagonal_mask.cuda()  


        negativeSamplesDistances = clustersChunkedAllEmbeddingsCosineDistances.masked_select(non_diagonal_mask).view(numOfClusters, numOfClusters - 1, -1)

        print("negativeSamplesDistances.shape={}".format(negativeSamplesDistances.shape))


        # print("positiveClusterDistances={}".format(positiveClusterDistances))


        positiveSamplesLoss = torch.max(positiveClusterDistances.new_zeros(1), positiveClusterDistances.reshape(-1, 1) - self.reference_distance + (self.margin / 2))
        negativeSamplesLoss = torch.max(negativeSamplesDistances.new_zeros(1), self.reference_distance + (self.margin / 2) - negativeSamplesDistances.reshape(-1, 1))

        print("positiveSamplesLoss.shape={}".format(positiveSamplesLoss.shape))
        print("negativeSamplesLoss.shape={}".format(negativeSamplesLoss.shape))



        if self.aggregation == "valid":
            nonZeroedPositiveSamplesLosses = (positiveSamplesLoss > self.epsilon).float().sum()
            nonZeroedNegativeSamplesLosses = (negativeSamplesLoss > self.epsilon).float().sum()

            print("nonZeroedPositiveSamplesLosses={}".format(nonZeroedPositiveSamplesLosses))
            print("nonZeroedNegativeSamplesLosses={}".format(nonZeroedNegativeSamplesLosses))

            if nonZeroedPositiveSamplesLosses > 0.0:
                out['loss'] = torch.sum(positiveSamplesLoss) / nonZeroedPositiveSamplesLosses
            else:
                out['loss'] = torch.mean(positiveSamplesLoss)

            if nonZeroedNegativeSamplesLosses > 0.0:
                out['loss'] += torch.sum(negativeSamplesLoss) / nonZeroedNegativeSamplesLosses
            else:
                out['loss'] += torch.mean(negativeSamplesLoss)
        else:
            out['loss'] = torch.mean(positiveSamplesLoss) + torch.mean(negativeSamplesLoss)

        return out