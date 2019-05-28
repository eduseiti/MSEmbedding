import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import MSEmbedding as m
import spectra as s
import batchLoader as b

import numpy as np

import random

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import time



os.chdir('../PXD000561')

originalData = s.PXD000561('Gel_Elite_49.csv', 'initialTest2.pkl')

originalData.load_identifications(True, singleFile = 'b01')

if not originalData.totalSpectra.spectra:
    originalData.read_spectra(s.MGF())

totalSpectra = originalData.totalSpectra

bl = b.batchLoader(totalSpectra)
bl.listMultipleScansSequences()

multipleScansSequences = bl.multipleScansSequences



testData = s.PXD000561('Gel_Elite_49.csv', 'initialTest2_b02.pkl')

testData.load_identifications(True, singleFile = 'b02')

if not testData.totalSpectra.spectra:
    testData.read_spectra(s.MGF())

testSpectra = testData.totalSpectra

blTest = b.batchLoader(testSpectra)
blTest.listMultipleScansSequences()

unrecognizedTestLen = len(testSpectra.spectra[s.scan.UNRECOGNIZED_SEQUENCE])

print('Test sequences size {}'.format(len(blTest.multipleScansSequences)))




# taken from Cross-Modal retrieval article
LOSS_MARGIN = 0.3 

embeddingNet = m.MSEmbeddingNet(2000, 16, 10, False).cuda()

optmizer = optim.SGD(embeddingNet.parameters(), lr = 0.01)


criterion = nn.TripletMarginLoss(margin = LOSS_MARGIN)

try:
    embeddingNet.load_state_dict(torch.load('embeddingNet.state'))
except Exception as e:
    print('Could not open trained model files')

newBatch = bl.createTripletBatch()

groupedBatch = bl.groupTriplets(newBatch)

embeddingNet.train()

HOW_MANY_SAMPLES = 100

embeddings    = torch.empty(HOW_MANY_SAMPLES, 3, m.LSTM_OUT_DIM)
embeddingsSeq = torch.empty(HOW_MANY_SAMPLES, 3, m.LSTM_OUT_DIM)

for i, sample in enumerate(groupedBatch[:HOW_MANY_SAMPLES]):
# for i, tripletKey in enumerate(list(newBatch.keys())[:HOW_MANY_SAMPLES]):

    print('\nSequence {}'.format(i))
    # print('\nSequence {}: {}'.format(i, tripletKey))

    # triplet = newBatch[tripletKey]

    # anchor   = totalSpectra.spectra[tripletKey][triplet['anchor']]['nzero_peaks'].cuda()
    # positive = totalSpectra.spectra[tripletKey][triplet['positive']]['nzero_peaks'].cuda()
    # negative = totalSpectra.spectra[s.scan.UNRECOGNIZED_SEQUENCE][triplet['negative']]['nzero_peaks'].cuda()

    startTime = time.time()

    # anchorEmbedding, positiveEmbedding, negativeEmbedding = embeddingNet(anchor, 
    #                                                                      positive, 
    #                                                                      negative)
    anchorEmbedding, positiveEmbedding, negativeEmbedding = embeddingNet(sample[0], 
                                                                         sample[1], 
                                                                         sample[2])

    print('T1: {}'.format(time.time() - startTime))

    print(anchorEmbedding.shape)

    embeddings[i][0] = anchorEmbedding[:, -1, :]
    embeddings[i][1] = positiveEmbedding[:, -1, :]
    embeddings[i][2] = negativeEmbedding[:, -1, :]

    # loss = criterion(anchorEmbedding[:, -1, :], positiveEmbedding[:, -1, :], negativeEmbedding[:, -1, :])

    # print('- Loss = {}'.format(loss))

    # loss.backward()
    # optmizer.step()


    startTime = time.time()
    # anchorEmbeddingSeq, positiveEmbeddingSeq, negativeEmbeddingSeq = embeddingNetSeq(anchor, 
    #                                                                                 positive, 
    #                                                                                 negative)
    anchorEmbeddingSeq, positiveEmbeddingSeq, negativeEmbeddingSeq = embeddingNetSeq(sample[0], 
                                                                                     sample[1], 
                                                                                     sample[2])

    print('T2: {}'.format(time.time() - startTime))

    embeddingsSeq[i][0] = anchorEmbeddingSeq
    embeddingsSeq[i][1] = positiveEmbeddingSeq
    embeddingsSeq[i][2] = negativeEmbeddingSeq



    # lossSeq = criterion(anchorEmbeddingSeq[0], positiveEmbeddingSeq[0], negativeEmbeddingSeq[0])

    # print('- LossSeq = {}'.format(lossSeq))

    # lossSeq.backward()
    # optmizerSeq.step()

loss = criterion(embeddings[:,0], embeddings[:,1], embeddings[:, 2])

print('- Loss = {}'.format(loss))

startTime = time.time()
optmizer.zero_grad()
loss.backward()
optmizer.step()
print('- Backprop time: {}'.format(time.time() - startTime))


lossSeq = criterion(embeddingsSeq[:,0], embeddingsSeq[:, 1], embeddingsSeq[:, 2])

print('- LossSeq = {}'.format(lossSeq))

startTime = time.time()
optmizerSeq.zero_grad()
lossSeq.backward()
optmizerSeq.step()
print('- Seq Backprop time: {}'.format(time.time() - startTime))



torch.save(embeddingNet.state_dict(), 'embeddingNet.state')
torch.save(embeddingNetSeq.state_dict(), 'embeddingNetSeq.state')



embeddingNet.eval()
embeddingNetSeq.eval()

random.shuffle(blTest.multipleScansSequences)

resultEmbedding = np.zeros((len(blTest.multipleScansSequences), 2), dtype = bool)
resultEmbeddingSeq = np.zeros((len(blTest.multipleScansSequences), 2), dtype = bool)

for i, testSequence in enumerate(blTest.multipleScansSequences[:50]):

    print('\nTest sequence {} = {}'.format(i, testSequence))

    anchor   = testSpectra.spectra[testSequence][0]['nzero_peaks'].cuda()
    positive = testSpectra.spectra[testSequence][1]['nzero_peaks'].cuda()
    negative = testSpectra.spectra[s.scan.UNRECOGNIZED_SEQUENCE][random.randrange(unrecognizedTestLen)]['nzero_peaks'].cuda()
    
    anchorEmbedding, positiveEmbedding, negativeEmbedding = embeddingNet(anchor, 
                                                                         positive, 
                                                                         negative)

    if torch.dist(anchorEmbedding[:, -1, :], positiveEmbedding[:, -1, :]) < LOSS_MARGIN:
        resultEmbedding[i] = (True, True)
    else:
        resultEmbedding[i] = (True, False)
    
    if torch.dist(anchorEmbedding[:, -1, :], negativeEmbedding[:, -1, :]) < LOSS_MARGIN:
        resultEmbedding[i] = (True, False)
    else:
        resultEmbedding[i] = (True, True)


    anchorEmbeddingSeq, positiveEmbeddingSeq, negativeEmbeddingSeq = embeddingNetSeq(anchor, 
                                                                                     positive, 
                                                                                     negative)

    if torch.dist(anchorEmbeddingSeq[0], positiveEmbeddingSeq[0]) < LOSS_MARGIN:
        resultEmbeddingSeq[i] = (True, True)
    else:
        resultEmbeddingSeq[i] = (True, False)
    
    if torch.dist(anchorEmbeddingSeq[0], negativeEmbeddingSeq[0]) < LOSS_MARGIN:
        resultEmbeddingSeq[i] = (True, False)
    else:
        resultEmbeddingSeq[i] = (True, True)



print('Validation accuracy: {}'.format(accuracy_score(resultEmbedding[:, 0], resultEmbedding[:, 1])))
print('Validation accuracy Seq: {}'.format(accuracy_score(resultEmbeddingSeq[:, 0], resultEmbeddingSeq[:, 1])))

