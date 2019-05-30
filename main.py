import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import MSEmbedding as m
import spectra as s
import BatchLoader as b

import numpy as np

import random

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import time


HOW_MANY_SAMPLES = 128


os.chdir('../PXD000561')


#
# training data
#

originalData = s.PXD000561('Gel_Elite_49.csv', 'initialTest2.pkl')

originalData.load_identifications(True, singleFile = 'b01')

if not originalData.totalSpectra.spectra:
    originalData.read_spectra(s.MGF())

totalSpectra = originalData.totalSpectra

bl = b.BatchLoader(totalSpectra, randomSeed = time.time())
bl.listMultipleScansSequences()

multipleScansSequences = bl.multipleScansSequences


#
# testing data
#

testData = s.PXD000561('Gel_Elite_49.csv', 'initialTest2_b02.pkl')

testData.load_identifications(True, singleFile = 'b02')

if not testData.totalSpectra.spectra:
    testData.read_spectra(s.MGF())

testSpectra = testData.totalSpectra

blTest = b.BatchLoader(testSpectra)
blTest.listMultipleScansSequences()

unrecognizedTestLen = len(testSpectra.spectra[s.Scan.UNRECOGNIZED_SEQUENCE])

print('Test sequences size {}'.format(len(blTest.multipleScansSequences)))




# taken from Cross-Modal retrieval article
LOSS_MARGIN = 0.3 

embeddingNet = m.MSEmbeddingNet(2000, 16, 10, False)



if torch.cuda.is_available():
    embeddingNet = embeddingNet.cuda()

optmizer = optim.SGD(embeddingNet.parameters(), lr = 0.01)


criterion = nn.TripletMarginLoss(margin = LOSS_MARGIN)

try:
    embeddingNet.load_state_dict(torch.load('embeddingNet.state'))
except Exception as e:
    print('Could not open trained model files')


while True:

    newBatch = bl.createTripletBatch()

    loadedBatch = bl.loadTripletsBatch(newBatch, HOW_MANY_SAMPLES)

    print('\nLoaded train batch shape: {}\n'.format(loadedBatch.shape))

    # print('\n**Memory after batch train: {}'.format(torch.cuda.memory_allocated()))


    embeddingNet.train()

    startTime = time.time()
    embeddings = embeddingNet(loadedBatch[:(HOW_MANY_SAMPLES * 3), :, :])
    # print('- Embedding time: {}'.format(time.time() - startTime))

    # print('\n**Memory after train embeddings: {}'.format(torch.cuda.memory_allocated()))

    startTime = time.time()
    for i in range(len(embeddings) // 3):
        loss = criterion(embeddings[i * 3], embeddings[i * 3 + 1], embeddings[i * 3 + 2])

    print('- LOSS = {}. Time = {}'.format(loss, time.time() - startTime))


    startTime = time.time()
    optmizer.zero_grad()
    loss.backward()
    optmizer.step()
    # print('- Backprop time: {}'.format(time.time() - startTime))

    torch.save(embeddingNet.state_dict(), 'embeddingNet.state')


    #
    # Free some GPU memory
    #

    del loadedBatch
    del embeddings

    torch.cuda.empty_cache()

    # print('\n**Memory after train: {}'.format(torch.cuda.memory_allocated()))

    embeddingNet.eval()


    newTestBatch    = blTest.createTripletBatch()
    loadedTestBatch = blTest.loadTripletsBatch(newTestBatch, HOW_MANY_SAMPLES)

    print('\nLoaded test batch shape: {}\n'.format(loadedTestBatch.shape))

    # print('\n**Memory after batch test: {}'.format(torch.cuda.memory_allocated()))


    startTime = time.time()

    with torch.no_grad():
        resultEmbeddings = embeddingNet(loadedTestBatch[:(HOW_MANY_SAMPLES * 3), :, :])

    # print('- Test embedding time: {}'.format(time.time() - startTime))

    # print('\n**Memory after test embeddings: {}'.format(torch.cuda.memory_allocated()))

    validationResult = np.zeros((HOW_MANY_SAMPLES * 2, 2), dtype = bool)

    for i in range(len(resultEmbeddings) // 3):

        positiveDistance = torch.dist(resultEmbeddings[i * 3], resultEmbeddings[i * 3 + 1])
        negativeDistance = torch.dist(resultEmbeddings[i * 3], resultEmbeddings[i * 3 + 2])

        # print('{} - Positive distance={}, Negative distance={}'.format(i, positiveDistance, negativeDistance))

        if torch.lt(positiveDistance, LOSS_MARGIN):
            validationResult[i * 2] = (True, True)
        else:
            validationResult[i * 2] = (True, False)
        
        if torch.lt(negativeDistance, LOSS_MARGIN):
            validationResult[i * 2 + 1] = (True, False)
        else:
            validationResult[i * 2 + 1] = (True, True)


    print('Validation accuracy: {}'.format(accuracy_score(validationResult[:, 0], validationResult[:, 1])))

    del loadedTestBatch
    del resultEmbeddings

    torch.cuda.empty_cache()

    # print('\n**Memory after test: {}'.format(torch.cuda.memory_allocated()))
