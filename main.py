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



os.chdir('../PXD000561')

originalData = s.PXD000561('Gel_Elite_49.csv', 'initialTest2.pkl')

originalData.load_identifications(True, singleFile = 'b01')

if not originalData.totalSpectra.spectra:
    originalData.read_spectra(s.MGF())

totalSpectra = originalData.totalSpectra

bl = b.BatchLoader(totalSpectra)
bl.listMultipleScansSequences()

multipleScansSequences = bl.multipleScansSequences



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

newBatch = bl.createTripletBatch()

loadedBatch = bl.loadTripletsBatch(newBatch)

print('Batch shape: {}'.format(loadedBatch.shape))

embeddingNet.train()

HOW_MANY_SAMPLES = 100

startTime = time.time()
embeddings = embeddingNet(loadedBatch)
print('Embedding time: {}'.format(time.time() - startTime))

startTime = time.time()
for i in range(len(embeddings) // 3):
    loss = criterion(embeddings[i * 3], embeddings[i * 3 + 1], embeddings[i * 3 + 2])

print('- Loss = {}. Time = {}'.format(loss, time.time() - startTime))

startTime = time.time()
optmizer.zero_grad()
loss.backward()
optmizer.step()
print('- Backprop time: {}'.format(time.time() - startTime))

torch.save(embeddingNet.state_dict(), 'embeddingNet.state')


embeddingNet.eval()


newTestBatch = blTest.createTripletBatch()

loadedTestBatch = blTest.loadTripletsBatch(newTestBatch)

startTime = time.time()
resultEmbeddings = embeddingNet(loadedTestBatch)
print('Embedding time: {}'.format(time.time() - startTime))

validationResult = np.zeros((len(blTest.multipleScansSequences), 2), dtype = bool)

for i in range(len(resultEmbeddings) // 3):
    if torch.dist(resultEmbeddings[i * 3], resultEmbeddings[i * 3 + 1]) < LOSS_MARGIN:
        validationResult[i] = (True, True)
    else:
        validationResult[i] = (True, False)
    
    if torch.dist(resultEmbeddings[i * 3], resultEmbeddings[i * 3 + 2]) < LOSS_MARGIN:
        validationResult[i] = (True, False)
    else:
        validationResult[i] = (True, True)


print('Validation accuracy: {}'.format(accuracy_score(validationResult[:, 0], validationResult[:, 1])))

