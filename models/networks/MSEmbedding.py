import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from bootstrap.lib.options import Options

import numpy as np



class MSEmbeddingNet(nn.Module):
    
    def __init__(self):

        self.fcOutDim = Options()['model']['network']['fc_out_dim']
        self.lstmOutDim = Options()['model']['network']['lstm_out_dim']
        self.bidirecionalLstm = Options()['model']['network']['bidirecional_lstm']
        self.window = Options()['model']['network']['window']

        super(MSEmbeddingNet, self).__init__()

        self.fcMZ1 = nn.Linear(self.window, 32)
        self.fcIntensity1 = nn.Linear(self.window, 32)
        self.fcCombined = nn.Linear(32 * 2, self.fcOutDim * 2)

        self.lstm = nn.LSTM(self.fcOutDim * 2, self.lstmOutDim, 
                            batch_first = True, 
                            bidirectional = self.bidirecionalLstm)

        if self.bidirecionalLstm:
            self.fusion = nn.Linear(self.lstmOutDim * 2, self.lstmOutDim)




    #
    # Receives a batch with the shape [<batch size>, <sequence len>, <2 = m/z + intensity pair>]
    #

    def forward(self, batch):

        # print (">>> batch={}".format(batch))
        # print (">>> peaks.shape={}".format(batch['peaks'].shape)) 
        # print (">>> peaksLen.shape={}".format(batch['peaksLen'].shape)) 

        originalPeaksLen = batch['peaksLen']

        indexesSortedPeaks = torch.argsort(originalPeaksLen, descending = True)

        sortedPeaks = batch['peaks'][indexesSortedPeaks]

        x = sortedPeaks

        print("--> Data shape: {}".format(x.shape))

        originalPeaksMaxLen = x.shape[1]

        xMZ = F.relu(self.fcMZ1(x[:, :, 0].unfold(1, self.window, self.window)))
        xIntensity = F.relu(self.fcIntensity1(x[:, :, 1].unfold(1, self.window, self.window)))

        print("xMZ.shape={}, xIntensity.shape={}".format(xMZ.shape, xIntensity.shape))

        # xMZ_xIntesity_combined = torch.empty(xMZ.shape[0], xMZ.shape[1], self.fcOutDim * 2)

        # if torch.cuda.is_available():
        #     xMZ_xIntesity_combined = xMZ_xIntesity_combined.cuda()                  

        xMZ_xIntesity_combined = torch.stack((xMZ, xIntensity), 2).view(xMZ.shape[0], xMZ.shape[1], -1)

        print("xMZ_xIntesity_combined.shape={}".format(xMZ_xIntesity_combined.shape))
            
        transform = F.relu(self.fcCombined(xMZ_xIntesity_combined))

        print('-- Before pack: Len = {}, shape = {}'.format(len(transform), transform.shape))

        transform = torch.nn.utils.rnn.pack_padded_sequence(transform, originalPeaksLen[indexesSortedPeaks] // self.window - 1, batch_first = True)

        x, _ = self.lstm(transform)

        # print('Output={}, hidden={}'.format(x.shape, hidden))

        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first = True, total_length = originalPeaksMaxLen // self.window)

        print('-- After pack: Len = {}, shape = {}'.format(len(x), x.shape))


        # for i in range(x.shape[1]):
        #     print("x[0,{}]={}; x[1,{}]={}; x[2,{}]={}".format(i, x[0, i], i, x[1, i], i, x[2, i]))

        # #
        # # Now, return the embeddings to their original sort order, to recover the triplets sequence ― (anchor, positive example, negative example)
        # #

        # originalIndexes = x.new_zeros(len(indexesSortedPeaks), dtype = torch.int32)

        # for i in range(len(indexesSortedPeaks)):
        #     originalIndexes[indexesSortedPeaks[i]] = i

        # x = x[originalIndexes.tolist()]

        #
        # Apply the fusion layer if using bi-LSTM
        #

        if self.bidirecionalLstm:
            print("-- shape last={}".format(x[range(x.shape[0]), originalPeaksLen[indexesSortedPeaks] // self.window - 1, :].shape))

            # selects the last internal state of each direction

            x = F.relu(self.fusion(x[range(x.shape[0]), originalPeaksLen[indexesSortedPeaks] // self.window - 1, :]))

        print("x={}".format(x))

        return (x, indexesSortedPeaks)





