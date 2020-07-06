import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

import numpy as np



class MSEmbeddingNet(nn.Module):
    
    def __init__(self):
        super(MSEmbeddingNet, self).__init__()

        self.fcOutDim = Options()['model']['network']['fc_out_dim']
        self.lstmOutDim = Options()['model']['network']['lstm_out_dim']
        self.bidirecionalLstm = Options()['model']['network']['bidirecional_lstm']
        self.numOfLayers = Options()['model']['network'].get('num_of_layers', 1)
        self.applyPepmass = Options()['model']['network'].get('apply_pepmass', False)
        
        self.pepmassFinalDim = 16

        self.fcMZ1 = nn.Linear(1, 32)
        self.fcMZ2 = nn.Linear(32, self.fcOutDim)

        self.fcIntensity1 = nn.Linear(1, 32)
        self.fcIntensity2 = nn.Linear(32, self.fcOutDim)

        self.lstm = nn.LSTM(self.fcOutDim * 2, self.lstmOutDim, 
                            batch_first = True, 
                            bidirectional = self.bidirecionalLstm,
                            num_layers = self.numOfLayers)

        if self.bidirecionalLstm:
            self.fusion = nn.Linear(self.lstmOutDim * 2, self.lstmOutDim)


        if self.applyPepmass:
            #
            # For handling the pepmass information
            #

            self.fcPepmass1 = nn.Linear(1, 32)
            self.fcPepmass2 = nn.Linear(32, self.pepmassFinalDim)

            self.pepmassCombination = nn.Linear(self.lstmOutDim + self.pepmassFinalDim, self.lstmOutDim)

        Logger()("Number of parameters={}".format(sum(p.numel() for p in self.parameters())))


    #
    # Receives a batch with two elements:
    #   peaks: shape [<batch size>, <sequence len>, <2 = m/z and intensity pair>]
    #   peakslen: [<batch size>]
    #   pepmass: [<batch size>]
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

        transform = torch.empty(x.shape[0], x.shape[1], self.fcOutDim * 2)

        if torch.cuda.is_available():
            transform = transform.cuda()           
        
        xMZ = F.relu(self.fcMZ1(x[:, :, 0].view(x.shape[0], -1, 1)))
        xMZ = F.relu(self.fcMZ2(xMZ))

        xIntensity = F.relu(self.fcIntensity1(x[:, :, 1].view(x.shape[0], -1, 1)))
        xIntensity = F.relu(self.fcIntensity2(xIntensity))

        transform = torch.stack((xMZ, xIntensity), 2).view(x.shape[0], x.shape[1], -1)

        print('-- Before pack: Len = {}, shape = {}'.format(len(transform), transform.shape))

        transform = torch.nn.utils.rnn.pack_padded_sequence(transform, originalPeaksLen[indexesSortedPeaks], batch_first = True)

        x, (hidden_state, cell_state) = self.lstm(transform)

        print('Output type={}, hidden_state.shape={}, cell_state.shape={}'.format(type(x), hidden_state.shape, cell_state.shape))

        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first = True, total_length = originalPeaksMaxLen)

        print('-- After pack: Len = {}, shape = {}'.format(len(x), x.shape))


        #
        # Return the hidden states to their original order
        #

        originalIndexes = torch.zeros_like(indexesSortedPeaks)

        for i in range(len(indexesSortedPeaks)):
            originalIndexes[indexesSortedPeaks[i]] = i

        # x = x[originalIndexes]

        # x = x.view(x.shape[0], x.shape[1], -1, self.lstmOutDim)

        # hidden_state = hidden_state.view(self.numOfLayers, -1, hidden_state.shape[1], hidden_state.shape[2])

        cell_state = cell_state.view(self.numOfLayers, -1, cell_state.shape[1], cell_state.shape[2])


        # print("shape lstm output={}".format(x.shape))
        # print("shape lstm output accessed={}".format(x[range(x.shape[0]), originalPeaksLen - 1].shape))

        # print("hidden_state.shape={}".format(hidden_state.shape))
        # print("last hidden layer.shape={}".format(hidden_state[self.numOfLayers - 1].shape))


        # for i in range(40):
        #     print("Dim {}: LSTM={}, hidden={}".format(i, x[range(x.shape[0]), originalPeaksLen - 1][0][0][i], hidden_state[2, 0][originalIndexes][0][i]))

        # print("\n")

        # for i in range(40):
        #     print("Dim {}: LSTM={}, h0={}".format(i, x[range(x.shape[0]), 0][0][1][i], hidden_state[2, 1][originalIndexes][0][i]))


        #
        # Apply the fusion layer if using bi-LSTM
        #

        if self.bidirecionalLstm:

            print("cell_state.shape={}".format(cell_state.shape))
            print("last cell layer.shape={}".format(cell_state[self.numOfLayers - 1].shape))
            # print("concatenated.shape={}".format(torch.cat((cell_state[self.numOfLayers - 1, 0], cell_state[self.numOfLayers - 1, 1]), 1).shape))

            # selects the last internal state of each direction

            x = F.relu(self.fusion(torch.cat((cell_state[self.numOfLayers - 1, 0], cell_state[self.numOfLayers - 1, 1]), 1)))

            print("final x.shape={}".format(x.shape))

        else:
            x = cell_state[self.numOfLayers - 1, 0]


        x = x[originalIndexes]


        if self.applyPepmass:
            #
            # Process the pepmass
            #

            xPepmass = batch['pepmass']

            print("-- shape pepmass={}".format(xPepmass.shape))

            xPepmass = F.relu(self.fcPepmass1(batch['pepmass'].view(xPepmass.shape[0], -1)))
            xPepmass = F.relu(self.fcPepmass2(xPepmass))


            #
            # Combine pepmass and the LSTM internal state
            #

            x = F.relu(self.pepmassCombination(torch.cat((xPepmass, x), 1)))

        return x






