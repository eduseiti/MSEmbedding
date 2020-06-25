import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from bootstrap.lib.options import Options

import numpy as np

from .custom_lstms import script_lnlstm, LSTMState, PADDING_32


def join_and_split_output(list):
    return [torch.stack([item[0] for item in list]), torch.stack([item[1] for item in list])]  


class MSEmbeddingNormNet(nn.Module):
    
    def __init__(self):
        super(MSEmbeddingNormNet, self).__init__()

        self.fcOutDim = Options()['model']['network']['fc_out_dim']
        self.lstmOutDim = Options()['model']['network']['lstm_out_dim']
        self.bidirecionalLstm = Options()['model']['network']['bidirecional_lstm']
        self.numOfLayers = Options()['model']['network'].get('num_of_layers', 1)
        self.applyPepmass = Options()['model']['network'].get('apply_pepmass', False)
        
        self.batchSize = Options().get('dataset.batch_size', 250)

        self.pepmassFinalDim = 16

        self.fcMZ1 = nn.Linear(1, 32)
        self.fcMZ2 = nn.Linear(32, self.fcOutDim)

        self.fcIntensity1 = nn.Linear(1, 32)
        self.fcIntensity2 = nn.Linear(32, self.fcOutDim)


        self.lstm = script_lnlstm(self.fcOutDim * 2, self.lstmOutDim, self.numOfLayers,
                                  bidirectional=self.bidirecionalLstm, decompose_layernorm=True)

        if self.bidirecionalLstm:

            self.lstm_states = [[LSTMState(torch.randn(self.batchSize, self.lstmOutDim).cuda(),
                                           torch.randn(self.batchSize, self.lstmOutDim).cuda())
                                for _ in range(2)]
                                for _ in range(self.numOfLayers)]

            self.fusion = nn.Linear(self.lstmOutDim * 2, self.lstmOutDim)
        else:
            self.lstm_states = [LSTMState(torch.randn(self.batchSize, self.lstmOutDim).cuda(),
                                          torch.randn(self.batchSize, self.lstmOutDim).cuda())
                                for _ in range(self.numOfLayers)]



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

        x = batch['peaks']

        print("--> Data shape: {}".format(x.shape))

        originalPeaksMaxLen = x.shape[1]

        transform = torch.empty(x.shape[0], x.shape[1], self.fcOutDim * 2)

        transform = transform.cuda()           
        
        xMZ = F.relu(self.fcMZ1(x[:, :, 0].view(x.shape[0], -1, 1)))
        xMZ = F.relu(self.fcMZ2(xMZ))

        xIntensity = F.relu(self.fcIntensity1(x[:, :, 1].view(x.shape[0], -1, 1)))
        xIntensity = F.relu(self.fcIntensity2(xIntensity))


        for i in range(x.shape[0]):
            xMZ[i, -originalPeaksLen[i]:] = PADDING_32
            xIntensity[i, -originalPeaksLen[i]:] = PADDING_32


        transform = torch.stack((xMZ, xIntensity), 2).view(x.shape[0], x.shape[1], -1)

        x, output_states = self.lstm(torch.transpose(transform, 0, 1), self.lstm_states)

        if self.bidirecionalLstm:
            consolidated_directions = []

            for layer in output_states:
                consolidated_directions.append(join_and_split_output(layer))

            hidden_state = torch.stack([consolidated[0] for consolidated in consolidated_directions])
            cell_state = torch.stack([consolidated[1] for consolidated in consolidated_directions])

            print('Output type={}, hidden_state.shape={}, cell_state.shape={}'.format(type(x), hidden_state.shape, cell_state.shape))

            print("cell_state.shape={}".format(cell_state.shape))
            print("last cell layer.shape={}".format(cell_state[self.numOfLayers - 1].shape))
            # print("concatenated.shape={}".format(torch.cat((cell_state[self.numOfLayers - 1, 0], cell_state[self.numOfLayers - 1, 1]), 1).shape))

            # selects the last internal state of each direction

            x = F.relu(self.fusion(torch.cat((cell_state[self.numOfLayers - 1, 0], cell_state[self.numOfLayers - 1, 1]), 1)))

        else:
            print("Original x.shape={}".format(x.shape))

            cell_states = torch.transpose(x, 0, 1)

            x = cell_states[range(len(originalPeaksLen)), originalPeaksLen - 1]


        print("final x.shape={}".format(x.shape))


        return x






