import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MSEmbeddingNet(nn.Module):
    
    def __init__(self, maxSequenceLen, fcOutDim, lstmOutDim, bidirecionalLstm = False):
        super(MSEmbeddingNet, self).__init__()

        self.fcMZ1 = nn.Linear(1, 32)
        self.fcMZ2 = nn.Linear(32, MSEmbeddingNet.fcsOutDim)

        self.fcIntensity1 = nn.Linear(1, 32)
        self.fcIntensity2 = nn.Linear(32, MSEmbeddingNet.fcsOutDim)

        self.lstm = nn.LSTM(MSEmbeddingNet.fcsOutDim * 2, MSEmbeddingNet.lstmOutDim, 
                            batch_first = True, 
                            bidirectional = MSEmbeddingNet.bidirecionalLstm)


    #
    # Receives a batch with the shape [<batch size>, <sequence len>, <2 = m/z + intensity pair>]
    #

    def forward(self, x):

        transform = torch.empty(x.shape[0], x.shape[1], 1).cuda()
        
        xMZ = F.relu(self.fcMZ1(x[:, :, 0]))
        xMZ = F.relu(self.fcMZ2(xMZ))

        xIntensity = F.relu(self.fcIntensity1(x[:, :, 1))
        xIntensity = F.relu(self.fcIntensity2(xIntensity))

        transform = torch.cat((xMZ, xIntensity))
            
        print('-- Len = {}, shape = {}'.format(len(transform), transform.shape))
            
        x, _ = self.lstm(transform)

        # print('Output={}, hidden={}'.format(x.shape, hidden))

        return x





