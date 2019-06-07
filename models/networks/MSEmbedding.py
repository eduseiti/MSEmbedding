import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MSEmbeddingNet(nn.Module):
    
    def __init__(self, maxSequenceLen, fcOutDim, lstmOutDim, bidirecionalLstm = False):

        self.maxSequenceLen = maxSequenceLen
        self.fcOutDim = fcOutDim
        self.lstmOutDim = lstmOutDim
        self.bidirecionalLstm = bidirecionalLstm

        super(MSEmbeddingNet, self).__init__()

        self.fcMZ1 = nn.Linear(1, 32)
        self.fcMZ2 = nn.Linear(32, self.fcOutDim)

        self.fcIntensity1 = nn.Linear(1, 32)
        self.fcIntensity2 = nn.Linear(32, self.fcOutDim)

        self.lstm = nn.LSTM(self.fcOutDim * 2, self.lstmOutDim, 
                            batch_first = True, 
                            bidirectional = self.bidirecionalLstm)


    #
    # Receives a batch with the shape [<batch size>, <sequence len>, <2 = m/z + intensity pair>]
    #

    def forward(self, x):

        transform = torch.empty(x.shape[0], x.shape[1], self.fcOutDim * 2)

        if torch.cuda.is_available():
            transform = transform.cuda()           
        
        xMZ = F.relu(self.fcMZ1(x[:, :, 0].view(x.shape[0], -1, 1)))
        xMZ = F.relu(self.fcMZ2(xMZ))

        xIntensity = F.relu(self.fcIntensity1(x[:, :, 1].view(x.shape[0], -1, 1)))
        xIntensity = F.relu(self.fcIntensity2(xIntensity))

        transform = torch.stack((xMZ, xIntensity), 2).view(x.shape[0], x.shape[1], -1)
            
        print('-- Len = {}, shape = {}'.format(len(transform), transform.shape))
            
        x, _ = self.lstm(transform)

        # print('Output={}, hidden={}'.format(x.shape, hidden))

        return x





