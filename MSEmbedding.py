import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

MAX_SEQUENCE_LEN = 2000
FCS_OUT_DIM = 16
LSTM_OUT_DIM = 10
BIDIRECIONAL_LSTM = False


class MSEmbeddingNet(nn.Module):

    maxSequenceLen   = MAX_SEQUENCE_LEN
    fcsOutDim        = FCS_OUT_DIM
    lstmOutDim       = LSTM_OUT_DIM
    bidirecionalLstm = BIDIRECIONAL_LSTM
    
    
    def __init__(self):
        super(MSEmbeddingNet, self).__init__()

        self.fcMZ1 = nn.Linear(1, 32)
        self.fcMZ2 = nn.Linear(32, MSEmbeddingNet.fcsOutDim)

        self.fcIntensity1 = nn.Linear(1, 32)
        self.fcIntensity2 = nn.Linear(32, MSEmbeddingNet.fcsOutDim)

        self.lstm = nn.LSTM(MSEmbeddingNet.fcsOutDim * 2, MSEmbeddingNet.lstmOutDim, 
                            batch_first = True, 
                            bidirectional = MSEmbeddingNet.bidirecionalLstm)



    def forward(self, x):

        transform = torch.empty(x.shape[0], MSEmbeddingNet.fcsOutDim * 2).cuda()
        
        for i, pair in enumerate(x):
            xMZ = F.relu(self.fcMZ1(pair[0].view(-1)))
            xMZ = F.relu(self.fcMZ2(xMZ))

            xIntensity = F.relu(self.fcIntensity1(pair[1].view(-1)))
            xIntensity = F.relu(self.fcIntensity2(xIntensity))

            transform[i] = torch.cat((xMZ, xIntensity))
            
        print('-- Len = {}, shape = {}'.format(len(transform), transform.shape))
            
        x, _ = self.lstm(transform.unsqueeze(0))

        # print('Output={}, hidden={}'.format(x.shape, hidden))

        return x


    
class MSEmbeddingNetSeq(nn.Module):

    maxSequenceLen   = MAX_SEQUENCE_LEN
    fcsOutDim        = FCS_OUT_DIM
    lstmOutDim       = LSTM_OUT_DIM
    bidirecionalLstm = BIDIRECIONAL_LSTM
    
    
    def __init__(self):
        super(MSEmbeddingNetSeq, self).__init__()

        self.fcMZ1 = nn.Linear(1, 32)
        self.fcMZ2 = nn.Linear(32, MSEmbeddingNet.fcsOutDim)

        self.fcIntensity1 = nn.Linear(1, 32)
        self.fcIntensity2 = nn.Linear(32, MSEmbeddingNet.fcsOutDim)

        self.lstm = nn.LSTM(MSEmbeddingNet.fcsOutDim * 2, MSEmbeddingNet.lstmOutDim, 
                            batch_first = True, 
                            bidirectional = MSEmbeddingNet.bidirecionalLstm)



    def forward(self, x):
        for i, pair in enumerate(x):
            xMZ = F.relu(self.fcMZ1(pair[0].view(-1)))
            xMZ = F.relu(self.fcMZ2(xMZ))

            xIntensity = F.relu(self.fcIntensity1(pair[1].view(-1)))
            xIntensity = F.relu(self.fcIntensity2(xIntensity))
                      
            x, _ = self.lstm(torch.cat((xMZ, xIntensity)).unsqueeze(0).unsqueeze(0))

        return x
    

#
# Receives a grouped sample triplets to embed: (anchor, positive, negative), as a 3 dimensional tensor. 
# Process each sequence pair individually in the MLP and then sends the triplet to the LSTM.
#

class MSEmbeddingNetGroupedTriplets(nn.Module):

    maxSequenceLen   = MAX_SEQUENCE_LEN
    fcsOutDim        = FCS_OUT_DIM
    lstmOutDim       = LSTM_OUT_DIM
    bidirecionalLstm = BIDIRECIONAL_LSTM
    
    
    def __init__(self):
        super(MSEmbeddingNetGroupedTriplets, self).__init__()

        self.fcMZ1 = nn.Linear(1, 32)
        self.fcMZ2 = nn.Linear(32, MSEmbeddingNet.fcsOutDim)

        self.fcIntensity1 = nn.Linear(1, 32)
        self.fcIntensity2 = nn.Linear(32, MSEmbeddingNet.fcsOutDim)

        self.lstm = nn.LSTM(MSEmbeddingNet.fcsOutDim * 2, MSEmbeddingNet.lstmOutDim, 
                            batch_first = True, 
                            bidirectional = MSEmbeddingNet.bidirecionalLstm)


    #
    # x: Dimension (<batches>, <peaks>, <2 = (m/z, intensity)>)
    #

    def forward(self, x):

        transform = torch.empty(x.shape[0], x.shape[1], MSEmbeddingNet.fcsOutDim * 2).cuda()

        for batch in range(len(x.shape[0])):
            for i, pair in enumerate(x):
                xMZ = F.relu(self.fcMZ1(pair[0].view(-1)))
                xMZ = F.relu(self.fcMZ2(xMZ))

                xIntensity = F.relu(self.fcIntensity1(pair[1].view(-1)))
                xIntensity = F.relu(self.fcIntensity2(xIntensity))

                transform[batch, i] = torch.cat((xMZ, xIntensity))

        x, _ = self.lstm(transform)

        return x




    
class TripletNet(nn.Module):
    def __init__(self, embeddingNet):
        super(TripletNet, self).__init__()
        self.embeddingNet = embeddingNet

    def forward(self, anchor, positive, negative):
        embeddedAnchor   = self.embeddingNet(anchor)
        embeddedPositive = self.embeddingNet(positive)
        embeddedNegative = self.embeddingNet(negative)
        return embeddedAnchor, embeddedPositive, embeddedNegative






