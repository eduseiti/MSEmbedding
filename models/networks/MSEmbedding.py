import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from bootstrap.lib.options import Options


class MSEmbeddingNet(nn.Module):
    
    def __init__(self):

        self.fcOutDim = Options()['model']['network']['fc_out_dim']
        self.lstmOutDim = Options()['model']['network']['lstm_out_dim']
        self.bidirecionalLstm = Options()['model']['network']['bidirecional_lstm']

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

        x, _ = self.lstm(transform)

        # print('Output={}, hidden={}'.format(x.shape, hidden))

        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first = True, total_length = originalPeaksMaxLen)

        print('-- After pack: Len = {}, shape = {}'.format(len(x), x.shape))

        # for i in range(x.shape[1]):
        #     print("x[0,{}]={}; x[1,{}]={}; x[2,{}]={}".format(i, x[0, i], i, x[1, i], i, x[2, i]))

        return (x, indexesSortedPeaks)





