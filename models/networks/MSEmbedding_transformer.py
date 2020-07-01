import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from bootstrap.lib.options import Options

import math

import numpy as np

#
# Code based in https://pytorch.org/tutorials/beginner/transformer_tutorial.html
#


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class MSEmbeddingTransformerNet(nn.Module):
    
    def __init__(self):
        super(MSEmbeddingTransformerNet, self).__init__()

        self.classificationToken = Options().get("dataset.classification_token", 0)

        self.fcIntermediateDim = Options()['model']['network']['fc_intermediate_dim']
        self.fcOutDim = Options()['model']['network']['fc_out_dim']
        self.lstmOutDim = Options()['model']['network']['lstm_out_dim']
        self.numOfLayers = Options()['model']['network'].get('num_of_layers', 1)

        self.numOfHeads = Options()['model']['network'].get('num_of_heades', 2)
        self.dimFeedForward = Options()['model']['network'].get('dim_feedforward', 40)
        self.transformerDropout = Options()['model']['network'].get('transformer_dropout', 0.0)

        self.fcMZ1 = nn.Linear(1, self.fcIntermediateDim)
        self.fcMZ2 = nn.Linear(self.fcIntermediateDim, self.fcOutDim)

        self.fcIntensity1 = nn.Linear(1, self.fcIntermediateDim)
        self.fcIntensity2 = nn.Linear(self.fcIntermediateDim, self.fcOutDim)

        self.positionalEncoder = PositionalEncoding(self.fcOutDim * 2)

        encoder_layers = nn.TransformerEncoderLayer(self.fcOutDim * 2, self.numOfHeads, self.dimFeedForward, self.transformerDropout)

        self.transformerEncoder = nn.TransformerEncoder(encoder_layers, self.numOfLayers)

        print("Number of parameters={}".format(sum(p.numel() for p in self.parameters())))


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

        transform = torch.empty(x.shape[0], x.shape[1], self.fcOutDim * 2)

        if torch.cuda.is_available():
            transform = transform.cuda()           
        
        xMZ = F.relu(self.fcMZ1(x[:, :, 0].view(x.shape[0], -1, 1)))
        xMZ = F.relu(self.fcMZ2(xMZ))

        xIntensity = F.relu(self.fcIntensity1(x[:, :, 1].view(x.shape[0], -1, 1)))
        xIntensity = F.relu(self.fcIntensity2(xIntensity))

        transform = torch.stack((xMZ, xIntensity), 2).view(x.shape[0], x.shape[1], -1)


        # print('-- Before adding classification token: shape = {}'.format(transform.shape))

        classification = torch.empty(transform.shape[0], 1, transform.shape[2])
        classification.fill_(self.classificationToken)

        x = torch.cat([classification.cuda(), transform], dim=1)


        # Create a mask to ignore the positions greater than the sequence length

        keyPaddingMask = torch.empty(x.shape[0], x.shape[1], dtype=torch.bool).fill_(False).cuda()
        
        for i in range(x.shape[0]):
            keyPaddingMask[i, originalPeaksLen[i]:] = True

        # print('-- keyPaddingMask.shape={}'.format(keyPaddingMask.shape))

        # print("keyPaddingMask={}".format(keyPaddingMask))


        # print('-- Before entering positionalEncoder: shape = {}'.format(x.shape))

        x = self.positionalEncoder(x.transpose(0, 1))

        x = self.transformerEncoder(x, src_key_padding_mask=keyPaddingMask)

        # print('Transformer output shape={}'.format(x.shape))

        x = x.transpose(0, 1)

        # print('Transformer transposed output shape={}'.format(x.shape))

        x = x[:,0]

        # print('Model output shape={}'.format(x.shape))

        # print("x[0]={}".format(x[0]))
        # print("x[1]={}".format(x[1]))
        # print("x[2]={}".format(x[2]))
        # print("x[3]={}".format(x[3]))

        # print("x[0]==x[2]{}".format(x[0] == x[2]))
        # print("x[1]==x[3]{}".format(x[1] == x[3]))

        return x






