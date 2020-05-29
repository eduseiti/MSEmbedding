import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from bootstrap.lib.options import Options

import numpy as np



class MSEmbedding_MLP_Net(nn.Module):
    
    def __init__(self):

        super(MSEmbedding_MLP_Net, self).__init__()        

        embeddings_size = Options().get('model.network.embeddings_size', 20)

        self.fc01 = nn.Linear(2000, 2000)
        self.batchNorm01 = nn.BatchNorm1d(num_features=2000)
        self.fc02 = nn.Linear(2000, embeddings_size)
        self.batchNorm02 = nn.BatchNorm1d(num_features=embeddings_size)


    def forward(self, batch):

        print("MSEmbedding_MLP_Net: batch.shape={}".format(batch['peaks'].shape))

        x = F.relu(self.batchNorm01(self.fc01(batch['peaks'])))
        x = F.relu(self.batchNorm02(self.fc02(x)))

        return x






