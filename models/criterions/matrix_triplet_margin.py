import torch
from torch import nn
from bootstrap.lib.options import Options

class MatrixTripletMargin(nn.Module):

    def __init__(self):

        super(MatrixTripletMargin, self).__init__()

        self.margin = Options()['model']['criterion']['loss_margin']


    def forward(self, networkOutput, batch):

        out = {}

        anchors = nn.functional.normalize(networkOutput[::3])
        positive = nn.functional.normalize(networkOutput[1::3])
        negative = nn.functional.normalize(networkOutput[2::3])

        anchors = anchors.reshape(anchors.shape[0], -1)
        positive = positive.reshape(positive.shape[0], -1)
        negative = negative.reshape(negative.shape[0], -1)

        allCosineDistances = 1 - torch.mm(anchors, torch.cat((positive, negative)).t())

        anchorPositiveDistance = allCosineDistances.diag().unsqueeze(1)

        loss = torch.max(anchorPositiveDistance - allCosineDistances + self.margin, torch.zeros(1).cuda())

        loss[range(loss.shape[0]), range(loss.shape[0])] = 0.

        out['loss'] = torch.mean(loss)

        return out