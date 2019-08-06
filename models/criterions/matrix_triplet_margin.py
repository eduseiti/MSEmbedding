from torch import nn
from bootstrap.lib.options import Options

class MatrixTripletMargin(nn.Module):

    def __init__(self):

        super(MatrixTripletMargin, self).__init__()

        self.margin = Options()['model']['criterion']['loss_margin']


    def forward(self, networkOutput, batch):

        out = {}

        anchors = networkOutput[::3]
        po


        out['loss'] = super(TripletMargin, self).forward(networkOutput[::3], 
                                                         networkOutput[1::3], 
                                                         networkOutput[2::3])

        return out