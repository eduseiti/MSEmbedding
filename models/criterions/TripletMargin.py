from torch import nn
from bootstrap.lib.options import Options

class TripletMargin(nn.TripletMarginLoss):

    def __init__(self):

        margin = Options()['model']['criterion']['loss_margin']
        p = 2
        eps = 1e-6
        swap = False
        size_average = None
        reduce = None
        reduction = 'mean'

        super(TripletMargin, self).__init__(margin, p, eps, swap, size_average,
                                            reduce, reduction)


    def forward(self, networkOutput, batch):

        out = {}

        out['loss'] = super(TripletMargin, self).forward(networkOutput[::3], 
                                                         networkOutput[1::3], 
                                                         networkOutput[2::3])

        return out