from torch import nn

class TripletMargin(mm.TripletMarginLoss):

    def __init__(self, margin=1.0, p=2., eps=1e-6, swap=False, size_average=None,
                 reduce=None, reduction='mean'):

        super(TripletMargin, self).__init__(margin, p, eps, swap, size_average,
                                            reduce, reduction)


    def forward(self, networkOutput, batch):

        return super(TripletMargin, self).forward(networkOutput[::3], networkOutput[1::3], networkOutput[2::3])