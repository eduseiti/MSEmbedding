from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

from .TripletMargin import TripletMargin
from .matrix_triplet_margin import MatrixTripletMargin

#
# mode: depending on the split (?)
#

def factory(engine=None, mode=None):

    selectedCriterion = Options()['model']['criterion']['name']

    Logger()('Creating {} criterion...'.format(selectedCriterion))

    if (selectedCriterion == 'tripletMargin'):
        if mode == 'train':
            criterion = TripletMargin()
        else:
            criterion = None
    elif (selectedCriterion == 'MatrixTripletMargin'):
        if mode == 'train':
            criterion = MatrixTripletMargin()
    else:
        raise ValueError()

    return criterion
