from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

from .TripletMargin import TripletMargin
from .matrix_triplet_margin import MatrixTripletMargin
from .n_pair import NPair
from .double_n_pair import DoubleNPair
from .quad_n_pair import QuadNPair
from .double_n_pair_margin import DoubleNPairMargin

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
        criterion = MatrixTripletMargin()
    elif (selectedCriterion == 'NPair'):
        criterion = NPair()
    elif (selectedCriterion == 'DoubleNPair'):
        criterion = DoubleNPair()
    elif (selectedCriterion == 'DoubleNPairMargin'):
        criterion = DoubleNPairMargin()
    elif (selectedCriterion == 'QuadNPair'):
        criterion = QuadNPair()
    else:
        raise ValueError()

    return criterion
