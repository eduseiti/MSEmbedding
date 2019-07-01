from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

from .TripletMargin import TripletMargin

#
# mode: depending on the split (?)
#

def factory(engine=None, mode=None):

    Logger()('Creating TripletMargin criterion...')

    if (Options()['model']['criterion']['name'] == 'tripletMargin'):
        if mode == 'train':
            criterion = TripletMargin()
        else:
            criterion = None
    else:
        raise ValueError()

    return criterion
