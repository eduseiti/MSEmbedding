from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

from .madamw import MAdamW

#
# mode: depending on the split (?)
#

def factory(engine=None, mode=None):

    Logger()('Creating MAdamW optimizer...')

    if (Options()['optimizer']['name'] == 'madamw'):
        if mode == 'eval':
            optimizer = MAdamW()
        else:
            optimizer = None
    else:
        raise ValueError()

    return optimizer
