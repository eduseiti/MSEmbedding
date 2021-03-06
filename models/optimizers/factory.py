from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

from .madamw import MAdamW
from .radam import RAdam

#
# mode: depending on the split (?)
#

def factory(model=None, engine=None):


    if (Options()['optimizer']['name'] == 'madamw'):
        Logger()('Creating MAdamW optimizer...')

        optimizer = MAdamW(filter(lambda p: p.requires_grad, model.network.parameters()),
                           lr = Options()['optimizer']['lr'],
                           weight_decay = Options()['optimizer'].get('weight_decay', 0))
    elif (Options()['optimizer']['name'] == 'radam'):
        Logger()('Creating radam optimizer...')

        optimizer = RAdam(filter(lambda p: p.requires_grad, model.network.parameters()),
                          lr = Options()['optimizer']['lr'],
                          weight_decay = Options()['optimizer'].get('weight_decay', 0))
    else:
        raise ValueError()

    return optimizer
