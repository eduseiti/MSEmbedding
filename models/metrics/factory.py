from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

from .EmbeddingsDistance import EmbeddingsDistance

#
# mode: depending on the split (?)
#

def factory(engine=None, mode=None):

    Logger()('Creating EmbeddingDistance metric...')

    if (Options()['model']['metric']['name'] == 'embeddingsDistance'):
        if mode == 'eval':
            metric = EmbeddingsDistance(engine = engine, mode = mode)
        else:
            metric = None
    else:
        raise ValueError()

    return metric
