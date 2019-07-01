from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

from .EmbeddingsDistance import EmbeddingDistance

#
# mode: depending on the split (?)
#

def factory(engine=None, mode=None):

    Logger()('Creating EmbeddingDistance metric...')

    if (Options()['model']['metric']['name'] == 'embeddingsDistance'):
        if mode == 'eval':
            metric = EmbeddingDistance()
        else:
            metric = None
    else:
        raise ValueError()

    return metric
