from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

from .EmbeddingsDistance import EmbeddingsDistance
from .SaveEmbeddings import SaveEmbeddings

#
# mode: depending on the split (?)
#

def factory(engine=None, mode=None):

    if (Options()['model']['metric']['name'] == 'embeddingsDistance'):
        if mode == 'eval':
            Logger()('Creating EmbeddingsDistance metric...')
            metric = EmbeddingsDistance(engine = engine, mode = mode)
        else:
            metric = None
    elif (Options()['model']['metric']['name'] == 'saveEmbeddings'):
        if mode == 'test':
            Logger()('Creating SaveEmbeddings metric...')

            metric = SaveEmbeddings(engine = engine, mode = mode)
        else:
            metric = None
    else:
        raise ValueError()

    return metric
