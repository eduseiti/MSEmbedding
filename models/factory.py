from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

from .MSEmbedding import MSEmbedding


def factory(engine=None):

    Logger()('Creating MSEmbedding model...')

    if Options()['model']['name'] == 'MSEmbedding':
        model = MSEmbedding(Options()['model'],
                            engine.dataset.keys(),
                            engine)
    else:
        raise ValueError()

    return model
