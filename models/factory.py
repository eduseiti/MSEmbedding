from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

from .MSEmbedding import MSEmbedding


def factory(engine=None):

    Logger()('Creating MSEmbedding model...')

    if Options()['model']['name'] == 'MSEmbedding':
        model = MSEmbedding(Options()['model'],
                            engine.dataset.keys(),
                            engine)
    elif Options()['model']['name'] == 'MSEmbedding_encoding':
        model = MSEmbedding(Options()['model'],
                            'test',
                            engine)
    else:
        raise ValueError()

    return model
