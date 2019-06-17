import bootstrap.lib.utils as utils
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
from bootstrap.models.networks.data_parallel import DataParallel

from .MSEmbedding import MSEmbeddingNet


def factory(engine=None):

    Logger()('Creating MSEmbedding network...')

    if Options()['model']['network']['name'] == 'MSEmbeddingNet':
        network = MSEmbeddingNet()

        if Options()['misc']['cuda'] and len(utils.available_gpu_ids()) >= 2:
            network = DataParallel(network)

    else:
        raise ValueError()

    return network
