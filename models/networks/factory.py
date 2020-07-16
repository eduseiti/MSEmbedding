import bootstrap.lib.utils as utils
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
from bootstrap.models.networks.data_parallel import DataParallel

from .MSEmbedding import MSEmbeddingNet
from .MSEmbedding_MLP import MSEmbedding_MLP_Net
from .MSEmbedding_norm import MSEmbeddingNormNet
from .MSEmbedding_transformer import MSEmbeddingTransformerNet
from .MSEmbedding_transformer_2 import MSEmbeddingTransformer2Net

def factory(engine=None):

    Logger()('Creating MSEmbedding network...')

    if Options()['model']['network']['name'] == 'MSEmbeddingNet':
        network = MSEmbeddingNet()

        if Options()['misc']['cuda'] and len(utils.available_gpu_ids()) >= 2:
            network = DataParallel(network)

    elif Options()['model']['network']['name'] == 'MSEmbedding_MLP_Net':
        network = MSEmbedding_MLP_Net()

        if Options()['misc']['cuda'] and len(utils.available_gpu_ids()) >= 2:
            network = DataParallel(network)

    elif Options()['model']['network']['name'] == 'MSEmbeddingNormNet':
        network = MSEmbeddingNormNet()

        if Options()['misc']['cuda'] and len(utils.available_gpu_ids()) >= 2:
            network = DataParallel(network)

    elif Options()['model']['network']['name'] == 'MSEmbeddingTransformerNet':
        network = MSEmbeddingTransformerNet()

        if Options()['misc']['cuda'] and len(utils.available_gpu_ids()) >= 2:
            network = DataParallel(network)

    elif Options()['model']['network']['name'] == 'MSEmbeddingTransformer2Net':
        network = MSEmbeddingTransformer2Net()

        if Options()['misc']['cuda'] and len(utils.available_gpu_ids()) >= 2:
            network = DataParallel(network)

    else:
        raise ValueError()

    return network
