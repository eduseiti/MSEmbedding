import torch
import torch.nn as nn

from bootstrap.datasets import transforms
from bootstrap.models.model import Model

from . import networks
from . import criterions
from . import metrics


class MSEmbedding(Model):
    
    def __init__(self,
                 opt,
                 modes=['train', 'eval'],
                 engine=None,
                 cuda_tf=transforms.ToCuda):

        super(MSEmbedding, self).__init__(engine, cuda_tf=cuda_tf)

        self.network = networks.MSEmbeddingNet()

        self.criterions = {}
        self.metrics = {}

        if 'train' in modes:
            self.criterions['train'] = criterions.TripletMargin()

        if 'eval' in modes:
            self.criterions['eval'] = criterions.TripletMargin()
            self.metrics['eval'] = metrics.EmbeddingsDistance(engine = engine, mode = 'eval')
