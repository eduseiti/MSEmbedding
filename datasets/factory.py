from bootstrap.lib.options import Options
from .HumanProteome import HumanProteome
from .MixedSpectra import MixedSpectra
from .Linfeng import Linfeng

def factory(engine=None):
    dataset = {}

    if Options()['dataset']['name'] == 'humanProteome':
        
        if Options()['dataset'].get('train_split', None):
            dataset['train'] = factory_humanProteome(Options()['dataset']['train_split'])

        if Options()['dataset'].get('eval_split', None):
            if 'train' in dataset:
                dataset['eval'] = factory_humanProteome(Options()['dataset']['eval_split'], 
                                                        dataset['train'])
            else:
                dataset['eval'] = factory_humanProteome(Options()['dataset']['eval_split'])
    
    elif Options()['dataset']['name'] == "mixedSpectra":
        if Options()['dataset'].get('train_split', None):
            dataset['train'] = factory_mixedSpectra(Options()['dataset']['train_split'])

        if Options()['dataset'].get('eval_split', None):
            if 'train' in dataset:
                dataset['eval'] = factory_mixedSpectra(Options()['dataset']['eval_split'], dataset['train'])
            else:
                dataset['eval'] = factory_mixedSpectra(Options()['dataset']['eval_split'])

    elif Options()['dataset']['name'] == "linfeng":
        if Options()['dataset'].get('eval_split', None):
            dataset['test'] = factory_linfeng(Options()['dataset']['eval_split'])
    else:
        raise ValueError()

    return dataset


def factory_humanProteome(split, trainingDataset = None):
    dataset = HumanProteome(
        Options()['dataset']['dir'],
        split,
        batch_size = Options()['dataset']['batch_size'],
        nb_threads = Options()['dataset']['nb_threads'], 
        trainingDataset = trainingDataset)
    return dataset


def factory_mixedSpectra(split, trainingDataset = None):
    dataset = MixedSpectra(
        Options()['dataset']['dir'],
        split,
        batch_size = Options()['dataset']['batch_size'],
        nb_threads = Options()['dataset']['nb_threads'], 
        trainingDataset = trainingDataset)
    return dataset


def factory_linfeng(split):
    dataset = Linfeng(
        Options()['dataset']['dir'],
        split,
        batch_size = Options()['dataset']['batch_size'],
        nb_threads = Options()['dataset']['nb_threads'])
    return dataset        