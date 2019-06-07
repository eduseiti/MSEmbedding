from bootstrap.lib.options import Options
from .humanProteome import HumanProteome

def factory(engine=None):
    dataset = {}

    if Options()['dataset']['name'] == 'humanProteome':
        
        if Options()['dataset'].get('train_split', None):
            dataset['train'] = factory_humanProteome(Options()['dataset']['train_split'])

        if Options()['dataset'].get('eval_split', None): 
            dataset['eval'] = factory_humanProteome(Options()['dataset']['eval_split'])
    else:
        raise ValueError()

    return dataset


def factory_humanProteome(split):
    dataset = HumanProteome(
        Options()['dataset']['dir'],
        split,
        batch_size=Options()['dataset']['batch_size'],
        nb_threads=Options()['dataset']['nb_threads'])
    return dataset