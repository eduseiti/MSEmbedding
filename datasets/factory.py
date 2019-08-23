from bootstrap.lib.options import Options
from .HumanProteome import HumanProteome

def factory(engine=None):
    dataset = {}

    if Options()['dataset']['name'] == 'humanProteome':
        
        if Options()['dataset'].get('train_split', None):
            dataset['train'] = factory_humanProteome(Options()['dataset']['train_split'])

        if Options()['dataset'].get('eval_split', None):

            if dataset['train']:
                dataset['eval'] = factory_humanProteome(Options()['dataset']['eval_split'], 
                                                        dataset['train'])
            else:
                raise RuntimeError("There is no associated training dataset.")
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