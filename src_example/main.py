import argparse
import yaml
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', default='')
    parser.add_argument('--new_config', default='')
    # exclusive arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--eval', action='store_true')
    group.add_argument('--sample', action='store_true')

    return parser.parse_args()

def apply_update(config, new_config):
    for key, value in new_config.items():
        if isinstance(value, dict):
            if key not in config:
                raise ValueError(f'key {key} not in config')
            apply_update(config[key], value)
        else:
            config[key] = value
    return config

def need_mode(mode, config, mode_set=set()):
    mode_set.add(mode)
    if 'need_other_modes' in config[mode]:
        for other_mode in config[mode]['need_other_modes']:
            mode_set = need_mode(other_mode, config, mode_set)
    return mode_set

def need_dataset(modes, config):
    dataset_set = set()
    for mode in modes:
        if 'need_datasets' in config[mode]:
            for dataset in config[mode]['need_datasets']:
                dataset_set.add(dataset)
    return dataset_set

def find_config_file(config_name):
    if config_name == '':
        raise ValueError('config name is empty')
    if Path(config_name).exists():
        config_path = Path(config_name)
    else:
        config_path = Path(__file__).parent / Path('configs', config_name + '.yml')
        if not config_path.exists():
            raise ValueError(f'config file {config_path} not found')
    return config_path
        
        
args = parse_args()

config_path = find_config_file(args.config)
with open(config_path) as f:
    config = yaml.safe_load(f)

if args.new_config != '':
    new_config_path = find_config_file(args.new_config)
    with open(new_config_path) as f:
        new_config = yaml.safe_load(f)
    config = apply_update(config, new_config)

if args.train:
    config['mode'] = 'train'
elif args.eval:
    config['mode'] = 'eval'
elif args.sample:
    config['mode'] = 'sample'

need_modes = need_mode(config['mode'], config)
for mode in set('train eval sample'.split()) - need_modes:
    del config[mode]
    
need_datasets = need_dataset(need_modes, config)
for dataset in set(['train', 'eval']) - need_datasets:
    del config['dataset'][dataset]

from easydict import EasyDict
from trainers import ClassificationTrainer

trainer = ClassificationTrainer(EasyDict(config))
if args.train:
    trainer.train()
elif args.eval:
    trainer.eval()