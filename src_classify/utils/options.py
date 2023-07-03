import argparse
import yaml
from pathlib import Path
from easydict import EasyDict
from pprint import pprint

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', default='base_config', help='path to config file')
    parser.add_argument('--new_config', default='', help='path to new config file, used to update `lr`, `batch_size`, etc.')
    # exclusive arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--eval', action='store_true')
    group.add_argument('--sample', action='store_true')
    # print arguments
    parser.add_argument('--print', action='store_true')

    return parser.parse_args()

def update_config(config: dict, new_config: dict):
    """ 
    Recursively update values in `config` with those in `new_config`.
    
    Structures of `new_config` must be a subtree of `config`, i.e. all keys 
    in `new_config` must be in `config`, and all values in `new_config` must 
    be of the same type as those in `config`.
    
    Examples:
    >>> update_config(config={'a': 1, 'b': 2}, new_config={'b': 0})
    {'a': 1, 'b': 0}
    
    >>> update_config(config={'a': {'b': 1, 'c': 2}}, new_config={'a': {'b': 0}})
    {'a': {'b': 0, 'c': 2}}
    
    >>> update_config(config={'a': 1}, new_config={'b': 2})
    AssertionError: key b not in config
    
    >>> update_config(config={'a': 1}, new_config={'a': {'b': 2}})
    AssertionError: type of a is <class 'int'>, not <class 'dict'>
    """
    for key, value in new_config.items():
        assert key in config, f'key {key} not in config'
        assert type(value) == type(config[key]), f'type of {key} is {type(config[key])}, not {type(value)}'
        
        if type(value) == type(config[key]) == dict:
            update_config(config[key], value)
        else:
            config[key] = value
    return config


ROOT_DIR = Path(__file__).parent.parent

def config_name2path(config_name: str):
    """ 
    return the path of config file given its name or path.
    """
    if config_name == '':
        raise ValueError('config name is empty')
    if Path(config_name).exists():
        config_path = Path(config_name)
    else:
        config_path = ROOT_DIR / Path('configs', config_name + '.yml')
        if not config_path.exists():
            raise ValueError(f'config file {config_path} not found')
    return config_path

def read_config_file(config_path: Path):
    assert config_path.exists(), f'config file {config_path} not found'
    assert config_path.suffix == '.yml', f'config file {config_path} is not a yaml file'
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def useful_mode(mode: str, config: dict, mode_set: set = set()):
    """ 
    extract useful mode from config file. Those modes in `need_other_modes` argument will not be deleted.
    """
    mode_set.add(mode)
    if 'need_other_modes' in config[mode]:
        for other_mode in config[mode]['need_other_modes']:
            mode_set = useful_mode(other_mode, config, mode_set)
    return mode_set

def useful_dataset(modes: set, config: dict):
    """ 
    extract useful dataset from config file. Those datasets in `need_other_datasets` argument will not be deleted.
    """
    dataset_set = set()
    for mode in modes:
        if 'need_datasets' in config[mode]:
            for dataset in config[mode]['need_datasets']:
                dataset_set.add(dataset)
    return dataset_set

def config_filter(config: dict, mode: str):
    need_modes = useful_mode(mode, config)
    redundant_modes = set('train eval sample'.split()) - need_modes
    for del_mode in redundant_modes:
        del config[del_mode]
    redundant_datasets = set('train eval'.split()) - useful_dataset(need_modes, config)
    
    for del_dataset in redundant_datasets:
        del config['dataset'][del_dataset]
    
    return config

def get_config() -> EasyDict:
    """ 
    Process command line arguments and config files. Delete those redundant key-value pairs.
    """
    args = parse_args()
    
    config_path = config_name2path(args.config)
    config = read_config_file(config_path)
    
    if args.new_config != '':
        new_config_path = config_name2path(args.new_config)
        new_config = read_config_file(new_config_path)
        config = update_config(config, new_config)
    
    if args.train:
        config['mode'] = 'train'
    elif args.eval:
        config['mode'] = 'eval'
    elif args.sample:
        config['mode'] = 'sample'
    else:
        raise ValueError('invalid mode')
    
    config = config_filter(config, config['mode'])
    
    assert 'experiment_name' in config and config['experiment_name'] is not None, 'experiment_name is empty'
    assert 'trial_index' in config and config['trial_index'] is not None, 'trial_index is empty'
    
    if args.print:
        pprint(config)
        exit()
        
    return EasyDict(config)

def edict2dict(edict: EasyDict):
    """ 
    Convert EasyDict to dict.
    """
    d = {}
    for k, v in edict.items():
        if isinstance(v, EasyDict):
            d[k] = edict2dict(v)
        else:
            d[k] = v
    return d

def save_config(config: EasyDict, path):
    with open(path, 'w') as f:
        yaml.dump(edict2dict(config), f)