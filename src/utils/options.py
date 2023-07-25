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
    # debug
    debug_gp = parser.add_mutually_exclusive_group()
    debug_gp.add_argument('--debug_epoch', action='store_true')
    debug_gp.add_argument('--debug_iter',action='store_true')
    
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
        assert type(value) == type(config[key]) or config[key] is None, f'type of {key} is {type(config[key])}, not {type(value)}'
        
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
    assert config_path.suffix in ['.yml', ".yaml"], f'config file {config_path} is not a yaml file'
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def config_filter(config: dict, mode: str):
    
    nece_modes = set([mode])
    if mode == "train" and config[mode]["need_eval"]:
        nece_modes.add("eval")
        
    nece_datasets = set()
    for mode in nece_modes:
        if config[mode]["need_dataset"]:
            nece_datasets.add(mode)
            
    for del_mode in set(['train', 'eval']) - nece_modes:
        if del_mode in config:
            del config[del_mode]
    for del_dataset in set(['train', 'eval']) - nece_datasets:
        if del_dataset in config['dataset']:
            del config['dataset'][del_dataset]
    
    return config

def get_config() -> EasyDict:
    """ 
    Process command line arguments and config files. Delete those redundant key-value pairs.
    """
    args = parse_args()
    return get_config_(
        config=args.config,
        new_config=args.new_config,
        train=args.train,
        eval=args.eval,
        sample=args.sample,
        print=args.print,
        debug_epoch=args.debug_epoch,
        debug_iter=args.debug_iter
    )
    
def get_config_(config, new_config="", train=True, eval=False, sample=False, print=False, debug_epoch=False, debug_iter=False):
    # 读取基础配置文件
    config_path = config_name2path(config)
    config = read_config_file(config_path)
    # 读取更新的配置文件
    if new_config != '':
        new_config_path = config_name2path(new_config)
        new_config = read_config_file(new_config_path)
        config = update_config(config, new_config)
    # 确定模式
    if train:
        config['mode'] = 'train'
    elif eval:
        config['mode'] = 'eval'
    else:
        raise ValueError('invalid mode')
    # 删去无用的key-value
    config = config_filter(config, config['mode'])
    
    assert 'experiment_name' in config and config['experiment_name'] is not None, 'experiment_name is empty'
    assert 'trial_index' in config and config['trial_index'] is not None, 'trial_index is empty'
    
    if print:
        pprint(config)
        exit()
            
    config['debug_epoch'] = debug_epoch
    config['debug_iter'] = debug_iter
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