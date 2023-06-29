
import trainers
from utils.options import get_config

def init_trainer(config):
    return getattr(trainers, config.trainer_name)(config)

def main():
    config = get_config()
    trainer = init_trainer(config)
    getattr(trainer, config.mode)()
    
if __name__ == '__main__':
    main()