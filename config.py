import os.path
import time

from dotmap import DotMap
from yaml import load, FullLoader


def get_config_from_yaml(yaml_file: str) -> object:
    with open(yaml_file, 'r', encoding='utf8') as config_file:
        return DotMap(load(config_file, Loader=FullLoader))

def set_config_dirs(yaml_file: str) -> object:
    config = get_config_from_yaml(yaml_file)
    config.callbacks.tensorboard.log_dir = os.path.join(
        'experiments',
        time.strftime("%Y-%m-%d/", time.localtime()),
        config.experiment.name,
        "logs/")
    config.callbacks.checkpoint_dir = os.path.join(
        "experiments",
        time.strftime("%Y-%m-%d/", time.localtime()),
        config.experiment.name,
        "checkpoints/")
    return config


cfg = set_config_dirs('configs/cnn_model_1.yaml')

print(cfg.callbacks.modelcheckpoint.filepath, cfg.callbacks.write_graf, cfg.callbacks.tensorboard.log_dir)
