"""
Config utility functions.
"""

import yaml
from pathlib import Path


def get_dataset_cfgs():
    """
    Get the dataset configurations from the dataset_cfgs.yaml file.
    """
    project_root = Path(__file__).resolve().parents[2]
    cfg_file = project_root / 'configs' / 'dataset_cfgs.yaml'
    with open(cfg_file, 'r') as f:
        dataset_cfgs = yaml.load(f, Loader=yaml.FullLoader)

    return dataset_cfgs['deep_learning_dataset_cfg'], dataset_cfgs['statistical_dataset_cfg']
