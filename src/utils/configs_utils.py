"""
Config utility functions.
"""
import yaml
from pathlib import Path
import yaml
from pathlib import Path


def resolve_paths(cfg: dict, base_path: Path) -> dict:
    """
    Recursively resolve all string paths in a config dict to absolute paths
    relative to base_path. Only modifies strings that look like paths.
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            resolve_paths(v, base_path)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, str):
                    path_candidate = Path(item)
                    if not path_candidate.is_absolute() and path_candidate.exists():
                        v[i] = str(base_path / item)
        elif isinstance(v, str):
            path_candidate = Path(v)
            if not path_candidate.is_absolute():
                cfg[k] = str(base_path / v)
    return cfg

def get_dataset_cfgs():
    """
    Get the dataset configurations from the dataset_cfgs.yaml file.
    Handles absolute path resolution for notebook compatibility.
    """
    project_root = Path(__file__).resolve().parents[2]
    cfg_file = project_root / 'configs' / 'dataset_cfgs.yaml'
    with open(cfg_file, 'r') as f:
        dataset_cfgs = yaml.load(f, Loader=yaml.FullLoader)
    dataset_cfgs = resolve_paths(dataset_cfgs, project_root)
    return dataset_cfgs['nonparametric_dataset_cfgs'], dataset_cfgs['parametric_dataset_cfg']

def get_model_cfgs():
    """
    Get all model configurations from the model_cfgs.yaml file.
    Resolves file paths relative to project root.
    """
    project_root = Path(__file__).resolve().parents[2]
    cfg_file = project_root / 'configs' / 'model_cfgs.yaml'
    with open(cfg_file, 'r') as f:
        model_cfgs = yaml.load(f, Loader=yaml.FullLoader)
    model_cfgs = resolve_paths(model_cfgs, project_root)
    return model_cfgs['TimeGAN_model_cfg']


