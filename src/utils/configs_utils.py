"""
Config utility functions.
"""
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
                if isinstance(item, str) and 'path' in k.lower():
                    path_candidate = Path(item)
                    if not path_candidate.is_absolute():
                        v[i] = str(base_path / item)
        elif isinstance(v, str) and 'path' in k.lower():
            # Only resolve strings that are in keys containing 'path'
            path_candidate = Path(v)
            if not path_candidate.is_absolute():
                cfg[k] = str(base_path / v)
    return cfg

def replace_ticker_in_cfg(cfg_dict, ticker_value):
    """
    Replace '${ticker}' in all string values in the given config dict with the provided ticker_value.
    """
    for k, v in cfg_dict.items():
        if isinstance(v, dict):
            replace_ticker_in_cfg(v, ticker_value)
        elif isinstance(v, str):
            cfg_dict[k] = v.replace('${ticker}', ticker_value)
    if 'ticker' in cfg_dict:
        cfg_dict['ticker'] = ticker_value

def get_dataset_cfgs(ticker=None):
    """
    Get the dataset configurations from the dataset_cfgs.yaml file.
    """
    project_root = Path(__file__).resolve().parents[2]
    cfg_file = project_root / 'configs' / 'dataset_cfgs.yaml'
    with open(cfg_file, 'r') as f:
        dataset_cfgs = yaml.load(f, Loader=yaml.FullLoader)

    if ticker is None:
        ticker = dataset_cfgs['nonparametric_dataset_cfg'].get('ticker', 'AAPL')
    
    replace_ticker_in_cfg(dataset_cfgs['nonparametric_dataset_cfg'], ticker)
    replace_ticker_in_cfg(dataset_cfgs['parametric_dataset_cfg'], ticker)

    dataset_cfgs = resolve_paths(dataset_cfgs, project_root)
    return dataset_cfgs['nonparametric_dataset_cfg'], dataset_cfgs['parametric_dataset_cfg']