from __future__ import annotations

import importlib
from typing import Dict, Type

from src.experiments.adapters.base_adapter import ModelAdapter

STATISTICAL_MODEL_KEYS = frozenset(
    {"gbm_adapter", "block_bootstrap", "ou_process", "merton_jump_diffusion", "de_jump_diffusion", "garch11"}
)

ADAPTER_REGISTRY: Dict[str, str] = {
    "gbm_adapter": "src.experiments.adapters.statistical_adapter.StatisticalGBMAdapter",
    "block_bootstrap": "src.experiments.adapters.statistical_adapter.BlockBootstrapAdapter",
    "ou_process": "src.experiments.adapters.statistical_adapter.StatisticalOUAdapter",
    "merton_jump_diffusion": "src.experiments.adapters.statistical_adapter.StatisticalMertonAdapter",
    "de_jump_diffusion": "src.experiments.adapters.statistical_adapter.StatisticalDEJDAdapter",
    "garch11": "src.experiments.adapters.statistical_adapter.StatisticalGARCH11Adapter",
    "quantgan": "src.experiments.adapters.deep_learning.quantgan_adapter.QuantGANAdapter",
    "timegan": "src.experiments.adapters.deep_learning.timegan_adapter.TimeGANAdapter",
    "timegrad": "src.experiments.adapters.deep_learning.timegrad_adapter.TimeGradAdapter",
    "timevae": "src.experiments.adapters.deep_learning.timevae_adapter.TimeVAEAdapter",
    "unconditional_tsdiffusion": "src.experiments.adapters.deep_learning.utsd_adapter.UnconditionalTSDiffusionAdapter",
    "vrnn": "src.experiments.adapters.deep_learning.vrnn_adapter.VRNNAdapter",
}


def _load_adapter_class(dotted_path: str) -> Type[ModelAdapter]:
    module_name, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_adapter(model_key: str) -> ModelAdapter:
    key = model_key.lower()
    if key not in ADAPTER_REGISTRY:
        supported = ", ".join(sorted(ADAPTER_REGISTRY.keys()))
        raise KeyError(f"Unknown model key '{model_key}'. Supported: {supported}")
    adapter_cls = _load_adapter_class(ADAPTER_REGISTRY[key])
    return adapter_cls()
