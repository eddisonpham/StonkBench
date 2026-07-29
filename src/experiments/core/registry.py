from __future__ import annotations

import importlib
from typing import Dict, Type

from src.experiments.adapters.base_adapter import ModelAdapter

STATISTICAL_MODEL_KEYS = frozenset(
    {
        "block_bootstrap",
        "stationary_block_bootstrap",
        "merton_jump_diffusion",
        "de_jump_diffusion",
        "garch11",
    }
)

ADAPTER_REGISTRY: Dict[str, str] = {
    "block_bootstrap": "src.experiments.adapters.statistical_adapter.BlockBootstrapAdapter",
    "stationary_block_bootstrap": "src.experiments.adapters.statistical_adapter.StationaryBlockBootstrapAdapter",
    "merton_jump_diffusion": "src.experiments.adapters.statistical_adapter.StatisticalMertonAdapter",
    "de_jump_diffusion": "src.experiments.adapters.statistical_adapter.StatisticalDEJDAdapter",
    "garch11": "src.experiments.adapters.statistical_adapter.StatisticalGARCH11Adapter",
    "quantgan": "src.experiments.adapters.deep_learning.quantgan_adapter.QuantGANAdapter",
    "kalman_vae": "src.experiments.adapters.deep_learning.kalman_vae_adapter.KalmanVAEAdapter",
    "unconditional_tsdiffusion": "src.experiments.adapters.deep_learning.utsd_adapter.UnconditionalTSDiffusionAdapter",
    "conditional_tsdiffusion": "src.experiments.adapters.deep_learning.cond_tsd_adapter.ConditionalTSDiffusionAdapter",
    "vrnn": "src.experiments.adapters.deep_learning.vrnn_adapter.VRNNAdapter",
    "pcf_gan": "src.experiments.adapters.deep_learning.pcf_gan_adapter.PCFGANAdapter",
    "cond_sig_wgan": "src.experiments.adapters.deep_learning.cond_sig_wgan_adapter.ConditionalSigWGANAdapter",
    "timegrad": "src.experiments.adapters.deep_learning.timegrad_adapter.TimeGradAdapter",
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
