from __future__ import annotations

from typing import Dict, Type

from src.experiments.adapters.base_adapter import ModelAdapter
from src.experiments.adapters.deep_learning import (
    QuantGANAdapter,
    TimeGANAdapter,
    TimeGradAdapter,
    TimeVAEAdapter,
    UnconditionalTSDiffusionAdapter,
    VRNNAdapter,
)
from src.experiments.adapters.statistical_adapter import (
    BlockBootstrapAdapter,
    StatisticalDEJDAdapter,
    StatisticalGARCH11Adapter,
    StatisticalGBMAdapter,
    StatisticalMertonAdapter,
    StatisticalOUAdapter,
)

STATISTICAL_MODEL_KEYS = frozenset(
    {"gbm_adapter", "block_bootstrap", "ou_process", "merton_jump_diffusion", "de_jump_diffusion", "garch11"}
)

ADAPTER_REGISTRY: Dict[str, Type[ModelAdapter]] = {
    "gbm_adapter": StatisticalGBMAdapter,
    "block_bootstrap": BlockBootstrapAdapter,
    "ou_process": StatisticalOUAdapter,
    "merton_jump_diffusion": StatisticalMertonAdapter,
    "de_jump_diffusion": StatisticalDEJDAdapter,
    "garch11": StatisticalGARCH11Adapter,
    "quantgan": QuantGANAdapter,
    "timegan": TimeGANAdapter,
    "timegrad": TimeGradAdapter,
    "timevae": TimeVAEAdapter,
    "unconditional_tsdiffusion": UnconditionalTSDiffusionAdapter,
    "vrnn": VRNNAdapter,
}


def get_adapter(model_key: str) -> ModelAdapter:
    key = model_key.lower()
    if key not in ADAPTER_REGISTRY:
        supported = ", ".join(sorted(ADAPTER_REGISTRY.keys()))
        raise KeyError(f"Unknown model key '{model_key}'. Supported: {supported}")
    return ADAPTER_REGISTRY[key]()
