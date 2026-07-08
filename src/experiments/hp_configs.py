"""Vendor-aligned hyperparameter grids for HP search and full training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

DL_MODEL_KEYS = [
    "quantgan",
    "timegan",
    "timegrad",
    "timevae",
    "unconditional_tsdiffusion",
    "vrnn",
]


@dataclass(frozen=True)
class HPConfig:
    config_id: str
    is_vendor_default: bool
    learning_rate: float
    batch_size: int
    patience: int

    def metadata(self, max_epochs: int) -> Dict[str, float | int | bool]:
        return {
            "config_id": self.config_id,
            "is_vendor_default": self.is_vendor_default,
            "max_epochs": max_epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "use_calibration": False,
        }


# Epoch budgets: HP search uses shorter budgets; final training uses vendor-scale budgets.
HP_SEARCH_EPOCHS: Dict[str, int] = {
    "quantgan": 30,
    "timegan": 30,
    "timegrad": 40,
    "timevae": 40,
    "unconditional_tsdiffusion": 40,
    "vrnn": 30,
}

FULL_TRAIN_EPOCHS: Dict[str, int] = {
    "quantgan": 60,
    "timegan": 60,
    "timegrad": 100,
    "timevae": 80,
    "unconditional_tsdiffusion": 100,
    "vrnn": 50,
}

# Vendor defaults (QuantGANConfig, TimeGAN options.py, timegrad trainer, timeVAE yaml,
# unconditional tsdiff train_tsdiff.yaml, VRNN train.py).
MODEL_HP_CONFIGS: Dict[str, List[HPConfig]] = {
    "quantgan": [
        HPConfig("vendor_default", True, 2e-4, 30, 12),
        HPConfig("lr1e-4_bs30_p12", False, 1e-4, 30, 12),
        HPConfig("lr5e-4_bs30_p12", False, 5e-4, 30, 12),
        HPConfig("lr2e-4_bs16_p12", False, 2e-4, 16, 12),
        HPConfig("lr2e-4_bs64_p12", False, 2e-4, 64, 12),
        HPConfig("lr2e-4_bs30_p8", False, 2e-4, 30, 8),
        HPConfig("lr2e-4_bs30_p16", False, 2e-4, 30, 16),
        HPConfig("lr1e-4_bs64_p8", False, 1e-4, 64, 8),
        HPConfig("lr5e-4_bs16_p16", False, 5e-4, 16, 16),
    ],
    "timegan": [
        HPConfig("vendor_default", True, 1e-3, 128, 12),
        HPConfig("lr5e-4_bs128_p12", False, 5e-4, 128, 12),
        HPConfig("lr2e-3_bs128_p12", False, 2e-3, 128, 12),
        HPConfig("lr1e-3_bs64_p12", False, 1e-3, 64, 12),
        HPConfig("lr1e-3_bs256_p12", False, 1e-3, 256, 12),
        HPConfig("lr1e-3_bs128_p8", False, 1e-3, 128, 8),
        HPConfig("lr1e-3_bs128_p16", False, 1e-3, 128, 16),
        HPConfig("lr5e-4_bs64_p8", False, 5e-4, 64, 8),
        HPConfig("lr2e-3_bs256_p16", False, 2e-3, 256, 16),
    ],
    "timegrad": [
        HPConfig("vendor_default", True, 1e-3, 32, 12),
        HPConfig("lr5e-4_bs32_p12", False, 5e-4, 32, 12),
        HPConfig("lr2e-3_bs32_p12", False, 2e-3, 32, 12),
        HPConfig("lr1e-3_bs16_p12", False, 1e-3, 16, 12),
        HPConfig("lr1e-3_bs64_p12", False, 1e-3, 64, 12),
        HPConfig("lr1e-3_bs32_p8", False, 1e-3, 32, 8),
        HPConfig("lr1e-3_bs32_p16", False, 1e-3, 32, 16),
        HPConfig("lr5e-4_bs16_p8", False, 5e-4, 16, 8),
        HPConfig("lr2e-3_bs64_p16", False, 2e-3, 64, 16),
    ],
    "timevae": [
        HPConfig("vendor_default", True, 1e-3, 16, 12),
        HPConfig("lr5e-4_bs16_p12", False, 5e-4, 16, 12),
        HPConfig("lr2e-3_bs16_p12", False, 2e-3, 16, 12),
        HPConfig("lr1e-3_bs8_p12", False, 1e-3, 8, 12),
        HPConfig("lr1e-3_bs32_p12", False, 1e-3, 32, 12),
        HPConfig("lr1e-3_bs16_p8", False, 1e-3, 16, 8),
        HPConfig("lr1e-3_bs16_p16", False, 1e-3, 16, 16),
        HPConfig("lr5e-4_bs8_p8", False, 5e-4, 8, 8),
        HPConfig("lr2e-3_bs32_p16", False, 2e-3, 32, 16),
    ],
    "unconditional_tsdiffusion": [
        HPConfig("vendor_default", True, 1e-3, 64, 12),
        HPConfig("lr5e-4_bs64_p12", False, 5e-4, 64, 12),
        HPConfig("lr2e-3_bs64_p12", False, 2e-3, 64, 12),
        HPConfig("lr1e-3_bs32_p12", False, 1e-3, 32, 12),
        HPConfig("lr1e-3_bs128_p12", False, 1e-3, 128, 12),
        HPConfig("lr1e-3_bs64_p8", False, 1e-3, 64, 8),
        HPConfig("lr1e-3_bs64_p16", False, 1e-3, 64, 16),
        HPConfig("lr5e-4_bs32_p8", False, 5e-4, 32, 8),
        HPConfig("lr2e-3_bs128_p16", False, 2e-3, 128, 16),
    ],
    "vrnn": [
        HPConfig("vendor_default", True, 1e-3, 8, 12),
        HPConfig("lr5e-4_bs8_p12", False, 5e-4, 8, 12),
        HPConfig("lr2e-3_bs8_p12", False, 2e-3, 8, 12),
        HPConfig("lr1e-3_bs4_p12", False, 1e-3, 4, 12),
        HPConfig("lr1e-3_bs16_p12", False, 1e-3, 16, 12),
        HPConfig("lr1e-3_bs8_p8", False, 1e-3, 8, 8),
        HPConfig("lr1e-3_bs8_p16", False, 1e-3, 8, 16),
        HPConfig("lr5e-4_bs4_p8", False, 5e-4, 4, 8),
        HPConfig("lr2e-3_bs16_p16", False, 2e-3, 16, 16),
    ],
}


def configs_for_model(model_key: str) -> List[HPConfig]:
    if model_key not in MODEL_HP_CONFIGS:
        raise KeyError(f"No HP configs for model '{model_key}'")
    configs = MODEL_HP_CONFIGS[model_key]
    if len(configs) != 9:
        raise ValueError(f"Expected 9 configs (vendor + 8 variants) for {model_key}, got {len(configs)}")
    return configs


def full_train_metadata(model_key: str, hp_summary_entry: Dict) -> Dict[str, float | int | bool]:
    best = hp_summary_entry["best_config"]
    return {
        "config_id": best.get("config_id", "hp_winner"),
        "max_epochs": FULL_TRAIN_EPOCHS[model_key],
        "learning_rate": float(best["learning_rate"]),
        "batch_size": int(best["batch_size"]),
        "patience": int(best["patience"]),
        "use_calibration": False,
        "hp_search_val_loss": float(best["mean_best_val_loss"]),
    }
