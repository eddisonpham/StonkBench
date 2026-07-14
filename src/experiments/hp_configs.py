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
    "pcf_gan",
    "sig_wgan",
]

# Cap full-train patience so early-stopping does not collapse training too soon.
# Collapsed GAN/RNN families need a higher floor.
FULL_TRAIN_PATIENCE_CAP = 12

# Models that systematically under-dispersed on z-scored returns in run 2026-07-11.
CALIBRATE_ON_GENERATE = {
    "quantgan",
    "vrnn",
    "pcf_gan",
    # sig_wgan: batch moment calibration amplified collapse (zero band + wave outliers).
}


@dataclass(frozen=True)
class HPConfig:
    config_id: str
    is_vendor_default: bool
    learning_rate: float
    batch_size: int
    patience: int

    def metadata(self, max_epochs: int, model_key: str = "") -> Dict[str, float | int | bool]:
        return {
            "config_id": self.config_id,
            "is_vendor_default": self.is_vendor_default,
            "max_epochs": max_epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "use_calibration": model_key in CALIBRATE_ON_GENERATE,
        }


# Longer HP budgets; GAN/RNN families need more epochs before early-stop can fire.
HP_SEARCH_EPOCHS: Dict[str, int] = {
    "quantgan": 80,
    "timegan": 80,
    "timegrad": 60,
    "timevae": 60,
    "unconditional_tsdiffusion": 60,
    "vrnn": 80,
    "pcf_gan": 80,
    "sig_wgan": 80,
}

FULL_TRAIN_EPOCHS: Dict[str, int] = {
    "quantgan": 150,
    "timegan": 150,
    "timegrad": 150,
    "timevae": 120,
    "unconditional_tsdiffusion": 150,
    "vrnn": 150,
    "pcf_gan": 150,
    "sig_wgan": 150,
}

# Patience raised for collapse-prone models (avoid epoch-2 / epoch-13 stops).
MODEL_HP_CONFIGS: Dict[str, List[HPConfig]] = {
    "quantgan": [
        HPConfig("vendor_default", True, 2e-4, 30, 10),
        HPConfig("lr1e-4_bs30_p10", False, 1e-4, 30, 10),
        HPConfig("lr5e-4_bs30_p10", False, 5e-4, 30, 10),
        HPConfig("lr2e-4_bs16_p10", False, 2e-4, 16, 10),
        HPConfig("lr2e-4_bs64_p10", False, 2e-4, 64, 10),
        HPConfig("lr2e-4_bs30_p8", False, 2e-4, 30, 8),
        HPConfig("lr2e-4_bs30_p12", False, 2e-4, 30, 12),
        HPConfig("lr1e-4_bs64_p8", False, 1e-4, 64, 8),
        HPConfig("lr5e-4_bs16_p12", False, 5e-4, 16, 12),
    ],
    # Prefer mid batch sizes (32–64). Prior 2026-07-11 "winner" bs=256 collapsed
    # (spurious val loss on z-score scale + GRU batch/time swap). Cap stays ≤128 in adapter.
    "timegan": [
        HPConfig("vendor_default", True, 1e-3, 64, 10),
        HPConfig("lr5e-4_bs64_p10", False, 5e-4, 64, 10),
        HPConfig("lr1e-3_bs32_p10", False, 1e-3, 32, 10),
        HPConfig("lr1e-3_bs64_p12", False, 1e-3, 64, 12),
        HPConfig("lr5e-4_bs32_p12", False, 5e-4, 32, 12),
        HPConfig("lr2e-3_bs64_p10", False, 2e-3, 64, 10),
        HPConfig("lr1e-3_bs64_p8", False, 1e-3, 64, 8),
        HPConfig("lr5e-4_bs64_p12", False, 5e-4, 64, 12),
        HPConfig("lr1e-3_bs128_p12", False, 1e-3, 128, 12),
    ],
    "timegrad": [
        HPConfig("vendor_default", True, 1e-3, 32, 5),
        HPConfig("lr5e-4_bs32_p5", False, 5e-4, 32, 5),
        HPConfig("lr2e-3_bs32_p5", False, 2e-3, 32, 5),
        HPConfig("lr1e-3_bs16_p5", False, 1e-3, 16, 5),
        HPConfig("lr1e-3_bs64_p5", False, 1e-3, 64, 5),
        HPConfig("lr1e-3_bs32_p4", False, 1e-3, 32, 4),
        HPConfig("lr1e-3_bs32_p6", False, 1e-3, 32, 6),
        HPConfig("lr5e-4_bs16_p4", False, 5e-4, 16, 4),
        HPConfig("lr2e-3_bs64_p6", False, 2e-3, 64, 6),
    ],
    "timevae": [
        HPConfig("vendor_default", True, 1e-3, 16, 5),
        HPConfig("lr5e-4_bs16_p5", False, 5e-4, 16, 5),
        HPConfig("lr2e-3_bs16_p5", False, 2e-3, 16, 5),
        HPConfig("lr1e-3_bs8_p5", False, 1e-3, 8, 5),
        HPConfig("lr1e-3_bs32_p5", False, 1e-3, 32, 5),
        HPConfig("lr1e-3_bs16_p4", False, 1e-3, 16, 4),
        HPConfig("lr1e-3_bs16_p6", False, 1e-3, 16, 6),
        HPConfig("lr5e-4_bs8_p4", False, 5e-4, 8, 4),
        HPConfig("lr2e-3_bs32_p6", False, 2e-3, 32, 6),
    ],
    "unconditional_tsdiffusion": [
        HPConfig("vendor_default", True, 1e-3, 64, 5),
        HPConfig("lr5e-4_bs64_p5", False, 5e-4, 64, 5),
        HPConfig("lr2e-3_bs64_p5", False, 2e-3, 64, 5),
        HPConfig("lr1e-3_bs32_p5", False, 1e-3, 32, 5),
        HPConfig("lr1e-3_bs128_p5", False, 1e-3, 128, 5),
        HPConfig("lr1e-3_bs64_p4", False, 1e-3, 64, 4),
        HPConfig("lr1e-3_bs64_p6", False, 1e-3, 64, 6),
        HPConfig("lr5e-4_bs32_p4", False, 5e-4, 32, 4),
        HPConfig("lr2e-3_bs128_p6", False, 2e-3, 128, 6),
    ],
    "vrnn": [
        HPConfig("vendor_default", True, 1e-3, 32, 10),
        HPConfig("lr5e-4_bs32_p10", False, 5e-4, 32, 10),
        HPConfig("lr2e-3_bs32_p10", False, 2e-3, 32, 10),
        HPConfig("lr1e-3_bs16_p10", False, 1e-3, 16, 10),
        HPConfig("lr1e-3_bs64_p10", False, 1e-3, 64, 10),
        HPConfig("lr1e-3_bs32_p8", False, 1e-3, 32, 8),
        HPConfig("lr1e-3_bs32_p12", False, 1e-3, 32, 12),
        HPConfig("lr5e-4_bs16_p8", False, 5e-4, 16, 8),
        HPConfig("lr2e-3_bs64_p12", False, 2e-3, 64, 12),
    ],
    # Vendor stock/OU defaults: lr_G=1e-3, batch_size=64, GAN-like patience.
    "pcf_gan": [
        HPConfig("vendor_default", True, 1e-3, 64, 10),
        HPConfig("lr5e-4_bs64_p10", False, 5e-4, 64, 10),
        HPConfig("lr2e-3_bs64_p10", False, 2e-3, 64, 10),
        HPConfig("lr1e-3_bs32_p10", False, 1e-3, 32, 10),
        HPConfig("lr1e-3_bs128_p10", False, 1e-3, 128, 10),
        HPConfig("lr1e-3_bs64_p8", False, 1e-3, 64, 8),
        HPConfig("lr1e-3_bs64_p12", False, 1e-3, 64, 12),
        HPConfig("lr5e-4_bs32_p8", False, 5e-4, 32, 8),
        HPConfig("lr2e-3_bs128_p12", False, 2e-3, 128, 12),
    ],
    # Vendor SigWGAN.json: lr=1e-3; batch capped vs paper's 2000 for our window counts.
    "sig_wgan": [
        HPConfig("vendor_default", True, 1e-3, 128, 10),
        HPConfig("lr5e-4_bs128_p10", False, 5e-4, 128, 10),
        HPConfig("lr2e-3_bs128_p10", False, 2e-3, 128, 10),
        HPConfig("lr1e-3_bs64_p10", False, 1e-3, 64, 10),
        HPConfig("lr1e-3_bs256_p10", False, 1e-3, 256, 10),
        HPConfig("lr1e-3_bs128_p8", False, 1e-3, 128, 8),
        HPConfig("lr1e-3_bs128_p12", False, 1e-3, 128, 12),
        HPConfig("lr5e-4_bs64_p8", False, 5e-4, 64, 8),
        HPConfig("lr2e-3_bs256_p12", False, 2e-3, 256, 12),
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
    patience = min(int(best["patience"]), FULL_TRAIN_PATIENCE_CAP)
    return {
        "config_id": best.get("config_id", "hp_winner"),
        "max_epochs": FULL_TRAIN_EPOCHS[model_key],
        "learning_rate": float(best["learning_rate"]),
        "batch_size": int(best["batch_size"]),
        "patience": patience,
        "use_calibration": model_key in CALIBRATE_ON_GENERATE,
        "hp_search_val_loss": float(best["mean_best_val_loss"]),
    }
