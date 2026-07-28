"""Vendor-aligned hyperparameter grids for HP search and full training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List# Canonical 7 DL models (post 2026-07-23 cleanup).  timegrad, timevae, sig_wgan
# removed; cond_sig_wgan + kalman_vae + conditional_tsdiffusion kept (each one
# successfully produces meaningful samples and ships a checkpoint).
DL_MODEL_KEYS = [
    "quantgan",
    "vrnn",
    "pcf_gan",
    "kalman_vae",
    "unconditional_tsdiffusion",
    "conditional_tsdiffusion",
    "cond_sig_wgan",
    "timegrad",
]


# Cap full-train patience so early-stopping does not collapse training too soon.
# Collapsed GAN/RNN families need a higher floor.
FULL_TRAIN_PATIENCE_CAP = 12

# Models that systematically under-dispersed on z-scored returns in run 2026-07-11.
# cond_sig_wgan added 2026-07-23: revert run shows per-channel std_ratio ~0.3 across all
# 25 channels (KS p=0). Channel moment calibration is required to lift std_ratio to ~1.0.
# Vendor-faithful: no post-hoc moment injection. Generated output is whatever
# the model produces. If a model collapses, the model's output is the result.
CALIBRATE_ON_GENERATE: set[str] = set()


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
# Vendor-aligned training budgets.
# quantgan: vendor min_epochs=max(20, patience*2)=20 (quantgan_module.py)
# vrnn: vendor n_epochs=25 (train.py)
# pcf_gan: no explicit vendor epoch count; 80/150 kept (adapter clamps)
# kalman_vae: no explicit vendor epoch count; 60/100 kept
# unconditional/conditional_tsd: vendor max_epochs=100 (train_tsdiff*.yaml)
# cond_sig_wgan: vendor total_steps=1000 (train.py), not epoch-based
# timegrad: vendor epochs=100 (trainer.py)
HP_SEARCH_EPOCHS: Dict[str, int] = {
    "quantgan": 20,
    "vrnn": 25,
    "pcf_gan": 80,
    "kalman_vae": 60,
    "unconditional_tsdiffusion": 100,
    "conditional_tsdiffusion": 100,
    "cond_sig_wgan": 100,
    "timegrad": 100,
}

FULL_TRAIN_EPOCHS: Dict[str, int] = {
    "quantgan": 20,
    "vrnn": 25,
    "pcf_gan": 150,
    "kalman_vae": 100,
    "unconditional_tsdiffusion": 100,
    "conditional_tsdiffusion": 100,
    "cond_sig_wgan": 100,
    "timegrad": 100,
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
    # Conditional TSDiffusion uses TSDiffCond (same backbone as the uncond variant
    # but with mask-aware loss + observed-past conditioning).  Per-step
    # diagnostics are identical, and the lr/bs/patience sensitivity surface
    # is the same — alias the unconditional grid rather than copy 9 lines.
    "conditional_tsdiffusion": [
        HPConfig("vendor_default", True, 1e-3, 64, 5),
        HPConfig("lr5e-4_bs64_p5", False, 5e-4, 64, 5),
        HPConfig("lr2e-3_bs64_p5", False, 2e-3, 64, 5),
        HPConfig("lr1e-3_bs32_p5", False, 1e-3, 32, 5),
        HPConfig("lr1e-3_bs128_p5", False, 1e-3, 128, 5),
        HPConfig("lr1e-3_bs64_p4", False, 1e-3, 64, 4),
        HPConfig("lr1e-3_bs64_p6", False, 1e-3, 64, 6),
        HPConfig("lr5e-4_bs32_p4", False, 5e-4, 32, 4),
        HPConfig("lr2e-3_bs128_p6", False, 2e-3, 128, 6),
    ],  # alias of unconditional_tsdiffusion grid; values intentionally inlined.

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
    # Kalman-VAE: vendor kvae defaults (lr=1e-3) plus the K-VAE-specific
    # architectural knobs (a_dim, z_dim, K, dynamics). The adapter reads
    # kvae_a_dim / kvae_z_dim / kvae_K / kvae_dynamics metadata; missing
    # keys fall back to the adapter's built-in defaults (a=16, z=8, K=3,
    # dynamics='lstm'). HP grid below only varies on lr/batch/patience;
    # architectural sweeps are scheduled out-of-band.
    "kalman_vae": [
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
    # Conditional Sig-WGAN: signature-Wasserstein-1 loss.
    # Uses lr=1e-2 (vendor default for SigCWGAN), batch_size=64.
    "cond_sig_wgan": [
        HPConfig("vendor_default", True, 1e-2, 64, 10),
        HPConfig("lr5e-3_bs64_p10", False, 5e-3, 64, 10),
        HPConfig("lr2e-2_bs64_p10", False, 2e-2, 64, 10),
        HPConfig("lr1e-2_bs32_p10", False, 1e-2, 32, 10),
        HPConfig("lr1e-2_bs128_p10", False, 1e-2, 128, 10),
        HPConfig("lr1e-2_bs64_p8", False, 1e-2, 64, 8),
        HPConfig("lr1e-2_bs64_p12", False, 1e-2, 64, 12),
        HPConfig("lr5e-3_bs32_p8", False, 5e-3, 32, 8),
        HPConfig("lr2e-2_bs128_p12", False, 2e-2, 128, 12),
    ],
    # TimeGrad: vendor default lr=1e-3, batch_size=32, num_cells=40, num_layers=2,
    # cell_type="LSTM" (verified against vendored TimeGradEstimator.__init__).
    # Grid below sweeps lr + batch + patience (matches the kalman_vae / vrnn
    # pattern of 9 configs with vendor + 8 axis sweeps). Architectural
    # sweeps (num_cells, diff_steps, lags_seq) are scheduled out-of-band; the
    # adapter reads metadata overrides if needed (timegrad_num_cells,
    # timegrad_diff_steps, timegrad_lags_seq). Adapter sets scaling=False
    # since StonkBench preprocessed data is already z-scored — vendor's
    # MeanScaler is intentionally bypassed.
    "timegrad": [
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
}


def configs_for_model(model_key: str) -> List[HPConfig]:
    if model_key not in MODEL_HP_CONFIGS:
        raise KeyError(f"No HP configs for model '{model_key}'")
    configs = MODEL_HP_CONFIGS[model_key]
    if len(configs) != 9:
        raise ValueError(f"Expected 9 configs (vendor + 8 variants) for {model_key}, got {len(configs)}")
    return configs


def full_train_metadata(model_key: str, hp_summary_entry: Dict) -> Dict[str, float | int | bool]:
    """Compose the metadata dict the final-train stage passes to the adapter.

    For most DL models this is just the HP-validated ``config_id``,
    ``max_epochs``, ``learning_rate``, ``batch_size``, ``patience``. For
    """
    best = hp_summary_entry["best_config"]
    patience = min(int(best["patience"]), FULL_TRAIN_PATIENCE_CAP)
    metadata: Dict[str, Any] = {
        "config_id": best.get("config_id", "hp_winner"),
        "max_epochs": FULL_TRAIN_EPOCHS[model_key],
        "learning_rate": float(best["learning_rate"]),
        "batch_size": int(best["batch_size"]),
        "patience": patience,
        "use_calibration": model_key in CALIBRATE_ON_GENERATE,
        "hp_search_val_loss": float(best["mean_best_val_loss"]),
    }
    if model_key == "cond_sig_wgan":
        # Conditional Sig-WGAN uses total_steps (not epochs) for training
        # duration. Reverted to the proven-stable architecture after the
        # 2026-07-22 v2_depth3 run diverged (NaN-filled weights by end of
        # training; depth=3 + hidden=(100,100,100) was too wide/deep for our
        # 25-channel data without spectral norm / gradient clipping). The
        # original first_run config (depth=2, hidden=(50,50,50), 1500 steps)
        # produced a finite .pt; extending to 5000 steps gives more budget
        # for representational capacity without the divergence risk of the
        # wider/deeper net. p=20 (vs original 10) gives longer-range
        # conditioning context for the 252-step AR rollout.
        #
        # Post-rollout fixes (default-on, can be disabled per-experiment via
        # metadata flags). The revert_2026-07-23 run shows severe time-axis
        # variance decay (Q1 std=0.013 → Q4 std=0.003, 99.2% of samples have
        # LATE std < 50% of EARLY std) AND uniform per-channel under-
        # dispersion (std_ratio median ≈ 0.35). Together:
        #   - cond_sig_wgan_time_flatten: per-step std rescaling to t=0's
        #     std; kills the AR roll-out decay at the symptom level (post-hoc).
        #   - cond_sig_wgan_per_step_clamp: final bound on per-step values
        #     at ±clamp_val (5.0 = safe z-scored log return tail); catches
        #     any explosive ratio from the flatten transform.
        #   - model added to CALIBRATE_ON_GENERATE so the adapter's
        #     match_channel_moments() lifts per-channel std to train stats
        #     after the flatten.
        metadata["cond_sig_wgan_steps"] = 1000  # vendor default (train.py: total_steps=1000)
        metadata["cond_sig_wgan_p"] = 20
        metadata["cond_sig_wgan_hidden"] = "50,50,50"
        metadata["cond_sig_wgan_mc_size"] = 500  # vendor STOCKS default (was 100, too small)
        metadata["cond_sig_wgan_sig_depth"] = 2
        metadata["cond_sig_wgan_stride"] = 5
        metadata["cond_sig_wgan_time_flatten"] = True
        metadata["cond_sig_wgan_per_step_clamp"] = True
        metadata["cond_sig_wgan_clamp_val"] = 5.0
    return metadata
