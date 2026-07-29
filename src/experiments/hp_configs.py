"""Vendor-aligned hyperparameter grids for HP search and full training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
# Canonical 7 DL models (post 2026-07-23 cleanup).  timegrad, timevae, sig_wgan
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


@dataclass(frozen=True)
class HPConfig:
    config_id: str
    is_vendor_default: bool
    learning_rate: float
    batch_size: int
    patience: int
    # Per-trial smoking-gun knob overrides (clip_value, d_steps_per_g_step,
    # noise_dim, ...). These are merged into fit metadata by full_train_metadata
    # and TAKE PRECEDENCE over MODEL_FIXED_HP, but only for the keys present in
    # this dict. Other models leave extras={} so they continue to use
    # MODEL_FIXED_HP / vendor defaults.
    extras: Dict[str, Any] = field(default_factory=dict)

    def metadata(self, max_epochs: int, model_key: str = "") -> Dict[str, float | int | bool]:
        m: Dict[str, Any] = {
            "config_id": self.config_id,
            "is_vendor_default": self.is_vendor_default,
            "max_epochs": max_epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "patience": self.patience,
        }
        m.update(self.extras)
        return m


# HP search and full-train budgets:
# quantgan: vendor min_epochs=max(20, patience*2)=20 (quantgan_module.py)
# vrnn: vendor n_epochs=25 (train.py)
# pcf_gan: no explicit vendor epoch count; 80/150 kept (adapter clamps)
# kalman_vae: no explicit vendor epoch count; 60/100 kept
# unconditional/conditional_tsd: vendor max_epochs=100 (train_tsdiff*.yaml)
# cond_sig_wgan: vendor total_steps=1000 (train.py), not epoch-based
# timegrad: vendor epochs=100 (trainer.py)
HP_SEARCH_EPOCHS: Dict[str, int] = {
    "quantgan": 80,
    "vrnn": 25,
    "pcf_gan": 80,
    "kalman_vae": 60,
    "unconditional_tsdiffusion": 100,
    "conditional_tsdiffusion": 100,
    "cond_sig_wgan": 100,
    "timegrad": 100,
}

FULL_TRAIN_EPOCHS: Dict[str, int] = {
    "quantgan": 200,
    # VRNN: vendor's published n_epochs=25 (VariationalRecurrentNeuralNetwork/train.py).
    # Earlier we doubled to 50 to "escape plateau" — that was a vendor-deviation
    # fix. Per the vendor-faithful mandate, revert to 25. If VRNN collapses on
    # 25-channel data with the vendor's natural budget, report it as a
    # legitimate vendor failure mode rather than engineering a custom KL-warmup.
    "vrnn": 25,
    "pcf_gan": 150,
    "kalman_vae": 100,
    "unconditional_tsdiffusion": 100,
    "conditional_tsdiffusion": 100,
    "cond_sig_wgan": 100,
    "timegrad": 100,
}


# ---- Per-model FIXED (non-tunable) smoking-gun hyperparameters ----
# These knobs ARE exposed in vendor's own configuration surface but are NOT
# varied in our focused HP search — we lock them at the best-known stable
# values from vendor's published defaults so the search tests only the
# relevant lr/bs/patience axis. Architecture is NOT varied here.
# Rationale for each model's smoking-gun knob documented inline.
MODEL_FIXED_HP: Dict[str, Dict[str, Any]] = {
    "quantgan": {
        # Vendor defaults (per QuantGANConfig). Earlier we widened clip=0.05
        # + d_steps=10 to "prevent critic-pooling collapse" — that was a
        # vendor-deviation anti-collapse patch, and the 2026-07-29 retrain
        # proved it causes the model to converge to a flat-line attractor
        # (best_epoch=2, std_ratio=0.0002). Per the vendor-faithful mandate
        # (2026-07-29 user directive), we revert to the vendor's stock
        # WGAN-CP defaults. If QuantGAN collapses on 25-channel data with
        # vendor defaults, that's the model's responsibility to report —
        # not ours to patch around.
        "clip_value": 0.01,
        "d_steps_per_g_step": 5,
    },
    "vrnn": {
        # Vendor's VRNN has no extra training knobs exposed beyond lr/bs/patience;
        # architectural knobs (hidden_dim, z_dim) are intentionally NOT varied.
    },
    "pcf_gan": {
        # Path-characteristic-function training is vendor-stabilized; no extra
        # knobs beyond the standard optimizer+early-stopping axis.
    },
    "kalman_vae": {
        # Architectural knobs at vendor's documented values; hp_search only
        # varies lr/bs/patience. Read by kalman_vae_adapter via
        # meta.get("kvae_K", 3) etc.
        "kvae_K": 3,
        "kvae_dynamics": "lstm",
        "kvae_a_dim": 16,
        "kvae_z_dim": 8,
    },
    # unconditional_tsdiffusion, conditional_tsdiffusion, timegrad:
    # No MODEL_FIXED_HP entries because their adapters (utsd_adapter.py,
    # cond_tsd_adapter.py, timegrad_adapter.py) currently use hardcoded
    # constants / class-body attributes and do NOT read `grad_clip`,
    # `num_batches_per_epoch`, `num_cells`, `num_layers`, `diff_steps` from
    # fit metadata. Adding fixed knobs here would be dead metadata. If we ever
    # want hp_search to vary those, the adapter must first read them.
    "cond_sig_wgan": {
        # cond_sig_wgan_steps: vendor's stock/OU setting is 5000 steps (too long
        # for 25-channel data; depth=3 + hidden=(100,100,100) earlier diverged
        # with NaN-filled weights after 4-5k steps). 1500 steps is the proven-
        # stable value from the 2026-07-23 first_run that produced a finite .pt.
        # All keys below are read by cond_sig_wgan_adapter via fit metadata.
        "cond_sig_wgan_steps": 1500,
        "cond_sig_wgan_p": 20,
        "cond_sig_wgan_hidden": "50,50,50",
        # mc_size lowered from 500 -> 100 to match the adapter's documented
        # default and to fit the 25-channel signature-calibration matrix in
        # GPU memory during smoke runs. 100 MC samples is sufficient for a
        # stable Sig-Wasserstein-1 estimate per the original paper.
        "cond_sig_wgan_mc_size": 100,
        "cond_sig_wgan_sig_depth": 2,
        "cond_sig_wgan_stride": 5,
        # POST-ROLLOUT FIXES DISABLED (2026-07-29 vendor-faithful revert).
        # Earlier defaults of time_flatten=True, per_step_clamp=True were
        # "minimal sufficient adapter-level fixes" that rescaled the
        # generator output to kill Q1→Q4 variance decay and per-channel
        # under-dispersion. Per the vendor-faithful mandate, these are
        # vendor-deviations and must be removed. If the raw vendor output
        # has these pathologies, we report them in the analysis — we do
        # not silently patch them.
    },
}


# Single BEST HP per model (no HP tuning). Each model has exactly one
# 'vendor_best' HPConfig chosen by:
#   1. Vendor's published default (only deviating when there's a SPECIFIC
#      reason documented in MODEL_FIXED_HP).
#   2. Dataset scale adjustment for our 25-channel, ~1970-window z-scored
#      log returns (much smaller than vendor's typical datasets like MNIST).
#   3. Failure-mode prevention (e.g. quantgan clip_value=0.05 to prevent
#      critic-pooling collapse; vrnn doubled epochs to escape plateau).
# Smoking-gun architectural knobs (clip_value, d_steps, kvae_K, etc.)
# stay in MODEL_FIXED_HP and survive into run_final_training unchanged.
MODEL_HP_CONFIGS: Dict[str, List[HPConfig]] = {
    # QuantGAN: single vendor_best HP per "NO HP TUNING" directive (2026-07-30).
    # Knobs are the published QuantGANConfig dataclass defaults: noise_dim=3,
    # clip_value=0.01, d_steps_per_g_step=5. The 6-trial skew-fix grid was
    # reverted because (a) the user explicitly requested no HP tuning, and
    # (b) the rounds of vendor-deviation moment-matching penalty were
    # rejected. We accept the model's vendor-faithful output as-is.
    "quantgan": [
        HPConfig("vendor_best", True, 2e-4, 30, 20,
                 extras={"noise_dim": 3, "clip_value": 0.01, "d_steps_per_g_step": 5}),
    ],
    # VRNN: Vendor LR/BS/epochs doubled to 50 because 25-epoch budget plateaued
    # at val_loss ~178K (variational collapse). 50 epochs gives the model
    # room to actually converge with kl_warmup=10.
    "vrnn":                 [HPConfig("vendor_best", True, 1e-3, 32, 12)],
    # PCF-GAN: Vendor defaults. Characteristic-function metric is stable;
    # 150 epochs gives the joint generator+critic time to converge.
    "pcf_gan":              [HPConfig("vendor_best", True, 1e-3, 64, 12)],
    # Kalman-VAE: Vendor defaults. K=3, a_dim=16, z_dim=8, dynamics=lstm
    # pinned in MODEL_FIXED_HP. 100 epochs is enough for the 1970-window
    # dataset (vendor defaults don't specify).
    "kalman_vae":           [HPConfig("vendor_best", True, 1e-3, 32, 12)],
    # Unconditional TSDiffusion: Vendor defaults. 100 epochs × 128 batches
    # per epoch = 12,800 gradient steps (matches vendor's training budget).
    "unconditional_tsdiffusion":   [HPConfig("vendor_best", True, 1e-3, 64, 6)],
    # Conditional TSDiffusion: Same as uncond + noise_observed=False.
    "conditional_tsdiffusion":     [HPConfig("vendor_best", True, 1e-3, 64, 6)],
    # Cond-Sig-WGAN: Vendor LR (high lr=1e-2 is intentional for WGAN-GP
    # variants). steps/p/hidden pinned in MODEL_FIXED_HP at proven-stable
    # values (steps=1500, depth=2, hidden=(50,50,50)).
    "cond_sig_wgan":        [HPConfig("vendor_best", True, 1e-2, 64, 12)],
    # TimeGrad: Vendor defaults. num_cells=40, num_layers=2, diff_steps=100
    # pinned in adapter class body. 100 epochs × 50 batches = 5,000 steps.
    "timegrad":             [HPConfig("vendor_best", True, 1e-3, 32, 12)],
}


def configs_for_model(model_key: str) -> List[HPConfig]:
    """Single 'vendor_best' config per model. HP tuning is decommissioned."""
    if model_key not in MODEL_HP_CONFIGS:
        raise KeyError(f"No HP configs for model '{model_key}'")
    return MODEL_HP_CONFIGS[model_key]


def full_train_metadata(model_key: str, hp_summary_entry: Dict) -> Dict[str, float | int | bool]:
    """Compose the metadata dict the final-train stage passes to the adapter.

    For most DL models this is just the HP-validated ``config_id``,
    ``max_epochs``, ``learning_rate``, ``batch_size``, ``patience``.
    Per-model smoking-gun knobs (clip_value, d_steps_per_g_step, grad_clip,
    conditioning_length, cond_sig_wgan_steps, ...) come from
    ``MODEL_FIXED_HP``.
    """
    best = hp_summary_entry["best_config"]
    patience = min(int(best["patience"]), FULL_TRAIN_PATIENCE_CAP)
    metadata: Dict[str, Any] = {
        "config_id": best.get("config_id", "hp_winner"),
        "max_epochs": FULL_TRAIN_EPOCHS[model_key],
        "learning_rate": float(best["learning_rate"]),
        "batch_size": int(best["batch_size"]),
        "patience": patience,
        "hp_search_val_loss": float(best["mean_best_val_loss"]),
    }
    # Merge per-model fixed (smoking-gun) hyperparameters — these aren't
    # varied by hp_search; we lock them at vendor-published defaults above.
    for k, v in MODEL_FIXED_HP.get(model_key, {}).items():
        metadata[k] = v
    # Per-trial HPConfig.extras (e.g. quantgan's 6-trial skew-fix grid) take
    # PRECEDENCE over MODEL_FIXED_HP for the keys they specify. This is how
    # we vary clip_value / d_steps_per_g_step / noise_dim per trial without
    # touching the loss function.
    for k, v in best.get("extras", {}).items():
        metadata[k] = v
    return metadata
