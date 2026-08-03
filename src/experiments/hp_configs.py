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
# kalman_vae: no explicit vendor epoch count; 60/100 kept
# unconditional/conditional_tsd: vendor max_epochs=100 (train_tsdiff*.yaml)
# cond_sig_wgan: vendor total_steps=1000 (train.py), not epoch-based
# timegrad: vendor epochs=100 (trainer.py)
HP_SEARCH_EPOCHS: Dict[str, int] = {
    "quantgan": 80,
    "vrnn": 25,
    "kalman_vae": 60,
    "unconditional_tsdiffusion": 100,
    "conditional_tsdiffusion": 100,
    "cond_sig_wgan": 100,
    "timegrad": 100,
}

FULL_TRAIN_EPOCHS: Dict[str, int] = {
    "quantgan": 200,
    # VRNN: Wave 1 diagnostics proved vendor's n_epochs=25 is insufficient
    # (collapsed with std_ratio=0.01). 50 epochs gives the model room to
    # escape variational collapse with kl_warmup=10 (was ATROPHIED not
    # COLLAPSED at std_ratio=0.56, KS=0.094).
    "vrnn": 50,
    "kalman_vae": 100,
    # kalman_vae_safe: halved epoch cap (50 vs 100) to skip the late-stage
    # MultivariateNormal.loc divergence that bit the prior 1h53m run.
    # Adapter hardcoded clip_grad_norm=10.0 still applies.
    "kalman_vae_safe": 50,
    "unconditional_tsdiffusion": 100,
    "conditional_tsdiffusion": 100,
    "cond_sig_wgan": 100,
    "timegrad": 100,
    # ----- Wave 1 fix variants: full-budget override on max_epochs -----
    # vrnn_epochs50 explicitly doubles vendor's 25 → 50 to test the “escape
    # plateau” hypothesis. Other variants inherit the base model's epoch cap.
    "quantgan_clipfix": 200,
    "quantgan_nstep": 200,
    "vrnn_epochs50": 50,
    # ----- Wave 5 (2026-07-30): timegrad noise-distribution swap -----
    # Replaces vendor's Gaussian reverse-time kernel with Student-t df=5
    # via the OPTION A patch in timegrad_adapter:generate() (monkey-patch
    # on vendor's `module.noise_like` tightly scoped via try/finally).
    # Same epoch budget as base timegrad (100); the noise change, not the
    # step count, is the recovery axis.
    "timegrad_t_noise": 100,
    # Wave 6 (2026-07-30): same Student-t axis, df 5 -> 10 — pull variance
    # and kurt down toward reference after df=5 OVERSHOT (mean |kurt diff|
    # = 449.9, std_ratio = 1.377 vs healthy 0.85-1.15). Diagnosis from
    # GENERATED artifact (job 694886): the unbounded Conv1d output in
    # epsilon_theta + 100-step diffusion chain + Student-t df=5's wide
    # tails compound chaotically. df=10 is lighter-than-df=5 but heavier
    # than vendor Gaussian — the sweet spot is expected between 5 and 15.
    "timegrad_t_noise_df10": 100,
    # vrnn_klsched: same epochs as base but slower KL ramp (warmup=20
    # vs vendor max(10, epochs/5)=10), drives more expressive z-latents
    # and pulls std_ratio above the atrophied baseline for 25-channel data.
    "vrnn_klsched": 50,
    # vrnn_klsched_v2: DEPRECATED 2026-07-30 — superseded by vrnn_klsched.
    # Mean std_ratio 0.640 (under-dispersed) vs v1's 0.783 on the denormalized
    # ground-truth scale; the more-aggressive kl_dim_scale=0.25 under-saturated
    # the prior instead of pushing compressed channels above the healthy
    # threshold. Entry kept here so existing .pt artifacts under
    # results/latest/vrnn_klsched_v2/ continue to resolve via configs_for_model.
    "vrnn_klsched_v2": 50,
    # ----- Wave 3: Architecture-fix GAN variants (2026-07-30) -----
    # QuantGAN: 5 directions to escape variance-starvation collapse (ch1,2,7,8,10,11,22 compressed).
    "quantgan_wgangp": 200,        # F1: WGAN-CP clip -> WGAN-GP
    "quantgan_per_asset": 100,     # F2: drop joint-TCN, train 25 parallel univariate TCNs
    "quantgan_hidden512": 200,     # F3: TCN hidden 80 -> 512 for 25-channel joint regression
    "quantgan_wgangp_h512": 200,   # F1+F3 composite
    "quantgan_dstep20": 150,       # F4: critic 20 steps/gen (last-ditch HP)
    # PCF-GAN: 6 directions to escape HS-norm pathology (kurt=153, skew=-9.16).
    # Cond-Sig-WGAN: 5 directions to escape high-d signature OOM + MC noise.
    "cond_sig_wgan_mc500_d2": 150, # F1: mc 500 + depth 2
    "cond_sig_wgan_mc500_d2_h100": 150, # F1+F3: wider hidden
    "cond_sig_wgan_p5_q50": 150,   # F2: short past p=5, wider hidden
    "cond_sig_wgan_mc300_d2": 150, # F1 mid-range
    "cond_sig_wgan_mc500_d2_p10_h50x3": 150, # F1 paper-faithful recipe
    "timegrad_cells80": 100,
    "timegrad_lr5e4": 100,       # Wave 1 — same epoch cap; just lowers lr to 5e-4
    "utsd_gclip": 100,
    "cond_tsd_gclip": 100,        # Wave 1 — re-uses utsd grad_clip hook via inheritance
    # ----- Wave 7 (2026-07-30): GAN architecture-first recovery -----
    # Each variant pairs the BASE model's prior collapsed verdict with one
    # targeted architecture fix from the gan_tests/ROADMAP.md proposal set.
    # Editor: Buffy, after per-channel baseline diagnostic on 2026-07-30
    # cond_sig_wgan SEVERE (0.03). HP-only waves were exhausted.
    "quantgan_tanhbound":       200,  # Wave-7: soft-tanh clamp on TCN output (cap variance)
    "quantgan_wgangp_tanh":     200,  # Wave-7: WGAN-GP loss + soft-tanh clamp (composite)
    "cond_sig_wgan_qtrain5":    150,  # Wave-7: decouple train q from gen q (train at q=5)
    "cond_sig_wgan_qtrain5_h100": 150,  # Wave-7: qtrain=5 + wider hidden (100,100,100)
    # ----- Wave 2: GAN collapse recovery variants (2026-07-30) -----
    # quantgan: 3 directions to escape variance-starvation collapse
    "quantgan_clip05_d3": 200,
    "quantgan_clip10_n10": 200,
    # cond_sig_wgan: 2 directions to improve under-trained generator
    "cond_sig_wgan_st3k_h100": 100,
    "cond_sig_wgan_p10_s2": 100,
    # ----- SigWGAN variance recovery (2026-08-01): 6 directions -----
    "csigwgan_ols_s1": 100,
    "csigwgan_noscale": 100,
    "csigwgan_ridge001": 100,
    "csigwgan_mc500_lr2": 100,
    "csigwgan_q50": 100,
    "csigwgan_ols_s1_ns": 100,
    # ----- SigWGAN loss/architecture recovery round 2 (2026-08-03) -----
    # Round 1 (calibration-side) plateaued at std_ratio ~0.21-0.25: the
    # expectation-matching loss lets the AR-FNN ignore its latent noise over
    # long horizons. These variants act on the generator input / loss, not
    # the calibration target. Same epoch budget as base cond_sig_wgan (100).
    "csigwgan_noise5":         100,  # D1: amplify latent noise to σ_z=5
    "csigwgan_noise10":        100,  # D2: σ_z=10 (stronger probe)
    "csigwgan_varreg1":        100,  # D3: variance-matching loss λ=1.0
    "csigwgan_noise5_varreg1": 100,  # D4: composite σ_z=5 + λ=1.0
    "csigwgan_q50_noise5":     100,  # D5: train q=50 + σ_z=5
    "csigwgan_noise5_lr3":     100,  # D6: σ_z=5 + lr=3e-3
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
    "kalman_vae": {
        # Architectural knobs at vendor's documented values; hp_search only
        # varies lr/bs/patience. Read by kalman_vae_adapter via
        # meta.get("kvae_K", 3) etc.
        "kvae_K": 3,
        "kvae_dynamics": "lstm",
        "kvae_a_dim": 16,
        "kvae_z_dim": 8,
    },
    # unconditional_tsdiffusion: Wave 1 proved grad_clip=2.0 (vs vendor 0.5)
    # fixes the variance-collapse (HEALTHY: std_ratio=0.82, KS=0.05). The
    # adapter reads `grad_clip` from fit metadata (utsd_adapter.py line ~90).
    "unconditional_tsdiffusion": {
        "grad_clip": 2.0,
    },
    "conditional_tsdiffusion": {
        # Inherits grad_clip=2.0 from parent utsd adapter (HEALTHY:
        # std_ratio=0.78, KS=0.046 in Wave 1).
        "grad_clip": 2.0,
    },
    # timegrad: Wave 1 proved num_cells=80 (vs vendor 40) reduces variance
    # atrophy (ATROPHIED: std_ratio=0.67 vs 0.01 collapsed). The adapter
    # reads `timegrad_num_cells` from fit metadata (timegrad_adapter.py).
    "timegrad": {
        "timegrad_num_cells": 80,
    },
    "cond_sig_wgan": {
        # Full-horizon training (q = generation_length - p ≈ 242). The
        # qtrain=5 recipe (Wave-7) collapsed at 252 steps because the AR-FNN
        # trained on 5-step windows then rolled out 50× autoregressively —
        # error compounds and the generator converges to near-zero noise.
        # This restores the original "good enough" setup: full 242-step
        # horizon training + Ridge calibration (α=1.0, vendor default for
        # high-dim signature features).
        "cond_sig_wgan_steps": 1500,
        "cond_sig_wgan_p": 10,
        "cond_sig_wgan_hidden": "100,100,100",
        "cond_sig_wgan_mc_size": 100,
        "cond_sig_wgan_sig_depth": 2,
        "cond_sig_wgan_stride": 5,
        "cond_sig_wgan_calibration_alpha": 1.0,
        # SigWGAN variance recovery round (2026-08-01): 6 calibration-side
        # variants ALL plateaued at std_ratio ~0.21-0.25 (expectation-matching
        # loss is the bottleneck, not calibration). Best variant was
        # csigwgan_noscale (scale=1.0, std_ratio 0.246). Baked into the base
        # config so bare-name cond_sig_wgan reproduces it. The Scale(0.5)
        # augmentation halved returns -> signatures ~4x smaller; removing it
        # restores the raw target scale.
        "cond_sig_wgan_scale": 1.0,
        # SigWGAN recovery round 2 (2026-08-03): latent-noise amplification.
        # Round-1 calibration knobs plateaued at 0.21-0.25 because the
        # expectation-matching loss averages the ArFNN's latent noise away
        # over the 242-step horizon, so the generator learns to ignore z.
        # Scaling the latent z by sigma_z=10 BEFORE it enters the network
        # revives the variance signal (the variance-correction terms in
        # E_z[sig(G)] scale ~sigma_z^2): csigwgan_noise10 hit mean std_ratio
        # 0.97 (seq21/42/126/252 = 0.98/0.95/0.96/0.97) with NO rollout decay
        # (per-step std quartiles 0.0183-0.0188, q4/q1 = 1.03). Baked into
        # the base config so a bare-name cond_sig_wgan retrain reproduces it.
        # The adapter monkey-patches SimpleGenerator.sample to scale z and
        # persists noise_std as a state_dict buffer (verified round-trip).
        "cond_sig_wgan_noise_std": 10.0,
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
    # Kalman-VAE: Vendor defaults. K=3, a_dim=16, z_dim=8, dynamics=lstm
    # pinned in MODEL_FIXED_HP. 100 epochs is enough for the 1970-window
    # dataset (vendor defaults don't specify).
    "kalman_vae":           [HPConfig("vendor_best", True, 1e-3, 32, 12)],
    # Unconditional TSDiffusion: Vendor defaults. 100 epochs × 128 batches
    # per epoch = 12,800 gradient steps (matches vendor's training budget).
    "unconditional_tsdiffusion":   [HPConfig("vendor_best", True, 1e-3, 64, 6)],
    # Conditional TSDiffusion: Same as uncond + noise_observed=False.
    "conditional_tsdiffusion":     [HPConfig("vendor_best", True, 1e-3, 64, 6)],
    # Cond-Sig-WGAN: accepted qtrain5_h100 recipe; bare name reproduces the
    # previously accepted output under results/latest/cond_sig_wgan/.
    "cond_sig_wgan":        [HPConfig("vendor_best", True, 1e-3, 64, 6)],
    # TimeGrad: Vendor defaults. num_cells=40, num_layers=2, diff_steps=100
    # pinned in adapter class body. 100 epochs × 50 batches = 5,000 steps.
    "timegrad":             [HPConfig("vendor_best", True, 1e-3, 32, 12)],

    # ----- Wave 1 fix variants (multi-pronged recovery; 2026-07-29) -----
    # Each variant reuses the base model's adapter; only the smoking-gun
    # knob changes via metadata.extras. Adapter & registry dispatch via
    # VARIANT_TO_BASE so the .pt filename carries the variant suffix.
    "quantgan_clipfix":   [HPConfig("vendor_best_clipfix", True, 2e-4, 30, 20,
                                     extras={"clip_value": 0.05})],
    "vrnn_epochs50":      [HPConfig("vendor_best_epochs50", True, 1e-3, 32, 12)],
    # Atrophy-fix variant: only knob changed is KL annealing ramp length.
    # vrnn_adapter.py line 89 reads `vrnn_kl_warmup` from metadata to
    # override the vendor default max(10, max_epochs // 5) = 10.
    "vrnn_klsched":       [HPConfig("vendor_best_klsched", True, 1e-3, 32, 12,
                                     extras={"vrnn_kl_warmup": 20,
                                             "vrnn_kl_dim_scale": 0.5})],
    # Wave 5 (2026-07-30): TimeGrad tail-shape recovery. Replaces the
    # vendor's Gaussian reverse-time kernel with Student-t (df=5) via the
    # OPTION A monkey-patch on module.noise_like in
    # timegrad_adapter.generate(). Adapter reads `timegrad_noise_dist` +
    # `timegrad_noise_df` from metadata and persists them in the .pt
    # checkpoint so generate() can re-install the patch on load.
    "timegrad_t_noise":   [HPConfig("vendor_best_t_noise", True, 1e-3, 32, 12,
                                     extras={"timegrad_noise_dist": "student_t",
                                             "timegrad_noise_df": 5.0})],
    # Wave 6 (2026-07-30): df=5 OVERSHOT (cf. Wave 5 diagnostic for
    # timegrad_t_noise: mean |kurt diff| = 449.9, std_ratio = 1.377 vs
    # healthy 0.85-1.15). Hypothesis: lighter Student-t (df=10) brings
    # tail mass closer to ref without the chaotic amplification in the
    # 100-step diffusion chain. Same architecture + same cached-sampler
    # OPTION A patch; only the df changes.
    "timegrad_t_noise_df10":   [HPConfig("vendor_best_t_noise_df10", True, 1e-3, 32, 12,
                                     extras={"timegrad_noise_dist": "student_t",
                                             "timegrad_noise_df": 10.0})],
    # vrnn_klsched_v2 (kl_dim_scale=0.25): DEPRECATED 2026-07-30; superseded
    # by vrnn_klsched. Under-dispersed worse than v1; see
    # src.experiments.core.registry.SUPERSEDED_VARIANTS.
    # KEPT IN MODEL_HP_CONFIGS / FULL_TRAIN_EPOCHS so existing on-disk .pt
    # artifacts under results/latest/vrnn_klsched_v2/ continue to resolve via
    # configs_for_model(...) for historical eval. NOT retrained (no longer in
    # any STONKBENCH_FIX_VARIANTS dispatch).
    "vrnn_klsched_v2":    [HPConfig("vendor_best_klsched_v2", True, 1e-3, 32, 12,
                                     extras={"vrnn_kl_warmup": 20,
                                             "vrnn_kl_dim_scale": 0.25})],
    # ----- Wave 3 (2026-07-30): GAN architecture-fix variants -----
    # QuantGAN (Wiese 2019 base; vendor_quantgan_module.Generator/Discriminator TCN)
    "quantgan_wgangp":     [HPConfig("vendor_best_wgangp",    True, 2e-4, 32, 12,
                                     extras={"clip_value": 0.0,
                                             "gan_loss": "wgangp",
                                             "clip_violations_logged": True})],
    "quantgan_per_asset":  [HPConfig("vendor_best_per_asset", True, 2e-4, 32, 12,
                                     extras={"channel_strategy": "per_asset",
                                             "per_asset_hidden": 16,
                                             "per_asset_n_channels": 1})],
    "quantgan_hidden512":  [HPConfig("vendor_best_hidden512", True, 2e-4, 16, 12,
                                     extras={"tcn_hidden": 512,
                                             "clip_value": 0.01})],
    "quantgan_wgangp_h512":[HPConfig("vendor_best_wgangp_h512",True, 2e-4, 16, 12,
                                     extras={"gan_loss": "wgangp",
                                             "clip_value": 0.0,
                                             "tcn_hidden": 512})],
    "quantgan_dstep20":    [HPConfig("vendor_best_dstep20",   True, 2e-4, 32, 12,
                                     extras={"d_steps_per_g_step": 20,
                                             "clip_value": 0.01})],
    # PCF-GAN (Becker 2021 base; vendor pcfgan_src.LSTMGenerator + char_func_path HS critic)
    # Conditional Sig-WGAN (Ni 2020 base; vendor lib.algos.sigcwgan.SigCWGAN)
    "cond_sig_wgan_mc500_d2":           [HPConfig("vendor_best_mc500_d2",           True, 1e-3, 64, 6,
                                                  extras={"cond_sig_wgan_mc_size": 500,
                                                          "cond_sig_wgan_sig_depth": 2})],
    "cond_sig_wgan_mc500_d2_h100":      [HPConfig("vendor_best_mc500_d2_h100",      True, 1e-3, 64, 6,
                                                  extras={"cond_sig_wgan_mc_size": 500,
                                                          "cond_sig_wgan_sig_depth": 2,
                                                          "cond_sig_wgan_hidden": "100,100,100,100"})],
    "cond_sig_wgan_p5_q50":             [HPConfig("vendor_best_p5_q50",             True, 1e-3, 64, 6,
                                                  extras={"cond_sig_wgan_p": 5,
                                                          "cond_sig_wgan_hidden": "100,100,100,100"})],
    "cond_sig_wgan_mc300_d2":           [HPConfig("vendor_best_mc300_d2",           True, 1e-3, 64, 6,
                                                  extras={"cond_sig_wgan_mc_size": 300,
                                                          "cond_sig_wgan_sig_depth": 2})],
    "cond_sig_wgan_mc500_d2_p10_h50x3": [HPConfig("vendor_best_mc500_d2_p10_h50x3", True, 1e-3, 64, 6,
                                                  extras={"cond_sig_wgan_mc_size": 500,
                                                          "cond_sig_wgan_sig_depth": 2,
                                                          "cond_sig_wgan_p": 10,
                                                          "cond_sig_wgan_hidden": "50,50,50"})],
    # fields inherit vendor defaults — adapter uses hardcoded fallback
    # (kvae_K=3, a_dim=16, z_dim=8, dynamics=lstm) when metadata isn't
    # carrying kvae_* keys.
    "kalman_vae_safe":    [HPConfig("vendor_best_safe", True, 1e-3, 32, 6)],
    "timegrad_cells80":   [HPConfig("vendor_best_cells80", True, 1e-3, 32, 12,
                                     extras={"timegrad_num_cells": 80})],
    "utsd_gclip":         [HPConfig("vendor_best_gclip", True, 1e-3, 64, 6,
                                     extras={"grad_clip": 2.0})],
    # Second direction per collapsed model (Wave 1 expansion):
    "quantgan_nstep":     [HPConfig("vendor_best_nstep", True, 2e-4, 30, 20,
                                     extras={"d_steps_per_g_step": 10})],
    "timegrad_lr5e4":     [HPConfig("vendor_best_lr5e4", True, 5e-4, 32, 12)],
    "cond_tsd_gclip":     [HPConfig("vendor_best_cond_tsd_gclip", True, 1e-3, 64, 6,
                                     extras={"grad_clip": 2.0})],

    # ----- Wave 2: GAN collapse recovery (2026-07-30) -----
    # quantgan: relax WGAN-CP weight clamp + rebalance D/G ratio
    "quantgan_clip05_d3":  [HPConfig("wave2_clip05_d3", True, 2e-4, 30, 20,
                                     extras={"clip_value": 0.05, "d_steps_per_g_step": 3})],
    "quantgan_clip10_n10": [HPConfig("wave2_clip10_n10", True, 2e-4, 30, 20,
                                     extras={"clip_value": 0.10, "noise_dim": 10})],
    # cond_sig_wgan: more capacity + more training steps; or more data + shorter context
    "cond_sig_wgan_st3k_h100": [HPConfig("wave2_st3k_h100", True, 1e-2, 64, 12,
                                     extras={"cond_sig_wgan_steps": 3000,
                                             "cond_sig_wgan_hidden": "100,100,100"})],
    "cond_sig_wgan_p10_s2":    [HPConfig("wave2_p10_s2", True, 1e-2, 64, 12,
                                     extras={"cond_sig_wgan_p": 10,
                                             "cond_sig_wgan_stride": 2})],

    # ----- SigWGAN variance recovery (2026-08-01): 6 directions -----
    # Root cause: Scale(0.5) halves signal + Ridge(α=1.0) shrinks
    # calibration target → generator converges to near-zero variance.
    # Each variant changes ONE axis, keeping the rest vendor-faithful.
    #
    # Direction 1 — OLS (α=0.0) + stride=1: ~1970 windows, well-conditioned
    # LinearRegression, no shrinkage. Most paper-faithful fix.
    "csigwgan_ols_s1":       [HPConfig("vfix_ols_s1", True, 1e-3, 64, 6,
                                   extras={"cond_sig_wgan_calibration_alpha": 0.0,
                                           "cond_sig_wgan_stride": 1})],
    # Direction 2 — No Scale augmentation: remove the 0.5× signal halving.
    # Cumsum still applied. Return magnitudes flow through at full scale.
    "csigwgan_noscale":      [HPConfig("vfix_noscale", True, 1e-3, 64, 6,
                                   extras={"cond_sig_wgan_scale": 1.0})],
    # Direction 3 — Ridge α=0.001: negligible shrinkage while keeping
    # numerical stability of Ridge vs raw OLS.
    "csigwgan_ridge001":     [HPConfig("vfix_ridge001", True, 1e-3, 64, 6,
                                   extras={"cond_sig_wgan_calibration_alpha": 0.001})],
    # Direction 4 — MC=500 + lr=1e-2: vendor-grade Monte Carlo + learning
    # rate. Better gradient estimates reduce "safe collapse" incentive.
    "csigwgan_mc500_lr2":    [HPConfig("vfix_mc500_lr2", True, 1e-2, 64, 6,
                                   extras={"cond_sig_wgan_mc_size": 500})],
    # Direction 5 — Train at q=50 intermediate horizon (not 242): avoids
    # full-horizon gradient death without extreme q=5 rollouts.
    "csigwgan_q50":          [HPConfig("vfix_q50", True, 1e-3, 64, 6,
                                   extras={"cond_sig_wgan_train_q": 50})],
    # Direction 6 — OLS + stride=1 + Scale(1.0): composite of D1+D2.
    "csigwgan_ols_s1_ns":    [HPConfig("vfix_ols_s1_ns", True, 1e-3, 64, 6,
                                   extras={"cond_sig_wgan_calibration_alpha": 0.0,
                                           "cond_sig_wgan_stride": 1,
                                           "cond_sig_wgan_scale": 1.0})],

    # ----- SigWGAN loss/architecture recovery round 2 (2026-08-03): 6 -----
    # Round 1 proved the calibration target is NOT the bottleneck (every
    # α/stride/scale knob landed on the same 0.21-0.25 plateau). The loss
    # only matches E_z[sig(G(past,z))]; over a 242-step rollout the noise's
    # variance signal is averaged away, so the generator learns to ignore z.
    # These variants act on the generator input / loss, keeping the vendor
    # ArFNN + sig-W1 loss otherwise intact:
    #   D1 — noise_std=5: latent z scaled 5× BEFORE the network. The
    #     variance-correction terms in E_z[sig] scale ~σ_z², so the noise
    #     channel gets a strong, survivable gradient signal through the
    #     long-horizon averaging. Pure input scaling — no loss change.
    "csigwgan_noise5":         [HPConfig("vfix2_noise5", True, 1e-3, 64, 6,
                                      extras={"cond_sig_wgan_noise_std": 5.0})],
    #   D2 — noise_std=10: stronger probe of the same axis.
    "csigwgan_noise10":        [HPConfig("vfix2_noise10", True, 1e-3, 64, 6,
                                      extras={"cond_sig_wgan_noise_std": 10.0})],
    #   D3 — var_reg=1.0: gated variance-matching loss term
    #     λ·MSE(log1p(std_z[G]), log1p(std_real)) per (t, channel),
    #     directly pinning generated path std to the training windows.
    #     Off by default (base model keeps the pure vendor loss).
    "csigwgan_varreg1":        [HPConfig("vfix2_varreg1", True, 1e-3, 64, 6,
                                      extras={"cond_sig_wgan_var_reg": 1.0})],
    #   D4 — composite: noise σ_z=5 + variance-matching λ=1.0.
    "csigwgan_noise5_varreg1": [HPConfig("vfix2_noise5_varreg1", True, 1e-3, 64, 6,
                                      extras={"cond_sig_wgan_noise_std": 5.0,
                                              "cond_sig_wgan_var_reg": 1.0})],
    #   D5 — train horizon q=50 + σ_z=5: q50 alone regressed (0.142); with
    #     amplified noise the short-horizon variance signal should survive
    #     AND the 50-step training should stabilize the AR rollout vs q=5.
    "csigwgan_q50_noise5":     [HPConfig("vfix2_q50_noise5", True, 1e-3, 64, 6,
                                      extras={"cond_sig_wgan_train_q": 50,
                                              "cond_sig_wgan_noise_std": 5.0})],
    #   D6 — noise σ_z=5 + lr=3e-3: modest LR bump (vendor lr=1e-2, base
    #     1e-3) gives the amplified channel time to tune its gain within
    #     the 1500-step budget without the instability that lr=1e-2 showed
    #     in round 1 (mc500_lr2 → 0.000 total collapse).
    "csigwgan_noise5_lr3":     [HPConfig("vfix2_noise5_lr3", True, 3e-3, 64, 6,
                                      extras={"cond_sig_wgan_noise_std": 5.0})],

    # ----- Wave 7 (2026-07-30): GAN architecture-first recovery (6 variants, 2 per GAN) -----
    # These variants DO NOT change the loss / data pipeline — they activate
    # explicit architecture patches in the adapter layer that close the
    # collapse mode surfaced by today's per-channel baseline diagnostic.
    #
    # QuantGAN — Wave-4 per-asset unleashes 25 independent TCNs which over-
    # shoot mean std_ratio 1.478 (vs healthy 1.0). Fix: soft tanh-clamp the
    # per-asset Generator output. Default tanh-bound = 5.0 maps ~99.5% of
    # z-scored log-return mass into ±5σ while preserving gradient flow at
    # the origin (dy/dx = 1 when raw==0). Adapter hook:
    # `quantgan_generator_bound_std=5.0` -> Generator.forward returns
    # 5.0 * tanh(self.net(x) / 5.0).
    "quantgan_tanhbound":     [HPConfig("vendor_best_tanhbound", True, 2e-4, 30, 20,
                                       extras={"quantgan_generator_bound_std": 5.0})],
    # Composite: existing WGAN-GP loss + the new bound. Two architecture fixes
    # applied simultaneously — provides a stronger alternative if WGAN-GP alone
    # (Wave-3) still couldn't un-collapse the per-asset version.
    "quantgan_wgangp_tanh":   [HPConfig("vendor_best_wgangp_tanh", True, 2e-4, 32, 12,
                                       extras={"gan_loss": "wgangp",
                                               "clip_value": 0.0,
                                               "quantgan_generator_bound_std": 5.0})],
    # because the Hilbert-Schmidt distance critic over-focuses on PC1
    # (largest variance component) and flatlines PC2..PC5. Fix: standardize
    # each latent PC axis (divide by its empirical std) BEFORE the HS critic
    # sees the data, then un-standardize BEFORE inverse-PCA in generate().
    # K=10 alternative: less compression; if K=5 was too lossy (loses 20
    # channels of structure) the K=10 version recovers more signal while
    # still enjoying the standardization fix.
    # Cond-Sig-WGAN — Wave-3 `cond_sig_wgan_mc500_d2` (mc=500, depth=2) drove
    # mean std_ratio down to 0.030 (SEVERE) because the ArFNN trained at
    # q=252 has Wasserstein-1 gradient flow die once signature exponentiation
    # hits depth-2 in 25-D for 252 steps. Fix: decouple train horizon q
    # from generation horizon — train at q=5 (paper-faithful; the vendor
    # ArFNN can roll out autoregressively to any gen_length at inference).
    "cond_sig_wgan_qtrain5":  [HPConfig("vendor_best_qtrain5", True, 1e-3, 64, 6,
                                       extras={"cond_sig_wgan_train_q": 5})],
    # Composite: shorter train horizon + wider hidden layer (100,100,100
    # instead of 50,50,50) for added representational capacity.
    "cond_sig_wgan_qtrain5_h100": [HPConfig("vendor_best_qtrain5_h100", True, 1e-3, 64, 6,
                                            extras={"cond_sig_wgan_train_q": 5,
                                                    "cond_sig_wgan_hidden": "100,100,100"})],
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

    Variant keys (e.g. ``csigwgan_ols_s1``) resolve to their base model
    (e.g. ``cond_sig_wgan``) via VARIANT_TO_BASE before looking up
    MODEL_FIXED_HP, so smoking-gun defaults are inherited correctly.
    """
    from src.experiments.core.registry import VARIANT_TO_BASE

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
    # Variants inherit MODEL_FIXED_HP from their BASE model so that
    # smoking-gun defaults (sig_depth, hidden, steps, ...) survive
    # when the variant key itself isn't in MODEL_FIXED_HP.
    base_key = VARIANT_TO_BASE.get(model_key, model_key)
    for k, v in MODEL_FIXED_HP.get(base_key, {}).items():
        metadata[k] = v
    # Per-trial HPConfig.extras (e.g. quantgan's 6-trial skew-fix grid) take
    # PRECEDENCE over MODEL_FIXED_HP for the keys they specify. This is how
    # we vary clip_value / d_steps_per_g_step / noise_dim per trial without
    # touching the loss function.
    for k, v in best.get("extras", {}).items():
        metadata[k] = v
    return metadata
