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
    "cond_sig_wgan": "src.experiments.adapters.deep_learning.cond_sig_wgan_adapter.ConditionalSigWGANAdapter",
    "timegrad": "src.experiments.adapters.deep_learning.timegrad_adapter.TimeGradAdapter",
}


# Multi-pronged recovery variants. Each variant reuses the BASE model's adapter
# class but overrides a different smoking-gun knob via metadata.extras. See
# src/experiments/hp_configs.py for per-variant HPConfig entries.
# Naming convention: "{base}_{change}"  → .pt file becomes
# {variant}_seq{L}_final.pt and the result dir is results/{run}/{variant}/.
VARIANT_TO_BASE: Dict[str, str] = {
    # quantgan: WGAN-CP clip is too tight on 25-dim data
    "quantgan_clipfix":        "quantgan",
    "quantgan_nstep":          "quantgan",
    "quantgan_minEPO":         "quantgan",
    # vrnn: vendor n_epochs=25 is too short; kl_warmup=10 too steep
    "vrnn_epochs50":           "vrnn",
    "vrnn_klsched":            "vrnn",
    # vrnn_klsched_v2 (kl_dim_scale=0.25): SUPERSEDED 2026-07-30 by vrnn_klsched
    # (kl_dim_scale=0.5). v2 produced UNDER-DISPERSED mean std_ratio=0.640 vs
    # v1's 0.783, so the more-aggressive latent pressure under-saturated the
    # prior instead of fighting collapse. Kept in VARIANT_TO_BASE so existing
    # .pt artifacts under results/latest/vrnn_klsched_v2/ still resolve, but
    # NOT dispatched by any final_training.sh STONKBENCH_FIX_VARIANTS case.
    "vrnn_klsched_v2":         "vrnn",
    # Wave 3 (2026-07-30): GAN architecture-fix variants map to base GAN.
    "quantgan_wgangp":              "quantgan",
    "quantgan_per_asset":           "quantgan",
    "quantgan_hidden512":           "quantgan",
    "quantgan_wgangp_h512":         "quantgan",
    "quantgan_dstep20":             "quantgan",
    "cond_sig_wgan_mc500_d2":       "cond_sig_wgan",
    "cond_sig_wgan_mc500_d2_h100":  "cond_sig_wgan",
    "cond_sig_wgan_p5_q50":         "cond_sig_wgan",
    "cond_sig_wgan_mc300_d2":       "cond_sig_wgan",
    "cond_sig_wgan_mc500_d2_p10_h50x3": "cond_sig_wgan",
    "vrnn_hdim":               "vrnn",
    # kalman_vae_safe: retry with halved epoch cap to dodge late-stage
    # MultivariateNormal.loc divergence (2026-07-30 cleanup).
    "kalman_vae_safe":         "kalman_vae",
    # kalman_vae: pre-emptive
    "kalman_vae_K5":           "kalman_vae",
    "kalman_vae_adim32":       "kalman_vae",
    "kalman_vae_ep200":        "kalman_vae",
    # utsd/cond_tsd: grad_clip=0.5 too tight; min_epochs too short
    "utsd_gclip":              "unconditional_tsdiffusion",
    "utsd_minEPO":             "unconditional_tsdiffusion",
    "utsd_sched":              "unconditional_tsdiffusion",
    "cond_tsd_gclip":          "conditional_tsdiffusion",
    "cond_tsd_ep200":          "conditional_tsdiffusion",
    # cond_sig_wgan: depth=2 sig too compressed
    "cond_sig_wgan_mc500":     "cond_sig_wgan",
    "cond_sig_wgan_depth3":    "cond_sig_wgan",
    "cond_sig_wgan_steps3k":   "cond_sig_wgan",
    # ----- Wave 2: GAN collapse recovery (2026-07-30) -----
    # quantgan: relax WGAN-CP clamp + rebalance D/G ratio
    "quantgan_clip05_d3":      "quantgan",
    "quantgan_clip10_n10":     "quantgan",
    # cond_sig_wgan: more capacity+steps or more data+shorter context
    "cond_sig_wgan_st3k_h100": "cond_sig_wgan",
    "cond_sig_wgan_p10_s2":    "cond_sig_wgan",
    # timegrad: num_cells=40 too small for 25 channels
    "timegrad_cells80":        "timegrad",
    "timegrad_lr5e4":          "timegrad",
    "timegrad_ep200":          "timegrad",
    "timegrad_minEPO":         "timegrad",
    # ----- Wave 5 (2026-07-30): TimeGrad tail-shape recovery -----
    # Vendor's reverse-time Gaussian kernel can't sample fat tails (KS_p=0
    # on 25 channels). Variant swaps noise_like -> StudentT(df=5) via
    # the OPTION A monkey-patch in timegrad_adapter.generate() (tightly
    # scoped by try/finally so vendor's kernel is restored on every
    # generate() exit, including exceptions). Reuses TimeGradAdapter;
    # the only knob change is the metadata-flagged noise distribution +
    # degrees-of-freedom, persisted in the checkpoint.
    "timegrad_t_noise":        "timegrad",
    # ----- Wave 6 (2026-07-30): TimeGrad df=10 (lighter Student-t) -----
    # df=5 OVERSHOT (mean |kurt diff| = 449.9, std_ratio = 1.377 vs
    # healthy 0.85-1.15). Unbounded Conv1d output in epsilon_theta +
    # 100-step diffusion chain + heavy-tail Student-t df=5 amplified
    # rare 4σ events chaotically. df=10 is the next probe — lighter
    # tails than df=5 but heavier than vendor Gaussian. Same adapter +
    # same cached-sampler ON-DEVICE monkey-patch; only the df metadata
    # knobs change (10.0 vs 5.0).
    "timegrad_t_noise_df10":   "timegrad",
    # ----- Wave 7 (2026-07-30): GAN architecture-first recovery -----
    # Six variants — 2 per GAN — paired with the architecture patches in
    # the adapter layer:
    #   - QuantGAN: per-asset Wave-4 over-shoots mean_std_ratio 1.478.
    #     `quantgan_tanhbound` soft-clamps per-asset Generator forward()
    #     via tanh/5.0. `quantgan_wgangp_tanh` adds WGAN-GP on top.
    #   - PCF-GAN: K=5 PCA Wave-3 hits mean_std_ratio 0.452 (HS critic over-
    #   - Cond-Sig-WGAN: Wave-3 trained at q=252 with depth=2 -> mean_std_
    #     ratio 0.030 (SEVERE). `cond_sig_wgan_qtrain5` decouples train q
    #     to 5; `cond_sig_wgan_qtrain5_h100` widens hidden to (100,100,100).
    "quantgan_tanhbound":         "quantgan",
    "quantgan_wgangp_tanh":       "quantgan",
    "cond_sig_wgan_qtrain5":      "cond_sig_wgan",
    "cond_sig_wgan_qtrain5_h100": "cond_sig_wgan",
    # ----- SigWGAN variance recovery (2026-08-01): 6 directions -----
    "csigwgan_ols_s1":           "cond_sig_wgan",
    "csigwgan_noscale":          "cond_sig_wgan",
    "csigwgan_ridge001":         "cond_sig_wgan",
    "csigwgan_mc500_lr2":        "cond_sig_wgan",
    "csigwgan_q50":              "cond_sig_wgan",
    "csigwgan_ols_s1_ns":        "cond_sig_wgan",
    # ----- SigWGAN loss/architecture recovery round 2 (2026-08-03) -----
    # Round-1 calibration knobs plateaued; round 2 acts on the generator
    # input (noise_std) / loss (var_reg). See hp_configs.py comments.
    "csigwgan_noise5":           "cond_sig_wgan",
    "csigwgan_noise10":          "cond_sig_wgan",
    "csigwgan_varreg1":          "cond_sig_wgan",
    "csigwgan_noise5_varreg1":   "cond_sig_wgan",
    "csigwgan_q50_noise5":       "cond_sig_wgan",
    "csigwgan_noise5_lr3":       "cond_sig_wgan",
}


def _load_adapter_class(dotted_path: str) -> Type[ModelAdapter]:
    module_name, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


# Variants that have been retired from the SLURM dispatch list (entries with
# benchmark results that did not improve on their predecessor). The keys MUST
# remain in VARIANT_TO_BASE so existing on-disk .pt artifacts under
# results/latest/<variant>/artifacts/ continue to resolve via
# get_adapter(); consumers can consult this map to skip them in eval sweeps.
#
# Replacement policy: a deprecated variant name points to its single canonical
# successor. Future code (evaluation, plotting, summary aggregation) should
# filter deprecated variants out unless explicitly requested for ablation.
SUPERSEDED_VARIANTS: Dict[str, str] = {
    # vrnn_klsched_v2: experimentally worse than v1 (mean std_ratio 0.640 vs
    # 0.783 on a denormalized ground-truth scale). Use vrnn_klsched instead;
    # see gan_tests/ROADMAP.md for the ablation logic.
    "vrnn_klsched_v2": "vrnn_klsched",
    # ----- Wave-7 winner lock-in (2026-07-30) -----
    # Cond-Sig-WGAN: `cond_sig_wgan_qtrain5_h100` is the canonical variant
    # after today's per-channel diagnostic (mean std_ratio 0.512, 3/25 in-band
    # vs baseline mc500_d2's 0.030 SEVERE collapse). The qtrain5 decoupling
    # (train at q=5 instead of q=252) restores W-1 gradient flow; the
    # hidden=(100,100,100) widening gives the remaining representational
    # capacity. All older cond_sig_wgan_* variants are retired in favour of
    # this one. See gan_tests/ROADMAP.md.
    "cond_sig_wgan_mc500_d2": "cond_sig_wgan_qtrain5_h100",
    "cond_sig_wgan_qtrain5":  "cond_sig_wgan_qtrain5_h100",
    # QuantGAN baseline (`quantgan`) is canonical per the user (2026-07-30):
    # "the original quantgan was excellent — keep it". Wave-7 tanh-clamp
    # variants (quantgan_tanhbound, quantgan_wgangp_tanh) remain available
    # for ablation but are NOT promoted.
}


def get_adapter(model_key: str) -> ModelAdapter:
    key = model_key.lower()
    # Variant keys fall back to their BASE class via VARIANT_TO_BASE. The
    # variant-specific smoking-gun changes live in MODEL_HP_CONFIGS (extras),
    # not in the adapter class itself, so each variant transparently re-uses
    # the base's fit() / generate() with the overridden metadata.
    if key not in ADAPTER_REGISTRY and key in VARIANT_TO_BASE:
        key = VARIANT_TO_BASE[key]
    if key not in ADAPTER_REGISTRY:
        supported = ", ".join(sorted(ADAPTER_REGISTRY.keys()))
        raise KeyError(f"Unknown model key '{model_key}'. Supported: {supported}")
    adapter_cls = _load_adapter_class(ADAPTER_REGISTRY[key])
    return adapter_cls()
