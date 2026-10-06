"""QuantGAN adapter — per-asset univariate vendor-faithful port (Wave-4 restart, 2026-07-30).

Wave-3 root-cause summary:
- The vendor code at src/models/deep_learning/quantgan_module.py was designed
  strictly for univariate log-return series (1 channel).
- Prior adapter work (MultivariateQuantGANTrainer subclass) forced the vendor
  TCN to handle 25 channels jointly. The single 1D conv had to learn 25×25
  cross-channel correlations the paper never tested; the Wasserstein critic
  under WGAN-CP weight-clip was not Lipschitz-constrained enough to express
  them.
- Wave-3 retrain (2026-07-30) tried WGAN-GP + tcn_hidden=512 architecture fixes:
    * quantgan_wgangp       : 7× improvement (mean_ratio 0.002 → 0.066), still
                              SEVERE_COLLAPSE on 11/25 channels.
    * quantgan_wgangp_h512  : overshot to 5.8× real variance (28× reference).
    * quantgan_hidden512    : overshot to 6.4× real variance (29× reference).
  None of the three produced a healthy (mean_ratio ~1.0) collapse-fix.

This Wave-4 adapter (2026-07-30, "from scratch") does NOT modify vendor code.
It runs C independent vendor-faithful univariate QuantGAN trainers, one per
channel, and stacks the outputs at generate() time. Result: a (N, L, 25)
tensor that preserves the per-asset univariate dynamics the paper validated,
while sidestepping the multivariate joint-TCN pathology entirely.

Reference: Wiese et al. 2019, "QuantGAN: Generating Continuous-Valued Stock
Prices via Temporal Convolutional Networks" — vendor module reproduces the
architecture described there.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from src.experiments.adapters.base_adapter import ModelAdapter
from src.experiments.core.contracts import AdapterFitInput, AdapterGenerateOutput
from src.experiments.adapters.deep_learning.training_utils import (
    resolve_device,
    resolve_final_checkpoint,
)

# Vendor-faithful: import only the trainer class + dataclasses. The per-asset
# loop instantiates vendor code internally via QuantGANTrainer; we never touch
# Generator / Discriminator / TCN / TemporalBlock at this layer.
from src.models.deep_learning.quantgan_module import (
    QuantGANConfig,
    QuantGANFitResult,
    QuantGANTrainer,
)


class QuantGANAdapter(ModelAdapter):
    """Per-asset univariate QuantGAN adapter.

    Trains ``num_channels`` independent vendor-faithful QuantGAN models, one
    per channel. The output of generate() is a (N, L, C) tensor where C is the
    number of input channels (25 for the stonkbench preprocessing pipeline).

    Each per-asset trainer is a fresh vendor ``QuantGANTrainer`` with its own
    TCN Generator + TCN Discriminator (n_hidden=80, 7 TemporalBlocks with
    dilations 1,2,4,8,16,32,64). No vendor code is patched.
    """

    model_name = "QuantGAN"
    can_regenerate_from_checkpoint = True

    def __init__(self) -> None:
        super().__init__()
        self.trainers: List[QuantGANTrainer] = []
        self.base_length: int = 1
        self.num_channels: int = 1
        self.checkpoints: list[Path] = []

    def fit(
        self,
        fit_input: AdapterFitInput,
        checkpoints_dir: Path,
        logs_dir: Path,
    ) -> Dict[str, Any]:
        windows = fit_input.batch.train_windows
        valid_windows = fit_input.batch.valid_windows
        if windows is None or windows.ndim != 3:
            raise ValueError(
                "QuantGANAdapter expects train_windows shaped (N, L, C)"
            )
        if valid_windows is None or valid_windows.shape[0] == 0:
            raise ValueError("QuantGANAdapter requires non-empty valid_windows.")

        from src.experiments.adapters.deep_learning.training_utils import (
            parse_training_params,
        )

        params = parse_training_params(fit_input)
        self.base_length = int(windows.shape[1])
        self.num_channels = int(windows.shape[2])

        # Smoking-gun knobs from fit.metadata (set by hp_configs.full_train_metadata
        # from MODEL_FIXED_HP + per-trial HPConfig.extras). Defaults match
        # vendor's QuantGANConfig dataclass so vendor-faithful behavior is
        # preserved when no metadata override is present.
        metadata = fit_input.metadata or {}
        clip_value = float(metadata.get("clip_value", 0.01))
        d_steps_per_g_step = int(metadata.get("d_steps_per_g_step", 5))
        noise_dim = int(metadata.get("noise_dim", 3))
        # Wave-7 (2026-07-30): soft tanh-clamp on generator output — cures
        # the Wave-4 per-asset overshoot (mean std_ratio 1.478). Only
        # `quantgan_tanhbound` and `quantgan_wgangp_tanh` pass this; the base
        # `quantgan` variant keeps output_bound=0.0 (vendor-faithful
        # unbounded) so backwards-compat with on-disk .pt artifacts is
        # preserved.
        output_bound = float(metadata.get("quantgan_generator_bound_std", 0.0))

        # Per-asset loop: instantiate ``num_channels`` independent vendor-faithful
        # QuantGANTrainer objects (each wrapping a univariate TCN with
        # n_hidden=80) and fit each on its channel slice.
        model_key_early = str(
            (fit_input.metadata or {}).get("model_key", self.model_name)
        )
        self.trainers = []
        per_asset_results: List[QuantGANFitResult] = []
        for c in range(self.num_channels):
            cfg = QuantGANConfig(
                noise_dim=noise_dim,
                epochs=params.max_epochs,
                batch_size=max(8, min(params.batch_size, windows.shape[0])),
                lr=params.learning_rate,
                patience=params.patience,
                clip_value=clip_value,
                d_steps_per_g_step=d_steps_per_g_step,
                output_bound=output_bound,
            )
            trainer = QuantGANTrainer(device=fit_input.device, cfg=cfg)
            # Logger label so live epoch events distinguish per-asset trainers
            # (only written when $STONKBENCH_MINI_EVENTS is set).
            trainer._emit_label = f"{model_key_early}.ch{c}"
            train_slice = windows[..., c]                       # (N, L)
            valid_slice = (
                valid_windows[..., c] if valid_windows is not None else None
            )
            result = trainer.fit(train_slice, valid_slice)
            self.trainers.append(trainer)
            per_asset_results.append(result)

        # Aggregate adapter-contract fields from per-asset results.
        best_val = float(np.mean([r.best_val_loss for r in per_asset_results]))
        best_epoch = int(max(r.best_epoch for r in per_asset_results))
        stopped_early = bool(any(r.stopped_early for r in per_asset_results))
        # Channel-averaged epoch curves (logger pass-through; identical for
        # every channel count since each trainer is trained independently).
        n_ep = min((len(r.train_loss_history) for r in per_asset_results), default=0)
        n_vp = min((len(r.val_loss_history) for r in per_asset_results), default=0)
        agg_train = [
            float(np.mean([r.train_loss_history[i] for r in per_asset_results]))
            for i in range(n_ep)
        ]
        agg_val = [
            float(np.mean([r.val_loss_history[i] for r in per_asset_results]))
            for i in range(n_vp)
        ]

        # Per-asset consolidated checkpoint: all C generator + discriminator
        # state_dicts in a single file. The downstream pipeline still loads it
        # as a single artifact per (model_key, seq_length).
        meta_ = fit_input.metadata or {}
        model_key_ = str(meta_.get("model_key", self.model_name))
        final_ckpt = checkpoints_dir / f"{model_key_}_seq{self.base_length}_final.pt"
        torch.save(
            {
                "model_name": model_key_,
                "num_channels": self.num_channels,
                "base_length": self.base_length,
                "per_asset_state_dicts": [
                    t.generator.state_dict() for t in self.trainers
                ],
                "per_asset_disc_state_dicts": [
                    t.discriminator.state_dict() for t in self.trainers
                ],
                "config_snapshot": {
                    "noise_dim": noise_dim,
                    "clip_value": clip_value,
                    "d_steps_per_g_step": d_steps_per_g_step,
                },
                "training_mode": "per_asset_univariate",
            },
            final_ckpt,
        )
        self.checkpoints = [final_ckpt]
        self._is_fitted = True

        return {
            "num_channels": self.num_channels,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
            "per_asset_best_val_loss": [
                r.best_val_loss for r in per_asset_results
            ],
            "per_asset_best_epoch": [r.best_epoch for r in per_asset_results],
            "train_loss_history": agg_train,
            "val_loss_history": agg_val,
        }

    def load_state(self, checkpoints: List[Path]) -> Dict[str, Any]:
        """Rebuild the per-asset trainers from a consolidated QuantGAN checkpoint.

        The file stores ``config_snapshot`` (noise_dim / clip / d_steps) plus
        one generator ``state_dict`` per channel, so the vendor trainers can be
        reconstructed exactly as ``fit`` built them and the generators reloaded.
        Discriminators are restored too when present (never used by generate,
        but keeps the trainer complete).
        """
        if not checkpoints:
            raise ValueError("QuantGANAdapter.load_state received an empty checkpoint list.")
        path = resolve_final_checkpoint(checkpoints)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "per_asset_state_dicts" not in payload:
            raise ValueError(
                f"{path.name} is not a consolidated QuantGAN checkpoint "
                "(missing 'per_asset_state_dicts')."
            )

        snap = payload.get("config_snapshot") or {}
        cfg = QuantGANConfig(
            noise_dim=int(snap.get("noise_dim", 3)),
            clip_value=float(snap.get("clip_value", 0.01)),
            d_steps_per_g_step=int(snap.get("d_steps_per_g_step", 5)),
            output_bound=float(snap.get("output_bound", 0.0)),
        )
        device = resolve_device()
        gen_states = payload["per_asset_state_dicts"]
        disc_states = payload.get("per_asset_disc_state_dicts") or []

        trainers: List[QuantGANTrainer] = []
        for idx, gen_state in enumerate(gen_states):
            trainer = QuantGANTrainer(device=device, cfg=cfg)
            trainer._init_models()
            trainer.generator.load_state_dict(gen_state)
            if idx < len(disc_states):
                trainer.discriminator.load_state_dict(disc_states[idx])
            trainer.generator.eval()
            trainers.append(trainer)

        self.trainers = trainers
        self.num_channels = int(payload.get("num_channels", len(trainers)))
        self.base_length = int(payload.get("base_length", 1))
        self.checkpoints = [path]
        self._is_fitted = True
        return {
            "restored": True,
            "checkpoint": str(path),
            "num_channels": self.num_channels,
            "base_length": self.base_length,
            "noise_dim": cfg.noise_dim,
            "device": str(device),
        }

    def generate(
        self,
        num_samples: int,
        generation_length: int,
        seed: int,
    ) -> AdapterGenerateOutput:
        if not self._is_fitted or not self.trainers:
            raise RuntimeError("Call fit() before generate().")
        # Per-asset univariate generate: each trainer returns (N, L); we stack
        # along a new channel axis to recover (N, L, C). Seed offset per channel
        # gives independent noise samples across assets — otherwise all 25
        # channels would share the same noise pattern and the joint simulator
        # output would carry spurious cross-channel correlation.
        per_asset = []
        for c, trainer in enumerate(self.trainers):
            data = trainer.generate(num_samples, self.base_length, seed=seed + c)
            per_asset.append(data)
        stacked = torch.stack(per_asset, dim=-1)
        return AdapterGenerateOutput(
            data=stacked.float(),
            checkpoints=self.checkpoints,
            logs={"trainer": "quantgan_per_asset"},
            extra_metadata={
                "num_channels": stacked.shape[-1],
                "training_mode": "per_asset_univariate",
            },
        )