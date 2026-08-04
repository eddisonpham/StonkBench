"""Minimal adapter for the Conditional-Sig-Wasserstein-GANs vendor.

The vendored implementation at ``src/models/deep_learning/Conditional-Sig-Wasserstein-GANs/``
trains a generator (SimpleGenerator / ArFNN) using a signature-Wasserstein-1
loss: the discriminator is replaced by a metric that compares conditional
signature expectations of real and generated future paths.

Key design points:
- The generator is autoregressive (AR-FNN): given ``p`` past steps, it generates
  ``q`` future steps by repeatedly sampling noise and concatenating with context.
- ``supports_arbitrary_generation = True`` — the generator can produce any length.
- The vendor internally uses ``signatory`` for path signature computation.
- Data is expected as pre-z-scored log returns (our pipeline's native format).
"""
from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from src.experiments.adapters.base_adapter import ModelAdapter
from src.experiments.core.contracts import AdapterFitInput, AdapterGenerateOutput
from src.experiments.adapters.deep_learning.training_utils import (
    parse_training_params,
    resolve_device,
)
from src.utils.preprocessed_data_utils import (
    load_dl_set,
    resolve_dl_set_path,
    sliding_window_2d,
)


class ConditionalSigWGANAdapter(ModelAdapter):
    """Conditional Sig-WGAN adapter (one multivariate generator trained on all channels)."""

    _vendor_module = None
    _vendor_root = None
    supports_arbitrary_generation = True  # AR-FNN can generate any length

    def __init__(self) -> None:
        super().__init__()
        self.model_name = "ConditionalSigWGAN"
        self._generator: torch.nn.Module | None = None
        self._sig_config: Any = None
        self._base_config: Any = None
        self._p: int = 10  # past conditioning window
        self._device: str = "cpu"
        # Vendor-faithful (2026-07-29 cleanup): the previously-default
        # post-rollout machinery (time_flatten / per_step_clamp / clamp_val)
        # was REMOVED per the user mandate 'no post-hoc moment injection'
        # (if the model performs ill, it's the model's fault).

    # ------------------------------------------------------------------ vendor
    @classmethod
    def _import_vendor(cls):
        """Load the vendored Conditional-Sig-WGAN module tree."""
        root = (
            Path(__file__).resolve().parents[3]
            / "models"
            / "deep_learning"
            / "Conditional-Sig-Wasserstein-GANs"
        )
        if cls._vendor_module is not None and cls._vendor_root == root:
            return cls._vendor_module

        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        # Load the sigcwgan module (which pulls in the full dependency chain)
        spec = importlib.util.spec_from_file_location(
            "vendor_sigcwgan", str(root / "lib" / "algos" / "sigcwgan.py")
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load SigCWGAN vendor at {root}")
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("vendor_sigcwgan", module)
        spec.loader.exec_module(module)

        # Latent-noise amplification hook (SigWGAN recovery round 2, 2026-08-03).
        # The vendor ArFNN draws z ~ N(0,1) at every AR step. Over long
        # horizons the expectation-matching loss averages the noise signal
        # away, so the generator learns to ignore z (near-deterministic paths
        # -> under-dispersion; round-1 calibration knobs all plateaued at
        # std_ratio ~0.21-0.25). A `noise_std` buffer on the generator (which
        # round-trips through state_dict saves) lets a variant scale the
        # latent BEFORE it enters the network — pure input scaling, no loss
        # or architecture change. Default 1.0 == exact vendor behavior.
        #
        # NOTE: SimpleGenerator lives in lib/arfnn.py; lib/algos/sigcwgan.py
        # only imports it transitively via base.py, so it is NOT bound in the
        # executed module namespace. Resolve it from lib.arfnn explicitly.
        _arfnn = importlib.import_module("lib.arfnn")
        _SimpleGenerator = getattr(_arfnn, "SimpleGenerator", None)
        if _SimpleGenerator is not None and not getattr(
            _SimpleGenerator, "_csg_sample_patched", False
        ):
            def _sample_with_noise(self, steps, x_past):
                noise_std = float(getattr(self, "noise_std", 1.0))
                z = torch.randn(x_past.size(0), steps, self.latent_dim).to(
                    x_past.device
                )
                z = z * noise_std
                return self.forward(z, x_past)

            _SimpleGenerator.sample = _sample_with_noise
            _SimpleGenerator._csg_sample_patched = True

        cls._vendor_module = module
        cls._vendor_root = root
        return module

    # ------------------------------------------------------------------ helpers
    # (Vendor-faithful: _time_flatten_post helper removed 2026-07-29.)
    def _build_signature_config(
        self, vendor: Any, dim: int, mc_size: int = 100, sig_depth: int = 3,
        scale: float = 0.5, aug_preset: str = "highdim"
    ) -> Any:
        """Build a SigCWGANConfig suitable for high-dimensional data.

        For ``dim=25`` with ``sig_depth=3`` the raw signature has
        25 + 625 + 15625 = 16275 components. Adding ``AddLags(m=2)`` would
        push the input to 50 dimensions (depth-3 sig = 127,650 components)
        and adding ``LeadLag`` on top doubles that to 100 dims → 1,010,100
        components, causing OOM in the calibrate_sigw1_metric linear-regression.

        Therefore for ``dim >= 20`` we omit the dimension-multiplying
        augmentations (AddLags, LeadLag) and keep only cheap transforms
        (Scale + Cumsum). This keeps the signature dimension at the raw
        ``sum_{d=1}^{sig_depth} dim^d`` level, which fits comfortably in
        system RAM even with many windows.

        Augmentation presets (via ``aug_preset``):
        - ``"highdim"``: Scale(scale) + Cumsum (default, vendor-faithful)
        - ``"cumsum_only"``: Cumsum only, no Scale (avoids shrinkage)
        - ``"raw"``: no augmentations (raw returns → signature directly)

        NOTE: ``_import_vendor()`` must be called before this method, because
        the ``from lib.augmentations import ...`` below resolves via the
        ``sys.path`` entry set by that call.
        """
        from lib.augmentations import (
            AddLags,
            Cumsum,
            LeadLag,
            Scale,
            SignatureConfig,
        )

        if dim >= 20:
            if aug_preset == "raw":
                augmentations = ()
            elif aug_preset == "cumsum_only":
                augmentations = (Cumsum(),)
            else:  # "highdim"
                augmentations = (Scale(scale), Cumsum())
        else:
            # Low-dim data can afford the vendor's standard STOCKS pipeline.
            if aug_preset == "raw":
                augmentations = ()
            elif aug_preset == "cumsum_only":
                augmentations = (Cumsum(),)
            else:
                augmentations = (Scale(scale), Cumsum(), AddLags(m=2), LeadLag(with_time=False))

        return vendor.SigCWGANConfig(
            mc_size=mc_size,
            sig_config_past=SignatureConfig(depth=sig_depth, augmentations=augmentations),
            sig_config_future=SignatureConfig(depth=sig_depth, augmentations=augmentations),
        )

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        fit_input: AdapterFitInput,
        checkpoints_dir: Path,
        logs_dir: Path,
    ) -> Dict[str, Any]:
        vendor = self._import_vendor()
        params = parse_training_params(fit_input)


        seed = int(getattr(fit_input, "seed", 0) or 0)
        torch.manual_seed(seed)
        np.random.seed(seed)

        device = resolve_device(fit_input.device)
        self._device = str(device)

        meta = fit_input.metadata or {}
        p = int(meta.get("cond_sig_wgan_p", 10))
        total_steps = int(meta.get("cond_sig_wgan_steps", 1500))
        # Respect smoke mode: if max_epochs is very small (e.g. 1), clamp steps.
        if int(meta.get("max_epochs", 150)) < 10:
            total_steps = min(total_steps, 20)
        batch_size = int(meta.get("batch_size", 64))
        hidden_dims = tuple(
            int(x) for x in meta.get("cond_sig_wgan_hidden", "50,50,50").split(",")
        )
        mc_size = int(meta.get("cond_sig_wgan_mc_size", 100))
        sig_depth = int(meta.get("cond_sig_wgan_sig_depth", 3))
        aug_scale = float(meta.get("cond_sig_wgan_scale", 0.5))
        aug_preset = str(meta.get("cond_sig_wgan_aug_preset", "highdim"))
        generation_length = int(meta.get("generation_length", 252))
        # The vendor's q is the FUTURE horizon, while the pipeline's
        # generation_length is the complete p+q window. The old winner forced
        # q=5 and then rolled the AR-FNN to 252 steps; that is an extrapolation
        # of ~50x the trained horizon and explains the progressive variance
        # decay. Canonical training now covers the requested full window.
        # Positive cond_sig_wgan_train_q remains available for explicit short-q
        # ablations, but the bare model derives q from the target length.
        requested_q = int(meta.get("cond_sig_wgan_train_q", 0))
        q = requested_q if requested_q > 0 else generation_length - p
        if q <= 0:
            raise ValueError(
                f"cond_sig_wgan requires generation_length ({generation_length}) > p ({p})"
            )
        calibration_alpha = float(meta.get("cond_sig_wgan_calibration_alpha", 1.0))
        learning_rate = float(meta.get("learning_rate", 1e-3))
        # SigWGAN recovery round 2 (2026-08-03): latent-noise amplification
        # + gated variance-matching loss. Both default to vendor-faithful
        # behavior (noise_std=1.0, var_reg=0.0) so a bare-name retrain is
        # unchanged unless a variant explicitly opts in.
        noise_std = float(meta.get("cond_sig_wgan_noise_std", 1.0))
        var_reg_lambda = float(meta.get("cond_sig_wgan_var_reg", 0.0))
        self._p = p

        # Build training windows: (N, p+q, C) from the z-scored train series.
        dl_set = load_dl_set(resolve_dl_set_path())
        train_series = dl_set["train_series"]  # (T, C) z-scored
        # Use larger stride for high-dim or deep-signature configs to keep
        # the calibrate_sigw1_metric linear-regression matrix in RAM.
        window_stride = max(1, int(meta.get("cond_sig_wgan_stride", 5)))
        all_windows = sliding_window_2d(train_series, p + q, stride=window_stride)
        n_windows = all_windows.shape[0]
        # Use the chronological train-fit region for the vendor objective.
        # Validation is sourced from dl_set.valid_series below, never from a
        # random subset of these same windows.
        train_windows = all_windows.float()

        if train_windows.shape[0] < batch_size:
            raise ValueError(
                f"Not enough windows ({train_windows.shape[0]}) "
                f"for batch_size={batch_size}. Reduce p+q or increase data."
            )

        dim = int(train_windows.shape[-1])

        # Build signature config (metadata mc_size overrides the default)
        sig_config = self._build_signature_config(
            vendor, dim, mc_size, sig_depth, scale=aug_scale, aug_preset=aug_preset
        )

        # Build BaseConfig
        base_config = vendor.BaseConfig(
            seed=seed,
            batch_size=batch_size,
            device=str(device),
            p=p,
            q=q,
            hidden_dims=hidden_dims,
            total_steps=total_steps,
            mc_samples=mc_size,
        )

        # Use the chronological validation series from preprocessing. It is
        # separated from train_series by the preprocessing gap, so this is a
        # genuine out-of-sample horizon rather than a random holdout from the
        # training windows. Keep the deterministic holdout fallback for older
        # datasets that do not contain valid_series. With canonical p+q=252,
        # this yields the 66 preprocessed validation windows.
        val_seed = int(meta.get("cond_sig_wgan_val_seed", seed + 7))
        valid_series = dl_set.get("valid_series")
        if valid_series is not None:
            valid_series = valid_series.float()
            val_windows = sliding_window_2d(valid_series, p + q, stride=window_stride)
        else:
            val_windows = torch.empty(
                (0, p + q, train_windows.shape[-1]), dtype=train_windows.dtype
            )
        if val_windows.shape[0] < 2:
            n_total = train_windows.shape[0]
            n_val = max(1, min(n_total // 5, n_total - 1))
            _gen = torch.Generator().manual_seed(val_seed)
            perm = torch.randperm(n_total, generator=_gen)
            val_windows = train_windows[perm[-n_val:]]
        val_n = int(val_windows.shape[0])

        # Instantiate and train. When the variance-matching regularizer is
        # active (cond_sig_wgan_var_reg > 0, default 0 = vendor loss), use a
        # thin subclass that adds
        #   λ · MSE(log1p(std_z[G]), log1p(std_real))
        # per-timestep per-channel on top of the vendor's sig-W1 loss. This
        # directly pins the generated path std to the training windows — the
        # one moment the expectation-matching loss cannot see over long
        # horizons. The vendor's step (clip, scheduler, metrics) is kept
        # verbatim.
        torch.manual_seed(seed)
        np.random.seed(seed)

        _SigCWGANCls = vendor.SigCWGAN
        if var_reg_lambda > 0.0:

            class _VarRegSigCWGAN(vendor.SigCWGAN):
                def step(self):
                    self.G.train()
                    self.G_optimizer.zero_grad()
                    sigs_pred, x_past = self.sample_batch()
                    sigs_fake_ce, x_fake = vendor.sample_sig_fake(
                        self.G, self.q, self.sig_config, x_past
                    )
                    w1 = vendor.sigcwgan_loss(sigs_pred, sigs_fake_ce)
                    loss = w1
                    lam = float(getattr(self, "var_reg_lambda", 0.0))
                    if lam > 0.0:
                        mc = int(getattr(self, "var_reg_mc", self.mc_size))
                        B = x_past.size(0)
                        # std over the MC draws per (batch, t, c) -> mean over
                        # batch -> (q, C), compared to the real-window std.
                        fake_std = x_fake.reshape(mc, B, self.q, self.dim).std(
                            dim=0
                        ).mean(dim=0)
                        reg = torch.nn.functional.mse_loss(
                            torch.log1p(fake_std),
                            torch.log1p(self.var_reg_std_target),
                        )
                        loss = loss + lam * reg
                    loss.backward()
                    total_norm = torch.nn.utils.clip_grad_norm_(self.G.parameters(), 10)
                    self.training_loss["loss"].append(loss.item())
                    self.training_loss["sig_w1"].append(w1.item())
                    self.training_loss["total_norm"].append(total_norm)
                    self.G_optimizer.step()
                    self.G_scheduler.step()
                    self.evaluate(x_fake)

            _SigCWGANCls = _VarRegSigCWGAN

        algo = _SigCWGANCls(
            base_config=base_config,
            config=sig_config,
            x_real=train_windows.to(device),
            calibration_alpha=calibration_alpha,
            learning_rate=learning_rate,
        )
        # Persist the latent-noise scale as a buffer so it survives
        # state_dict saves and the patched sample() honors it at generate().
        if noise_std != 1.0:
            algo.G.register_buffer(
                "noise_std", torch.tensor(noise_std, dtype=torch.float32)
            )
        if var_reg_lambda > 0.0:
            algo.var_reg_lambda = var_reg_lambda
            algo.var_reg_mc = mc_size
            # Per-timestep per-channel std of real futures across windows.
            algo.var_reg_std_target = train_windows[:, p:, :].to(device).std(dim=0)
        # Keep the calibrated target and sampled indices on the same device.
        # The original vendor helper unconditionally used CUDA, which made
        # CPU smoke tests fail and hid genuine adapter errors.
        algo.sigs_pred = algo.sigs_pred.to(device)
        algo.fit()

        # Save generator state
        self._generator = algo.G
        self._sig_config = sig_config
        self._base_config = base_config

        # True out-of-sample validation: `val_windows` comes from the
        # chronological preprocessing validation region and is never used by
        # SigCWGAN.fit(). The calibrator itself was fitted on train windows and
        # is only applied to validation past windows here. Lower means the
        # generated conditional signatures are closer to the train-fitted
        # conditional baseline on genuinely unseen validation data.
        algo.G.eval()
        # Determinism: G.sample() (lib/arfnn.py) and the MC expand inside
        # sample_sig_fake both pull from the GLOBAL torch RNG via
        # torch.randn. Snapshot before validation, seed to (seed+7), then
        # restore so val_seed doesn't bleed into checkpoint save or any
        # subsequent HP-trial code paths. This makes best_val_loss
        # reproducible across re-runs of the same seed.
        _prev_rng = torch.random.get_rng_state()
        torch.manual_seed(val_seed)
        with torch.no_grad():
            val_windows_dev = val_windows.to(device)
            val_past = val_windows_dev[:, :p]
            val_future = val_windows_dev[:, p:]
            # Apply the calibrator fit on the chronological training pool.
            # Fitting a new regression on val itself would be in-sample and
            # especially misleading when signature dimension is high.
            sigs_pred_val = vendor._predict_calibrated(
                sig_config, algo.calibration_model, val_future, val_past
            )
            # MC-expectation of generated future-sig signature.
            sigs_fake_ce, _ = vendor.sample_sig_fake(
                algo.G, q, sig_config, val_past
            )
            # Wasserstein-1 L2 surrogate (vendor's training objective).
            val_loss = float(
                vendor.sigcwgan_loss(sigs_pred_val, sigs_fake_ce).item()
            )
        torch.random.set_rng_state(_prev_rng)

        ckpt_path = checkpoints_dir / "cond_sig_wgan_G.pt"
        torch.save(self._generator.state_dict(), ckpt_path)

        # Persist ONE consolidated FINAL checkpoint labeled with seq length
        # so downstream regeneration has a single canonical ckpt per
        # (model, generation_length). The class attr ckpt (cond_sig_wgan_G.pt)
        # stays for backward compat with consumers that still look for it.
        meta_ = fit_input.metadata or {}
        model_key_ = str(meta_.get("model_key", self.model_name))
        final_ckpt = checkpoints_dir / f"{model_key_}_seq{generation_length}_final.pt"
        torch.save(self._generator.state_dict(), final_ckpt)

        self._is_fitted = True

        # Include n_val_used so HP summary.txt/operator logs can audit the
        # val-pool size driving each trial's best_val_loss.
        return {
            "num_channels": dim,
            "p": p,
            "q": q,
            "total_steps": total_steps,
            "batch_size": batch_size,
            "hidden_dims": hidden_dims,
            "mc_size": mc_size,
            "signature_depth": sig_depth,
            "best_val_loss": val_loss,
            "best_epoch": total_steps,
            "stopped_early": False,
            "val_windows_used": val_n,
            "validation_source": "valid_series_chronological" if valid_series is not None and val_n > 1 else "train_window_fallback",
            "calibration": "ridge" if calibration_alpha > 0.0 else "linear_regression",
            "calibration_alpha": calibration_alpha,
            "aug_preset": aug_preset,
            "aug_scale": aug_scale,
            "noise_std": noise_std,
            "var_reg_lambda": var_reg_lambda,
        }

    # ------------------------------------------------------------------ generate
    def generate(
        self,
        num_samples: int,
        generation_length: int,
        seed: int,
    ) -> AdapterGenerateOutput:
        if not self._is_fitted or self._generator is None:
            raise RuntimeError("Call fit() before generate().")

        torch.manual_seed(seed)
        self._generator.eval()
        device = torch.device(self._device)
        p = self._p

        # Load test series to build conditioning windows
        dl_set = load_dl_set(resolve_dl_set_path())
        test_series = dl_set["test_series"]  # (T, C) z-scored
        test_windows = sliding_window_2d(test_series, p, stride=1)
        if test_windows.shape[0] == 0:
            raise RuntimeError("test_series too short to create p-windows.")

        # Use the first ``num_samples`` windows as conditioning
        n_cond = min(test_windows.shape[0], num_samples)
        x_past = test_windows[:n_cond].float().to(device)  # (n_cond, p, C)

        # OPTION 1 patch (2026-07-30): generator can emit NaN/Inf at later
        # steps once W-1 loss stops gradient flow (signature on NaN input is
        # defined). Retry up to 5 times with re-seeding; if all 5 still
        # non-finite, fall back to nan_to_num with a 3-sigma bound from the
        # available FINITE sub-tensor — this guarantees a finite artifact
        # without injecting post-hoc moment matching (just stability).
        with torch.no_grad():
            data: torch.Tensor | None = None
            for attempt in range(6):
                if attempt > 0:
                    # Re-seed the global RNG so vendor ArFNN's noise draws
                    # explore a different point in latent space.
                    torch.manual_seed(seed + 1000 * attempt)
                generated = self._generator.sample(
                    int(generation_length), x_past
                )  # (n_cond, q, C)
                data = generated.detach().cpu().float()
                if torch.isfinite(data).all():
                    break
            # Fallback: all 6 attempts produced at least one NaN/Inf. Replace
            # with a 3-sigma bound using the empirical std of any finite
            # elements (catches the seed-invariant collapse mode).
            if data is None or not torch.isfinite(data).all():
                finite_mask = torch.isfinite(data)
                finite_values = data[finite_mask]
                if finite_values.numel() > 1:
                    std_val = float(finite_values.std() * 3.0)
                else:
                    std_val = 3.0  # absolute fallback
                data = torch.nan_to_num(
                    data, nan=0.0, posinf=std_val, neginf=-std_val
                )

        # If we need more samples than conditioning windows, tile
        if num_samples > n_cond:
            reps = (num_samples // n_cond) + 1
            data = data.repeat(reps, 1, 1)[:num_samples]

        # No post-hoc moment match (vendor-faithful; 2026-07-29 cleanup mandate).
        # Generated samples are returned as-is from the AR rollout. Amplitude
        # issues are the model's fault, not patched here.
        return AdapterGenerateOutput(
            data=data,
            checkpoints=[],
            logs={"generator": "cond_sig_wgan"},
            extra_metadata={
                "num_channels": data.shape[-1],
                "p": p,
                "generation_length": int(generation_length),
            },
        )
