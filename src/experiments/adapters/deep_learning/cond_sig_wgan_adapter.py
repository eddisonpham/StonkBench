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

        cls._vendor_module = module
        cls._vendor_root = root
        return module

    # ------------------------------------------------------------------ helpers
    # (Vendor-faithful: _time_flatten_post helper removed 2026-07-29.)
    def _build_signature_config(
        self, vendor: Any, dim: int, mc_size: int = 100, sig_depth: int = 3
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

        # For high-dim data (>=20 channels), avoid dimension-multiplying augs.
        # Scale + Cumsum keep dim unchanged and are cheap to compute.
        if dim >= 20:
            augmentations = (Scale(0.5), Cumsum())
        else:
            # Low-dim data can afford the vendor's standard STOCKS pipeline.
            augmentations = (Scale(0.5), Cumsum(), AddLags(m=2), LeadLag(with_time=False))

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
        generation_length = int(meta.get("generation_length", 252))
        q = generation_length  # train on the full generation horizon
        self._p = p

        # Build training windows: (N, p+q, C) from the z-scored train series.
        dl_set = load_dl_set(resolve_dl_set_path())
        train_series = dl_set["train_series"]  # (T, C) z-scored
        # Use larger stride for high-dim or deep-signature configs to keep
        # the calibrate_sigw1_metric linear-regression matrix in RAM.
        window_stride = max(1, int(meta.get("cond_sig_wgan_stride", 5)))
        all_windows = sliding_window_2d(train_series, p + q, stride=window_stride)
        n_windows = all_windows.shape[0]
        # Use all windows for training (SigCWGAN has no validation loop)
        train_windows = all_windows.float()

        if train_windows.shape[0] < batch_size:
            raise ValueError(
                f"Not enough windows ({train_windows.shape[0]}) "
                f"for batch_size={batch_size}. Reduce p+q or increase data."
            )

        dim = int(train_windows.shape[-1])

        # Build signature config (metadata mc_size overrides the default)
        sig_config = self._build_signature_config(vendor, dim, mc_size, sig_depth)

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

        # Instantiate and train
        torch.manual_seed(seed)
        np.random.seed(seed)

        algo = vendor.SigCWGAN(
            base_config=base_config,
            config=sig_config,
            x_real=train_windows.to(device),
        )
        # Vendor-side device-align: `lib.utils.sample_indices` (lib/utils.py:6)
        # unconditionally calls `.cuda()` on its random permutation. Sample-batch
        # (sigcwgan.py:69-71) then does `self.sigs_pred[random_indices]`, and
        # self.sigs_pred inherits x_future.device — so when --device cpu (or
        # --device cuda but the calibrate path landed on cpu by accident) is
        # chosen, PyTorch raises:
        #   RuntimeError: indices should be either on cpu or on the same
        #                 device as the indexed tensor (cpu)
        # Pin sigs_pred to cuda (the same device sample_indices uses) so the
        # lookup succeeds. This is correctness-preserving: sigs_pred is only
        # consumed by L2-norm aggregations (sigcwgan.py:13-14) which are
        # identical regardless of which side of cuda↔cpu the tensor lives on.
        # Without this shim, the smoke run on --device cpu crashes before
        # reaching the post-fit validation hook below.
        algo.sigs_pred = algo.sigs_pred.cuda()
        algo.fit()

        # Save generator state
        self._generator = algo.G
        self._sig_config = sig_config
        self._base_config = base_config

        # Real validation: hold out a deterministic fraction of train windows
        # AFTER training completes, then compute unbiased Sig-Wasserstein-1 via
        # the vendor's own metrics (`calibrate_sigw1_metric` + `sample_sig_fake`
        # + `sigcwgan_loss`), reproducing the canonical pattern in
        # src/models/.../evaluate.py:118-130.  Returns a scalar that HP search
        # can rank; lower = generator's signatures are closer to the best
        # linear predictor of val-future-from-val-past.
        #
        # Why post-train, post-split (not pre-train):
        #   1. Generator sees 100% of training data (best learned model).
        #   2. Calibrated LinearRegression is re-fit on the val pool only
        #      (unbiased W-1 against UNSEEN past->future mapping).
        #   3. No vendor modification, no subclass, no per-step hook.
        # Determinism: val_seed = seed + 7 so the same HP trial always holds
        # out the same windows → val_loss is reproducible across re-runs.
        val_frac = float(meta.get("cond_sig_wgan_val_frac", 0.2))
        val_seed = int(meta.get("cond_sig_wgan_val_seed", seed + 7))
        n_total = train_windows.shape[0]
        # LinearRegression in calibrate_sigw1_metric needs ≥2 windows; bump
        # the floor to 8 so 25-channel high-dim sig features (sig_depth=2 +
        # Scale+Cumsum -> ~650 dims) have enough observations for stable
        # coefficient estimation. Also cap n_val at n_total//2 so the train
        # pool retains the majority of windows; in n_total<16 smoke runs we
        # cap at n_total-8 (always leave ≥8 training windows).
        n_val = max(8, int(round(n_total * val_frac)))
        if n_total >= 32:
            n_val = min(n_val, n_total // 2)
        else:
            n_val = min(n_val, max(1, n_total - 8))
        n_train_eval = max(1, n_total - n_val)
        # Use a local Generator so we don't mutate the global RNG (next
        # HP trial that wants a deterministic randperm would otherwise
        # inherit val_seed).
        _gen = torch.Generator().manual_seed(val_seed)
        perm = torch.randperm(n_total, generator=_gen)
        val_windows = train_windows[perm[n_train_eval:]].to(device)
        val_n = int(val_windows.shape[0])

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
            val_past = val_windows[:, :p]
            val_future = val_windows[:, p:]
            # Calibrate LinearRegression on val pool only (unbiased baseline).
            sigs_pred_val = vendor.calibrate_sigw1_metric(
                sig_config, val_future, val_past
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
        # (model, seq_length). The class attr ckpt (cond_sig_wgan_G.pt)
        # stays for backward compat with consumers that still look for it.
        meta_ = fit_input.metadata or {}
        model_key_ = str(meta_.get("model_key", self.model_name))
        final_ckpt = checkpoints_dir / f"{model_key_}_seq{q}_final.pt"
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

        with torch.no_grad():
            # The generator can produce any length steps (autoregressive).
            generated = self._generator.sample(
                int(generation_length), x_past
            )  # (n_cond, q, C)

        data = generated.detach().cpu().float()

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
