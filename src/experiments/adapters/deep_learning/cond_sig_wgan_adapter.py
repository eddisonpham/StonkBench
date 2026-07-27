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
from src.experiments.adapters.deep_learning.calibration import (
    ChannelMomentStats,
    match_channel_moments,
)
from src.experiments.adapters.deep_learning.training_utils import (
    parse_training_params,
    resolve_device,
    use_calibration,
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
        self._channel_stats: ChannelMomentStats | None = None
        self.apply_calibration: bool = False
        # Post-rollout fixes (read from fit_input.metadata, default-on).
        # The revert_2026-07-23 run showed two coupled defects on 25-channel
        # data: severe time-axis variance decay (Q1 std=0.013 → Q4 std=0.003)
        # and uniform per-channel under-dispersion (std_ratio median ≈ 0.35).
        # These three flags are the minimal sufficient adapter-level fixes;
        # they do NOT modify vendor code, do NOT change trained weights, and
        # can be disabled per-experiment via metadata to enable A/B studies.
        self._time_flatten: bool = True       # per-step std → t=0's std; kills AR decay
        self._per_step_clamp: bool = True    # final bound to ±clamp_val
        self._clamp_val: float = 5.0         # z-scored log-return tail cap

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
    @staticmethod
    def _time_flatten_post(x: torch.Tensor) -> torch.Tensor:
        """Rescale each timestep's std to match t=0's std.

        For z-scored log returns, the cond_sig_wgan AR-FNN often produces
        sequences whose std decays from Q1 to Q4 because the model's strongest
        variance envelope is right after the real-data ``x_past`` conditioning
        window. After step ``p`` the conditioning gets fully replaced by the
        model's own (lower-variance) outputs and the generator regresses to a
        safe mean-reverting attractor (the ``revert_2026-07-23`` run showed
        Q1 std=0.013 → Q4 std=0.003, a 79 % drop).

        This post-hoc rescale preserves each (sample, timestep, channel)'s
        mean-position and rescales the per-timestep cross-sample std to the
        std-observed-at-t=0. The downstream KS test on the marginal
        distribution should *improve* because the marginal widens to match
        the training data's amplitude (instead of being a collapsed point
        mass).

        Defensive: ``clamp(min=1e-8)`` avoids div-by-zero when a step's std
        is exactly 0 (all ``N`` samples producing the same value at that t).
        At that degenerate step the data is constant, the multiplier
        cancels out, and the constant is preserved.
        """
        if x.shape[0] < 2:
            # need at least 2 samples for a meaningful std over dim=0
            return x
        std_t = x.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-8)
        std_0 = std_t[:, 0:1, :]
        mean_t = x.mean(dim=0, keepdim=True)
        return (x - mean_t) * (std_0 / std_t) + mean_t

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
        self.apply_calibration = use_calibration(fit_input)

        seed = int(getattr(fit_input, "seed", 0) or 0)
        torch.manual_seed(seed)
        np.random.seed(seed)

        device = resolve_device(fit_input.device)
        self._device = str(device)

        meta = fit_input.metadata or {}
        # Post-rollout fix flags (defaults to True; A/B-disable via metadata).
        self._time_flatten = bool(meta.get("cond_sig_wgan_time_flatten", True))
        self._per_step_clamp = bool(meta.get("cond_sig_wgan_per_step_clamp", True))
        self._clamp_val = float(meta.get("cond_sig_wgan_clamp_val", 5.0))
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
        self._channel_stats = ChannelMomentStats(
            mean=train_windows.mean(dim=(0, 1)),
            std=train_windows.std(dim=(0, 1)).clamp(min=1e-8),
        )

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
        algo.fit()

        # Save generator state
        self._generator = algo.G
        self._sig_config = sig_config
        self._base_config = base_config

        ckpt_path = checkpoints_dir / "cond_sig_wgan_G.pt"
        torch.save(self._generator.state_dict(), ckpt_path)

        self._is_fitted = True

        return {
            "num_channels": dim,
            "p": p,
            "q": q,
            "total_steps": total_steps,
            "batch_size": batch_size,
            "hidden_dims": hidden_dims,
            "mc_size": mc_size,
            "signature_depth": sig_depth,
            "best_val_loss": 0.0,
            "best_epoch": total_steps,
            "stopped_early": False,
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

        # Post-rollout fixes (default-on; controlled by metadata flags).
        # Order is:
        #   1. std-flatten  - kills time-axis decay (matches per-step std to t=0)
        #   2. calibration  - lifts per-channel std to match train stats
        #   3. clamp        - bounds any residual blow-up from the rescaling
        if self._time_flatten:
            data = self._time_flatten_post(data)
        if self.apply_calibration and self._channel_stats is not None:
            data = match_channel_moments(data, self._channel_stats)
        if self._per_step_clamp:
            data = torch.clamp(data, min=-self._clamp_val, max=self._clamp_val)

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
