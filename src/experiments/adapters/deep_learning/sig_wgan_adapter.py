"""Sig-Wasserstein GAN adapter: signature W1 metric + LSTM generator, per channel."""

from __future__ import annotations

import copy
import math
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from src.experiments.adapters.base_adapter import ModelAdapter
from src.experiments.adapters.deep_learning.calibration import (
    ChannelMomentStats,
    match_channel_moments,
)
from src.experiments.adapters.deep_learning.training_utils import (
    EarlyStopping,
    FitTrainingInfo,
    parse_training_params,
    resolve_device,
    use_calibration,
)
from src.experiments.core.contracts import AdapterFitInput, AdapterGenerateOutput
from src.utils.artifact_utils import stitch_sequences


def _torch_signature(path: torch.Tensor, depth: int) -> torch.Tensor:
    """
    Truncated stream signature via iterated integrals (piecewise-linear / Chen).

    path: (N, L, C) -> (N, sum_{k=1}^{depth} C^k)
    Used as a drop-in when the ``signatory`` wheel is unavailable for the installed torch.
    """
    if path.ndim != 3:
        raise ValueError(f"signature expects (N, L, C), got {tuple(path.shape)}")
    if depth < 1:
        raise ValueError("depth must be >= 1")
    dx = path[:, 1:] - path[:, :-1]
    n, t, c = dx.shape
    level = dx
    parts = [level.sum(dim=1)]
    for _ in range(2, depth + 1):
        running = torch.cumsum(level, dim=1)
        delayed = torch.nn.functional.pad(running[:, :-1], (0, 0, 1, 0))
        prev_flat = delayed.reshape(n, t, -1)
        level = (prev_flat.unsqueeze(-1) * dx.unsqueeze(-2)).reshape(n, t, -1)
        parts.append(level.sum(dim=1))
    return torch.cat(parts, dim=-1)


def _ensure_signatory_shim() -> None:
    """Install a minimal signatory-compatible module if real signatory is missing."""
    if "signatory" in sys.modules:
        return
    try:
        import signatory  # noqa: F401

        return
    except Exception:
        pass

    mod = types.ModuleType("signatory")

    def signature(path: torch.Tensor, depth: int, **kwargs):  # noqa: ANN003
        del kwargs
        return _torch_signature(path, depth)

    def logsignature_channels(in_channels: int, depth: int) -> int:
        total = 0
        dim = in_channels
        for k in range(1, depth + 1):
            total += dim**k
        return int(total)

    mod.signature = signature  # type: ignore[attr-defined]
    mod.logsignature_channels = logsignature_channels  # type: ignore[attr-defined]
    sys.modules["signatory"] = mod


class SigWGANAdapter(ModelAdapter):
    """
    Integrates vendor SigWGAN (signature Wasserstein-1) with an LSTM generator.

    Trains one univariate model per channel (signature dim explodes for C=50).
    Native length = window length; stitch for longer horizons.
    """

    model_name = "Sig-Wasserstein-GAN"
    _vendor_cache: Dict[str, Any] = {}

    def __init__(self) -> None:
        super().__init__()
        self.generators: List[nn.Module] = []
        self.checkpoints: List[Path] = []
        self.base_length = 1
        self.device = "cpu"
        self.channel_stats: List[ChannelMomentStats] = []
        self.apply_calibration = False
        self.sig_depth = 3
        self.g_input_dim = 5
        self.hidden_dim = 50
        self.n_layers = 2

    @classmethod
    def _import_sig_wgan(cls) -> Tuple[Any, Any, Any, Any]:
        if all(
            k in cls._vendor_cache
            for k in ("LSTMGenerator", "SigW1Metric", "parse_augmentations", "apply_augmentations")
        ):
            return (
                cls._vendor_cache["LSTMGenerator"],
                cls._vendor_cache["SigW1Metric"],
                cls._vendor_cache["parse_augmentations"],
                cls._vendor_cache["apply_augmentations"],
            )

        _ensure_signatory_shim()
        # parents[4] = repo root (NOT src/). Vendored Sig-WGAN package lives at
        # <repo>/models/deep_learning/Sig-Wasserstein-GANs, NOT under src/.
        root = Path(__file__).resolve().parents[4] / "models" / "deep_learning" / "Sig-Wasserstein-GANs"
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        for key in list(sys.modules):
            if key == "lib" or key.startswith("lib."):
                mod = sys.modules[key]
                mod_file = str(getattr(mod, "__file__", "") or "")
                if "Sig-Wasserstein-GANs" not in mod_file:
                    del sys.modules[key]

        from lib.augmentations import apply_augmentations, parse_augmentations  # type: ignore
        from lib.networks.generators import LSTMGenerator  # type: ignore
        from lib.trainers.sig_wgan import SigW1Metric  # type: ignore

        for key in list(sys.modules):
            if key == "lib" or key.startswith("lib."):
                mod = sys.modules.pop(key)
                sys.modules[f"_sigwgan_vendor.{key}"] = mod

        cls._vendor_cache.update(
            {
                "LSTMGenerator": LSTMGenerator,
                "SigW1Metric": SigW1Metric,
                "parse_augmentations": parse_augmentations,
                "apply_augmentations": apply_augmentations,
            }
        )
        return LSTMGenerator, SigW1Metric, parse_augmentations, apply_augmentations

    def _default_augmentations(self, parse_augmentations):
        # Keep path dim small: univariate + time (+ optional scale). LeadLag/VisiTrans explode dim.
        aug_cfg = [
            {"name": "Scale", "scale": 1.0},
            {"name": "AddTime"},
        ]
        return parse_augmentations([dict(x) for x in aug_cfg])

    @staticmethod
    def _normalize_signature(expected_sig: torch.Tensor, depth: int, dim: int) -> torch.Tensor:
        out = expected_sig.clone()
        count = 0
        for i in range(depth):
            block = dim ** (i + 1)
            out[count : count + block] = out[count : count + block] * math.factorial(i + 1)
            count += block
        return out

    @torch.no_grad()
    def _eval_val_loss(
        self,
        generator: nn.Module,
        x_valid: torch.Tensor,
        augmentations,
        apply_augmentations,
        depth: int,
        device: torch.device,
        batch_size: int,
    ) -> float:
        generator.eval()
        n_lags = x_valid.shape[1]
        x_fake = generator(batch_size=min(batch_size, x_valid.shape[0]), n_lags=n_lags, device=device)
        aug_real = apply_augmentations(x_valid.to(device), augmentations)
        aug_fake = apply_augmentations(x_fake, augmentations)
        mu = _torch_signature(aug_real, depth).mean(0)
        nu = _torch_signature(aug_fake, depth).mean(0)
        dim = aug_real.shape[-1]
        mu = self._normalize_signature(mu, depth, dim)
        nu = self._normalize_signature(nu, depth, dim)
        sig_loss = (mu - nu).pow(2).sum().sqrt()
        # Slightly stronger variance floor than original to reduce zero-band collapse.
        std_pen = torch.relu(x_valid.to(device).std() * 0.35 - x_fake.std())
        return float((sig_loss + 8.0 * std_pen).item())

    def _train_channel(
        self,
        generator: nn.Module,
        x_train: torch.Tensor,
        x_valid: torch.Tensor,
        SigW1Metric,
        augmentations,
        apply_augmentations,
        params,
        device: torch.device,
    ) -> FitTrainingInfo:
        metric = SigW1Metric(
            depth=self.sig_depth,
            x_real=x_train.to(device),
            mask_rate=0.01,
            augmentations=augmentations,
            normalise=True,
        )
        optimizer = torch.optim.Adam(generator.parameters(), lr=params.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.95, step_size=128)
        batch_size = max(8, min(int(params.batch_size), 256, int(x_train.shape[0])))

        if params.max_epochs <= 2:
            steps_per_epoch = 1
        elif params.max_epochs <= 5:
            steps_per_epoch = max(4, x_train.shape[0] // batch_size)
        else:
            # Minimal bump over baseline so full train is not done in ~3 minutes.
            steps_per_epoch = max(40, x_train.shape[0] // batch_size)

        early_stop = EarlyStopping(
            patience=params.patience,
            min_epochs=0 if params.max_epochs <= 5 else max(30, params.patience * 3),
        )
        best_state = copy.deepcopy(generator.state_dict())
        info = FitTrainingInfo(best_val_loss=float("inf"), best_epoch=0, stopped_early=False)
        n_lags = int(x_train.shape[1])

        for epoch in range(params.max_epochs):
            generator.train()
            epoch_loss = 0.0
            for _ in range(steps_per_epoch):
                optimizer.zero_grad(set_to_none=True)
                x_fake = generator(batch_size=batch_size, n_lags=n_lags, device=device)
                loss = metric(x_fake)
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += float(loss.item())

            info.train_loss_history.append(epoch_loss / max(steps_per_epoch, 1))
            val_loss = self._eval_val_loss(
                generator,
                x_valid,
                augmentations,
                apply_augmentations,
                self.sig_depth,
                device,
                batch_size,
            )
            info.val_loss_history.append(val_loss)
            if val_loss < info.best_val_loss:
                info.best_val_loss = val_loss
                info.best_epoch = epoch + 1
                best_state = copy.deepcopy(generator.state_dict())
            if early_stop.step(val_loss, epoch + 1):
                info.stopped_early = True
                break

        generator.load_state_dict(best_state)
        return info

    def fit(self, fit_input: AdapterFitInput, checkpoints_dir: Path, logs_dir: Path) -> Dict[str, Any]:
        del logs_dir
        windows = fit_input.batch.train_windows
        valid_windows = fit_input.batch.valid_windows
        if windows is None or windows.ndim != 3:
            raise ValueError("SigWGANAdapter expects train_windows shaped (N, L, C)")
        if valid_windows is None or valid_windows.shape[0] == 0:
            raise ValueError("SigWGANAdapter requires non-empty valid_windows for model selection.")

        LSTMGenerator, SigW1Metric, parse_augmentations, apply_augmentations = self._import_sig_wgan()
        params = parse_training_params(fit_input)
        self.apply_calibration = use_calibration(fit_input)
        self.base_length = int(windows.shape[1])
        num_channels = int(windows.shape[2])
        device = resolve_device(fit_input.device)
        self.device = str(device)

        self.generators = []
        self.checkpoints = []
        self.channel_stats = []
        channel_val_losses: List[float] = []
        best_epochs: List[int] = []
        stopped_early = False
        augmentations = self._default_augmentations(parse_augmentations)

        for c in range(num_channels):
            channel_train = windows[:, :, c : c + 1].float()
            channel_valid = valid_windows[:, :, c : c + 1].float()
            self.channel_stats.append(ChannelMomentStats.from_univariate(channel_train.squeeze(-1)))

            generator = LSTMGenerator(
                input_dim=self.g_input_dim,
                output_dim=1,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_layers,
                init_fixed=False,
            ).to(device)
            info = self._train_channel(
                generator,
                channel_train,
                channel_valid,
                SigW1Metric,
                augmentations,
                apply_augmentations,
                params,
                device,
            )
            channel_val_losses.append(info.best_val_loss)
            best_epochs.append(info.best_epoch)
            stopped_early = stopped_early or info.stopped_early
            self.generators.append(generator.cpu())

            ckpt = checkpoints_dir / f"sig_wgan_checkpoint_{c + 1}.pt"
            torch.save(
                {
                    "channel": c,
                    "generator": generator.state_dict(),
                    "base_length": self.base_length,
                    "sig_depth": self.sig_depth,
                    "g_input_dim": self.g_input_dim,
                },
                ckpt,
            )
            self.checkpoints.append(ckpt)

        self._is_fitted = True
        return {
            "num_channels": num_channels,
            "best_val_loss": float(sum(channel_val_losses) / len(channel_val_losses)),
            "best_epoch": int(max(best_epochs)),
            "stopped_early": stopped_early,
            "sig_depth": self.sig_depth,
        }

    def generate(self, num_samples: int, generation_length: int, seed: int) -> AdapterGenerateOutput:
        if not self._is_fitted or not self.generators:
            raise RuntimeError("Call fit() before generate().")

        torch.manual_seed(seed)
        device = resolve_device(self.device)
        per_channel = []
        for c, generator in enumerate(self.generators):
            generator = generator.to(device)
            generator.eval()
            with torch.no_grad():
                channel = generator(
                    batch_size=int(num_samples),
                    n_lags=self.base_length,
                    device=device,
                ).detach().cpu().squeeze(-1)
            if self.apply_calibration and self.channel_stats:
                channel = match_channel_moments(channel, self.channel_stats[c])
            if generation_length != self.base_length:
                channel = stitch_sequences(channel, generation_length, seed=seed + c)
            per_channel.append(channel.unsqueeze(-1))
            generator.cpu()

        data = torch.cat(per_channel, dim=-1).float()
        return AdapterGenerateOutput(
            data=data,
            checkpoints=self.checkpoints,
            logs={"trainer": "sig_wgan"},
            extra_metadata={"num_channels": data.shape[-1]},
        )
