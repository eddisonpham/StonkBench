"""PCF-GAN adapter: PathChar_GAN (unitary path characteristic function) per channel."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.experiments.adapters.base_adapter import ModelAdapter
from src.experiments.adapters.deep_learning.calibration import (
    ChannelMomentStats,
    match_channel_moments,
)
from src.experiments.adapters.deep_learning.training_utils import (
    EarlyStopping,
    FitTrainingInfo,
    make_loader,
    parse_training_params,
    resolve_device,
    use_calibration,
)
from src.experiments.core.contracts import AdapterFitInput, AdapterGenerateOutput
from src.utils.artifact_utils import stitch_sequences


def _toggle_grad(model: nn.Module, requires_grad: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(requires_grad)


class PCFGANAdapter(ModelAdapter):
    """
    Integrates vendor PathChar_GAN (PCF-GAN without reconstruction embedding).

    Trains one univariate LSTM generator + path CF critic per channel, then stacks
    to (R, L, C). Native length equals window length (typically 100); longer
    horizons use stitch_sequences.
    """

    model_name = "PCF-GAN"
    _vendor_cache: Dict[str, Any] = {}

    def __init__(self) -> None:
        super().__init__()
        self.generators: List[nn.Module] = []
        self.checkpoints: List[Path] = []
        self.base_length = 1
        self.device = "cpu"
        self.channel_stats: List[ChannelMomentStats] = []
        self.apply_calibration = False
        self.g_input_dim = 5
        self.noise_scale = 0.05

    @classmethod
    def _import_pcf(cls) -> Tuple[Any, Any]:
        """
        Import vendor LSTMGenerator + char_func_path without hijacking StonkBench ``src``.

        Loads PathChar critic pieces from ``PCFGAN/nn.py`` / ``PCFGAN.py`` under a
        temporary ``src`` mount, then restores StonkBench modules.
        """
        if "LSTMGenerator" in cls._vendor_cache and "char_func_path" in cls._vendor_cache:
            return cls._vendor_cache["LSTMGenerator"], cls._vendor_cache["char_func_path"]

        # parents[4] = repo root (NOT src/). The vendored PCF-GAN package lives at
        # <repo>/models/deep_learning/PCF-GAN, NOT under src/. Use parents[4].
        root = Path(__file__).resolve().parents[4] / "models" / "deep_learning" / "PCF-GAN"
        root_str = str(root)

        saved = {k: sys.modules[k] for k in list(sys.modules) if k == "src" or k.startswith("src.")}
        for k in list(saved):
            del sys.modules[k]

        inserted = False
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
            inserted = True
        try:
            from src.networks.generators import LSTMGenerator  # type: ignore
            from src.PCFGAN.PCFGAN import char_func_path  # type: ignore

            vendor_keepalive = {}
            for k, mod in list(sys.modules.items()):
                if not (k == "src" or k.startswith("src.")):
                    continue
                mod_file = str(getattr(mod, "__file__", "") or "")
                if "PCF-GAN" in mod_file:
                    vendor_keepalive[f"_pcfgan_vendor.{k}"] = mod
            sys.modules.update(vendor_keepalive)
            cls._vendor_cache["LSTMGenerator"] = LSTMGenerator
            cls._vendor_cache["char_func_path"] = char_func_path
            return LSTMGenerator, char_func_path
        finally:
            if inserted and root_str in sys.path:
                sys.path.remove(root_str)
            # Drop whatever vendor ``src.*`` is currently mounted, then restore StonkBench.
            for k in list(sys.modules):
                if k == "src" or k.startswith("src."):
                    del sys.modules[k]
            sys.modules.update(saved)

    def _build_generator(self, LSTMGenerator, output_dim: int = 1) -> nn.Module:
        return LSTMGenerator(
            input_dim=self.g_input_dim,
            output_dim=output_dim,
            hidden_dim=32,
            n_layers=2,
            noise_scale=self.noise_scale,
            BM=True,
            activation=nn.Identity(),
        )

    def _build_char_func(self, char_func_path, input_size: int = 1):
        # Slightly below vendor (10/6) to keep unitary Lie algebra training tractable at C=50.
        return char_func_path(
            num_samples=4,
            hidden_size=6,
            input_size=input_size,
            add_time=True,
            init_range=1,
        )

    @torch.no_grad()
    def _eval_val_loss(
        self,
        generator: nn.Module,
        char_func: nn.Module,
        valid_loader: DataLoader,
        device: torch.device,
        n_lags: int,
    ) -> float:
        generator.eval()
        char_func.eval()
        losses: List[float] = []
        for (batch_x,) in valid_loader:
            batch_x = batch_x.to(device)
            x_fake = generator(batch_size=batch_x.shape[0], n_lags=n_lags, device=device)
            # Also penalize variance collapse relative to real batch.
            dist = char_func.distance_measure(batch_x, x_fake, Lambda=0.1)
            std_pen = torch.relu(batch_x.std() * 0.25 - x_fake.std())
            losses.append(float((dist + 5.0 * std_pen).item()))
        return float(sum(losses) / max(len(losses), 1))

    def _train_channel(
        self,
        generator: nn.Module,
        char_func: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        params,
        device: torch.device,
        n_lags: int,
    ) -> FitTrainingInfo:
        g_opt = torch.optim.Adam(generator.parameters(), lr=params.learning_rate, betas=(0.0, 0.9))
        m_opt = torch.optim.Adam(char_func.parameters(), lr=min(5e-3, params.learning_rate * 5.0), betas=(0.0, 0.9))
        # Unitary CF steps are expensive on CPU; keep updates light for short HP/smoke budgets.
        if params.max_epochs <= 2:
            d_steps, steps_per_epoch = 1, 1
        elif params.max_epochs <= 5:
            d_steps, steps_per_epoch = 1, max(4, len(train_loader))
        else:
            d_steps, steps_per_epoch = 2, max(20, len(train_loader))
        early_stop = EarlyStopping(
            patience=params.patience,
            min_epochs=0 if params.max_epochs <= 5 else max(20, params.patience * 2),
        )
        best_state = copy.deepcopy(generator.state_dict())
        info = FitTrainingInfo(best_val_loss=float("inf"), best_epoch=0, stopped_early=False)
        train_iter = iter(train_loader)

        def _next_batch() -> torch.Tensor:
            nonlocal train_iter
            try:
                (batch,) = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                (batch,) = next(train_iter)
            return batch.to(device)

        for epoch in range(params.max_epochs):
            generator.train()
            char_func.train()
            epoch_g = 0.0
            for _ in range(steps_per_epoch):
                for _d in range(d_steps):
                    x_real = _next_batch()
                    with torch.no_grad():
                        x_fake = generator(
                            batch_size=x_real.shape[0], n_lags=n_lags, device=device
                        )
                    _toggle_grad(char_func, True)
                    m_opt.zero_grad(set_to_none=True)
                    d_loss = -char_func.distance_measure(x_real, x_fake, Lambda=0.1)
                    d_loss.backward()
                    m_opt.step()
                    _toggle_grad(char_func, False)

                x_real = _next_batch()
                _toggle_grad(generator, True)
                g_opt.zero_grad(set_to_none=True)
                x_fake = generator(batch_size=x_real.shape[0], n_lags=n_lags, device=device)
                g_loss = char_func.distance_measure(x_real, x_fake, Lambda=0.1)
                g_loss.backward()
                g_opt.step()
                epoch_g += float(g_loss.item())

            info.train_loss_history.append(epoch_g / max(steps_per_epoch, 1))
            val_loss = self._eval_val_loss(generator, char_func, valid_loader, device, n_lags)
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
        del logs_dir  # unused; vendor plots are skipped
        windows = fit_input.batch.train_windows
        valid_windows = fit_input.batch.valid_windows
        if windows is None or windows.ndim != 3:
            raise ValueError("PCFGANAdapter expects train_windows shaped (N, L, C)")
        if valid_windows is None or valid_windows.shape[0] == 0:
            raise ValueError("PCFGANAdapter requires non-empty valid_windows for model selection.")

        LSTMGenerator, char_func_path = self._import_pcf()
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
        effective_bs = max(16, min(int(params.batch_size), 128, int(windows.shape[0])))

        for c in range(num_channels):
            channel_train = windows[:, :, c : c + 1].float()
            channel_valid = valid_windows[:, :, c : c + 1].float()
            self.channel_stats.append(ChannelMomentStats.from_univariate(channel_train.squeeze(-1)))

            generator = self._build_generator(LSTMGenerator).to(device)
            char_func = self._build_char_func(char_func_path).to(device)
            train_loader = make_loader(channel_train, effective_bs, shuffle=True)
            valid_loader = make_loader(channel_valid, effective_bs, shuffle=False)
            info = self._train_channel(
                generator, char_func, train_loader, valid_loader, params, device, self.base_length
            )
            channel_val_losses.append(info.best_val_loss)
            best_epochs.append(info.best_epoch)
            stopped_early = stopped_early or info.stopped_early
            self.generators.append(generator.cpu())

            ckpt = checkpoints_dir / f"pcf_gan_checkpoint_{c + 1}.pt"
            torch.save(
                {
                    "channel": c,
                    "generator": generator.state_dict(),
                    "base_length": self.base_length,
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
            logs={"trainer": "pcf_gan_pathchar"},
            extra_metadata={"num_channels": data.shape[-1]},
        )
