"""PCF-GAN adapter — native multivariate (single joint generator on full (N, L, C) data)."""
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


def _toggle_grad(model: nn.Module, requires_grad: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(requires_grad)


class PCFGANAdapter(ModelAdapter):
    """Multivariate PCF-GAN: single joint LSTMGenerator + path CF critic on full (N, L, C) data."""

    model_name = "PCF-GAN"
    _vendor_cache: Dict[str, Any] = {}

    def __init__(self) -> None:
        super().__init__()
        self.generator: nn.Module | None = None
        self.char_func: nn.Module | None = None
        self.checkpoints: List[Path] = []
        self.base_length = 1
        self.num_channels = 1
        self.device = "cpu"
        self.g_input_dim = 5
        self.noise_scale = 0.05

    @classmethod
    def _import_pcf(cls) -> Tuple[Any, Any]:
        if "LSTMGenerator" in cls._vendor_cache and "char_func_path" in cls._vendor_cache:
            return cls._vendor_cache["LSTMGenerator"], cls._vendor_cache["char_func_path"]

        root = Path(__file__).resolve().parents[3] / "models" / "deep_learning" / "PCF-GAN"
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        old_src = str(root / "src")
        if old_src in sys.path:
            sys.path.remove(old_src)

        from pcfgan_src.networks.generators import LSTMGenerator  # type: ignore
        from pcfgan_src.PCFGAN.PCFGAN import char_func_path  # type: ignore

        cls._vendor_cache["LSTMGenerator"] = LSTMGenerator
        cls._vendor_cache["char_func_path"] = char_func_path
        return LSTMGenerator, char_func_path

    def _build_generator(self, LSTMGenerator, output_dim: int) -> nn.Module:
        return LSTMGenerator(
            input_dim=self.g_input_dim,
            output_dim=output_dim,
            hidden_dim=32,
            n_layers=2,
            noise_scale=self.noise_scale,
            BM=True,
            activation=nn.Identity(),
        )

    def _build_char_func(self, char_func_path, input_size: int) -> nn.Module:
        return char_func_path(
            num_samples=4,
            hidden_size=6,
            input_size=input_size,
            add_time=True,
            init_range=1,
        )

    @torch.no_grad()
    def _eval_val_loss(self, generator, char_func, valid_loader, device, n_lags):
        generator.eval()
        char_func.eval()
        losses: List[float] = []
        for (batch_x,) in valid_loader:
            batch_x = batch_x.to(device)
            x_fake = generator(batch_size=batch_x.shape[0], n_lags=n_lags, device=device)
            dist = char_func.distance_measure(batch_x, x_fake, Lambda=0.1)
            std_pen = torch.relu(batch_x.std() * 0.50 - x_fake.std())
            losses.append(float((dist + 5.0 * std_pen).item()))
        return float(sum(losses) / max(len(losses), 1))

    def _train_joint(self, generator, char_func, train_loader, valid_loader, params, device, n_lags):
        g_opt = torch.optim.Adam(generator.parameters(), lr=params.learning_rate, betas=(0.0, 0.9))
        m_opt = torch.optim.Adam(char_func.parameters(), lr=min(5e-3, params.learning_rate * 5.0), betas=(0.0, 0.9))
        if params.max_epochs <= 2:
            d_steps, steps_per_epoch = 1, 1
        elif params.max_epochs <= 5:
            d_steps, steps_per_epoch = 1, max(4, len(train_loader))
        else:
            d_steps, steps_per_epoch = 2, max(20, len(train_loader))
        early_stop = EarlyStopping(
            patience=params.patience,
            min_epochs=0 if params.max_epochs <= 5 else max(80, params.patience * 3),
        )
        best_state = copy.deepcopy(generator.state_dict())
        info = FitTrainingInfo(best_val_loss=float("inf"), best_epoch=0, stopped_early=False)
        train_iter = iter(train_loader)

        def _next_batch():
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
                        x_fake = generator(batch_size=x_real.shape[0], n_lags=n_lags, device=device)
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
                std_pen = torch.relu(x_real.std() * 0.50 - x_fake.std())
                g_loss = g_loss + 2.0 * std_pen
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
        del logs_dir
        windows = fit_input.batch.train_windows
        valid_windows = fit_input.batch.valid_windows
        if windows is None or windows.ndim != 3:
            raise ValueError("PCFGANAdapter expects train_windows shaped (N, L, C)")
        if valid_windows is None or valid_windows.shape[0] == 0:
            raise ValueError("PCFGANAdapter requires non-empty valid_windows.")

        LSTMGenerator, char_func_path = self._import_pcf()
        params = parse_training_params(fit_input)
        self._last_fit_metadata = dict(getattr(fit_input, "metadata", {}) or {})
        self.base_length = int(windows.shape[1])
        num_channels = int(windows.shape[2])
        self.num_channels = num_channels
        device = resolve_device(fit_input.device)
        self.device = str(device)

        self.generator = self._build_generator(LSTMGenerator, output_dim=num_channels).to(device)
        self.char_func = self._build_char_func(char_func_path, input_size=num_channels).to(device)
        effective_bs = max(16, min(int(params.batch_size), 128, int(windows.shape[0])))
        train_loader = make_loader(windows.float(), effective_bs, shuffle=True)
        valid_loader = make_loader(valid_windows.float(), effective_bs, shuffle=False)
        info = self._train_joint(
            self.generator, self.char_func, train_loader, valid_loader, params, device, self.base_length
        )

        self._is_fitted = True
        meta_ = fit_input.metadata or {}
        model_key_ = str(meta_.get("model_key", self.model_name))
        final_ckpt = checkpoints_dir / f"{model_key_}_seq{self.base_length}_final.pt"
        torch.save(
            {
                "model_name": model_key_,
                "num_channels": num_channels,
                "base_length": self.base_length,
                "g_input_dim": self.g_input_dim,
                "noise_scale": self.noise_scale,
                "state_dict": self.generator.state_dict(),
                "char_func_state_dict": self.char_func.state_dict(),
            },
            final_ckpt,
        )
        self.checkpoints = [final_ckpt]

        return {
            "num_channels": num_channels,
            "best_val_loss": info.best_val_loss,
            "best_epoch": info.best_epoch,
            "stopped_early": info.stopped_early,
        }

    def generate(self, num_samples: int, generation_length: int, seed: int) -> AdapterGenerateOutput:
        if not self._is_fitted or self.generator is None:
            raise RuntimeError("Call fit() before generate().")

        torch.manual_seed(seed)
        device = resolve_device(self.device)
        self.generator = self.generator.to(device)
        self.generator.eval()
        with torch.no_grad():
            data = self.generator(
                batch_size=int(num_samples),
                n_lags=self.base_length,
                device=device,
            ).detach().cpu()
        self.generator.cpu()

        # Ensure output is (R, L, C)
        if data.ndim == 2:
            data = data.unsqueeze(-1)

        return AdapterGenerateOutput(
            data=data.float(),
            checkpoints=self.checkpoints,
            logs={"trainer": "pcf_gan_pathchar"},
            extra_metadata={"num_channels": data.shape[-1]},
        )
