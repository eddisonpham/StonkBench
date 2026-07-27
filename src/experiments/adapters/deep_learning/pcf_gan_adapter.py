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



def _toggle_grad(model: nn.Module, requires_grad: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(requires_grad)


class PCFGANAdapter(ModelAdapter):
    """
    Integrates vendor PathChar_GAN (PCF-GAN without reconstruction embedding).

    Trains one univariate LSTM generator + path CF critic per channel, then stacks
    to (R, L, C). Native length equals the trained window length.
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
        """Import vendor LSTMGenerator + char_func_path from the PCF-GAN vendored code."""
        if "LSTMGenerator" in cls._vendor_cache and "char_func_path" in cls._vendor_cache:
            return cls._vendor_cache["LSTMGenerator"], cls._vendor_cache["char_func_path"]

        root = Path(__file__).resolve().parents[3] / "models" / "deep_learning" / "PCF-GAN"
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        # Remove the pre-rename vendor path so a stale entry from an earlier
        # process cannot shadow the new pcfgan_src package.
        old_src = str(root / "src")
        if old_src in sys.path:
            sys.path.remove(old_src)

        from pcfgan_src.networks.generators import LSTMGenerator  # type: ignore
        from pcfgan_src.PCFGAN.PCFGAN import char_func_path  # type: ignore

        cls._vendor_cache["LSTMGenerator"] = LSTMGenerator
        cls._vendor_cache["char_func_path"] = char_func_path
        return LSTMGenerator, char_func_path

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
            # Floor at 0.50 * real std (was 0.25) — empirically prevents the
            # generator from settling near the legacy 0.25*floor that produced
            # std_ratio ~ 0.5-0.6 on high-vol channels (META/NFLX/NVDA/AMZN).
            std_pen = torch.relu(batch_x.std() * 0.50 - x_fake.std())
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
        # min_epochs bumped from max(20, patience*2) -> max(80, patience*3) so that
        # mode-collapsed high-vol channels (META/NFLX/NVDA/AMZN) cannot bail out at
        # ep ~30 — empirically the generator needs >50 epochs of W-distance training
        # before its per-channel std reaches the [0.50, 1.0] x target_std band.
        early_stop = EarlyStopping(
            patience=params.patience,
            min_epochs=0 if params.max_epochs <= 5 else max(80, params.patience * 3),
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
                # Inject the same std-floor into g_loss that we use in val_loss
                # so the generator receives a real variance gradient signal
                # (g_loss alone has no explicit variance term and the
                # W-distance critic can settle on a narrow distribution
                # that fooled the critic). 0.50*real_std floor matches val.
                std_pen = torch.relu(x_real.std() * 0.50 - x_fake.std())
                # 2.0× (not 5.0× like val_loss) so the variance term acts as
                # regularization, not a hammer. W-distance is still the primary
                # gradient signal.
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
        # Stash the metadata so generate() can pick up configuration knobs
        # like `pcf_gan_clamp_k` without threading them through every call.
        self._last_fit_metadata = dict(getattr(fit_input, "metadata", {}) or {})
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
            per_channel.append(channel.unsqueeze(-1))
            generator.cpu()

        data = torch.cat(per_channel, dim=-1).float()
        # Per-step safety clamp: bound every channel at ±k*target_std (k=4 by
        # default, override via metadata key "pcf_gan_clamp_k").  Without
        # this, co-occurrence of variance collapse + match_channel_moments
        # rescaling can drag an entire simulated path into always-negative
        # territory on a handful of channels (the symptom this commit
        # addresses). k=4 keeps paths inside ~4σ of train-mean, extreme but
        # not absurd; larger k is faithful, smaller k is safer.
        # default 6.0 (was 4.0) — empirically fat-tailed log returns routinely
        # produce |x| > 4σ events; 4σ hard-clip amputated them and dragged
        # std_ratio down to ~0.5 on META/NFLX/NVDA/AMZN/QQQ. 6σ still
        # numerically safe (P>|6σ| ≈ 2e-9 under normality) while preserving
        # the realistic heavy-tail distribution shape.
        clamp_k = float(
            (getattr(self, "_last_fit_metadata", {}) or {}).get("pcf_gan_clamp_k", 6.0)
        )
        if clamp_k > 0 and self.channel_stats:
            bound = clamp_k * torch.cat(
                [cs.std for cs in self.channel_stats]
            ).to(data)
            data = data.clamp(min=-bound.view(1, 1, -1), max=bound.view(1, 1, -1))
        return AdapterGenerateOutput(
            data=data,
            checkpoints=self.checkpoints,
            logs={"trainer": "pcf_gan_pathchar"},
            extra_metadata={"num_channels": data.shape[-1], "clamp_k": clamp_k},
        )
