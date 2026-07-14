from __future__ import annotations

import copy
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import torch

from src.experiments.adapters.base_adapter import ModelAdapter
from src.experiments.core.contracts import AdapterFitInput, AdapterGenerateOutput
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
from src.utils.artifact_utils import stitch_sequences


class TimeGANAdapter(ModelAdapter):
    """Real TimeGAN integration (one model per channel)."""

    # Warn when generated paths are near-constant in time (vs train temporal scale).
    _TEMPORAL_STD_WARN_FRAC = 0.02
    _TEMPORAL_STD_ABS_FLOOR = 1e-3

    def __init__(self) -> None:
        super().__init__()
        self.model_name = "TimeGAN"
        self.models: List = []
        self.checkpoints: List[Path] = []
        self.base_length = 1
        self.channel_stats: List[ChannelMomentStats] = []
        self.apply_calibration = False

    @staticmethod
    def _import_timegan():
        root = (
            Path(__file__).resolve().parents[3]
            / "models"
            / "deep_learning"
            / "TimeGAN-pytorch"
        )
        lib_root = root / "lib"
        for p in (root, lib_root):
            p_str = str(p)
            if p_str not in sys.path:
                sys.path.insert(0, p_str)
        import lib.timegan as timegan_module  # type: ignore
        from lib.timegan import TimeGAN  # type: ignore
        from utils import random_generator  # type: ignore

        orig_rg = timegan_module.random_generator

        def _patched_random_generator(
            batch_size, z_dim, T_mb, max_seq_len, mean: float = 0.0, std: float = 1.0
        ):
            if len(T_mb) < batch_size:
                T_mb = [int(max_seq_len)] * batch_size
            return orig_rg(batch_size, z_dim, T_mb, max_seq_len, mean, std)

        timegan_module.random_generator = _patched_random_generator
        return TimeGAN, random_generator

    def _build_opt(
        self,
        fit_input: AdapterFitInput,
        out_dir: Path,
        seq_len: int,
        z_dim: int,
        name: str,
        batch_size: int,
        learning_rate: float,
    ):
        device = resolve_device(fit_input.device)
        device_name = "cuda:0" if device.type == "cuda" else "cpu"
        # Vendor default hidden=24. Cap growth with L so L=100 does not explode capacity.
        hidden_dim = int(max(24, min(int(seq_len), 64)))
        meta = fit_input.metadata or {}
        num_layer = int(meta.get("timegan_num_layer", 3))
        return SimpleNamespace(
            manualseed=int(fit_input.seed),
            isTrain=True,
            data_name="custom",
            z_dim=int(z_dim),
            seq_len=int(seq_len),
            module="gru",
            hidden_dim=hidden_dim,
            num_layer=max(1, num_layer),
            iteration=1,
            batch_size=max(8, batch_size),
            metric_iteration=1,
            workers=0,
            device=device_name,
            gpu_ids=[0] if device.type == "cuda" else [],
            ngpu=1 if device.type == "cuda" else 0,
            model="TimeGAN",
            outf=str(out_dir),
            name=name,
            display=False,
            display_server="http://localhost",
            display_port=8097,
            display_id=0,
            print_freq=1000,
            load_weights=False,
            resume="",
            beta1=0.9,
            lr=learning_rate,
            w_gamma=1.0,
            w_es=0.1,
            w_e0=10.0,
            w_g=100.0,
        )

    @staticmethod
    def _normalize_like_model(arr: np.ndarray, model) -> np.ndarray:
        """Apply the same NormMinMax affine used at train time (train min/range)."""
        min_val = np.asarray(model.min_val, dtype=np.float32)
        max_val = np.asarray(model.max_val, dtype=np.float32)
        normed = (arr.astype(np.float32) - min_val) / (max_val + 1e-7)
        return np.clip(normed, 0.0, 1.0).astype(np.float32)

    @staticmethod
    @torch.no_grad()
    def _eval_val_loss(model, valid_np: np.ndarray, random_generator, batch_size: int) -> float:
        """Reconstruction + supervised loss on NormMinMax scale (matches training)."""
        model.nete.eval()
        model.netr.eval()
        model.nets.eval()
        model.netg.eval()

        losses: List[float] = []
        n = valid_np.shape[0]
        for start in range(0, n, batch_size):
            batch = valid_np[start : start + batch_size]
            x = torch.tensor(batch, dtype=torch.float32).to(model.device)
            h = model.nete(x)
            x_tilde = model.netr(h)
            err_er = model.l_mse(x_tilde, x)
            h_supervise = model.nets(h)
            err_s = model.l_mse(h_supervise[:, :-1, :], h[:, 1:, :])
            # Also score synthetic variance collapse after a cheap generation pass.
            z = random_generator(
                batch.shape[0],
                model.opt.z_dim,
                [batch.shape[1]] * batch.shape[0],
                batch.shape[1],
            )
            e_hat = model.netg(torch.tensor(np.asarray(z), dtype=torch.float32).to(model.device))
            h_hat = model.nets(e_hat)
            x_hat = model.netr(h_hat)
            # Prefer temporal std so constant-in-time paths are penalized.
            x_tstd = x.std(dim=1).mean()
            xhat_tstd = x_hat.std(dim=1).mean()
            std_pen = torch.relu(x_tstd * 0.5 - xhat_tstd)
            total = 10 * torch.sqrt(err_er) + 0.1 * err_s + 12.0 * std_pen
            losses.append(float(total.item()))
        return float(sum(losses) / max(len(losses), 1))

    def _train_channel(
        self,
        model,
        params,
        valid_np: np.ndarray,
        random_generator,
        steps_per_epoch: int | None = None,
        min_epochs: int | None = None,
    ) -> FitTrainingInfo:
        # Paper uses ~5e4 phase iterations; raise floor so L=100 is not starved.
        if steps_per_epoch is None:
            steps_per_epoch = max(60, model.data_num // max(model.opt.batch_size, 1))
        steps_per_epoch = max(1, int(steps_per_epoch))
        min_ep = (
            int(min_epochs)
            if min_epochs is not None
            else max(30, params.patience * 3)
        )
        early_stop = EarlyStopping(patience=params.patience, min_epochs=min_ep)
        best_state = {
            "nete": copy.deepcopy(model.nete.state_dict()),
            "netr": copy.deepcopy(model.netr.state_dict()),
            "netg": copy.deepcopy(model.netg.state_dict()),
            "netd": copy.deepcopy(model.netd.state_dict()),
            "nets": copy.deepcopy(model.nets.state_dict()),
        }
        info = FitTrainingInfo(best_val_loss=float("inf"), best_epoch=0, stopped_early=False)

        for epoch in range(params.max_epochs):
            # cuDNN RNN backward requires training mode; _eval_val_loss leaves nets in eval.
            model.nete.train()
            model.netr.train()
            model.nets.train()
            model.netg.train()
            model.netd.train()
            for _ in range(steps_per_epoch):
                model.train_one_iter_er()
            for _ in range(steps_per_epoch):
                model.train_one_iter_s()
            for _ in range(steps_per_epoch):
                for _ in range(2):
                    model.train_one_iter_g()
                    model.train_one_iter_er_()
                model.train_one_iter_d()

            val_loss = self._eval_val_loss(model, valid_np, random_generator, model.opt.batch_size)
            info.val_loss_history.append(val_loss)
            info.train_loss_history.append(val_loss)
            if val_loss < info.best_val_loss:
                info.best_val_loss = val_loss
                info.best_epoch = epoch + 1
                best_state = {
                    "nete": copy.deepcopy(model.nete.state_dict()),
                    "netr": copy.deepcopy(model.netr.state_dict()),
                    "netg": copy.deepcopy(model.netg.state_dict()),
                    "netd": copy.deepcopy(model.netd.state_dict()),
                    "nets": copy.deepcopy(model.nets.state_dict()),
                }
            if early_stop.step(val_loss, epoch + 1):
                info.stopped_early = True
                break

        model.nete.load_state_dict(best_state["nete"])
        model.netr.load_state_dict(best_state["netr"])
        model.netg.load_state_dict(best_state["netg"])
        model.netd.load_state_dict(best_state["netd"])
        model.nets.load_state_dict(best_state["nets"])
        return info

    @staticmethod
    def _winsorize_channel(arr: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
        """Clip extreme z-score outliers so TimeGAN min-max is not dominated by spikes."""
        flat = arr.reshape(-1)
        q_lo, q_hi = np.percentile(flat, [lo, hi])
        return np.clip(arr, q_lo, q_hi).astype(np.float32)

    @classmethod
    def _warn_if_collapsed(cls, channel: torch.Tensor, channel_idx: int, train_std: float) -> None:
        """Warn when generated paths are near-constant in time."""
        if channel.ndim != 2 or channel.shape[1] < 2:
            return
        temporal_std = channel.std(dim=1)
        mean_tstd = float(temporal_std.mean().item())
        frac_flat = float((temporal_std < 1e-4).float().mean().item())
        threshold = max(cls._TEMPORAL_STD_ABS_FLOOR, float(train_std) * cls._TEMPORAL_STD_WARN_FRAC)
        if mean_tstd < threshold or frac_flat > 0.5:
            warnings.warn(
                f"TimeGAN channel {channel_idx}: near-constant generations "
                f"(mean temporal std={mean_tstd:.4g}, frac_flat={frac_flat:.2f}, "
                f"train_std={train_std:.4g}). Check training / recovery saturation.",
                RuntimeWarning,
                stacklevel=2,
            )

    def fit(self, fit_input: AdapterFitInput, checkpoints_dir: Path, logs_dir: Path) -> Dict[str, Any]:
        windows = fit_input.batch.train_windows
        valid_windows = fit_input.batch.valid_windows
        if windows is None or windows.ndim != 3:
            raise ValueError("TimeGANAdapter expects train_windows shaped (N, L, C)")
        if valid_windows is None or valid_windows.shape[0] == 0:
            raise ValueError("TimeGANAdapter requires non-empty valid_windows for model selection.")

        TimeGAN, random_generator = self._import_timegan()
        params = parse_training_params(fit_input)
        self.apply_calibration = use_calibration(fit_input)
        self.base_length = windows.shape[1]
        num_channels = windows.shape[2]
        self.models = []
        self.checkpoints = []
        self.channel_stats = []

        channel_val_losses: List[float] = []
        stopped_early = False
        best_epochs: List[int] = []

        for c in range(num_channels):
            channel_windows = windows[:, :, c]
            channel_valid = valid_windows[:, :, c]
            self.channel_stats.append(ChannelMomentStats.from_univariate(channel_windows))
            train_np = self._winsorize_channel(
                channel_windows.unsqueeze(-1).detach().cpu().numpy().astype(np.float32)
            )
            valid_raw = self._winsorize_channel(
                channel_valid.unsqueeze(-1).detach().cpu().numpy().astype(np.float32)
            )
            # Cap batch size — TimeGAN with bs=256 on ~1.5k windows collapses easily.
            effective_bs = max(16, min(int(params.batch_size), 128, int(windows.shape[0])))
            opt = self._build_opt(
                fit_input=fit_input,
                out_dir=logs_dir,
                seq_len=self.base_length,
                z_dim=1,
                name=f"timegan_channel_{c + 1}",
                batch_size=effective_bs,
                learning_rate=params.learning_rate,
            )
            model = TimeGAN(opt, train_np)
            # Critical: early-stop on the same [0,1] scale the nets train/recover in.
            # Comparing sigmoid recovery to raw z-scores prefers near-zero constants.
            valid_np = self._normalize_like_model(valid_raw, model)
            meta = fit_input.metadata or {}
            info = self._train_channel(
                model,
                params,
                valid_np,
                random_generator,
                steps_per_epoch=meta.get("timegan_steps_per_epoch"),
                min_epochs=meta.get("timegan_min_epochs"),
            )
            channel_val_losses.append(info.best_val_loss)
            best_epochs.append(info.best_epoch)
            stopped_early = stopped_early or info.stopped_early
            model.save_weights(info.best_epoch)
            self.models.append(model)

            ckpt = checkpoints_dir / f"timegan_checkpoint_{c + 1}.pt"
            torch.save(
                {
                    "channel": c,
                    "encoder": model.nete.state_dict(),
                    "generator": model.netg.state_dict(),
                    "supervisor": model.nets.state_dict(),
                    "recovery": model.netr.state_dict(),
                    "discriminator": model.netd.state_dict(),
                    "min_val": np.asarray(model.min_val),
                    "max_val": np.asarray(model.max_val),
                    "hidden_dim": int(model.opt.hidden_dim),
                    "seq_len": int(model.opt.seq_len),
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
        if not self._is_fitted:
            raise RuntimeError("Call fit() before generate().")

        np.random.seed(seed)
        torch.manual_seed(seed)
        per_channel = []
        for c, model in enumerate(self.models):
            generated = model.generation(num_samples=num_samples, mean=0.0, std=1.0)
            channel_np = np.stack([np.asarray(x).squeeze(-1) for x in generated], axis=0)
            channel = torch.from_numpy(channel_np).float()
            train_std = float(self.channel_stats[c].std.reshape(-1)[0].item())
            self._warn_if_collapsed(channel, c, train_std)
            if self.apply_calibration:
                # Use (N, L, 1) so calibration matches global channel moments, not per-t.
                channel = match_channel_moments(
                    channel.unsqueeze(-1), self.channel_stats[c]
                ).squeeze(-1)
            if generation_length != channel.shape[1]:
                channel = stitch_sequences(channel, generation_length, seed=seed + c)
            per_channel.append(channel.unsqueeze(-1))

        data = torch.cat(per_channel, dim=-1)
        return AdapterGenerateOutput(
            data=data,
            checkpoints=self.checkpoints,
            logs={"trainer": "timegan_real"},
            extra_metadata={"num_channels": data.shape[-1]},
        )
