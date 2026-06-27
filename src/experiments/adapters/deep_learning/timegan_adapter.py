from __future__ import annotations

import copy
import sys
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
    shuffle_time_within_windows,
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
            return orig_rg(batch_size, z_dim, T_mb, max_seq_len)

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
        return SimpleNamespace(
            manualseed=int(fit_input.seed),
            isTrain=True,
            data_name="custom",
            z_dim=int(z_dim),
            seq_len=int(seq_len),
            module="gru",
            hidden_dim=max(16, int(seq_len)),
            num_layer=3,
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
    @torch.no_grad()
    def _eval_val_loss(model, valid_np: np.ndarray, random_generator, batch_size: int) -> float:
        model.nete.eval()
        model.netr.eval()
        model.nets.eval()
        model.netg.eval()
        model.netd.eval()

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
            z = random_generator(batch.shape[0], model.opt.z_dim, [batch.shape[1]] * batch.shape[0], batch.shape[1])
            e_hat = model.netg(torch.tensor(z, dtype=torch.float32).to(model.device))
            h_hat = model.nets(e_hat)
            x_hat = model.netr(h_hat)
            y_fake = model.netd(h_hat)
            err_g_u = model.l_bce(y_fake, torch.ones_like(y_fake))
            err_g_v1 = torch.mean(
                torch.abs(
                    torch.sqrt(torch.std(x_hat, [0])[1] + 1e-6)
                    - torch.sqrt(torch.std(x, [0])[1] + 1e-6)
                )
            )
            err_g_v2 = torch.mean(torch.abs(torch.mean(x_hat, [0])[0] - torch.mean(x, [0])[0]))
            err_g = err_g_u + err_g_v1 * model.opt.w_g + err_g_v2 * model.opt.w_g + torch.sqrt(err_s)
            total = 10 * torch.sqrt(err_er) + 0.1 * err_s + err_g
            losses.append(float(total.item()))
        return float(sum(losses) / max(len(losses), 1))

    def _train_channel(
        self,
        model,
        params,
        valid_np: np.ndarray,
        random_generator,
    ) -> FitTrainingInfo:
        steps_per_epoch = max(10, model.data_num // model.opt.batch_size)
        early_stop = EarlyStopping(patience=params.patience)
        best_state = {
            "nete": copy.deepcopy(model.nete.state_dict()),
            "netr": copy.deepcopy(model.netr.state_dict()),
            "netg": copy.deepcopy(model.netg.state_dict()),
            "netd": copy.deepcopy(model.netd.state_dict()),
            "nets": copy.deepcopy(model.nets.state_dict()),
        }
        info = FitTrainingInfo(best_val_loss=float("inf"), best_epoch=0, stopped_early=False)

        for epoch in range(params.max_epochs):
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
            train_np = channel_windows.unsqueeze(-1).detach().cpu().numpy().astype(np.float32)
            valid_np = channel_valid.unsqueeze(-1).detach().cpu().numpy().astype(np.float32)
            opt = self._build_opt(
                fit_input=fit_input,
                out_dir=logs_dir,
                seq_len=self.base_length,
                z_dim=1,
                name=f"timegan_channel_{c + 1}",
                batch_size=params.batch_size,
                learning_rate=params.learning_rate,
            )
            model = TimeGAN(opt, train_np)
            info = self._train_channel(model, params, valid_np, random_generator)
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
            if self.apply_calibration:
                channel = match_channel_moments(channel, self.channel_stats[c])
            if generation_length != channel.shape[1]:
                channel = stitch_sequences(channel, generation_length, seed=seed + c)
            per_channel.append(channel.unsqueeze(-1))

        data = torch.cat(per_channel, dim=-1)
        if self.apply_calibration:
            data = shuffle_time_within_windows(data, seed)
        return AdapterGenerateOutput(
            data=data,
            checkpoints=self.checkpoints,
            logs={"trainer": "timegan_real"},
            extra_metadata={"num_channels": data.shape[-1]},
        )
