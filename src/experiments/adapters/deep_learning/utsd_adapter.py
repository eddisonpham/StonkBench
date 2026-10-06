"""Unconditional time-series diffusion adapter — native multivariate.

Trains a single TSDiff model on the full (N, L, C) multivariate input.
The backbone's ``input_init`` is a ``nn.Linear(input_dim, hidden_dim)`` that
processes each timestep independently across channels, so setting
``input_dim=C`` gives native multivariate denoising.  The S4 residual blocks
use ``Conv1d`` across the time axis, preserving cross-channel structure.

Key change vs the old per-channel adapter: ``sample_n(return_lags=True)``
returns the full ``(R, L, input_dim)`` tensor instead of the hardcoded
``samples[..., 0]`` slice.
"""
from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from src.experiments.adapters.base_adapter import ModelAdapter
from src.experiments.adapters.deep_learning.training_utils import (
    EarlyStopping,
    FitTrainingInfo,
    make_loader,
    parse_training_params,
    reconcile_gluonts,
    resolve_device,
    resolve_final_checkpoint,
)
from src.experiments.core.contracts import AdapterFitInput, AdapterGenerateOutput
from src.utils.epoch_events import emit_epoch


class UnconditionalTSDiffusionAdapter(ModelAdapter):
    """Native-multivariate unconditional time-series diffusion adapter.

    Trains one diffusion model on the full (N, L, C) tensor.  The backbone
    accepts any ``input_dim`` so multivariate works out of the box.
    """

    can_regenerate_from_checkpoint = True

    def __init__(self) -> None:
        super().__init__()
        self.model_name = "UnconditionalTSDiffusion"
        self.model: Any = None
        self.num_channels: int = 0
        self.base_length: int = 0
        self.context_length: int = 0
        self.prediction_length: int = 0
        self.checkpoints: List[Path] = []

    # ------------------------------------------------------------------
    # Vendor bootstrap (unchanged — sets up sys.path + gluonts shim)
    # ------------------------------------------------------------------
    @staticmethod
    def _import_utsd():
        # Cross-vendor hygiene: if TimeGrad already ran in this process its
        # vendored gluonts shadows the installed one that uncond_ts_diff
        # needs (ImportError: DataEntry). Drop it before importing.
        reconcile_gluonts(
            Path(__file__).resolve().parents[3] / "models" / "deep_learning" / "timegrad",
            prefer_vendor=False,
        )
        root = (
            Path(__file__).resolve().parents[3]
            / "models"
            / "deep_learning"
            / "unconditional-time-series-diffusion"
            / "src"
        )
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        pkg_root = root / "uncond_ts_diff"
        model_root = pkg_root / "model"
        diffusion_root = model_root / "diffusion"

        if "gluonts.torch.modules.scaler" not in sys.modules:
            try:
                from gluonts.torch.scaler import MeanScaler, NOPScaler  # type: ignore

                scaler_mod = ModuleType("gluonts.torch.modules.scaler")
                scaler_mod.MeanScaler = MeanScaler
                scaler_mod.NOPScaler = NOPScaler
                sys.modules["gluonts.torch.modules.scaler"] = scaler_mod
            except Exception:
                pass

        import uncond_ts_diff.configs as diffusion_configs  # type: ignore

        model_pkg = ModuleType("uncond_ts_diff.model")
        model_pkg.__path__ = [str(model_root)]  # type: ignore[attr-defined]
        sys.modules["uncond_ts_diff.model"] = model_pkg
        diffusion_pkg = ModuleType("uncond_ts_diff.model.diffusion")
        diffusion_pkg.__path__ = [str(diffusion_root)]  # type: ignore[attr-defined]
        sys.modules["uncond_ts_diff.model.diffusion"] = diffusion_pkg

        tsdiff_module = importlib.import_module(
            "uncond_ts_diff.model.diffusion.tsdiff"
        )
        return diffusion_configs, tsdiff_module.TSDiff

    # ------------------------------------------------------------------
    # Model construction hook — subclasses override for TSDiffCond etc.
    # ------------------------------------------------------------------
    def _build_model(
        self,
        backbone_params: Dict[str, Any],
        cfg: Dict[str, Any],
        context_length: int,
        prediction_length: int,
        lr: float,
        device: torch.device,
    ) -> Any:
        """Build the diffusion model.  Subclasses may override to swap
        the model class (e.g. TSDiffCond) or add kwargs."""
        diffusion_configs, TSDiff = self._import_utsd()
        return TSDiff(
            backbone_parameters=backbone_params,
            timesteps=cfg["timesteps"],
            diffusion_scheduler=cfg["diffusion_scheduler"],
            freq="h",
            use_features=False,
            use_lags=False,
            normalization="none",
            context_length=context_length,
            prediction_length=prediction_length,
            lr=lr,
            init_skip=True,
        ).to(device)

    # ------------------------------------------------------------------
    # Validation loss (operates on full multivariate batch)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval_val_loss(
        self, model: Any, loader: DataLoader, device: torch.device
    ) -> float:
        model.eval()
        total = 0.0
        count = 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)  # (B, L, C)
            t = torch.randint(
                0, model.timesteps, (batch_x.shape[0],), device=device
            ).long()
            loss, _, _ = model.p_losses(batch_x, t, features=None, loss_type="l2")
            total += float(loss.item())
            count += 1
        return total / max(count, 1)

    # ------------------------------------------------------------------
    # fit — single joint model on (N, L, C)
    # ------------------------------------------------------------------
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
                "UnconditionalTSDiffusionAdapter expects train_windows shaped (N, L, C)"
            )
        if valid_windows is None or valid_windows.shape[0] == 0:
            raise ValueError(
                "UnconditionalTSDiffusionAdapter requires non-empty valid_windows."
            )

        diffusion_configs, _TSDiff_cls = self._import_utsd()
        params = parse_training_params(fit_input)

        self.base_length = int(windows.shape[1])
        self.num_channels = int(windows.shape[2])
        self.context_length = max(2, self.base_length // 2)
        self.prediction_length = max(1, self.base_length - self.context_length)

        device = resolve_device(fit_input.device)

        # --- Build single joint model with input_dim=C ---
        cfg = copy.deepcopy(diffusion_configs.diffusion_small_config)
        backbone_params = cfg["backbone_parameters"].copy()
        backbone_params["input_dim"] = self.num_channels
        backbone_params["output_dim"] = self.num_channels

        model = self._build_model(
            backbone_params=backbone_params,
            cfg=cfg,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            lr=params.learning_rate,
            device=device,
        )

        # --- Data loaders (full multivariate, no per-channel split) ---
        train_loader = make_loader(
            windows.float(), params.batch_size, shuffle=True
        )
        valid_loader = make_loader(
            valid_windows.float(), params.batch_size, shuffle=False
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
        # Vendor uses gradient_clip_val=0.5 (train_tsdiff.yaml); variant hook for utsd_gclip lets
        # the smoke/HPC engine widen the clip (0.5 → 2.0 or None) so the diffusion model can
        # actually move off the flat-line attractor on multivariate data.
        _meta = fit_input.metadata or {}
        grad_clip_raw = _meta.get("grad_clip", 0.5)
        grad_clip = None if grad_clip_raw in (None, "None") else float(grad_clip_raw)
        # Vendor uses num_batches_per_epoch=128 — each epoch samples 128
        # random batches rather than processing the full dataset. This
        # matches the original Lightning-based training loop.
        num_batches_per_epoch = 128
        # min_epochs guard mirrors vrnn/kalman/quantgan: without it the
        # diffusion models early-stop on noisy validation before ever seeing
        # the improvement window (production runs stopped at best_epoch=2;
        # with this guard the mini run's val curve kept improving through
        # epoch 21). Same formula as the sibling adapters.
        early_stop = EarlyStopping(
            patience=params.patience, min_epochs=max(20, params.patience * 2)
        )
        best_state = copy.deepcopy(model.state_dict())
        info = FitTrainingInfo(
            best_val_loss=float("inf"), best_epoch=0, stopped_early=False
        )

        for epoch in range(params.max_epochs):
            model.train()
            train_loss = 0.0
            batch_count = 0
            # Infinite iterator to sample num_batches_per_epoch batches
            train_iter = iter(train_loader)
            for _ in range(num_batches_per_epoch):
                try:
                    (batch_x,) = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    (batch_x,) = next(train_iter)
                batch_x = batch_x.to(device)
                optimizer.zero_grad()
                t = torch.randint(
                    0, model.timesteps, (batch_x.shape[0],), device=device
                ).long()
                loss, _, _ = model.p_losses(
                    batch_x, t, features=None, loss_type="l2"
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                train_loss += float(loss.item())
                batch_count += 1
            info.train_loss_history.append(train_loss / max(batch_count, 1))

            val_loss = self._eval_val_loss(model, valid_loader, device)
            info.val_loss_history.append(val_loss)
            # Live training-curve event (no-op unless $STONKBENCH_MINI_EVENTS set).
            emit_epoch(
                str(_meta.get("model_key", self.model_name)),
                epoch + 1,
                train_loss=train_loss / max(batch_count, 1),
                val_loss=val_loss,
            )
            if val_loss < info.best_val_loss:
                info.best_val_loss = val_loss
                info.best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
            if early_stop.step(val_loss, epoch + 1):
                info.stopped_early = True
                break

        model.load_state_dict(best_state)
        self.model = model

        # --- Persist single checkpoint ---
        meta_ = fit_input.metadata or {}
        model_key_ = str(meta_.get("model_key", self.model_name))
        final_ckpt = checkpoints_dir / f"{model_key_}_seq{self.base_length}_final.pt"
        torch.save(
            {
                "model_name": model_key_,
                "num_channels": self.num_channels,
                "base_length": self.base_length,
                "context_length": self.context_length,
                "prediction_length": self.prediction_length,
                "input_dim": self.num_channels,
                "state_dict": model.state_dict(),
            },
            final_ckpt,
        )
        self.checkpoints = [final_ckpt]
        self._is_fitted = True

        return {
            "num_channels": self.num_channels,
            "best_val_loss": info.best_val_loss,
            "best_epoch": info.best_epoch,
            "stopped_early": info.stopped_early,
            "train_loss_history": [float(x) for x in info.train_loss_history],
            "val_loss_history": [float(x) for x in info.val_loss_history],
        }

    # ------------------------------------------------------------------
    # load_state — rebuild from a consolidated ``*_seq{L}_final.pt``
    # ------------------------------------------------------------------
    def load_state(self, checkpoints: List[Path]) -> Dict[str, Any]:
        """Rebuild the diffusion model from the persisted training config.

        The checkpoint stores ``context_length`` / ``prediction_length`` /
        ``input_dim`` alongside the weights, so the model is reconstructed
        with the exact ``diffusion_small_config`` hyper-parameters ``fit``
        used (the vendored config is the only source of timesteps / scheduler /
        backbone shape) and the state is loaded verbatim.

        Inherited by :class:`ConditionalTSDiffusionAdapter`, whose
        ``_import_utsd`` / ``_build_model`` overrides swap in ``TSDiffCond``.
        """
        if not checkpoints:
            raise ValueError("TSDiffusionAdapter.load_state received an empty checkpoint list.")
        path = resolve_final_checkpoint(checkpoints)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise ValueError(
                f"{path.name} is not a consolidated TSDiffusion checkpoint "
                "(missing 'state_dict')."
            )

        diffusion_configs, _ = self._import_utsd()
        device = resolve_device()

        self.num_channels = int(payload["num_channels"])
        self.base_length = int(payload["base_length"])
        self.context_length = int(payload["context_length"])
        self.prediction_length = int(payload["prediction_length"])
        input_dim = int(payload.get("input_dim", self.num_channels))

        # Mirror fit(): deepcopy the vendored small config and override only
        # the channel dims. lr is optimizer-only (no state_dict impact).
        cfg = copy.deepcopy(diffusion_configs.diffusion_small_config)
        backbone_params = cfg["backbone_parameters"].copy()
        backbone_params["input_dim"] = input_dim
        backbone_params["output_dim"] = input_dim

        model = self._build_model(
            backbone_params=backbone_params,
            cfg=cfg,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            lr=1e-3,
            device=device,
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()

        self.model = model
        self.checkpoints = [path]
        self._is_fitted = True
        return {
            "restored": True,
            "checkpoint": str(path),
            "num_channels": self.num_channels,
            "base_length": self.base_length,
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            "input_dim": input_dim,
            "device": str(device),
        }

    # ------------------------------------------------------------------
    # generate — sample_n(return_lags=True) gives full (R, L, C)
    # ------------------------------------------------------------------
    def generate(
        self, num_samples: int, generation_length: int, seed: int
    ) -> AdapterGenerateOutput:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Call fit() before generate().")

        torch.manual_seed(seed)
        self.model.eval()
        with torch.no_grad():
            # return_lags=True → full (R, seq_len, input_dim) array
            sampled = self.model.sample_n(
                num_samples=num_samples, return_lags=True
            )

        data = torch.from_numpy(sampled).float()  # (R, seq_len, C)
        # Trim to requested generation_length
        L = min(generation_length, data.shape[1])
        data = data[:, :L, :]

        return AdapterGenerateOutput(
            data=data,
            checkpoints=self.checkpoints,
            logs={"trainer": "utsd_multivariate"},
            extra_metadata={"num_channels": data.shape[-1]},
        )
