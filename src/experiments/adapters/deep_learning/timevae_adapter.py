from __future__ import annotations



import copy

import sys

from pathlib import Path

from typing import Any, Dict



import torch

import torch.optim as optim

import yaml

from torch.utils.data import DataLoader, TensorDataset



from src.experiments.adapters.base_adapter import ModelAdapter

from src.experiments.core.contracts import AdapterFitInput, AdapterGenerateOutput

from src.experiments.adapters.deep_learning.calibration import ChannelMomentStats, match_channel_moments

from src.experiments.adapters.deep_learning.training_utils import (

    EarlyStopping,

    FitTrainingInfo,

    parse_training_params,

    resolve_device,

    use_calibration,

)

from src.utils.artifact_utils import stitch_sequences





class TimeVAEAdapter(ModelAdapter):

    """Real TimeVAE integration using timeVAE-pytorch-main implementation."""



    def __init__(self) -> None:

        super().__init__()

        self.model_name = "TimeVAE"

        self.model = None

        self.base_length = 1

        self.num_channels = 1

        self.checkpoint: Path | None = None

        self.channel_stats: ChannelMomentStats | None = None
        self.apply_calibration = False



    @staticmethod

    def _import_timevae_tools():        # parents[4] = repo root (NOT src/). Vendored TimeVAE package lives at
        # <repo>/models/deep_learning/timeVAE-pytorch-main, NOT under src/.
        root = (
            Path(__file__).resolve().parents[4]
            / "models"
            / "deep_learning"
            / "timeVAE-pytorch-main"
            / "src"
        )

        root_str = str(root)

        if root_str not in sys.path:

            sys.path.insert(0, root_str)



        from vae.vae_utils import get_prior_samples, instantiate_vae_model  # type: ignore



        return instantiate_vae_model, get_prior_samples



    @staticmethod

    def _load_hparams() -> dict:

        hp_path = (

            Path(__file__).resolve().parents[3]

            / "models"

            / "deep_learning"

            / "timeVAE-pytorch-main"

            / "src"

            / "config"

            / "hyperparameters.yaml"

        )

        with hp_path.open("r", encoding="utf-8") as f:

            cfg = yaml.safe_load(f)

        return cfg["timeVAE"]



    @staticmethod

    def _eval_elbo(model, loader: DataLoader) -> float:

        model.eval()

        total = 0.0

        count = 0

        with torch.no_grad():

            for (batch,) in loader:

                z_mean, z_log_var, z = model.encoder(batch)

                reconstruction = model.decoder(z)

                loss, _, _ = model.loss_function(batch, reconstruction, z_mean, z_log_var)

                total += float((loss / batch.size(0)).item())

                count += 1

        return total / max(count, 1)



    def _train_with_validation(

        self,

        model,

        train_np,

        valid_np,

        params,

        device: torch.device,

    ) -> FitTrainingInfo:

        train_tensor = torch.FloatTensor(train_np).to(device)

        valid_tensor = torch.FloatTensor(valid_np).to(device)

        train_loader = DataLoader(

            TensorDataset(train_tensor),

            batch_size=max(1, min(params.batch_size, train_tensor.shape[0])),

            shuffle=True,

        )

        valid_loader = DataLoader(

            TensorDataset(valid_tensor),

            batch_size=max(1, min(params.batch_size, valid_tensor.shape[0])),

            shuffle=False,

        )

        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

        early_stop = EarlyStopping(patience=params.patience)

        best_state = copy.deepcopy(model.state_dict())

        info = FitTrainingInfo(best_val_loss=float("inf"), best_epoch=0, stopped_early=False)



        for epoch in range(params.max_epochs):

            model.train()

            train_loss = 0.0

            for (batch,) in train_loader:

                optimizer.zero_grad()

                z_mean, z_log_var, z = model.encoder(batch)

                reconstruction = model.decoder(z)

                loss, _, _ = model.loss_function(batch, reconstruction, z_mean, z_log_var)

                loss = loss / batch.size(0)

                loss.backward()

                optimizer.step()

                train_loss += float(loss.item())

            info.train_loss_history.append(train_loss / max(len(train_loader), 1))



            val_loss = self._eval_elbo(model, valid_loader)

            info.val_loss_history.append(val_loss)

            if val_loss < info.best_val_loss:

                info.best_val_loss = val_loss

                info.best_epoch = epoch + 1

                best_state = copy.deepcopy(model.state_dict())

            if early_stop.step(val_loss, epoch + 1):

                info.stopped_early = True

                break



        model.load_state_dict(best_state)

        return info



    def fit(self, fit_input: AdapterFitInput, checkpoints_dir: Path, logs_dir: Path) -> Dict[str, Any]:

        windows = fit_input.batch.train_windows

        valid_windows = fit_input.batch.valid_windows

        if windows is None or windows.ndim != 3:

            raise ValueError("TimeVAEAdapter expects train_windows shaped (N, L, C)")

        if valid_windows is None or valid_windows.shape[0] == 0:

            raise ValueError("TimeVAEAdapter requires non-empty valid_windows for model selection.")



        instantiate_vae_model, _ = self._import_timevae_tools()

        hparams = self._load_hparams()

        params = parse_training_params(fit_input)

        hparams["batch_size"] = max(8, min(params.batch_size, windows.shape[0]))



        train_np = windows.detach().cpu().numpy()

        valid_np = valid_windows.detach().cpu().numpy()

        self.channel_stats = ChannelMomentStats.from_windows(windows)
        self.apply_calibration = use_calibration(fit_input)

        self.base_length = train_np.shape[1]

        self.num_channels = train_np.shape[2]

        device = resolve_device(fit_input.device)

        self.model = instantiate_vae_model(

            vae_type="timeVAE",

            sequence_length=self.base_length,

            feature_dim=self.num_channels,

            **hparams,

        ).to(device)



        training_info = self._train_with_validation(self.model, train_np, valid_np, params, device)



        self.checkpoint = checkpoints_dir / "timevae_checkpoint.pt"

        torch.save(self.model.state_dict(), self.checkpoint)

        self._is_fitted = True

        return training_info.as_dict(num_channels=self.num_channels)



    def generate(self, num_samples: int, generation_length: int, seed: int) -> AdapterGenerateOutput:

        if not self._is_fitted or self.model is None:

            raise RuntimeError("Call fit() before generate().")



        torch.manual_seed(seed)

        _, get_prior_samples = self._import_timevae_tools()

        samples = get_prior_samples(self.model, num_samples=num_samples)

        data = torch.from_numpy(samples).float()

        if self.apply_calibration and self.channel_stats is not None:

            data = match_channel_moments(data, self.channel_stats)

        if generation_length != data.shape[1]:

            data = stitch_sequences(data, generation_length, seed=seed)



        checkpoints = [self.checkpoint] if self.checkpoint is not None else []

        return AdapterGenerateOutput(

            data=data,

            checkpoints=checkpoints,

            logs={"trainer": "timevae_real"},

            extra_metadata={"num_channels": data.shape[-1]},

        )


