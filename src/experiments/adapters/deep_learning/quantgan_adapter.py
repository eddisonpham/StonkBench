from __future__ import annotations



from pathlib import Path

from typing import Any, Dict, List



import torch



from src.experiments.adapters.base_adapter import ModelAdapter

from src.experiments.core.contracts import AdapterFitInput, AdapterGenerateOutput

from src.experiments.adapters.deep_learning.calibration import ChannelMomentStats, match_channel_moments

from src.experiments.adapters.deep_learning.training_utils import parse_training_params, use_calibration

from src.models.deep_learning.quantgan_module import QuantGANConfig, QuantGANTrainer







class QuantGANAdapter(ModelAdapter):

    def __init__(self) -> None:

        super().__init__()

        self.model_name = "QuantGAN"

        self.trainers: List[QuantGANTrainer] = []

        self.channel_stats: List[ChannelMomentStats] = []

        self.base_length = 1

        self.checkpoints: List[Path] = []
        self.apply_calibration = False



    def fit(self, fit_input: AdapterFitInput, checkpoints_dir: Path, logs_dir: Path) -> Dict[str, Any]:

        windows = fit_input.batch.train_windows

        valid_windows = fit_input.batch.valid_windows

        if windows is None or windows.ndim != 3:

            raise ValueError("QuantGANAdapter expects train_windows shaped (N, L, C)")

        if valid_windows is None or valid_windows.shape[0] == 0:

            raise ValueError("QuantGANAdapter requires non-empty valid_windows for model selection.")



        params = parse_training_params(fit_input)
        self.apply_calibration = use_calibration(fit_input)

        self.base_length = windows.shape[1]

        channels = windows.shape[2]

        self.trainers = []

        self.channel_stats = []

        self.checkpoints = []



        channel_val_losses: List[float] = []

        stopped_early = False

        best_epochs: List[int] = []



        for c in range(channels):

            channel_train = windows[:, :, c]

            channel_valid = valid_windows[:, :, c]

            self.channel_stats.append(ChannelMomentStats.from_univariate(channel_train))

            trainer = QuantGANTrainer(

                device=fit_input.device,

                cfg=QuantGANConfig(

                    epochs=params.max_epochs,

                    batch_size=max(8, min(params.batch_size, windows.shape[0])),

                    lr=params.learning_rate,

                    patience=params.patience,

                ),

            )

            fit_result = trainer.fit(channel_train, channel_valid)

            channel_val_losses.append(fit_result.best_val_loss)

            best_epochs.append(fit_result.best_epoch)

            stopped_early = stopped_early or fit_result.stopped_early

            self.trainers.append(trainer)

        self._is_fitted = True

        # Persist ONE consolidated FINAL checkpoint (multi-channel dict)
        # labeled with the seq length. SOLE disk artifact for (model, seq) —
        # matching the cleaner contract used by pcf_gan / utsd / cond_tsd.
        # Generate() reads in-memory self.trainers so per-channel files
        # were never reloaded. Schema key "state_dict" is shared across all
        # four multi-channel adapters.
        meta_ = fit_input.metadata or {}
        model_key_ = str(meta_.get("model_key", self.model_name))
        final_ckpt = checkpoints_dir / f"{model_key_}_seq{self.base_length}_final.pt"
        torch.save(
            {
                "model_name": model_key_,
                "num_channels": channels,
                "base_length": self.base_length,
                "channels": [
                    {"channel": c, "state_dict": t.generator.state_dict()}
                    for c, t in enumerate(self.trainers)
                ],
            },
            final_ckpt,
        )
        # Track the consolidated ckpt as the sole model checkpoint.
        self.checkpoints = [final_ckpt]

        return {

            "num_channels": channels,

            "best_val_loss": float(sum(channel_val_losses) / len(channel_val_losses)),

            "best_epoch": int(max(best_epochs)),

            "stopped_early": stopped_early,

        }



    def generate(self, num_samples: int, generation_length: int, seed: int) -> AdapterGenerateOutput:

        if not self._is_fitted:

            raise RuntimeError("Call fit() before generate().")



        per_channel = []

        for c, trainer in enumerate(self.trainers):
            channel = trainer.generate(num_samples, self.base_length, seed=seed + c)
            if self.apply_calibration and self.channel_stats:
                channel = match_channel_moments(channel, self.channel_stats[c])
            per_channel.append(channel.unsqueeze(-1))

        data = torch.cat(per_channel, dim=-1)

        return AdapterGenerateOutput(

            data=data.float(),

            checkpoints=self.checkpoints,

            logs={"trainer": "quantgan"},

            extra_metadata={"num_channels": data.shape[-1]},

        )


