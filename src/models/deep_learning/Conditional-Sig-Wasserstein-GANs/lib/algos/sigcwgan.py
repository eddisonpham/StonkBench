from dataclasses import dataclass

import torch
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import optim

from lib.algos.base import BaseAlgo, BaseConfig
from lib.augmentations import SignatureConfig
from lib.augmentations import augment_path_and_compute_signatures
from lib.utils import sample_indices, to_numpy


def sigcwgan_loss(sig_pred: torch.Tensor, sig_fake_conditional_expectation: torch.Tensor):
    return torch.norm(sig_pred - sig_fake_conditional_expectation, p=2, dim=1).mean()


@dataclass
class SigCWGANConfig:
    mc_size: int
    sig_config_future: SignatureConfig
    sig_config_past: SignatureConfig

    def compute_sig_past(self, x):
        return augment_path_and_compute_signatures(x, self.sig_config_past)

    def compute_sig_future(self, x):
        return augment_path_and_compute_signatures(x, self.sig_config_future)


def fit_sigw1_calibrator(config, x_future, x_past, alpha: float = 0.0):
    """Fit the vendor conditional-signature baseline and return model + tensors.

    ``alpha=0`` preserves the original vendor ``LinearRegression`` behavior.
    A positive alpha uses Ridge regression, which is materially safer for the
    high-dimensional multivariate signature space used by StonkBench.
    """
    sigs_past = config.compute_sig_past(x_past)
    sigs_future = config.compute_sig_future(x_future)
    assert sigs_past.size(0) == sigs_future.size(0)
    X, Y = to_numpy(sigs_past), to_numpy(sigs_future)
    if alpha > 0.0:
        # Signature coordinates span very different numerical scales. Scale
        # predictors before Ridge so alpha has a consistent meaning across
        # first- and higher-order signature terms.
        model = make_pipeline(
            StandardScaler(), Ridge(alpha=float(alpha), fit_intercept=True)
        )
    else:
        model = LinearRegression()
    model.fit(X, Y)
    sigs_pred = torch.from_numpy(model.predict(X)).float().to(x_future.device)
    return model, sigs_pred


def _predict_calibrated(config, model, x_future, x_past):
    """Apply a previously fitted vendor/Ridge calibrator to new windows."""
    sigs_past = config.compute_sig_past(x_past)
    X = to_numpy(sigs_past)
    return torch.from_numpy(model.predict(X)).float().to(x_future.device)


def calibrate_sigw1_metric(config, x_future, x_past, alpha: float = 0.0):
    """Backward-compatible vendor helper returning in-sample predictions."""
    _, sigs_pred = fit_sigw1_calibrator(config, x_future, x_past, alpha=alpha)
    return sigs_pred


def sample_sig_fake(G, q, sig_config, x_past):
    x_past_mc = x_past.repeat(sig_config.mc_size, 1, 1).requires_grad_()
    x_fake = G.sample(q, x_past_mc)
    sigs_fake_future = sig_config.compute_sig_future(x_fake)
    sigs_fake_ce = sigs_fake_future.reshape(sig_config.mc_size, x_past.size(0), -1).mean(0)
    return sigs_fake_ce, x_fake


class SigCWGAN(BaseAlgo):
    def __init__(
            self,
            base_config: BaseConfig,
            config: SigCWGANConfig,
            x_real: torch.Tensor,
            calibration_alpha: float = 0.0,
            learning_rate: float = 1e-2,
    ):
        super(SigCWGAN, self).__init__(base_config, x_real)
        self.sig_config = config
        self.mc_size = config.mc_size
        self.calibration_alpha = float(calibration_alpha)
        self.learning_rate = float(learning_rate)

        self.x_past = x_real[:, :self.p]
        x_future = x_real[:, self.p:]
        self.calibration_model, self.sigs_pred = fit_sigw1_calibrator(
            config, x_future, self.x_past, alpha=self.calibration_alpha
        )

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate)
        self.G_scheduler = optim.lr_scheduler.StepLR(self.G_optimizer, step_size=100, gamma=0.9)

    def sample_batch(self, ):
        random_indices = sample_indices(
            self.sigs_pred.shape[0], self.batch_size, device=self.sigs_pred.device
        )  # sample indices
        # sample the least squares signature and the log-rtn condition
        sigs_pred = self.sigs_pred[random_indices].clone().to(self.device)
        x_past = self.x_past[random_indices].clone().to(self.device)
        return sigs_pred, x_past

    def step(self):
        self.G.train()
        self.G_optimizer.zero_grad()  # empty 'cache' of gradients
        sigs_pred, x_past = self.sample_batch()
        sigs_fake_ce, x_fake = sample_sig_fake(self.G, self.q, self.sig_config, x_past)
        loss = sigcwgan_loss(sigs_pred, sigs_fake_ce)
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(self.G.parameters(), 10)
        self.training_loss['loss'].append(loss.item())
        self.training_loss['total_norm'].append(total_norm)
        self.G_optimizer.step()
        self.G_scheduler.step()  # decaying learning rate slowly.
        self.evaluate(x_fake)
