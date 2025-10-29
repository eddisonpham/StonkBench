"""
Feature-based metrics and visualizations for evaluating the fidelity of generated time series data.

This module provides a set of metrics and visualizations for assessing how well generated data
matches the statistical and distributional characteristics of real/original data, focusing on
feature-level properties and summary statistics.

Feature-based Metrics:
- Marginal Distribution Distance (MDD): Histogram-based distance between real and generated data.
- Mean Distance (MD): Difference in means between real and generated data.
- Standard Deviation Distance (SDD): Difference in standard deviations.
- Skewness Distance (SD): Difference in skewness.
- Kurtosis Distance (KD): Difference in kurtosis.
- Autocorrelation Distance (ACD): Difference in autocorrelation structure.

Visualizations:
- t-SNE visualization: 2D projection to compare overall structure of real and generated samples.
- Marginal Distribution Plot: Kernel density estimate (KDE) comparing sample distributions.

All metrics are implemented as PyTorch modules for easy integration with deep learning workflows.
"""

import os
import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.math_utils import (
    histogram_torch,
    acf_torch,
    non_stationary_acf_torch,
    skew_torch,
    kurtosis_torch,
    acf_diff,
)
from src.utils.conversion_utils import (
    to_torch_features_abc,
    to_numpy_features_for_visualization,
)


class Loss(nn.Module):
    """
    Base class for feature-based loss metrics.

    Args:
        name (str): Name of the loss.
        reg (float): Regularization coefficient (default: 1.0).
        transform (callable): Optional transform to apply to input data.
        threshold (float): Success threshold for the loss.
        backward (bool): Whether to allow backward computation.
        norm_foo (callable): Function to aggregate or normalize the loss.
    """
    def __init__(self, name, reg=1.0, transform=lambda x: x, threshold=10., backward=False, norm_foo=lambda x: x):
        super(Loss, self).__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo

    def forward(self, x_fake):
        """
        Compute the loss given generated (fake) data.

        Args:
            x_fake (torch.Tensor): Generated data.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake):
        """
        Compute the componentwise loss. To be implemented by subclasses.
        """
        raise NotImplementedError()

    @property
    def success(self):
        """
        Whether the loss is below the threshold for all components.
        """
        return torch.all(self.loss_componentwise <= self.threshold)

# =======================================
# Marginal Distribution Distance (MDD)
class HistoLoss(Loss):
    """
    Marginal Distribution Distance (MDD) loss based on histogram comparison.

    Args:
        x_real (torch.Tensor): Real/original data of shape (batch, time, features).
        n_bins (int): Number of histogram bins.
        **kwargs: Additional arguments for Loss.
    """
    def __init__(self, x_real, n_bins, **kwargs):
        super(HistoLoss, self).__init__(**kwargs)
        self.n_bins = n_bins
        self.densities = list()
        self.locs = list()
        self.deltas = list()
        # Compute one histogram per feature, aggregating over all timesteps
        for i in range(x_real.shape[2]):
            # Flatten all timesteps for this feature
            x_i = x_real[:, :, i].reshape(-1, 1)
            d, b = histogram_torch(x_i, n_bins, density=True)
            self.densities.append(nn.Parameter(d).to(x_real.device))
            delta = b[1:2] - b[:1]
            loc = 0.5 * (b[1:] + b[:-1])
            self.locs.append(loc)
            self.deltas.append(delta)

    def compute(self, x_fake):
        """
        Compute the histogram-based loss between real and fake data.

        Args:
            x_fake (torch.Tensor): Generated data of shape (batch, time, features).

        Returns:
            torch.Tensor: Loss for each feature.
        """
        loss = list()

        def relu(x):
            return x * (x >= 0.).float()

        for i in range(x_fake.shape[2]):
            # Flatten all timesteps for this feature
            x_i = x_fake[:, :, i].reshape(-1, 1)
            loc = self.locs[i].view(1, -1).to(x_fake.device)
            x_i_expanded = x_i.repeat(1, loc.shape[1])
            dist = torch.abs(x_i_expanded - loc)
            counter = (relu(self.deltas[i].to(x_fake.device) / 2. - dist) > 0.).float()
            density = counter.mean(0) / self.deltas[i].to(x_fake.device)
            abs_metric = torch.abs(density - self.densities[i].to(x_fake.device))
            loss.append(torch.mean(abs_metric))
        loss_componentwise = torch.stack(loss)
        return loss_componentwise

    
def calculate_mdd(ori_data, gen_data):
    """
    Calculate Marginal Distribution Distance (MDD) between real and generated data.
    
    Computes histogram-based distance by aggregating over all timesteps for each feature.

    Args:
        ori_data (np.ndarray or torch.Tensor): Real data of shape (batch, time, features).
        gen_data (np.ndarray or torch.Tensor): Generated data of shape (batch, time, features).

    Returns:
        float: MDD value.
    """
    if not torch.is_tensor(ori_data):
        ori_data = torch.tensor(ori_data)
    if not torch.is_tensor(gen_data):
        gen_data = torch.tensor(gen_data)
    
    mdd = (HistoLoss(ori_data, n_bins=50, name='marginal_distribution')(gen_data)).detach().cpu().numpy()
    if mdd.size > 1:
        mdd = mdd.mean()
    return mdd.item() if mdd.size == 1 else float(mdd)

# =======================================
# Mean Distance (MD)
class MeanLoss(Loss):
    """
    Mean Distance (MD) loss.

    Computes the absolute difference in mean between real and generated data.

    Assumes input tensors are of shape (R, l, N), where:
        R: number of samples (batch size)
        l: length of each sample (sequence length)
        N: number of channels/features

    Args:
        x_real (torch.Tensor): Real/original data of shape (R, l, N).
        **kwargs: Additional arguments for Loss.
    """
    def __init__(self, x_real, **kwargs):
        # Compute mean over samples and time, for each channel
        super(MeanLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.mean_real = x_real.mean(dim=(0, 1), keepdim=False)

    def compute(self, x_fake):
        # Compute mean over samples and time, for each channel
        mean_fake = x_fake.mean(dim=(0, 1), keepdim=False)
        # Returns a vector of length N (channels)
        return self.norm_foo(mean_fake - self.mean_real)

def calculate_md(ori_data, gen_data):
    """
    Calculate Mean Distance (MD) between real and generated data.

    Assumes input arrays/tensors are of shape (R, l, N), where:
        R: number of samples (batch size)
        l: length of each sample (sequence length)
        N: number of channels/features

    Args:
        ori_data (np.ndarray or torch.Tensor): Real data of shape (R, l, N).
        gen_data (np.ndarray or torch.Tensor): Generated data of shape (R, l, N).

    Returns:
        float: MD value (scalar).
    """
    ori = to_torch_features_abc(ori_data)
    gen = to_torch_features_abc(gen_data)
    # Align time length if needed
    min_len = min(ori.shape[1], gen.shape[1])
    ori = ori[:, :min_len, :]
    gen = gen[:, :min_len, :]
    mean_loss = MeanLoss(x_real=ori, name='mean')
    md = mean_loss.compute(gen)
    if md.numel() > 1:
        md = md.mean()
    return float(md.detach().cpu().item())

# =======================================
# Standard Deviation Distance (SDD)
class StdLoss(Loss):
    """
    Standard Deviation Distance (SDD) loss.

    Computes the absolute difference in standard deviation between real and generated data.

    Assumes input tensors are of shape (R, l, N), where:
        R: number of samples (batch size)
        l: length of each sample (sequence length)
        N: number of channels/features

    Args:
        x_real (torch.Tensor): Real/original data of shape (R, l, N).
        **kwargs: Additional arguments for Loss.
    """
    def __init__(self, x_real, **kwargs):
        # Compute std over samples and time, for each channel
        super(StdLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.std_real = x_real.std(dim=(0, 1), unbiased=True, keepdim=False)

    def compute(self, x_fake):
        # Compute std over samples and time, for each channel
        std_fake = x_fake.std(dim=(0, 1), unbiased=True, keepdim=False)
        # Returns a vector of length N (channels)
        return self.norm_foo(std_fake - self.std_real)

def calculate_sdd(ori_data, gen_data):
    """
    Calculate Standard Deviation Distance (SDD) between real and generated data.

    Assumes input arrays/tensors are of shape (R, l, N), where:
        R: number of samples (batch size)
        l: length of each sample (sequence length)
        N: number of channels/features

    Args:
        ori_data (np.ndarray or torch.Tensor): Real data of shape (R, l, N).
        gen_data (np.ndarray or torch.Tensor): Generated data of shape (R, l, N).

    Returns:
        float: SDD value (scalar).
    """
    ori = to_torch_features_abc(ori_data)
    gen = to_torch_features_abc(gen_data)
    min_len = min(ori.shape[1], gen.shape[1])
    ori = ori[:, :min_len, :]
    gen = gen[:, :min_len, :]
    std_loss = StdLoss(x_real=ori, name='std')
    sdd = std_loss.compute(gen)
    if sdd.numel() > 1:
        sdd = sdd.mean()
    return float(sdd.detach().cpu().item())

# =======================================
# Autocorrelation Distance (ACD)
class ACFLoss(Loss):
    def __init__(self, x_real, max_lag=64, stationary=True, **kwargs):
        super(ACFLoss, self).__init__(norm_foo=acf_diff, **kwargs)
        self.max_lag = min(max_lag, x_real.shape[1])
        self.stationary = stationary
        if stationary:
            self.acf_real = acf_torch(self.transform(
                x_real), self.max_lag, dim=(0, 1))
        else:
            self.acf_real = non_stationary_acf_torch(self.transform(
                x_real), symmetric=False)  # Divide by 2 because it is symmetric matrix

    def compute(self, x_fake):
        if self.stationary:
            acf_fake = acf_torch(self.transform(x_fake), self.max_lag, dim=(0, 1))
        else:
            acf_fake = non_stationary_acf_torch(self.transform(
                x_fake), symmetric=False)
        return self.norm_foo(acf_fake - self.acf_real.to(x_fake.device))

def calculate_acd(ori_data, gen_data):
    ori = to_torch_features_abc(ori_data)
    gen = to_torch_features_abc(gen_data)
    min_len = min(ori.shape[1], gen.shape[1])
    ori = ori[:, :min_len, :]
    gen = gen[:, :min_len, :]
    acf = ACFLoss(ori, name='auto_correlation', stationary=True)(gen)
    return float(acf.detach().cpu().item())

# =======================================
# SD calculation and utilities functions
class SkewnessLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(SkewnessLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        # Compute skewness over batch and time, keep per feature
        self.skew_real = skew_torch(x_real, dim=(0, 1), dropdims=True)

    def compute(self, x_fake, **kwargs):
        skew_fake = skew_torch(x_fake, dim=(0, 1), dropdims=True)
        return self.norm_foo(skew_fake - self.skew_real)
    
def calculate_sd(ori_data, gen_data):
    ori = to_torch_features_abc(ori_data)
    gen = to_torch_features_abc(gen_data)
    min_len = min(ori.shape[1], gen.shape[1])
    ori = ori[:, :min_len, :]
    gen = gen[:, :min_len, :]
    skewness = SkewnessLoss(x_real=ori, name='skew')
    sd = skewness.compute(gen)
    if sd.numel() > 1:
        sd = sd.mean()
    return float(sd.detach().cpu().item())

# =======================================
# KD calculation and utilities functions
class KurtosisLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(KurtosisLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        # Compute kurtosis over batch and time, keep per feature
        self.kurtosis_real = kurtosis_torch(x_real, dim=(0, 1), dropdims=True)

    def compute(self, x_fake):
        kurtosis_fake = kurtosis_torch(x_fake, dim=(0, 1), dropdims=True)
        return self.norm_foo(kurtosis_fake - self.kurtosis_real)
    
def calculate_kd(ori_data, gen_data):
    ori = to_torch_features_abc(ori_data)
    gen = to_torch_features_abc(gen_data)
    min_len = min(ori.shape[1], gen.shape[1])
    ori = ori[:, :min_len, :]
    gen = gen[:, :min_len, :]
    kurtosis = KurtosisLoss(x_real=ori, name='kurtosis')
    kd = kurtosis.compute(gen)
    if kd.numel() > 1:
        kd = kd.mean()
    return float(kd.detach().cpu().item())


def make_sure_path_exist(path):
    if os.path.isdir(path) and not path.endswith(os.sep):
        dir_path = path
    else:
        # Extract the directory part of the path
        dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)

def visualize_tsne(ori_data, gen_data, result_path, save_file_prefix, max_samples=1000):
    """
    t-SNE visualization following the reference implementation:
    - Averages across feature dimensions for each sequence
    - Runs joint t-SNE on combined original and generated samples
    """

    # Convert to numpy and ensure shape (R, L, N)
    ori_data = to_numpy_features_for_visualization(ori_data)
    gen_data = to_numpy_features_for_visualization(gen_data)

    R_ori, L, N = ori_data.shape
    R_gen, _, _ = gen_data.shape

    # Subsample to reduce computation
    sample_num = min(max_samples, R_ori, R_gen)
    ori_idx = np.random.permutation(R_ori)[:sample_num]
    gen_idx = np.random.permutation(R_gen)[:sample_num]

    # Average across feature dimension (dim = N)
    # Each sample becomes a 1D vector of length L
    ori_avg = np.mean(ori_data[ori_idx, :, :], axis=2)  # shape (sample_num, L)
    gen_avg = np.mean(gen_data[gen_idx, :, :], axis=2)  # shape (sample_num, L)

    # Concatenate for joint t-SNE embedding
    prep_data_final = np.concatenate([ori_avg, gen_avg], axis=0)  # shape (2*sample_num, L)

    # Color labels for plotting
    colors = ["C0"] * sample_num + ["C1"] * sample_num

    # Run t-SNE jointly
    tsne = TSNE(
        n_components=2,
        verbose=1,
        perplexity=40,    # same as reference
        n_iter=300,       # same as reference
        random_state=0
    )
    tsne_results = tsne.fit_transform(prep_data_final)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(
        tsne_results[:sample_num, 0],
        tsne_results[:sample_num, 1],
        c="C0", alpha=0.4, s=8, label="Original"
    )
    ax.scatter(
        tsne_results[sample_num:, 0],
        tsne_results[sample_num:, 1],
        c="C1", alpha=0.4, s=8, label="Generated"
    )

    # Aesthetic cleanup
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for pos in ['top', 'bottom', 'left', 'right']:
        ax.spines[pos].set_visible(False)
    ax.legend(loc='best', fontsize=8)
    ax.set_title('t-SNE Visualization (Averaged Features)')

    # Save
    save_path = os.path.join(result_path, f'tsne_{save_file_prefix}.png')
    make_sure_path_exist(save_path)
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()

def visualize_distribution(ori_data, gen_data, result_path, save_file_name, max_samples=1000):
    # Convert to numpy and ensure 3D shape (batch, time, features)
    ori_data = to_numpy_features_for_visualization(ori_data)
    gen_data = to_numpy_features_for_visualization(gen_data)
    
    # Subsample independently
    ori_sample_num = min(max_samples, len(ori_data))
    gen_sample_num = min(max_samples, len(gen_data))
    
    ori_idx = np.random.permutation(len(ori_data))[:ori_sample_num]
    gen_idx = np.random.permutation(len(gen_data))[:gen_sample_num]
    
    ori_data = ori_data[ori_idx]
    gen_data = gen_data[gen_idx]
    
    # Flatten all data to get marginal distribution over all timesteps and features
    prep_ori = ori_data.flatten()
    prep_gen = gen_data.flatten()
    
    fig, ax = plt.subplots(1,1,figsize=(4,4))
    
    # KDE plots with normalized density
    sns.kdeplot(prep_ori, color='C0', linewidth=2, label='Original', ax=ax, fill=False)
    sns.kdeplot(prep_gen, color='C1', linewidth=2, linestyle='--', label='Generated', ax=ax, fill=False)
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Auto x-limits based on data range
    min_val = min(prep_ori.min(), prep_gen.min())
    max_val = max(prep_ori.max(), prep_gen.max())
    ax.set_xlim(min_val, max_val)
    
    for pos in ['top','right']:
        ax.spines[pos].set_visible(False)
    
    save_path = os.path.join(result_path, 'distribution_'+save_file_name+'.png')
    make_sure_path_exist(save_path)
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()